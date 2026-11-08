//! Cut classes or single objects out of a labelled splat.
//!
//! ```text
//! cargo run --release -p visloc-gsplat-core --example gsplat_extract -- \
//!     --ply scene.ply --semantic semantic.ply --classes labels/classes.txt \
//!     [--list] [--select chair:3]... [--out chair.ply] [--splat chair.splat]
//! ```
//!
//! `--semantic` is `gsplat_semantic_cuda`'s `--out` for `--ply`: the same
//! gaussians in the same order, with a `class` and, from `--instances`, an
//! `instance` per gaussian. `--list` prints each class's objects, numbered
//! from 1 in instance order, with their gaussians, centre and size.
//!
//! `--select <class>` keeps a whole class and `--select <class>:<n>` its
//! n-th object; repeat it for more. The kept gaussians, in their colours
//! from `--ply`, go to `--out` (with the `class` and `instance` properties)
//! and to `--splat` (antimatter15's `.splat`, largest first) for the web
//! viewer: `docs/euroc_splat/index.html?url=chair.splat`.

use std::path::PathBuf;

use visloc_gsplat_core::gaussian::Scene;
use visloc_gsplat_core::ply::{ply_properties, save_ply_with, scene_from_ply};
use visloc_gsplat_core::semantic::parse_classes;
use visloc_gsplat_core::splat::{importance_order, save_splat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut ply: Option<PathBuf> = None;
    let mut semantic: Option<PathBuf> = None;
    let mut classes: Option<PathBuf> = None;
    let mut list = false;
    let mut select: Vec<String> = Vec::new();
    let mut out: Option<PathBuf> = None;
    let mut splat: Option<PathBuf> = None;
    while let Some(a) = args.next() {
        let mut next = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--ply" => ply = Some(PathBuf::from(next()?)),
            "--semantic" => semantic = Some(PathBuf::from(next()?)),
            "--classes" => classes = Some(PathBuf::from(next()?)),
            "--list" => list = true,
            "--select" => select.push(next()?),
            "--out" => out = Some(PathBuf::from(next()?)),
            "--splat" => splat = Some(PathBuf::from(next()?)),
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    let ply = ply.ok_or("--ply <scene.ply> is required")?;
    let semantic = semantic.ok_or("--semantic <semantic.ply> is required")?;
    let classes = classes.ok_or("--classes <classes.txt> is required")?;
    if !list && (select.is_empty() || (out.is_none() && splat.is_none())) {
        return Err("give --list, or --select with --out or --splat".into());
    }

    let classes = parse_classes(&std::fs::read_to_string(&classes)?)?;
    let scene = scene_from_ply(&std::fs::read(&ply)?)?;
    let n = scene.len();
    let [class, instance] = <[Option<Vec<f32>>; 2]>::try_from(ply_properties(
        &std::fs::read(&semantic)?,
        &["class", "instance"],
    )?)
    .map_err(|_| "ply_properties returned the wrong number of columns")?;
    let class = class.ok_or(format!(
        "{} has no class (gsplat_semantic_cuda's --out)",
        semantic.display()
    ))?;
    if class.len() != n {
        return Err(format!(
            "{} has {} gaussians, {} has {n}",
            semantic.display(),
            class.len(),
            ply.display()
        )
        .into());
    }
    let class: Vec<usize> = class.iter().map(|&c| c as usize).collect();
    // Each gaussian's object, if --instances gave it one.
    let instance: Vec<Option<usize>> = instance
        .map(|v| {
            v.iter()
                .map(|&i| (i >= 0.0).then_some(i as usize))
                .collect()
        })
        .unwrap_or_else(|| vec![None; n]);

    // The objects in instance order: their class and gaussians.
    let count = instance.iter().flatten().max().map_or(0, |&m| m + 1);
    let mut objects: Vec<(usize, Vec<usize>)> = vec![(usize::MAX, Vec::new()); count];
    for (g, i) in instance.iter().enumerate() {
        if let Some(i) = *i {
            objects[i].0 = class[g];
            objects[i].1.push(g);
        }
    }
    // Per class its objects, numbered from 1.
    let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); classes.len()];
    for (i, (c, members)) in objects.iter().enumerate() {
        if !members.is_empty() && *c < classes.len() {
            by_class[*c].push(i);
        }
    }

    if list {
        for (c, cls) in classes.iter().enumerate() {
            let gaussians = class.iter().filter(|&&k| k == c).count();
            if gaussians == 0 {
                continue;
            }
            println!(
                "{}: {gaussians} gaussians, {} objects",
                cls.name,
                by_class[c].len()
            );
            for (number, &i) in by_class[c].iter().enumerate() {
                let (centre, size) = extent(&scene, &objects[i].1);
                println!(
                    "  {}:{:<4} {:>7} gaussians  centre ({:.2}, {:.2}, {:.2})  size {:.2} x {:.2} x {:.2}",
                    cls.name,
                    number + 1,
                    objects[i].1.len(),
                    centre[0],
                    centre[1],
                    centre[2],
                    size[0],
                    size[1],
                    size[2]
                );
            }
        }
    }
    if select.is_empty() {
        return Ok(());
    }

    let mut keep = vec![false; n];
    for spec in &select {
        let (name, number) = match spec.rsplit_once(':') {
            Some((name, number)) => (name, Some(number.parse::<usize>()?)),
            None => (spec.as_str(), None),
        };
        let c = classes
            .iter()
            .position(|cls| cls.name == name)
            .ok_or(format!("--select {spec}: no class {name}"))?;
        match number {
            None => (0..n)
                .filter(|&g| class[g] == c)
                .for_each(|g| keep[g] = true),
            Some(number) => {
                let &i = by_class[c].get(number.wrapping_sub(1)).ok_or(format!(
                    "--select {spec}: {name}'s {} objects are numbered from 1",
                    by_class[c].len()
                ))?;
                objects[i].1.iter().for_each(|&g| keep[g] = true);
            }
        }
    }
    let kept: Vec<usize> = (0..n).filter(|&g| keep[g]).collect();
    let part = Scene::new(
        kept.iter().map(|&g| scene.gaussians[g].clone()).collect(),
        scene.sh_degree,
    );
    println!("{} of {n} gaussians kept", part.len());
    if let Some(path) = &out {
        let kept_class: Vec<f32> = kept.iter().map(|&g| class[g] as f32).collect();
        let kept_instance: Vec<f32> = kept
            .iter()
            .map(|&g| instance[g].map_or(-1.0, |i| i as f32))
            .collect();
        save_ply_with(
            &part,
            &[("class", &kept_class), ("instance", &kept_instance)],
            path,
        )?;
        println!("wrote {}", path.display());
    }
    if let Some(path) = &splat {
        let ordered = Scene::new(
            importance_order(&part, part.len())
                .iter()
                .map(|&g| part.gaussians[g].clone())
                .collect(),
            part.sh_degree,
        );
        save_splat(&ordered, path)?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// The centre (mean of the means) of some gaussians and the size of the box
/// that holds 90% of their means along each axis (5th to 95th percentile).
fn extent(scene: &Scene, members: &[usize]) -> ([f32; 3], [f32; 3]) {
    let mut centre = [0.0f32; 3];
    let mut size = [0.0f32; 3];
    for k in 0..3 {
        let mut v: Vec<f32> = members
            .iter()
            .map(|&g| scene.gaussians[g].mean[k])
            .collect();
        centre[k] = v.iter().sum::<f32>() / v.len().max(1) as f32;
        v.sort_by(f32::total_cmp);
        let at = |q: f32| v[((v.len() - 1) as f32 * q).round() as usize];
        size[k] = at(0.95) - at(0.05);
    }
    (centre, size)
}
