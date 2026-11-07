//! Semantic classes per gaussian, voted from per-pixel labels.
//!
//! A label image holds one class index per pixel ([`NO_LABEL`]: none). The
//! renderer adds each splat's blending weight at a pixel (`T_i a_i`) to that
//! pixel's class, over every view (`visloc_gsplat_cuda::CudaGsplat::
//! vote_labels`). A gaussian then takes the class with the most weight
//! ([`label_gaussians`]), and [`colour_by_class`] paints it in that class's
//! colour, for a splat to overlay on the RGB one.
//!
//! The classes come from a `classes.txt`, one per line, `index r g b name`,
//! the indices 0, 1, 2, ... in order ([`parse_classes`]).

use crate::gaussian::Scene;
use crate::sh::SH_C0;

/// A label image's "no label" pixel, and a gaussian's class when no labelled
/// pixel saw it.
pub const NO_LABEL: u8 = 255;

/// The colour of a gaussian no labelled pixel saw.
pub const UNSEEN_COLOR: [u8; 3] = [40, 40, 40];

/// A class of the label images: its name and display colour.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticClass {
    pub name: String,
    pub color: [u8; 3],
}

#[derive(Debug, thiserror::Error)]
pub enum SemanticError {
    #[error("classes.txt line {line}: {message}")]
    Classes { line: usize, message: String },
    #[error("no classes")]
    NoClasses,
}

/// Parse a `classes.txt`: per line `index r g b name` (the name may have
/// spaces), indices 0, 1, 2, ... in order, at most 255 classes. Blank lines
/// and lines starting with `#` are skipped.
pub fn parse_classes(text: &str) -> Result<Vec<SemanticClass>, SemanticError> {
    let mut classes = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let err = |message: &str| SemanticError::Classes {
            line: i + 1,
            message: message.to_owned(),
        };
        let mut f = line.split_whitespace();
        let mut num = |what: &str| -> Result<u32, SemanticError> {
            f.next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| err(&format!("expected {what}")))
        };
        let index = num("an index")?;
        if index as usize != classes.len() {
            return Err(err(&format!("index {index}, expected {}", classes.len())));
        }
        let mut color = [0u8; 3];
        for c in &mut color {
            *c = u8::try_from(num("r g b")?).map_err(|_| err("a colour above 255"))?;
        }
        let name = f.collect::<Vec<_>>().join(" ");
        if name.is_empty() {
            return Err(err("expected a name"));
        }
        classes.push(SemanticClass { name, color });
    }
    if classes.is_empty() {
        return Err(SemanticError::NoClasses);
    }
    if classes.len() >= NO_LABEL as usize {
        return Err(SemanticError::Classes {
            line: 0,
            message: format!("{} classes, at most 254", classes.len()),
        });
    }
    Ok(classes)
}

/// A gaussian's class from its votes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianLabel {
    /// The class with the most weight ([`NO_LABEL`] without any).
    pub class: u8,
    /// That class's share of the gaussian's weight (0 without any).
    pub confidence: f32,
    /// The gaussian's weight over all classes and views.
    pub weight: f32,
}

/// Each gaussian's class from `votes`, `num_classes` floats per gaussian
/// (the weight each class's pixels gave it). Ties go to the lower class.
pub fn label_gaussians(votes: &[f32], num_classes: usize) -> Vec<GaussianLabel> {
    votes
        .chunks_exact(num_classes.max(1))
        .map(|v| {
            let weight: f32 = v.iter().sum();
            let (best, top) =
                v.iter().enumerate().fold(
                    (0, 0.0f32),
                    |acc, (k, &w)| if w > acc.1 { (k, w) } else { acc },
                );
            if top > 0.0 {
                GaussianLabel {
                    class: best as u8,
                    confidence: top / weight,
                    weight,
                }
            } else {
                GaussianLabel {
                    class: NO_LABEL,
                    confidence: 0.0,
                    weight: 0.0,
                }
            }
        })
        .collect()
}

/// Each gaussian's class from trained logits, `num_classes` per gaussian: the
/// class of the largest, its softmax probability as the confidence, weight 1.
/// A gaussian whose logits are all zero (no labelled pixel trained them) gets
/// [`NO_LABEL`].
pub fn label_from_logits(logits: &[f32], num_classes: usize) -> Vec<GaussianLabel> {
    logits
        .chunks_exact(num_classes.max(1))
        .map(|z| {
            if z.iter().all(|&v| v == 0.0) {
                return GaussianLabel {
                    class: NO_LABEL,
                    confidence: 0.0,
                    weight: 0.0,
                };
            }
            let top = z.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let p: Vec<f32> = z.iter().map(|&v| (v - top).exp()).collect();
            label_gaussians(&p, num_classes)[0]
        })
        .map(|l| GaussianLabel {
            weight: if l.class == NO_LABEL { 0.0 } else { 1.0 },
            ..l
        })
        .collect()
}

/// `scene` with every gaussian painted its class's colour ([`UNSEEN_COLOR`]
/// without a class): degree-0 SH, the geometry and opacity unchanged.
pub fn colour_by_class(
    scene: &Scene,
    labels: &[GaussianLabel],
    classes: &[SemanticClass],
) -> Scene {
    let colors: Vec<[u8; 3]> = labels
        .iter()
        .map(|l| {
            classes
                .get(l.class as usize)
                .map_or(UNSEEN_COLOR, |c| c.color)
        })
        .collect();
    paint(scene, &colors)
}

/// `scene` with gaussian `i` painted `colors[i]`: degree-0 SH, the geometry
/// and opacity unchanged.
pub fn paint(scene: &Scene, colors: &[[u8; 3]]) -> Scene {
    let dc = |c: [u8; 3]| c.map(|v| (v as f32 / 255.0 - 0.5) / SH_C0);
    let gaussians = scene
        .gaussians
        .iter()
        .zip(colors)
        .map(|(g, &c)| {
            let mut g = g.clone();
            g.sh_dc = dc(c);
            g.sh_rest.clear();
            g.sh_degree = 0;
            g
        })
        .collect();
    Scene::new(gaussians, 0)
}

/// A gaussian's 3D instance when it has none.
pub const NO_INSTANCE: u32 = u32::MAX;

/// One instance of one view's instance image: its class, and its members,
/// the gaussians that gave it most of their blending weight in that view,
/// each with that weight ([`view_instance_members`]).
#[derive(Debug, Clone, PartialEq)]
pub struct ViewInstance {
    pub view: u32,
    pub class: u8,
    pub members: Vec<(u32, f32)>,
}

/// From one view's votes over its instance image (`num + 1` floats per
/// gaussian: 0 for the pixels of no instance, then the instances), each
/// instance's members: the gaussians whose weight in the view goes at least
/// `share` to it.
pub fn view_instance_members(votes: &[f32], num: usize, share: f32) -> Vec<Vec<(u32, f32)>> {
    let mut members = vec![Vec::new(); num];
    for (g, v) in votes.chunks_exact(num + 1).enumerate() {
        let total: f32 = v.iter().sum();
        let (best, top) =
            v.iter().enumerate().fold(
                (0, 0.0f32),
                |acc, (i, &w)| if w > acc.1 { (i, w) } else { acc },
            );
        if best > 0 && top > 0.0 && top >= share * total {
            members[best - 1].push((g as u32, top));
        }
    }
    members
}

/// A 3D instance: view instances of one class merged ([`merge_instances`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Instance {
    pub class: u8,
    /// The views it appears in.
    pub views: usize,
    /// Its gaussians.
    pub gaussians: usize,
}

/// Merge view instances into 3D instances, and give each gaussian one.
///
/// Two instances of one class in different views join when their members'
/// weights, each instance's normalised to sum 1, overlap by at least
/// `min_overlap`: the sum over shared gaussians of the smaller normalised
/// weight (1 for the same weights on the same gaussians, 0 for none shared).
/// Joins chain (union-find). A gaussian then takes the 3D instance its
/// memberships give the most weight, counting only instances of its own
/// class (`gaussian_class`); 3D instances seen in fewer than `min_views`
/// views are dropped. The rest are numbered by size, largest first.
pub fn merge_instances(
    instances: &[ViewInstance],
    gaussian_class: &[u8],
    min_overlap: f32,
    min_views: usize,
) -> (Vec<u32>, Vec<Instance>) {
    let n = gaussian_class.len();
    // Per gaussian, its memberships: (instance, normalised weight, weight).
    let mut by_gaussian: Vec<Vec<(u32, f32, f32)>> = vec![Vec::new(); n];
    for (i, inst) in instances.iter().enumerate() {
        let sum: f32 = inst.members.iter().map(|m| m.1).sum();
        if sum <= 0.0 {
            continue;
        }
        for &(g, w) in &inst.members {
            if let Some(list) = by_gaussian.get_mut(g as usize) {
                list.push((i as u32, w / sum, w));
            }
        }
    }
    let mut overlap: std::collections::HashMap<(u32, u32), f32> = Default::default();
    for list in &by_gaussian {
        for (x, a) in list.iter().enumerate() {
            for b in &list[x + 1..] {
                let (ia, ib) = (&instances[a.0 as usize], &instances[b.0 as usize]);
                if ia.class == ib.class && ia.view != ib.view {
                    *overlap.entry((a.0.min(b.0), a.0.max(b.0))).or_default() += a.1.min(b.1);
                }
            }
        }
    }
    let mut parent: Vec<u32> = (0..instances.len() as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    let mut pairs: Vec<_> = overlap.into_iter().filter(|p| p.1 >= min_overlap).collect();
    pairs.sort_by(|a, b| a.0.cmp(&b.0));
    for ((a, b), _) in pairs {
        let (ra, rb) = (root(&mut parent, a), root(&mut parent, b));
        if ra != rb {
            parent[ra.max(rb) as usize] = ra.min(rb);
        }
    }
    // Each root's views.
    let mut views: std::collections::HashMap<u32, std::collections::BTreeSet<u32>> =
        Default::default();
    for i in 0..instances.len() as u32 {
        let r = root(&mut parent, i);
        views
            .entry(r)
            .or_default()
            .insert(instances[i as usize].view);
    }
    // Each gaussian's root by weight, among instances of its class.
    let mut assigned = vec![NO_INSTANCE; n];
    for (g, list) in by_gaussian.iter().enumerate() {
        let mut weight: Vec<(u32, f32)> = Vec::new();
        for &(i, _, w) in list {
            if instances[i as usize].class != gaussian_class[g] {
                continue;
            }
            let r = root(&mut parent, i);
            match weight.iter_mut().find(|e| e.0 == r) {
                Some(e) => e.1 += w,
                None => weight.push((r, w)),
            }
        }
        if let Some(&(r, _)) = weight
            .iter()
            .filter(|e| views[&e.0].len() >= min_views)
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
        {
            assigned[g] = r;
        }
    }
    // Number the roots by size, largest first (ties by root).
    let mut size: std::collections::HashMap<u32, usize> = Default::default();
    for &r in assigned.iter().filter(|&&r| r != NO_INSTANCE) {
        *size.entry(r).or_default() += 1;
    }
    let mut roots: Vec<(u32, usize)> = size.into_iter().collect();
    roots.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let number: std::collections::HashMap<u32, u32> = roots
        .iter()
        .enumerate()
        .map(|(k, &(r, _))| (r, k as u32))
        .collect();
    for a in assigned.iter_mut().filter(|a| **a != NO_INSTANCE) {
        *a = number[a];
    }
    let found = roots
        .iter()
        .map(|&(r, gaussians)| Instance {
            class: instances[r as usize].class,
            views: views[&r].len(),
            gaussians,
        })
        .collect();
    (assigned, found)
}

/// A colour for 3D instance `i`, far from its neighbours' (golden-ratio hues),
/// [`UNSEEN_COLOR`] for [`NO_INSTANCE`].
pub fn instance_color(i: u32) -> [u8; 3] {
    if i == NO_INSTANCE {
        return UNSEEN_COLOR;
    }
    let h = (i as f32 * 0.618_034).fract() * 6.0;
    let (s, v) = (
        0.75 - 0.25 * ((i / 7) % 2) as f32,
        1.0 - 0.2 * ((i / 3) % 2) as f32,
    );
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r, g, b].map(|t| ((t + m) * 255.0).round() as u8)
}

/// The instance overlay's legend: [`classes_json`] with each class's
/// `count` its number of 3D instances, marked `"objects": true`.
pub fn instances_json(classes: &[SemanticClass], instances_per_class: &[usize]) -> String {
    classes_json(classes, instances_per_class).replacen(
        "{\"classes\"",
        "{\"objects\":true,\"classes\"",
        1,
    )
}

/// The classes as JSON for the web viewer's legend: per class its name,
/// colour and `count` (e.g. gaussians), in index order.
pub fn classes_json(classes: &[SemanticClass], counts: &[usize]) -> String {
    let escape = |s: &str| {
        s.chars()
            .flat_map(|c| match c {
                '"' => vec!['\\', '"'],
                '\\' => vec!['\\', '\\'],
                c if (c as u32) < 0x20 => vec![' '],
                c => vec![c],
            })
            .collect::<String>()
    };
    let rows: Vec<String> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "{{\"index\":{i},\"name\":\"{}\",\"color\":[{},{},{}],\"count\":{}}}",
                escape(&c.name),
                c.color[0],
                c.color[1],
                c.color[2],
                counts.get(i).copied().unwrap_or(0)
            )
        })
        .collect();
    format!("{{\"classes\":[\n{}\n]}}\n", rows.join(",\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gaussian::Gaussian;
    use nalgebra::{Quaternion, Vector3};

    #[test]
    fn classes_parse_in_index_order() {
        let text = "# index r g b name\n0 128 128 128 other\n\n1 230 25 75 coffee table\n";
        let classes = parse_classes(text).unwrap();
        assert_eq!(
            classes,
            vec![
                SemanticClass {
                    name: "other".into(),
                    color: [128, 128, 128]
                },
                SemanticClass {
                    name: "coffee table".into(),
                    color: [230, 25, 75]
                },
            ]
        );
        assert!(parse_classes("1 0 0 0 chair\n").is_err());
        assert!(parse_classes("0 0 300 0 chair\n").is_err());
        assert!(parse_classes("0 0 0 0\n").is_err());
        assert!(parse_classes("# nothing\n").is_err());
    }

    #[test]
    fn a_gaussian_takes_the_class_with_the_most_weight() {
        let votes = [
            0.5, 1.5, 0.0, /**/ 0.0, 0.0, 0.0, /**/ 2.0, 0.0, 2.0,
        ];
        let labels = label_gaussians(&votes, 3);
        assert_eq!(labels[0].class, 1);
        assert!((labels[0].confidence - 0.75).abs() < 1e-6);
        assert!((labels[0].weight - 2.0).abs() < 1e-6);
        assert_eq!(labels[1].class, NO_LABEL);
        assert_eq!(labels[1].confidence, 0.0);
        // A tie goes to the lower class.
        assert_eq!(labels[2].class, 0);
    }

    #[test]
    fn trained_logits_give_their_softmax_class() {
        let logits = [
            0.0, 2.0, 0.0, /**/ 0.0, 0.0, 0.0, /**/ -1.0, -3.0, -2.0,
        ];
        let labels = label_from_logits(&logits, 3);
        assert_eq!(labels[0].class, 1);
        let e = 2.0f32.exp();
        assert!((labels[0].confidence - e / (e + 2.0)).abs() < 1e-6);
        assert_eq!(labels[0].weight, 1.0);
        // Untrained: no class.
        assert_eq!(labels[1].class, NO_LABEL);
        assert_eq!(labels[1].weight, 0.0);
        // All negative still has a largest.
        assert_eq!(labels[2].class, 0);
    }

    #[test]
    fn colouring_keeps_the_geometry_and_paints_the_class() {
        let g = Gaussian {
            mean: Vector3::new(1.0, 2.0, 3.0),
            scale_log: Vector3::new(-1.0, -2.0, -3.0),
            rotation: Quaternion::new(1.0, 0.0, 0.0, 0.0),
            opacity_logit: 0.7,
            sh_dc: [0.1, 0.2, 0.3],
            sh_rest: vec![0.01; 9],
            sh_degree: 1,
        };
        let scene = Scene::new(vec![g.clone(), g], 1);
        let classes = [SemanticClass {
            name: "wall".into(),
            color: [255, 0, 128],
        }];
        let labels = label_gaussians(&[1.0, 0.0], 1);
        let out = colour_by_class(&scene, &labels, &classes);
        assert_eq!(out.sh_degree, 0);
        let rgb = |g: &Gaussian| g.sh_dc.map(|v| v * SH_C0 + 0.5);
        let a = rgb(&out.gaussians[0]);
        assert!(
            (a[0] - 1.0).abs() < 1e-6 && a[1].abs() < 1e-6 && (a[2] - 128.0 / 255.0).abs() < 1e-6
        );
        let b = rgb(&out.gaussians[1]);
        assert!((b[0] - 40.0 / 255.0).abs() < 1e-6);
        assert_eq!(out.gaussians[0].mean, scene.gaussians[0].mean);
        assert_eq!(out.gaussians[0].opacity_logit, 0.7);
        assert!(out.gaussians[0].sh_rest.is_empty());
    }

    #[test]
    fn members_are_the_gaussians_an_instance_holds_most_of() {
        // Per gaussian: none, instance 1, instance 2.
        let votes = [
            0.1, 0.8, 0.1, // instance 1
            0.6, 0.4, 0.0, // mostly no instance
            0.1, 0.3, 0.3, // a tie below the share
            0.0, 0.0, 0.0, // unseen
            0.0, 0.1, 0.5, // instance 2
        ];
        let members = view_instance_members(&votes, 2, 0.5);
        assert_eq!(members, vec![vec![(0, 0.8)], vec![(4, 0.5)]]);
    }

    #[test]
    fn view_instances_of_one_object_merge_across_views() {
        let inst = |view, class, members: &[(u32, f32)]| ViewInstance {
            view,
            class,
            members: members.to_vec(),
        };
        let instances = [
            // A chair in views 0, 1 and 2, its gaussians 0..4 seen in turn.
            inst(0, 2, &[(0, 1.0), (1, 1.0), (2, 1.0)]),
            inst(1, 2, &[(1, 1.0), (2, 1.0), (3, 1.0)]),
            inst(2, 2, &[(2, 1.0), (3, 1.0), (4, 1.0)]),
            // Another chair, apart, in views 0 and 1.
            inst(0, 2, &[(5, 1.0), (6, 1.0)]),
            inst(1, 2, &[(5, 1.0), (6, 1.0)]),
            // A table over the first chair's gaussians, in view 1 only.
            inst(1, 1, &[(2, 0.5), (7, 1.0)]),
            // Two instances of one view never merge, even sharing gaussians.
            inst(2, 2, &[(8, 1.0)]),
            inst(2, 2, &[(8, 1.0)]),
        ];
        let classes = [2, 2, 2, 2, 2, 2, 2, 1, 2];
        let (assigned, found) = merge_instances(&instances, &classes, 0.3, 2);
        // The first chair (5 gaussians) is instance 0, the second 1.
        assert_eq!(&assigned[..7], &[0, 0, 0, 0, 0, 1, 1]);
        // The table and the one-view instances have fewer than 2 views.
        assert_eq!(assigned[7], NO_INSTANCE);
        assert_eq!(assigned[8], NO_INSTANCE);
        assert_eq!(
            found,
            vec![
                Instance {
                    class: 2,
                    views: 3,
                    gaussians: 5
                },
                Instance {
                    class: 2,
                    views: 2,
                    gaussians: 2
                },
            ]
        );
        // Every pair of the first chair overlaps by 1/3 or 2/3: at 0.7 none join.
        let (assigned, _) = merge_instances(&instances, &classes, 0.7, 2);
        assert_eq!(&assigned[..5], &[NO_INSTANCE; 5]);
        assert_eq!(&assigned[5..7], &[0, 0]);
    }

    #[test]
    fn instance_colours_differ_from_their_neighbours() {
        for i in 0..50 {
            let (a, b) = (instance_color(i), instance_color(i + 1));
            let d: i32 = (0..3).map(|c| (a[c] as i32 - b[c] as i32).abs()).sum();
            assert!(d > 60, "{i}: {a:?} {b:?}");
        }
        assert_eq!(instance_color(NO_INSTANCE), UNSEEN_COLOR);
    }

    #[test]
    fn the_legend_escapes_names() {
        let classes = [SemanticClass {
            name: "a \"b\"".into(),
            color: [1, 2, 3],
        }];
        let json = classes_json(&classes, &[7]);
        assert!(
            json.contains(r#"{"index":0,"name":"a \"b\"","color":[1,2,3],"count":7}"#),
            "{json}"
        );
    }
}
