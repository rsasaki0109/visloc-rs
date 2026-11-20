//! Occupancy grids built from the simulated apartment's mapping run, and
//! the map_server round trip.

use visloc_navigation::sim::{
    apartment_mapping_route, build_map, CameraRig, MappingConfig, MappingRun, World,
};
use visloc_navigation::{
    floor_z_from_cameras, occupancy_from_visual_map, GridGeometry, OccupancyConfig, OccupancyGrid,
    OccupancyMapper, FREE, OCCUPIED, UNKNOWN,
};

fn mapping_run() -> (World, MappingRun) {
    let world = World::apartment(1);
    let rig = CameraRig::forward(0.5);
    let run = build_map(
        &world,
        &rig,
        &apartment_mapping_route(),
        &MappingConfig::default(),
    );
    (world, run)
}

fn cells_in(grid: &OccupancyGrid, x0: f64, y0: f64, x1: f64, y1: f64) -> Vec<i8> {
    let r = grid.geometry.resolution;
    let mut values = Vec::new();
    let mut y = y0;
    while y <= y1 {
        let mut x = x0;
        while x <= x1 {
            values.push(grid.value_at(x, y).unwrap_or(UNKNOWN));
            x += r;
        }
        y += r;
    }
    values
}

#[test]
fn apartment_grid_has_walls_rooms_doorways_and_furniture() {
    let (world, run) = mapping_run();
    assert_eq!(world.walls.len(), 29);
    assert!(run.map.landmarks.len() > 800, "{}", run.map.landmarks.len());
    let floor = floor_z_from_cameras(&run.map, 0.5).expect("posed keyframes");
    assert!(floor.abs() < 0.01, "floor {floor}");

    let (grid, stats) =
        occupancy_from_visual_map(&run.map, OccupancyConfig::default(), 0.2, 0.5).unwrap();
    assert!(
        stats.rays > 5000 && stats.landmarks_out_of_band > 0,
        "{stats:?}"
    );
    let counts = grid.counts();
    assert!(counts.free > 15_000 && counts.occupied > 300, "{counts:?}");

    // Room interiors and doorways are free.
    for (x, y) in [
        (4.0, 1.0),
        (8.0, 3.0),
        (3.0, 6.0),
        (8.0, 6.0),
        (6.0, 2.0),
        (2.5, 4.0),
        (8.5, 5.0),
    ] {
        assert_eq!(grid.value_at(x + 0.01, y + 0.01), Some(FREE), "({x}, {y})");
    }
    // Textured walls show up as occupied cells along their length.
    let wall = cells_in(&grid, 5.9, 0.0, 6.1, 1.5);
    assert!(wall.iter().filter(|&&v| v == OCCUPIED).count() >= 8);
    // The low sofa (0.8 m, under the camera's line of sight over it) is not
    // free inside, though rays pass above it.
    assert!(cells_in(&grid, 1.2, 0.6, 2.8, 1.0)
        .iter()
        .all(|&v| v != FREE));
    // Outside the apartment is never free; the bare (textureless) hallway
    // wall is not seen through.
    assert!(cells_in(&grid, -0.5, -0.5, 10.5, -0.1)
        .iter()
        .all(|&v| v != FREE));
    let bare = cells_in(&grid, 2.5, 4.97, 7.5, 4.97);
    let bare_above = cells_in(&grid, 2.5, 5.03, 7.5, 5.03);
    assert!(
        bare.iter()
            .zip(&bare_above)
            .all(|(a, b)| *a != FREE || *b != FREE),
        "the bare wall must not have a free crossing"
    );
}

#[test]
fn rays_above_a_low_obstacle_keep_it_occupied() {
    let mut mapper = OccupancyMapper::new(OccupancyConfig::default());
    let camera = nalgebra::Point3::new(0.0, 0.0, 0.5);
    for i in 0..20 {
        // Hits on a low box face at x = 1 (z 0.2), rays over it to a wall at x = 3.
        let y = -0.2 + 0.02 * i as f64;
        mapper.insert_ray(&camera, &nalgebra::Point3::new(1.0, y, 0.2));
        mapper.insert_ray(&camera, &nalgebra::Point3::new(3.0, y, 1.2));
        mapper.insert_ray(&camera, &nalgebra::Point3::new(0.7, y, 0.12));
    }
    let grid = mapper.to_grid(0.0).unwrap();
    assert_eq!(grid.value_at(1.01, 0.0), Some(OCCUPIED));
    assert_eq!(grid.value_at(3.01, 0.0), Some(OCCUPIED));
    // Seen through low (the rays to x = 0.7 reach z < 0.35 there): free.
    assert_eq!(grid.value_at(0.5, 0.0), Some(FREE));
    // Behind the box, only its high slices were seen: unknown.
    assert_eq!(grid.value_at(2.0, 0.0), Some(UNKNOWN));
}

#[test]
fn map_server_round_trip() {
    let geometry = GridGeometry {
        resolution: 0.1,
        origin_x: -1.5,
        origin_y: 2.25,
        width: 7,
        height: 4,
    };
    let mut grid = OccupancyGrid::unknown(geometry);
    grid.set(0, 0, OCCUPIED);
    grid.set(3, 1, FREE);
    grid.set(6, 3, FREE);
    let dir = std::env::temp_dir().join(format!("visloc_nav_roundtrip_{}", std::process::id()));
    let yaml = grid.write_map_server(dir.join("map")).unwrap();
    let text = std::fs::read_to_string(&yaml).unwrap();
    assert!(text.contains("image: map.pgm") && text.contains("origin: [-1.5, 2.25, 0.0]"));
    let back = OccupancyGrid::read_map_server(&yaml).unwrap();
    assert_eq!(back, grid);
    // The first PGM row is the top (highest y) row of the grid.
    let pgm = grid.to_pgm();
    let pixels = &pgm[pgm.len() - geometry.len()..];
    assert_eq!(
        pixels[3 * 7],
        0,
        "occupied cell (0, 0) is the last image row's first pixel"
    );
    assert_eq!(pixels[0], 205, "unknown is grey");
    assert_eq!(pixels[6], 254, "free cell (6, 3) is white");
    std::fs::remove_dir_all(dir).ok();
}
