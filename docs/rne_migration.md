# Moving the street simulator to Robot Native Engine

Status: done (P0 to P4, below). The ROS 2 simulator is frozen as the Nav2 reference. This plan moves the 3DGS street simulator
(`ros2/visloc-ros2/gs/`: `gs_sim.py`, `people.py`, `kerbs.py`,
`chase_video.py`, and Nav2) into Robot Native Engine (RNE). visloc-rs keeps the
reconstruction and the localization.

## Why

The simulator has outgrown the ROS 2 package it lives in. `people.py` alone is
about 1,500 lines of crowd, car, crossing and pavement logic. A run is one
real-time ROS 2 launch of about 40 minutes, and nothing in it is replayable
bit for bit. RNE already has most of what the simulator needs, in Rust,
deterministic and headless:

| Needed | In visloc-rs today | Already in RNE |
| --- | --- | --- |
| Splat world | `gsplat_render_server` subprocess | `rne_render_3dgs` (`*.rne.splat.toml`; `rotation_xyzw` turns a z-up splat into RNE's y-up) |
| Moving things in the splat | people and cars regenerated as gaussians each frame (`people.py`) | splat actors: `from_environment_with_actors`, `set_actor_pose` (rigid, depth-sorted with the scene) |
| Car models | lofted bodies in `people.py` | `mesh_to_actor.py` (glTF to flat gaussians, baked light) |
| Planning and control | Nav2 (`gz/nav2_params.yaml`) | `rne_nav`: `OccupancyGrid`, costmap layers, `plan_path`, `DwaPlanner`, pure pursuit, diff drive, TF tree |
| Camera-only closed loop | `visloc_map_localize_node` | example 161 drives KITTI by visloc-rs fixes (subprocess per fix) |
| Traffic light | a timer in `people.py` | `rne_traffic` signals |
| Determinism, CI | none | `SimClock`, seeded `WorldRandom`, replay hash tests, headless CI |

## Boundary

- **visloc-rs** keeps everything from video to world: `real2sim.py` (SfM,
  levelling, 3DGS, the localization map), and the analysis of the splat that
  needs its colours: `lane_map.py` (lane, detours, crossings), `kerbs.find_kerbs`,
  `people.estimate_road`, `check_road`, `fit_road_lines`, `pavement_edges`,
  `walk_offsets`, `kerb_spans`. It writes them once, as a **street bundle**
  (below), and does no simulation.
- **RNE** gets a new example package (like `examples/160_kitti_splat_drive`):
  it loads a street bundle and simulates the robot, the crowd, the cars and the
  light, with navigation by `rne_nav`. It does not depend on visloc-rs crates.
  Localization runs as a visloc-rs subprocess, as in example 161. Nothing goes
  into `rne_core` or other core crates, except one small renderer addition
  (P3).
- The ROS 2 path (`gs_sim.py` with Nav2) stays until the RNE example reaches
  parity. It is then frozen as the ROS 2 / Nav2 reference rather than deleted.

## The street bundle (visloc-rs writes, RNE reads)

One directory, all plain files (no NumPy):

```text
street/
  street.rne.splat.toml   splat manifest: ply_path, rotation_xyzw (z-up to y-up)
  scene.ply               the splat (or a path to it)
  street.json             start, goals, walk, kerbs, road lines, pavement edges,
                          kerb spans, crossing position, robot radius, camera
  ground.f32 + ground.json  ground height grid (x0, y0, resolution, w, h)
  lane.map.json           rne_nav OccupancyGrid (map_io JSON) of the lane map
  map/                    the visloc-rs localization map
  CREDIT.txt              source footage and licence (OverMaps360, CC BY-NC 4.0)
```

`real2sim.py --export-rne DIR` writes it from the run it just built. All
coordinates stay in visloc-rs's z-up map frame. Only the manifest's rotation
and one conversion function in the RNE example change frames, so a pose can
be compared across the two simulators.

## Phases

Each phase ends with a headless test in RNE, and a short docs entry.

### P0: the bundle (visloc-rs)

- `real2sim.py --export-rne`: the files above, from the current street run.
- The splat analysis moves from `gs_sim.py`'s start-up into the export, so it
  runs once per street, not once per run.
- Exit: an export of the OverMaps360 street; a Python test reads it back and
  matches `scene_lane5.json` and the twelfth run's `people.json` (walk
  offsets, pavement edges, kerb spans, road lines).

### P1: robot on ground truth (RNE)

- New package `examples/163_street_splat_nav`, which loads the bundle.
- A diff-drive delivery robot (0.87 × 0.70 m), its pose from ground truth.
- `rne_nav` plans on the lane map with a 0.45 m footprint and follows the path
  with DWA.
- The robot rides on the ground grid and the kerbs (the 10 cm drop, pitch from
  front and rear wheels, 0.12 m/s while a wheel is on a kerb edge: a port of
  `kerbs.Kerbs`).
- Front camera and chase view rendered by `rne_render_3dgs`. The robot is drawn
  as a splat actor made from the current robot model.
- Exit:
  - a headless test reaches the 4 goals with no contact;
  - time on kerbs and speed there are as in the Nav2 runs;
  - two runs give the same world hash.

### P2: camera-only closed loop (RNE + visloc-rs)

- The robot steers on its estimate: wheel odometry with the current error
  model (scale, yaw bias) between fixes, each fix from a visloc-rs subprocess
  on the rendered front camera (example 161's pattern).
- One long-lived localizer process for the run, not one per fix: about 800
  fixes per run make start-up cost matter. visloc-rs would add a small
  stdin/stdout server mode to `visloc_map_localize_node`'s core, or a new demo
  binary.
- Exit: 4/4 goals on visloc fixes, pose RMSE in the range of the Nav2 runs
  (4-9 cm). Documented with the same table as `docs/navigation.md`.

### P3: the crowd and the cars (RNE)

- Port `people.Crowd` to Rust in the example package's library: walkers,
  cyclists, dog walkers, strollers, cars; the walk, pavement edges, kerb
  spans; the crossing with `rne_traffic`'s signal; footprints, squeeze and
  deadlock rules. It is pure logic, so it is headless and unit-testable.
- Cars become rigid splat actors (`mesh_to_actor.py` from the three body
  styles, or glTF samples), placed with `set_actor_pose`.
- People bend at the knees and swing their arms, which a rigid actor cannot
  do. Add `set_actor_gaussians(queue, index, &[Gaussian])` to `rne_render_3dgs`.
  It rewrites an actor's range in place, with the count fixed (as the
  simulator's fixed slots per kind are now). This is the one core addition.
- The crowd's figures (`people.Figure`) are ported or exported. Simplest
  first step: bake each kind's walk cycle into N key poses of gaussians in
  visloc-rs, and have RNE pick and place them. That needs no per-frame figure
  code in Rust.
- People and cars reach the robot's planner as obstacles. Either use a scan
  from proxy depth (`splat_proxy_depth_from_gaussians` with the actors
  included), or, first, their footprints into an `rne_nav` costmap layer.
- Exit: the offline checks of `docs/navigation.md` as Rust tests over seeds:
  - nobody on the road except on the crossing;
  - nobody more than 5 cm into a car;
  - nobody within 0.10 m of the robot's body;
  - nobody still longer than about 25 s;
  - between the kerbs, nobody more than 0.91 m from the walk.

### P4: media, docs, and retiring the ROS 2 path

- Chase video and GIFs from RNE, with the credit in every frame.
- Docs: RNE gets `docs/STREET_SPLAT_NAV.md`; visloc-rs `docs/navigation.md`
  points to it and keeps the Nav2 runs as the reference.
- Exit: RNE reproduces the current README GIFs' content, so the ROS 2
  simulator can be frozen.

## Open questions

- **Licence of the street.** The footage is CC BY-NC 4.0. The bundle is not
  committed to either repository, and RNE's CI uses a small stand-in (as
  `tsukuba_confirmation_fixture.ply` does).
- **Renderer parity.** Example 161 found visloc-rs maps built from
  `gsplat_render_server` renders match RNE's renders less well (KITTI segment
  b: 22 of 81 fixes against 34 of 81 with maps coloured by RNE's renderer).
  P2 may need a localization map rendered by RNE, as 161 does
  (`--render-poses`).
- **Speed.** RNE's splat renderer runs on CPU lavapipe in CI. A 400 s run at
  2 camera frames per second is about 800 renders plus the chase view. Check
  this in P1 before committing to per-frame actor uploads in P3.
- **Who drives.** RNE's `rne_nav` replaces Nav2 here. A Nav2 comparison stays
  possible through RNE's ROS 2 adapter (`adapters/ros2`) later. It is not part
  of this plan.

## Progress

- **P0 done.** `rne_bundle.py` (and `real2sim.py --export-rne DIR`) writes the
  bundle, and `rne_bundle_test.py` checks it. The street's road and pavement
  come from a run's `people.json` (`--people-json`), or, without one, from
  the splat in the export itself (`analyze_street`, written to the bundle's
  `people.json` for `rne_crowd.py`).
- **P1 done**, in RNE `examples/163_street_splat_nav` (`docs/STREET_SPLAT_NAV.md`
  there). On the OverMaps360 street, on ground truth:
  - 4/4 goals over 111 m in 284 s, with no contacts.
  - 9.9 s on the kerbs at no more than 0.12 m/s.
  - At least 0.61 m from any solid cell.
  - 0.30 m median and 1.03 m largest distance from the walk.
  - Headless the drive takes 2 s of wall time. With a front camera and a
    chase view each simulated second, it takes 7 minutes on CPU lavapipe.
  - The trajectory hash is the same with and without rendering.
  - The robot is still a box in the chase view.
- **P2 done.** visloc-rs `visloc_map_localize_server` (in `ros2/visloc-ros2`)
  is `visloc_map_localize_node`'s core behind stdin and stdout. It loads the
  map once and answers one image per line, with the estimate as the prior.
  RNE's example 163 runs it for the whole drive with `--localizer BIN`:
  - Wheel odometry with the simulator's errors, a fix every 0.5 s on the
    front camera RNE renders. Fixes are gated at 0.25 m (+0.1 m/s since the
    last one) and blended in halfway.
  - 4/4 goals with no contacts in both runs. 529 of 572 and 538 of 575
    fixes. Pose error RMSE 8.6 and 5.5 cm, median 2.0 cm. Nav2's thirteenth
    run: 8.9 / 2.3 cm.
  - The map was built from `gsplat_render_server` renders; RNE's renders
    still localize (97 % of frames with the prior).
  - Runs with fixes are not bit-for-bit repeatable. The localizer is
    deterministic on the same images, but RNE's splat renders on CPU
    lavapipe vary slightly between runs.
- **P3 done.** `rne_crowd.py` writes the bundle's `crowd.json` (the start of
  `people.Crowd` with the run's seed) and `crowd/*.ply` (every agent at 12
  gait phases, the crossing at each light state). RNE's `crowd.rs` ports
  `Crowd.step`, and `replace_actor_asset` (new in `rne_render_3dgs`) swaps
  the gait phase in place, so the crowd adds 41,000 gaussians. The robot
  bends its command round people with `rne_nav::avoid_velocities`. Over 4
  seeds on ground truth:
  - 4/4 goals with no contacts with walls or people.
  - Nobody on the road off the crossing.
  - People at least 6 cm clear of the robot's body, and none in a car.
  - The longest anyone stood still: 27 to 45 s.
  - Runs take 470 to 536 s, against 284 s without the crowd. The robot
    waits by a sign post when a cyclist and a stroller come the other way:
    it has no recoveries (backing up, turning round) as Nav2 has.
- **P4 done.**
  - Media: example 163's chase GIF and video, with the crowd, the delivery
    robot drawn from the bundle's `robot.ply`, and the credit in every
    frame (RNE `docs/media/street-splat-nav.gif`, `.mp4`).
  - Docs: RNE `docs/STREET_SPLAT_NAV.md`. Here, `docs/navigation.md`
    points to it, and this plan records the results.
  - The ROS 2 simulator (`gs_sim.py`, `people.py`, `chase_video.py`, with
    Nav2) is frozen, not deleted. It stays as the Nav2 reference, and its
    runs are the numbers RNE's are compared with.
  - Still open: the crowd with camera-only localization, and recoveries for
    the robot when people hold it up.
