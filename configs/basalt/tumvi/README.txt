TUM-VI (512 px) calibration and config for the Basalt port
==========================================================

Both JSON files are upstream Basalt's, copied unmodified from the commit this
port pins (pipelines/basalt/PROVENANCE.md):

  https://gitlab.com/VladyslavUsenko/basalt/-/raw/0f3b2b52c807f70ff4e2973ce253c73329eea7bc/data/

  tumvi_512_ds_calib.json  sha256 6286c43a43b2cce8f4c3974765106a8da13c357b888d4a5c44f085616e96a04c
  tumvi_512_config.json    sha256 3b4d22e20da0f19ada6af656ca29a931f010bd7eb2daf36fbe4b4e9a6b8c2b3d

Upstream's data/euroc_ds_calib.json at that commit is byte-identical to
benchmarks/basalt/release_inputs/euroc_ds_calib.json, and its euroc_config.json
has the same values as configs/basalt/euroc_config.json.

tumvi_512_ds_calib.json is Basalt's Double Sphere fit of the TUM-VI 512 px
stereo rig with its IMU noise model. tumvi_512_config.json differs from
euroc_config.json in four values: optical_flow_detection_grid_size 40 (EuRoC
50), vio_use_lm and mapper_use_lm false, vio_lm_lambda_initial 1e-7.

They are used by the "EuRoC mono scale" workflow with dataset tumvi
(.github/workflows/euroc_mono_scale.yml) on the room sequences fetched by
scripts/tumvi_fetch.py, whose mav0/mocap0/data.csv is the ground truth.

License: BSD 3-Clause, Copyright (c) 2019, Vladyslav Usenko and Nikolaus
Demmel; the full text is in LICENSE.basalt next to this file.
