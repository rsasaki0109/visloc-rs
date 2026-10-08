# ABS_QR provenance

The AOM row ordering and landmark-wise QR/nullspace reduction in
`src/vio/aom/` are a clean-room contract for the upstream Basalt fixed SHA
`0f3b2b52c807f70ff4e2973ce253c73329eea7bc`.  The navigation block is
`pose6, velocity3, gyro-bias3, accel-bias3`; each landmark is eliminated by
QR projection before assembling the reduced normal system.  This module is a
numeric fixture layer only and does not depend on the repository bundle
optimizer.
