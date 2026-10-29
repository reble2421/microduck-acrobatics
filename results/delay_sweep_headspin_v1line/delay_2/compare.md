# Policy comparison (2026-10-08 09:14)

role=headstand, 5 trials per cell, dr=train, delay=2, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 0/5 (0-43%) | 0.274 | 0.366 | 0.436 | 0.606 | 0.34 | 1.94 |
| job3_HeadSpinRobust_14997.onnx | 5/5 (57-100%) | 0.926 | 1.02 | 0.864 | 0.988 | 0.12 | 0.66 |
