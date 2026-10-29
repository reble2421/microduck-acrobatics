# Policy comparison (2026-10-08 09:15)

role=headstand, 5 trials per cell, dr=train, delay=4, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 5/5 (57-100%) | 1.054 | 1.626 | 0.986 | 1.332 | 0.32 | 0.98 |
| job3_HeadSpinRobust_14997.onnx | 5/5 (57-100%) | 1.53 | 1.518 | 1.25 | 1.2 | 0.04 | 0.7 |
