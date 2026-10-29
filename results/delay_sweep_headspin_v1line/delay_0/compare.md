# Policy comparison (2026-10-08 09:14)

role=headstand, 5 trials per cell, dr=train, delay=0, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 0/5 (0-43%) | 0.086 | 0.062 | 0.096 | 0.062 | 0.38 | 1.24 |
| job3_HeadSpinRobust_14997.onnx | 0/5 (0-43%) | 0.272 | 0.404 | 0.444 | 0.426 | 0.12 | 0.54 |
