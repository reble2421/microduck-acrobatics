# Policy comparison (2026-10-08 09:15)

role=headstand, 5 trials per cell, dr=train, delay=3, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 3/5 (23-88%) | 0.74 | 1.076 | 0.78 | 1.094 | 0.26 | 1.62 |
| job3_HeadSpinRobust_14997.onnx | 5/5 (57-100%) | 1.334 | 1.312 | 1.108 | 1.096 | 0.1 | 0.9 |
