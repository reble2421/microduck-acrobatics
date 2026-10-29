# Policy comparison (2026-10-08 09:15)

role=headstand, 5 trials per cell, dr=train, delay=6, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 5/5 (57-100%) | 1.476 | 2.008 | 1.324 | 1.624 | 0.4 | 1.64 |
| job3_HeadSpinRobust_14997.onnx | 5/5 (57-100%) | 1.738 | 1.684 | 1.382 | 1.312 | 0.16 | 1.98 |
