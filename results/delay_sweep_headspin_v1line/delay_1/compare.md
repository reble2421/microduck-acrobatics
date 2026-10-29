# Policy comparison (2026-10-08 09:14)

role=headstand, 5 trials per cell, dr=train, delay=1, seed=0. Success column: successes/trials (95% Wilson CI).
Metrics are means over trials (ratio = achieved / commanded speed).

| policy | headspin | headspin.right2_ratio | headspin.left2_ratio | headspin.right3_ratio | headspin.left3_ratio | headspin.stop_deg_s | headspin.right3_drift_mm_s |
|---|---|---|---|---|---|---|---|
| job2_HeadSpin_12998.onnx | 0/5 (0-43%) | 0.156 | 0.154 | 0.19 | 0.2 | 0.38 | 3.24 |
| job3_HeadSpinRobust_14997.onnx | 0/5 (0-43%) | 0.56 | 0.754 | 0.584 | 0.74 | 0.06 | 0.88 |
