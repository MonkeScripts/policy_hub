# velstand

Walk on a velocity command (vx, vy, yaw) and get back up after a fall, in one policy.

`eval.py` (v1.1) tests, with 256 robots and seed 0:

| test | what | ranked on |
|---|---|---|
| get-up | 10 s from face-down, face-up, crouch; up = trunk within 20 mm of the robot's own standing height, tilt < 20 deg, for the whole last second | success (face-down, face-up), face-down time |
| tracking | from standstill: yaw +-0.3, +-1.0 rad/s; vx 0.1 / 0.2 / 0.3 m/s; mean of the last 2 s | yaw +-0.3, vx 0.2 (closer to the command is better) |
| standing | zero command 6 s | - |
| walking | stand 3 s, vx 0.2 for 2 s, stand 3 s: nominal, low supply 4.5-5.5 V, low friction 0.15-0.3 | falls at low friction, falls at low supply |
| pushes | 0.3 m/s pushes every 2-3 s for 20 s, walking at vx 0.3 and standing | share up at the end, walking |

Falls = trunk tilt over 45 deg. Every condition is fixed in `eval.py`, not taken from the task;
changing one means a new VERSION, and versions are not compared with each other.
