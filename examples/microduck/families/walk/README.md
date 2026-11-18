# walk

Follow a velocity command (vx, vy, yaw) and stay up. A walk policy does not get up after a fall,
so every fall counts against it.

`eval.py` (v1.1) tests, with 256 robots and seed 0, every robot starting standing at HOME:

| test | what | ranked on |
|---|---|---|
| tracking | from standstill (2 s at zero command first): yaw +-0.3, +-1.0 rad/s; vx 0.1 / 0.2 / 0.3 m/s; mean of the last 2 s of 4 s | yaw +-0.3, yaw +1.0, vx 0.2, vx 0.3 (closer to the command is better) |
| standing | zero command 6 s: drift, double support, falls | - |
| walking | stand 3 s, vx 0.2 for 2 s, stand 3 s: nominal, low supply 4.5-5.5 V, low friction 0.15-0.3 | falls at low friction, falls at low supply |
| pushes | 0.3 m/s pushes every 2-3 s for 20 s, walking at vx 0.3 and standing | falls walking, falls standing |

`replay.mp4`: walk vx 0.2 and 0.3, turn yaw +0.3 and -1.0, pushes while walking.

Falls = trunk tilt over 45 deg. Every condition is fixed in `eval.py`, not taken from the task;
changing one means a new VERSION, and versions are not compared with each other.
