# Hector SLAM — Local Simulation

A Python implementation of [Hector SLAM](http://www.ares.tu-darmstadt.de/) as an interactive local simulation. Drive a virtual robot with the keyboard and watch it build a 2D occupancy map in real time.

## Demo

```
W/S: forward / back    A/D: strafe    Q/E: rotate    R: reset    ESC: quit
```

The robot scans its environment with a simulated 2D lidar. Your job is to implement the core SLAM functions that turn raw scans into a consistent map.

## Requirements

```bash
pip install -r requirements.txt
# numpy, matplotlib, scipy
```

## Run

```bash
python main.py
```

## Project Structure

```
├── main.py               # Entry point — simulation loop & keyboard control
├── config/
│   └── params.py         # All tunable parameters
├── core/                 # Provided utilities (do not modify)
│   ├── pose.py           # SE(2) pose representation
│   ├── map.py            # Log-odds occupancy grid
│   ├── scan.py           # Laser scan container
│   └── simulator.py      # Simulated 2D lidar
├── student/              # Your implementations
│   ├── motion_update.py
│   ├── ray_casting.py
│   ├── lidar_measurement.py
│   ├── gn_optimize.py
│   ├── map_update.py
│   └── down_sampling.py
└── tests/                # Unit tests for each function
```

## What to Implement

All six files in `student/` have `TODO` markers. Implement them in order:

| # | File | Task |
|---|------|------|
| 1 | `motion_update.py` | Integrate IMU velocity to predict new pose |
| 2 | `ray_casting.py` | Cast rays from robot pose into the occupancy map |
| 3 | `lidar_measurement.py` | Compare real vs. predicted scan to get a likelihood score |
| 4 | `gn_optimize.py` | Gauss-Newton pose correction minimizing scan-to-map error |
| 5 | `map_update.py` | Update occupancy cells with hit/miss log-odds |
| 6 | `down_sampling.py` | 2× downsample fine map → coarse map |

Run the tests to check your work:

```bash
python -m pytest tests/
```

## Algorithm Overview

```
Keyboard Input
    ↓
Motion Update        (1) predict pose from velocity
    ↓
Ray Casting          (2) expected scan from current map
    ↓
Lidar Measurement    (3) score pose against real scan
    ↓
GN Optimize ×2       (4) refine pose on coarse then fine map
    ↓
Map Update           (5) integrate scan into occupancy grid
    ↓
Down Sampling        (6) sync coarse map from fine map
    ↓
Visualization
```

**Two-level refinement**: Gauss-Newton runs on the coarse map first (fast, wide basin), then uses that result to initialize on the fine map (precise).

**Log-odds grid**: Each cell stores log P(occupied) / P(free). Range −2.0 (free) → +3.5 (occupied). Map grows dynamically as the robot explores new areas.

**Update threshold**: Map is only updated when the robot moves > 5 cm or rotates > ~5°, avoiding redundant noisy updates.

## Key Parameters (`config/params.py`)

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `FINE_RESOLUTION` | 0.05 m/cell | Fine map cell size |
| `COARSE_RESOLUTION` | 0.10 m/cell | Coarse map cell size |
| `NUM_RAYS` | 360 | Lidar rays per scan |
| `LASER_RANGE_MAX` | 10.0 m | Max sensor range |
| `HIT_LOG_ODDS` | 0.75 | Evidence weight for occupied |
| `MISS_LOG_ODDS` | −0.30 | Evidence weight for free |

## License

MIT — free for educational use.
