# Hector SLAM — Local Simulation

This repo demonstrate the workflow of Hector SLAM by driving a virtual robot with the keyboard and watch it build a 2D occupancy map in real time.

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
Motion Update            (1) predict pose from velocity
    ↓
GN Optimize ×2 + 
LiDAR Measurement Score  (2) refine pose on coarse then fine map by looking up the map score against real scan
    ↓
Map Update               (3) integrate scan into occupancy grid (fine map)
    ↓
Down Sampling            (4) sync coarse map from fine map for next iteration
    ↓
Visualization
```

1. Motion Update
   The robot here is a differential drive robot with kinematics below.
   ## Robot model — differential drive
    <table>
    <tr>
    <td>
    <img width="316" height="227" alt="differential drive" src="https://github.com/user-attachments/assets/39bdbe7c-f2ee-44b5-8d32-eb03b0a02256" />
    </td>
    <td>
    
    | Symbol | Meaning |
    |--------|---------|
    | vx | forward velocity (robot frame) |
    | vy | lateral velocity (robot frame) |
    | ω | angular velocity |
    | θ | robot heading |
    
    **State update (Euler integration):**
    <pre>
    x     ← x + (vx·cos(θ) − vy·sin(θ))·dt
    y     ← y + (vx·sin(θ) + vy·cos(θ))·dt
    theta ← theta + ω·dt
    </pre>
    
    </td>
    </tr>
    </table>


3.GN (Gaussian Newton) Optimization

<div align="center">
  <img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/07d97fc8-87f5-46b6-a69b-5b6dec932ed0" />
</div>

The cost function is:

$$F(\mathbf{p}) = \sum_{i=1}^{n} \left[1 - M\left(S_i(\mathbf{p})\right)\right]^2$$

where **p** = (x, y, θ) is the estimated robot pose, S_i(**p**) is the projection function that transforms scan points from robot frame to world coordinates, and M(·) is the occupancy probability from either the coarse or fine map.

By minimizing F, we find the pose where scan points best align with occupied cells in the current map.

The pose is optimized using Gauss-Newton, where the gradient is obtained by taking the partial derivative of F with respect to x, y, and θ:

$$\frac{\partial F}{\partial \mathbf{p}} = \sum_{i=1}^{n} 2\left[1 - M(S_i)\right] \cdot \nabla M(S_i) \cdot \frac{\partial S_i}{\partial \mathbf{p}}$$

where:

$$\frac{\partial S_i}{\partial \mathbf{p}} = \begin{bmatrix} 1 & 0 & -r_i \sin(\theta + \alpha_i) \\ 0 & 1 & r_i \cos(\theta + \alpha_i) \end{bmatrix}$$

4.Map Update
5.Down Sampling

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
