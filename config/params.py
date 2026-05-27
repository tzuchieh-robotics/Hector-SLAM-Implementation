# Global parameters for Hector SLAM system

# ===== Map Configuration =====
FINE_RESOLUTION = 0.05  # meters per cell
COARSE_RESOLUTION = 0.40  # 2x downsampled

MAP_ORIGIN = (-6.0, -1.0)
INITIAL_MAP_SIZE = 400

# Log-odds for occupancy update
HIT_LOG_ODDS = 1.5
MISS_LOG_ODDS = -0.2
LOG_ODDS_MIN = -2.0
LOG_ODDS_MAX = 3.5

# ===== Motion Model =====
KEYBOARD_SCALE = 0.05  # movement per key press (meters)
ROTATION_SCALE = 0.1  # rotation per key press (radians)
IMU_NOISE_LINEAR = 0.001  # meters
IMU_NOISE_ANGULAR = 0.001  # radians

# ===== Laser Configuration =====
NUM_RAYS = 360
LASER_RANGE_MIN = 0.1  # meters
LASER_RANGE_MAX = 10.0  # meters
LASER_FOV = 3.14159 * 2  # full 360 degrees

# ===== Update Thresholds =====
MIN_TRANSLATION_FOR_MAP_UPDATE = 0.05  # meters
MIN_ROTATION_FOR_MAP_UPDATE = 0.087  # radians (~5 degrees)

# ===== Optimization =====
GN_MAX_ITERATIONS = 10
GN_CONVERGENCE_THRESHOLD = 1e-3
COARSE_GN_MAX_ITERATIONS = 10

# ===== Simulation =====
SIMULATOR_MAP_FILE = None  # None = use default simple world
SIMULATOR_UPDATE_RATE = 20  # Hz
