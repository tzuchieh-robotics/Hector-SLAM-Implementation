import numpy as np
from config.params import *

class OccupancyGrid:
    """2D Occupancy Grid with log-odds representation"""

    def __init__(self, resolution: float, origin: tuple = (0.0, 0.0),
                 initial_size: int = 200):
        self.resolution = resolution
        self.origin_x, self.origin_y = origin
        self.width = initial_size
        self.height = initial_size

        # Initialize with NaN (unknown)
        self.data = np.full((self.height, self.width), 0.0, dtype=np.float32)

    def world_to_grid(self, wx: float, wy: float) -> tuple:
        """Convert world coordinates to grid indices"""
        gx = int((wx - self.origin_x) / self.resolution)
        gy = int((wy - self.origin_y) / self.resolution)
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> tuple:
        """Convert grid indices to world coordinates"""
        wx = self.origin_x + gx * self.resolution
        wy = self.origin_y + gy * self.resolution
        return wx, wy

    def _expand_if_needed(self, gx: int, gy: int):
        """Expand grid if point is out of bounds"""
        if gx < 0 or gx >= self.width or gy < 0 or gy >= self.height:
            new_width = max(self.width, int(abs(gx) * 1.5) + 50)
            new_height = max(self.height, int(abs(gy) * 1.5) + 50)

            old_data = self.data
            self.data = np.zeros((new_height, new_width), dtype=np.float32)

            offset_x = (new_width - self.width) // 2
            offset_y = (new_height - self.height) // 2
            self.data[offset_y:offset_y+self.height, offset_x:offset_x+self.width] = old_data

            self.origin_x -= offset_x * self.resolution
            self.origin_y -= offset_y * self.resolution
            self.width = new_width
            self.height = new_height

    def get_value(self, wx: float, wy: float, default: float = 0.0) -> float:
        """Get log-odds value with bilinear interpolation"""
        # 轉到格子座標（浮點數）
        gx = (wx - self.origin_x) / self.resolution - 0.5
        gy = (wy - self.origin_y) / self.resolution - 0.5

        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        x1, y1 = x0 + 1, y0 + 1

        # 邊界檢查
        if x0 < 0 or y0 < 0 or x1 >= self.width or y1 >= self.height:
            return default

        # 四個角的值
        q00 = self.data[y0, x0]
        q10 = self.data[y0, x1]
        q01 = self.data[y1, x0]
        q11 = self.data[y1, x1]

        # 插值權重
        tx = gx - x0
        ty = gy - y0

        return (q00 * (1-tx) * (1-ty) +
                q10 *    tx  * (1-ty) +
                q01 * (1-tx) *    ty  +
                q11 *    tx  *    ty)

    def set_value(self, wx: float, wy: float, value: float):
        """Set log-odds value at world position"""
        gx, gy = self.world_to_grid(wx, wy)
        self._expand_if_needed(gx, gy)
        gx, gy = self.world_to_grid(wx, wy)
        self.data[gy, gx] = np.clip(value, LOG_ODDS_MIN, LOG_ODDS_MAX)

    def add_value(self, wx: float, wy: float, delta: float):
        """Add to log-odds value (for updating)"""
        current = self.get_value(wx, wy)
        new_value = current + delta
        self.set_value(wx, wy, new_value)

    def to_probability(self, log_odds: float) -> float:
        """Convert log-odds to probability [0, 1]"""
        return 1.0 - 1.0 / (1.0 + np.exp(log_odds))

    def get_probability(self, wx: float, wy: float) -> float:
        """Get occupancy probability [0, 1]"""
        log_odds = self.get_value(wx, wy)
        return self.to_probability(log_odds)

    def downsample(self) -> 'OccupancyGrid':
        """Create a 2x downsampled version"""
        downsampled = OccupancyGrid(
            resolution=self.resolution * 2,
            origin=(self.origin_x, self.origin_y)
        )
        downsampled.width = (self.width + 1) // 2
        downsampled.height = (self.height + 1) // 2
        downsampled.data = np.zeros((downsampled.height, downsampled.width), dtype=np.float32)

        # Simple average downsampling
        for y in range(downsampled.height):
            for x in range(downsampled.width):
                cells = []
                for dy in range(2):
                    for dx in range(2):
                        py, px = y * 2 + dy, x * 2 + dx
                        if py < self.height and px < self.width:
                            cells.append(self.data[py, px])
                if cells:
                    downsampled.data[y, x] = np.mean(cells)

        return downsampled

    def to_json(self) -> dict:
        """Convert to JSON-serializable format"""
        return {
            'resolution': float(self.resolution),
            'origin': [float(self.origin_x), float(self.origin_y)],
            'width': int(self.width),
            'height': int(self.height),
            'data': self.data.tolist()
        }

    @staticmethod
    def from_json(data: dict) -> 'OccupancyGrid':
        """Create from JSON"""
        grid = OccupancyGrid(
            resolution=data['resolution'],
            origin=tuple(data['origin']),
            initial_size=max(data['width'], data['height'])
        )
        grid.width = data['width']
        grid.height = data['height']
        grid.data = np.array(data['data'], dtype=np.float32)
        return grid
