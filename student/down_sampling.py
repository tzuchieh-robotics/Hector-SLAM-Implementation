import numpy as np
from core.map import OccupancyGrid
from config.params import *

def down_sample(fine_map: OccupancyGrid) -> OccupancyGrid:
    """
    Create a 2x downsampled version of the occupancy map.

    Args:
        fine_map: High resolution occupancy grid

    Returns:
        Low resolution (2x) occupancy grid

    TODO: Implement downsampling
        1. Create a new OccupancyGrid with 2x resolution
        2. For each cell in coarse map, compute value from 2x2 block in fine map
        3. Use appropriate aggregation (mean, max, or other)
        4. Return coarse map

    Hint: OccupancyGrid already has a downsample() method you could use
          But try implementing it yourself for learning!
    """
    coarse_map = fine_map.downsample()  # This is the reference

    # YOUR CODE HERE
    # Implement your own downsampling logic

    coarse_map = OccupancyGrid(
        resolution=fine_map.resolution * 2,
        origin=(fine_map.origin_x, fine_map.origin_y),
        initial_size=max((fine_map.width + 1) // 2, (fine_map.height + 1) // 2)
    )
    coarse_map.width  = (fine_map.width  + 1) // 2
    coarse_map.height = (fine_map.height + 1) // 2
    coarse_map.data   = np.zeros((coarse_map.height, coarse_map.width), dtype=np.float32)

    # 用 numpy reshape 做 2x2 block mean
    h = coarse_map.height * 2
    w = coarse_map.width  * 2
    padded = np.zeros((h, w), dtype=np.float32)
    padded[:fine_map.height, :fine_map.width] = fine_map.data[:fine_map.height, :fine_map.width]

    coarse_map.data = padded.reshape(coarse_map.height, 2, coarse_map.width, 2).mean(axis=(1, 3))

    return coarse_map
