"""
Pure-Python Particle Filter (MCL) for use inside the F1Tenth gym.

No ROS dependencies. Operates on raw numpy arrays:
  - lidar scan (1-D float array of ranges)
  - odometry deltas (dx, dy, dtheta in local frame)
  - occupancy grid map (loaded from .png + .yaml)

Returns localized pose [x, y, theta].
"""

import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

try:
    import range_libc
    HAS_RANGE_LIBC = True
except ImportError:
    HAS_RANGE_LIBC = False


@dataclass
class MapInfo:
    """Minimal map metadata matching what range_libc / the PF needs."""
    resolution: float
    origin_x: float
    origin_y: float
    origin_theta: float
    width: int
    height: int


@dataclass
class PFConfig:
    """Particle filter configuration."""
    max_particles: int = 1500
    angle_step: int = 18          # downsample lidar by this factor
    max_range: float = 10.0       # meters
    squash_factor: float = 2.2
    theta_discretization: int = 112
    range_method: str = 'rm'      # 'rm' (CPU) or 'rmgpu' (GPU)
    rangelib_variant: int = 2

    # sensor model
    z_short: float = 0.01
    z_max: float = 0.07
    z_rand: float = 0.12
    z_hit: float = 0.75
    sigma_hit: float = 8.0

    # motion model noise
    motion_dispersion_x: float = 0.05
    motion_dispersion_y: float = 0.025
    motion_dispersion_theta: float = 0.25


class ParticleFilter:
    """
    Monte Carlo Localization using range_libc for fast ray casting.

    Usage:
        pf = ParticleFilter(config, map_yaml_path)
        pf.set_pose(x, y, theta)  # initial pose
        # each step:
        pose, duration = pf.update(scan_ranges, odom_delta, scan_angles)
    """

    def __init__(self, config: PFConfig, map_yaml_path: str) -> None:
        if not HAS_RANGE_LIBC:
            raise ImportError("range_libc is required but not installed")

        self.config = config
        self.inv_squash_factor = 1.0 / config.squash_factor

        # particles: [x, y, theta]
        self.particles = np.zeros((config.max_particles, 3))
        self.weights = np.ones(config.max_particles) / config.max_particles
        self.particle_indices = np.arange(config.max_particles)
        self.local_deltas = np.zeros((config.max_particles, 3))

        # state
        self.inferred_pose = np.array([0.0, 0.0, 0.0])
        self.initialized = False
        self.first_sensor_update = True
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.downsampled_angles = None

        # load map and initialize range method
        self._load_map(map_yaml_path)
        self._precompute_sensor_model()

    def _load_map(self, map_yaml_path: str) -> None:
        """Load occupancy grid from map .yaml + .png and initialize range_libc."""
        import yaml
        from PIL import Image

        yaml_path = Path(map_yaml_path)
        with open(yaml_path) as f:
            map_meta = yaml.safe_load(f)

        # load the image — flip vertically to match gym convention
        # (gym does Image.FLIP_TOP_BOTTOM so row 0 = world bottom)
        img_path = yaml_path.parent / map_meta['image']
        img = Image.open(str(img_path)).transpose(Image.FLIP_TOP_BOTTOM).convert('L')
        img_array = np.array(img)

        # store map info
        origin = map_meta['origin']
        self.map_info = MapInfo(
            resolution=float(map_meta['resolution']),
            origin_x=float(origin[0]),
            origin_y=float(origin[1]),
            origin_theta=float(origin[2]),
            width=img_array.shape[1],
            height=img_array.shape[0],
        )

        # Match gym's thresholding: <=128 -> occupied (0), >128 -> free (255)
        negate = map_meta.get('negate', 0)
        if negate:
            img_array = 255 - img_array

        gym_map = img_array.copy().astype(np.float64)
        gym_map[gym_map <= 128.0] = 0.0
        gym_map[gym_map > 128.0] = 255.0
        gym_map = gym_map.astype(np.uint8)

        # build permissible region for particle initialization
        self.permissible_region = (gym_map == 255)

        # range_libc: 0 = free, >0 = occupied
        # Transpose needed: range_libc expects grid[col][row] layout
        occ_grid = (255 - gym_map).astype(np.uint8).T.copy()

        self.MAX_RANGE_PX = int(self.config.max_range / self.map_info.resolution)

        # Create OMap from grid
        omap = range_libc.PyOMap(occ_grid)

        # initialize range method
        rm = self.config.range_method
        if rm == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(omap, self.MAX_RANGE_PX)
        elif rm == 'rm':
            self.range_method = range_libc.PyRayMarching(omap, self.MAX_RANGE_PX)
        elif rm == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(omap, self.MAX_RANGE_PX)
        elif 'cddt' in rm:
            self.range_method = range_libc.PyCDDTCast(
                omap, self.MAX_RANGE_PX, self.config.theta_discretization
            )
        else:
            raise ValueError(f"Unknown range method: {rm}")

    def _precompute_sensor_model(self) -> None:
        """Build the sensor model lookup table."""
        cfg = self.config
        table_width = int(self.MAX_RANGE_PX) + 1
        self.sensor_model_table = np.zeros((table_width, table_width))

        for d in range(table_width):
            norm = 0.0
            for r in range(table_width):
                prob = 0.0
                z = float(r - d)
                # hit
                prob += cfg.z_hit * np.exp(-(z * z) / (2.0 * cfg.sigma_hit ** 2)) / (
                    cfg.sigma_hit * np.sqrt(2.0 * np.pi)
                )
                # short
                if r < d:
                    prob += 2.0 * cfg.z_short * (d - r) / float(d) if d > 0 else 0.0
                # max
                if int(r) == int(self.MAX_RANGE_PX):
                    prob += cfg.z_max
                # random
                if r < int(self.MAX_RANGE_PX):
                    prob += cfg.z_rand * 1.0 / float(self.MAX_RANGE_PX)
                norm += prob
                self.sensor_model_table[int(r), int(d)] = prob

            if norm > 0:
                self.sensor_model_table[:, int(d)] /= norm

        # upload to range_libc
        if self.config.rangelib_variant > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def set_pose(self, x: float, y: float, theta: float, spread: float = 0.1) -> None:
        """Initialize particles around the given pose."""
        n = self.config.max_particles
        self.particles[:, 0] = x + np.random.normal(0.0, spread, n)
        self.particles[:, 1] = y + np.random.normal(0.0, spread, n)
        self.particles[:, 2] = theta + np.random.normal(0.0, 0.1, n)
        self.weights[:] = 1.0 / n
        self.inferred_pose = np.array([x, y, theta])
        self.initialized = True

    def update(
        self,
        scan_ranges: np.ndarray,
        odom_delta: np.ndarray,
        scan_angles: Optional[np.ndarray] = None,
        scan_angle_min: float = -2.3561945,
        scan_angle_max: float = 2.3561945,
    ) -> Tuple[np.ndarray, float]:
        """
        Run one MCL update step.

        Args:
            scan_ranges: raw lidar ranges (1-D array)
            odom_delta: [dx, dy, dtheta] in local frame since last update
            scan_angles: full array of scan angles (optional, computed from min/max)
            scan_angle_min: minimum scan angle (if scan_angles not provided)
            scan_angle_max: maximum scan angle (if scan_angles not provided)

        Returns:
            (pose, duration_ms): localized [x, y, theta] and processing time in ms
        """
        if not self.initialized:
            return self.inferred_pose, 0.0

        t_start = time.time()

        # setup angles on first call
        if self.downsampled_angles is None:
            if scan_angles is None:
                scan_angles = np.linspace(scan_angle_min, scan_angle_max, len(scan_ranges))
            self.downsampled_angles = np.copy(
                scan_angles[:: self.config.angle_step]
            ).astype(np.float32)

        # downsample observation and clip to max range
        observation = np.array(scan_ranges[:: self.config.angle_step], dtype=np.float32)
        observation = np.clip(observation, 0.0, self.config.max_range)

        # run MCL
        self._mcl(odom_delta, observation)

        # compute expected pose
        self.inferred_pose = self._expected_pose()

        duration_ms = (time.time() - t_start) * 1000.0
        return self.inferred_pose.copy(), duration_ms

    def _mcl(self, action: np.ndarray, observation: np.ndarray) -> None:
        """One step of Monte Carlo Localization."""
        # resample
        proposal_indices = np.random.choice(
            self.particle_indices, self.config.max_particles, p=self.weights
        )
        proposal_distribution = self.particles[proposal_indices, :]

        # motion model
        self._motion_model(proposal_distribution, action)

        # sensor model
        self._sensor_model(proposal_distribution, observation, self.weights)

        # normalize
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights[:] = 1.0 / self.config.max_particles

        self.particles = proposal_distribution

    def _motion_model(self, proposal_dist: np.ndarray, action: np.ndarray) -> None:
        """Apply odometry action with noise to particle distribution."""
        cfg = self.config
        n = cfg.max_particles

        cosines = np.cos(proposal_dist[:, 2])
        sines = np.sin(proposal_dist[:, 2])

        self.local_deltas[:, 0] = cosines * action[0] - sines * action[1]
        self.local_deltas[:, 1] = sines * action[0] + cosines * action[1]
        self.local_deltas[:, 2] = action[2]

        proposal_dist[:, :] += self.local_deltas
        proposal_dist[:, 0] += np.random.normal(0.0, cfg.motion_dispersion_x, n)
        proposal_dist[:, 1] += np.random.normal(0.0, cfg.motion_dispersion_y, n)
        proposal_dist[:, 2] += np.random.normal(0.0, cfg.motion_dispersion_theta, n)

    def _world_to_pixel(self, particles: np.ndarray) -> np.ndarray:
        """Convert world coordinates (meters) to pixel coordinates for range_libc.

        range_libc queries are (x_px, y_px, theta) with transposed grid.
        """
        mi = self.map_info
        px = (particles[:, 0] - mi.origin_x) / mi.resolution
        py = (particles[:, 1] - mi.origin_y) / mi.resolution
        return np.column_stack([px, py, particles[:, 2]]).astype(np.float32)

    def _sensor_model(
        self, proposal_dist: np.ndarray, obs: np.ndarray, weights: np.ndarray
    ) -> None:
        """Compute particle weights using range_libc sensor model."""
        num_rays = self.downsampled_angles.shape[0]

        if self.first_sensor_update:
            if self.config.rangelib_variant <= 1:
                self.queries = np.zeros((num_rays * self.config.max_particles, 3), dtype=np.float32)
            else:
                self.queries = np.zeros((self.config.max_particles, 3), dtype=np.float32)
            self.ranges = np.zeros(num_rays * self.config.max_particles, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.config.max_particles)
            self.first_sensor_update = False

        # convert observation from meters to pixels
        obs_px = obs / self.map_info.resolution

        # convert particles from world to pixel coordinates
        pixel_particles = self._world_to_pixel(proposal_dist)

        variant = self.config.rangelib_variant
        if variant == 2:
            self.queries[:, :] = pixel_particles[:, :]
            self.range_method.calc_range_repeat_angles(
                self.queries, self.downsampled_angles, self.ranges
            )
            self.range_method.eval_sensor_model(
                obs_px, self.ranges, self.weights, num_rays, self.config.max_particles
            )
            np.power(self.weights, self.inv_squash_factor, self.weights)
        elif variant == 3:
            self.queries[:, :] = pixel_particles[:, :]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(
                self.queries, self.downsampled_angles, obs_px, self.weights
            )
            np.power(self.weights, self.inv_squash_factor, self.weights)
        elif variant == 1:
            self.queries[:, 0] = np.repeat(pixel_particles[:, 0], num_rays)
            self.queries[:, 1] = np.repeat(pixel_particles[:, 1], num_rays)
            self.queries[:, 2] = np.repeat(pixel_particles[:, 2], num_rays)
            self.queries[:, 2] += self.tiled_angles
            self.range_method.calc_range_many(self.queries, self.ranges)
            self.range_method.eval_sensor_model(
                obs_px, self.ranges, self.weights, num_rays, self.config.max_particles
            )
            np.power(self.weights, self.inv_squash_factor, self.weights)

    def _expected_pose(self) -> np.ndarray:
        """Weighted average of particles."""
        return np.dot(self.particles.transpose(), self.weights)
