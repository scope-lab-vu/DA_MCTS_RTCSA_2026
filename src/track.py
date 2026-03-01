"""
Track and Reference Trajectory Utilities for MPPI Racing Controller.

Provides:
  - Track: loads centerline CSV, computes boundaries, normals, arc-length
  - ReferenceTrajectory: loads raceline CSV, provides velocity lookup windows
  - TrackTensor / ReferenceTrajectoryTensor: GPU tensor versions
  - TrackOccupancyMap: precomputed grid for O(1) off-track detection
"""

import logging
from math import ceil
from typing import Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class Track:
    """Track class that loads and manages track data from centerline CSV."""

    def __init__(self, centerline_path: str):
        self.centerline_path = centerline_path
        self._load_track()
        self._compute_track_properties()

    def _load_track(self):
        data = np.loadtxt(self.centerline_path, delimiter=',', skiprows=1)
        self.centerline = data[:, :2]
        self.width_right = data[:, 2]
        self.width_left = data[:, 3]
        self.n_points = len(self.centerline)

    def _compute_track_properties(self):
        diff = np.diff(self.centerline, axis=0, append=self.centerline[:1])
        self.tangent = diff / (np.linalg.norm(diff, axis=1, keepdims=True) + 1e-8)
        self.normal = np.column_stack([-self.tangent[:, 1], self.tangent[:, 0]])
        self.left_boundary = self.centerline + self.normal * self.width_left[:, np.newaxis]
        self.right_boundary = self.centerline - self.normal * self.width_right[:, np.newaxis]
        segment_lengths = np.linalg.norm(diff, axis=1)
        self.arc_length = np.concatenate([[0], np.cumsum(segment_lengths[:-1])])
        self.total_length = self.arc_length[-1] + segment_lengths[-1]
        self._compute_curvature()
        self.heading = -1.0 * (np.arctan2(self.tangent[:, 1], -1.0 * self.tangent[:, 0]) - np.pi) # formula for converting arctan2 -- (-pi,pi) to (0,2pi)

    def _compute_curvature(self):
        """Compute curvature at each centerline point using finite differences.

        κ = |x'·y'' - y'·x''| / (x'² + y'²)^(3/2)
        Uses central differences with wrapping for a closed track.
        """
        x = self.centerline[:, 0]
        y = self.centerline[:, 1]
        # First derivatives (central difference, wrapping for closed track)
        dx = np.gradient(x, edge_order=2)
        dy = np.gradient(y, edge_order=2)
        # Second derivatives
        ddx = np.gradient(dx, edge_order=2)
        ddy = np.gradient(dy, edge_order=2)
        # Curvature: κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**1.5 + 1e-10
        self.curvature = numerator / denominator

    def get_nearest_point(self, position: np.ndarray) -> Tuple[int, float, float]:
        distances = np.linalg.norm(self.centerline - position, axis=1)
        idx = np.argmin(distances)
        to_point = position - self.centerline[idx]
        lateral_error = np.dot(to_point, self.normal[idx])
        progress = self.arc_length[idx]
        return idx, lateral_error, progress

    def get_curvature_lookahead(self, position: np.ndarray, v: float,
                                horizon: int, dt: float) -> np.ndarray:
        """Get curvature values at future positions along the centerline.

        Args:
            position: Current (x, y) position.
            v: Current velocity (used to estimate distance per step).
            horizon: Number of lookahead steps.
            dt: Time step.

        Returns:
            Array of curvature values (length = horizon).
        """
        idx, _, _ = self.get_nearest_point(position)
        ds_per_step = max(abs(v), 1.0) * dt
        curvatures = np.zeros(horizon)
        current_arc = self.arc_length[idx]
        for t in range(horizon):
            future_arc = (current_arc + (t + 1) * ds_per_step) % self.total_length
            future_idx = np.argmin(np.abs(self.arc_length - future_arc))
            curvatures[t] = self.curvature[future_idx]
        return curvatures

    def get_heading_lookahead(self, position: np.ndarray, v: float,
                              horizon: int, dt: float) -> np.ndarray:
        """Get track forward heading at future positions along the centerline."""
        idx, _, _ = self.get_nearest_point(position)
        ds_per_step = max(abs(v), 1.0) * dt
        headings = np.zeros(horizon)
        current_arc = self.arc_length[idx]
        for t in range(horizon):
            future_arc = (current_arc + (t + 1) * ds_per_step) % self.total_length
            future_idx = np.argmin(np.abs(self.arc_length - future_arc))
            headings[t] = self.heading[future_idx]
        return headings

    def is_off_track(self, position: np.ndarray, vehicle_width: float = 0.0) -> bool:
        idx, lateral_error, _ = self.get_nearest_point(position)
        half_width = vehicle_width / 2
        return (lateral_error > self.width_left[idx] - half_width or
                lateral_error < -self.width_right[idx] + half_width)


class ReferenceTrajectory:
    """Reference trajectory that provides reference velocities along the track."""

    def __init__(self, trajectory_path: str):
        self.trajectory_path = trajectory_path
        self._load_trajectory()

    def _load_trajectory(self):
        data = np.loadtxt(self.trajectory_path, delimiter=';', comments='#')
        self.arc_length = data[:, 0]
        self.positions = data[:, 1:3]
        self.headings = data[:, 3]
        self.curvatures = data[:, 4]
        self.velocities = data[:, 5]
        self.accelerations = data[:, 6]
        self.n_points = len(self.arc_length)
        self.total_length = self.arc_length[-1]

    def get_nearest_idx(self, position: np.ndarray) -> int:
        distances = np.linalg.norm(self.positions - position, axis=1)
        return np.argmin(distances)

    def get_reference_velocity(self, position: np.ndarray) -> float:
        idx = self.get_nearest_idx(position)
        return self.velocities[idx]

    def get_reference_velocity_window(self, position: np.ndarray, horizon: int, dt: float) -> np.ndarray:
        start_idx = self.get_nearest_idx(position)
        v_ref = self.velocities[start_idx]
        ds_per_step = v_ref * dt
        ref_velocities = np.zeros(horizon)
        current_arc = self.arc_length[start_idx]
        for t in range(horizon):
            future_arc = (current_arc + (t + 1) * ds_per_step) % self.total_length
            idx = np.argmin(np.abs(self.arc_length - future_arc))
            ref_velocities[t] = self.velocities[idx]
        return ref_velocities


class TrackTensor:
    """GPU-accelerated track data for MPPI cost computation."""

    def __init__(self, track: Track, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.centerline = torch.tensor(track.centerline, device=device, dtype=dtype)
        self.width_left = torch.tensor(track.width_left, device=device, dtype=dtype)
        self.width_right = torch.tensor(track.width_right, device=device, dtype=dtype)
        self.normal = torch.tensor(track.normal, device=device, dtype=dtype)
        self.n_points = track.n_points

    def is_off_track_batch(self, positions: torch.Tensor, vehicle_width: float = 0.31, boundary_margin: float = 0.3) -> torch.Tensor:
        diff = positions.unsqueeze(1) - self.centerline.unsqueeze(0)
        distances = torch.norm(diff, dim=2)
        min_dist, nearest_idx = torch.min(distances, dim=1)
        normals = self.normal[nearest_idx]
        width_left = self.width_left[nearest_idx]
        width_right = self.width_right[nearest_idx]
        to_point = positions - self.centerline[nearest_idx]
        lateral_error = torch.sum(to_point * normals, dim=1)
        total_margin = vehicle_width + boundary_margin
        off_track = (lateral_error > width_left - total_margin) | \
                    (lateral_error < -width_right + total_margin)
        return off_track


class ReferenceTrajectoryTensor:
    """GPU-accelerated reference trajectory for MPPI."""

    def __init__(self, ref_traj: ReferenceTrajectory, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.device = device
        self.dtype = dtype
        self.positions = torch.tensor(ref_traj.positions, device=device, dtype=dtype)
        self.headings = torch.tensor(ref_traj.headings, device=device, dtype=dtype)
        self.velocities = torch.tensor(ref_traj.velocities, device=device, dtype=dtype)
        self.arc_length = torch.tensor(ref_traj.arc_length, device=device, dtype=dtype)
        self.total_length = ref_traj.total_length
        self.n_points = ref_traj.n_points


class TrackOccupancyMap:
    """Precomputed occupancy map for O(1) off-track detection."""

    def __init__(
        self,
        track: Track,
        cell_size: float = 0.05,
        lane_width: float = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.track = track
        self.cell_size = cell_size
        self.lane_width = lane_width
        self.use_fixed_width = lane_width is not None

        if torch.cuda.is_available() and device == "cuda":
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._dtype = dtype

        all_points = np.vstack([track.left_boundary, track.right_boundary])
        self.x_min = all_points[:, 0].min() - 2.0
        self.x_max = all_points[:, 0].max() + 2.0
        self.y_min = all_points[:, 1].min() - 2.0
        self.y_max = all_points[:, 1].max() + 2.0

        self.nx = ceil((self.x_max - self.x_min) / cell_size)
        self.ny = ceil((self.y_max - self.y_min) / cell_size)

        log.info(
            "Building TrackOccupancyMap: %dx%d grid, cell_size=%.3fm, lane_width=%s",
            self.nx, self.ny, cell_size, lane_width,
        )
        self._build_occupancy_map()
        self._map_torch = torch.tensor(self._map, device=self._device, dtype=self._dtype)
        drivable = int(np.sum(self._map == 0))
        off_track = int(np.sum(self._map == 1))
        log.info("TrackOccupancyMap built: %d drivable, %d off-track cells", drivable, off_track)

    def _build_occupancy_map(self):
        from scipy.ndimage import distance_transform_edt

        self._map = np.ones((self.nx, self.ny), dtype=np.float32)
        centerline = self.track.centerline

        if self.use_fixed_width:
            for point in centerline:
                ix = int(round((point[0] - self.x_min) / self.cell_size))
                iy = int(round((point[1] - self.y_min) / self.cell_size))
                if 0 <= ix < self.nx and 0 <= iy < self.ny:
                    self._map[ix, iy] = 0
            distance_map = distance_transform_edt(self._map)
            max_distance = (self.lane_width / 2) / self.cell_size
            self._map = np.where(distance_map <= max_distance, 0, 1).astype(np.float32)
        else:
            normal = self.track.normal
            width_left = self.track.width_left
            width_right = self.track.width_right
            for i in range(self.nx):
                x = self.x_min + (i + 0.5) * self.cell_size
                for j in range(self.ny):
                    y = self.y_min + (j + 0.5) * self.cell_size
                    pos = np.array([x, y])
                    distances = np.linalg.norm(centerline - pos, axis=1)
                    idx = np.argmin(distances)
                    to_point = pos - centerline[idx]
                    lateral_error = np.dot(to_point, normal[idx])
                    if (lateral_error <= width_left[idx] and
                            lateral_error >= -width_right[idx]):
                        self._map[i, j] = 0.0

    def compute_cost(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() == 3:
            positions = positions.squeeze(1)
        if positions.device != self._device or positions.dtype != self._dtype:
            positions = positions.to(self._device, self._dtype)
        ix = ((positions[:, 0] - self.x_min) / self.cell_size).long()
        iy = ((positions[:, 1] - self.y_min) / self.cell_size).long()
        is_out_of_bound = (ix < 0) | (ix >= self.nx) | (iy < 0) | (iy >= self.ny)
        ix = torch.clamp(ix, 0, self.nx - 1)
        iy = torch.clamp(iy, 0, self.ny - 1)
        costs = self._map_torch[ix, iy]
        costs[is_out_of_bound] = 1.0
        return costs
