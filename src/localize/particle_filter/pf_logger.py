"""
Logger for particle filter localization experiments.

Logs per-step:
  - Ground truth state [x, y, psi]
  - PF estimated state [x, y, psi]
  - Position error, heading error
  - PF computation time
  - CPU usage
"""

import csv
import os
import time
import math
import numpy as np
from pathlib import Path
from typing import Optional


class PFLogger:
    """Logs particle filter localization vs ground truth."""

    def __init__(self, output_path: str = "results/pf_localization_log.csv") -> None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._file = open(output_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'step',
            'wall_clock_ns',
            'gt_x', 'gt_y', 'gt_psi',
            'pf_x', 'pf_y', 'pf_psi',
            'pos_error_m', 'heading_error_rad',
            'pf_duration_ms',
            'remaining_budget_ms',
        ])
        self._step = 0

    def log(
        self,
        gt_x: float, gt_y: float, gt_psi: float,
        pf_x: float, pf_y: float, pf_psi: float,
        pf_duration_ms: float,
        dt_ms: float = 40.0,
    ) -> None:
        pos_error = math.sqrt((gt_x - pf_x) ** 2 + (gt_y - pf_y) ** 2)

        # normalize heading error to [-pi, pi]
        heading_error = gt_psi - pf_psi
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi

        remaining_budget = dt_ms - pf_duration_ms

        self._writer.writerow([
            self._step,
            time.time_ns(),
            f'{gt_x:.6f}', f'{gt_y:.6f}', f'{gt_psi:.6f}',
            f'{pf_x:.6f}', f'{pf_y:.6f}', f'{pf_psi:.6f}',
            f'{pos_error:.6f}', f'{heading_error:.6f}',
            f'{pf_duration_ms:.4f}',
            f'{remaining_budget:.4f}',
        ])
        self._file.flush()
        self._step += 1

    def close(self) -> None:
        self._file.close()
