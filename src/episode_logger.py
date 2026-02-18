"""
Episode logger — records state-action trajectory and saves to CSV.

Usage:
    logger = EpisodeLogger()
    for step in loop:
        logger.log_step(state, cmd)
    logger.save("results/", reason="off_track", metadata={...})
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.idl_messages import ControlCmdMsg, VehicleStateMsg

log = logging.getLogger(__name__)

# Column ordering matches the reference run_mppi_known_dynamics.py output.
TRAJECTORY_COLUMNS = [
    "S_x", "S_y", "S_delta", "S_vel", "S_psi", "S_omega", "S_beta",
    "A_a", "A_sv",
    "pred_vx", "pred_vy", "pred_omega",
]


class EpisodeLogger:
    """Collects per-step state-action data and writes a CSV when done."""

    def __init__(self) -> None:
        self._rows: List[Dict[str, float]] = []

    @property
    def num_steps(self) -> int:
        return len(self._rows)

    def log_step(
        self,
        state: VehicleStateMsg,
        action: ControlCmdMsg,
        pred_vx: float = float("nan"),
        pred_vy: float = float("nan"),
        pred_omega: float = float("nan"),
    ) -> None:
        """Record one timestep, optionally with dynamics-model predictions."""
        self._rows.append({
            "S_x": state.x,
            "S_y": state.y,
            "S_delta": state.delta,
            "S_vel": state.v,
            "S_psi": state.psi,
            "S_omega": state.omega,
            "S_beta": state.beta,
            "A_a": action.acceleration,
            "A_sv": action.steering_vel,
            "pred_vx": pred_vx,
            "pred_vy": pred_vy,
            "pred_omega": pred_omega,
        })

    def save(
        self,
        output_dir: str,
        reason: str = "unknown",
        metadata: Optional[Dict[str, object]] = None,
        filename_prefix: str = "traj",
    ) -> str:
        """Write trajectory CSV and return its path.

        Args:
            output_dir: Directory to write into (created if needed).
            reason: Termination reason — "off_track", "lap_completed",
                    "interrupted", etc.
            metadata: Extra key-value pairs written as comments at the
                      top of the CSV (track, mu, dt, …).
            filename_prefix: Prefix for the CSV filename (before timestamp).

        Returns:
            Absolute path of the saved CSV.
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)

        # Build header comments from metadata.
        header_lines = [f"# reason: {reason}"]
        if metadata:
            for k, v in metadata.items():
                header_lines.append(f"# {k}: {v}")
        header_lines.append(f"# steps: {self.num_steps}")
        header = "\n".join(header_lines) + "\n"

        df = pd.DataFrame(self._rows, columns=TRAJECTORY_COLUMNS)

        with open(filepath, "w") as f:
            f.write(header)
            df.to_csv(f, index=False)

        log.info(
            "Saved %d-step trajectory → %s  (reason=%s)",
            self.num_steps, filepath, reason,
        )
        return filepath
