"""
IDL-style message definitions.

These dataclasses define the data contracts between components.
Only messages that the current main loop actually uses are included.
Add new message types here as components that produce/consume them are built.
"""

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class VehicleStateMsg:
    """Matches MPPI state vector ordering: [x, y, delta, v, psi, omega, beta]."""
    x: float = 0.0
    y: float = 0.0
    delta: float = 0.0      # steering angle [rad]
    v: float = 0.0           # longitudinal velocity [m/s]
    psi: float = 0.0         # heading angle [rad]
    omega: float = 0.0       # yaw rate [rad/s]
    beta: float = 0.0        # slip angle [rad]

    def validate(self) -> None:
        for name in ("x", "y", "delta", "v", "psi", "omega", "beta"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise ValueError(f"VehicleStateMsg.{name} is not finite: {val}")


@dataclass
class ControlCmdMsg:
    acceleration: float = 0.0   # [m/s^2]
    steering_vel: float = 0.0   # steering rate [rad/s]

    def validate(self) -> None:
        for name in ("acceleration", "steering_vel"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise ValueError(f"ControlCmdMsg.{name} is not finite: {val}")


@dataclass
class WaypointsMsg:
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    def validate(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"WaypointsMsg length mismatch: x has {len(self.x)}, y has {len(self.y)}"
            )
        if len(self.x) == 0:
            raise ValueError("WaypointsMsg is empty")
