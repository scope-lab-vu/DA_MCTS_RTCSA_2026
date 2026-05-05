"""Stanley path-following controller (no avoidance).

Tracks a precomputed raceline (CSV referenced in map_config.wpt_path) using
the standard Stanley method on the front axle, plus a P-controller on the
reference velocity.
"""
from src.controller.stanley_method.controller import StanleyMethodController

__all__ = ["StanleyMethodController"]
