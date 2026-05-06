"""LQR lateral path-follower ported from f1tenth_planning.

Faithful port of LQRPlanner from
https://github.com/f1tenth/f1tenth_planning/blob/main/f1tenth_planning/control/lqr/lqr.py
"""
from src.controller.lqr.controller import LQRController

__all__ = ["LQRController"]
