"""
Typed parameter dataclasses and YAML loader.

Vehicle params and algorithm config are loaded from separate YAML files
into distinct dataclasses. This keeps physical constants decoupled from
algorithm-specific tuning.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Vehicle parameters (from vehicle_params.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleParams:
    wheelbase_m: float = 0.33
    mass_kg: float = 3.74
    lf_m: float = 0.165
    lr_m: float = 0.165
    max_steer_rad: float = 0.41
    max_speed_mps: float = 7.0
    max_accel_mps2: float = 9.51


# ---------------------------------------------------------------------------
# Algorithm config sub-sections (from config_*.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerConfig:
    type: str = "simple"
    target_speed_mps: float = 2.0


@dataclass(frozen=True)
class DynamicsConfig:
    type: str = "kinematic_bicycle"
    model_path: Optional[str] = None
    inference_device: str = "cpu"


@dataclass(frozen=True)
class SafetyConfig:
    type: str = "noop"


@dataclass(frozen=True)
class SimulationConfig:
    vehicle_interface: str = "gym"
    dt_s: float = 0.04
    log_every_n_steps: int = 25


@dataclass(frozen=True)
class AlgoConfig:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Empty YAML file: {path}")
    return data


def load_vehicle_params(path: Path) -> VehicleParams:
    data = _load_yaml(path)
    vehicle_section = data.get("vehicle", {})
    return VehicleParams(**vehicle_section)


def load_algo_config(path: Path) -> AlgoConfig:
    data = _load_yaml(path)
    return AlgoConfig(
        controller=ControllerConfig(**data.get("controller", {})),
        dynamics=DynamicsConfig(**data.get("dynamics", {})),
        safety=SafetyConfig(**data.get("safety", {})),
        simulation=SimulationConfig(**data.get("simulation", {})),
    )
