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
    mu: float = 1.0489
    C_Sf: float = 4.718
    C_Sr: float = 5.4562
    lf: float = 0.15875
    lr: float = 0.17145
    h: float = 0.074
    m: float = 3.74
    I: float = 0.04712
    s_min: float = -0.4189
    s_max: float = 0.4189
    sv_min: float = -3.2
    sv_max: float = 3.2
    v_switch: float = 7.319
    a_max: float = 9.51
    v_min: float = -5.0
    v_max: float = 20.0
    width: float = 0.31
    length: float = 0.58

    def to_gym_dict(self) -> dict:
        """Convert to the dict format expected by f1tenth_gym."""
        from dataclasses import asdict
        return asdict(self)


# ---------------------------------------------------------------------------
# Algorithm config sub-sections (from config_*.yaml)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControllerConfig:
    type: str = "simple"
    target_speed_mps: float = 2.0
    # MPPI-specific fields
    horizon: int = 20
    num_samples: int = 500
    lambda_: float = 1.0
    noise_sigma_accel: float = 2.0
    noise_sigma_steer: float = 1.0
    v_ref: float = 5.0
    device: str = "cpu"
    use_occupancy_map: bool = True
    lane_width: float = 1.1


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
    dt_s: float = 0.01
    log_every_n_steps: int = 100
    seed: int = 42


@dataclass(frozen=True)
class AlgoConfig:
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


# ---------------------------------------------------------------------------
# Map config (from maps/config_*_map.yaml)
# Paths in the yaml are relative to the config file's directory.
# The loader resolves them to absolute paths.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MapConfig:
    map_path: str = ""
    map_ext: str = ".png"
    sx: float = 0.0
    sy: float = 0.0
    stheta: float = 0.0
    wpt_path: str = ""
    wpt_delim: str = ";"
    wpt_rowskip: int = 3
    wpt_xind: int = 1
    wpt_yind: int = 2
    wpt_thind: Optional[int] = None
    wpt_vind: Optional[int] = None
    run_name: Optional[str] = None
    perf_num: Optional[int] = None
    centerline_path: Optional[str] = None
    track_name: Optional[str] = None
    mu_id: Optional[str] = None


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


def load_map_config(path: Path, mu_id_override: Optional[str] = None) -> MapConfig:
    """Load map config. Resolves relative paths against the config file's directory."""
    data = _load_yaml(path)
    config_dir = path.parent

    # Apply CLI override for mu_id before template expansion.
    if mu_id_override is not None:
        data["mu_id"] = mu_id_override

    # Expand {track_name} and {mu_id} placeholders in path fields.
    for key in ("map_path", "wpt_path", "centerline_path"):
        if key in data and data[key] is not None:
            data[key] = data[key].format_map(data)

    # Resolve relative paths to absolute.
    for key in ("map_path", "wpt_path", "centerline_path"):
        if key in data and data[key] is not None:
            data[key] = str((config_dir / data[key]).resolve())

    return MapConfig(**data)
