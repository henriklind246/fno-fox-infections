from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
import copy
import hashlib
import json

import yaml


Range = tuple[float, float]


def _range(value: Any, name: str) -> Range:
    low, high = (float(item) for item in value)
    if not low < high:
        raise ValueError(f"{name} must be an increasing [low, high] pair.")
    return (low, high)


@dataclass(frozen=True)
class DomainConfig:
    length: float = 10.0
    horizon: float = 5.0
    mortality: float = 0.3
    learning_cells: int = 256
    n_snapshots: int = 21

    def validate(self) -> None:
        if self.length <= 0.0 or self.horizon <= 0.0:
            raise ValueError("domain.length and domain.horizon must be positive.")
        if self.mortality < 0.0:
            raise ValueError("domain.mortality must be nonnegative.")
        if self.learning_cells < 2:
            raise ValueError("domain.learning_cells must be at least 2.")
        if self.n_snapshots < 3:
            raise ValueError(
                "domain.n_snapshots must be at least 3 so later-source pairs exist."
            )

    @property
    def save_times(self) -> list[float]:
        step = self.horizon / (self.n_snapshots - 1)
        return [index * step for index in range(self.n_snapshots)]


@dataclass(frozen=True)
class NewtonConfig:
    max_iterations: int = 25
    residual_tolerance: float = 1.0e-11
    update_tolerance: float = 1.0e-9
    update_relative_tolerance: float = 1.0e-9
    max_backtracks: int = 12
    armijo_c: float = 1.0e-4

    def validate(self) -> None:
        if self.max_iterations < 1 or self.max_backtracks < 0:
            raise ValueError("Newton iteration limits must be positive.")
        if min(
            self.residual_tolerance,
            self.update_tolerance,
        ) <= 0.0 or self.update_relative_tolerance < 0.0:
            raise ValueError("Newton tolerances must be positive.")
        if not 0.0 < self.armijo_c < 1.0:
            raise ValueError("newton.armijo_c must lie in (0, 1).")


@dataclass(frozen=True)
class SolverConfig:
    refinement_factor: int = 2
    dt: float = 0.01
    positivity_tolerance: float = 1.0e-10
    balance_tolerance: float = 1.0e-10
    newton: NewtonConfig = field(default_factory=NewtonConfig)

    def validate(self) -> None:
        if self.refinement_factor < 1:
            raise ValueError("solver.refinement_factor must be a positive integer.")
        if self.dt <= 0.0:
            raise ValueError("solver.dt must be positive.")
        if self.positivity_tolerance < 0.0 or self.balance_tolerance <= 0.0:
            raise ValueError("solver acceptance tolerances must be nonnegative.")
        self.newton.validate()


@dataclass(frozen=True)
class SamplingConfig:
    n_trajectories: int = 600
    diffusivity: Range = (0.02, 0.08)
    transmission: Range = (0.6, 1.2)
    infection_amplitude: Range = (0.01, 0.05)
    infection_center: Range = (4.5, 5.5)
    infection_width: Range = (0.2, 0.4)
    susceptible_constant_fraction: float = 0.5
    susceptible_perturbation_amplitude: Range = (0.05, 0.2)
    susceptible_modes: int = 4

    def validate(self, domain: DomainConfig) -> None:
        if self.n_trajectories < 3:
            raise ValueError("sampling.n_trajectories must allow three splits.")
        if self.diffusivity[0] <= 0.0 or self.transmission[0] <= 0.0:
            raise ValueError("Disease parameter ranges must be positive.")
        if self.infection_amplitude[0] <= 0.0 or self.infection_width[0] <= 0.0:
            raise ValueError("Infection amplitude and width must be positive.")
        if not (
            0.0 <= self.infection_center[0]
            and self.infection_center[1] <= domain.length
        ):
            raise ValueError("Infection centers must lie inside the domain.")
        if not 0.0 <= self.susceptible_constant_fraction <= 1.0:
            raise ValueError("susceptible_constant_fraction must lie in [0, 1].")
        low, high = self.susceptible_perturbation_amplitude
        if low < 0.0 or high >= 1.0:
            # S0 >= 1 - a because the cosine coefficients satisfy sum |c_k| = 1.
            raise ValueError(
                "susceptible_perturbation_amplitude must lie in [0, 1) "
                "to keep S0 positive."
            )
        if self.susceptible_modes < 1:
            raise ValueError("susceptible_modes must be positive.")


@dataclass(frozen=True)
class SplitConfig:
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)

    def validate(self) -> None:
        if len(self.fractions) != 3 or any(value <= 0.0 for value in self.fractions):
            raise ValueError("splits.fractions must contain three positive values.")
        if abs(sum(self.fractions) - 1.0) > 1.0e-9:
            raise ValueError("splits.fractions must sum to one.")


@dataclass(frozen=True)
class RefinementConfig:
    enabled: bool = True
    random_pilots: int = 4
    threshold: float = 0.005

    def validate(self) -> None:
        if self.random_pilots < 0 or self.threshold <= 0.0:
            raise ValueError("refinement settings must be nonnegative/positive.")


@dataclass(frozen=True)
class PairConfig:
    later_source_per_trajectory: int = 20

    def validate(self) -> None:
        if self.later_source_per_trajectory < 1:
            raise ValueError("pairs.later_source_per_trajectory must be positive.")


@dataclass(frozen=True)
class ModelConfig:
    modes: int = 16
    width: int = 32
    n_layers: int = 4
    cond_hidden: int = 64
    dropout: float = 0.0

    def validate(self) -> None:
        if min(self.modes, self.width, self.n_layers, self.cond_hidden) < 1:
            raise ValueError("model sizes must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must lie in [0, 1).")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    num_workers: int = 0
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("training epochs/batch_size must be positive.")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("training optimizer settings are invalid.")


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int = 256
    front_threshold: float = 0.01
    speed_window: Range = (2.5, 5.0)
    min_front_points: int = 4
    interpolation_trajectories: int = 10
    interpolation_source_times: tuple[float, ...] = (0.0, 2.5)
    representative_trajectories: int = 3

    def validate(self) -> None:
        if self.batch_size < 1 or self.front_threshold <= 0.0:
            raise ValueError("evaluation batch size and threshold must be positive.")
        if self.min_front_points < 2:
            raise ValueError("A speed fit requires at least two crossings.")
        if self.interpolation_trajectories < 0 or self.representative_trajectories < 1:
            raise ValueError("evaluation trajectory counts are invalid.")


@dataclass(frozen=True)
class InverseConfig:
    pilot_cases: int = 10
    sweep_cases: int = 100
    selection_seed: int = 0
    start_seed: int = 0
    grid_size: int = 25
    batch_size: int = 64
    max_iterations: int = 100
    max_evaluations: int = 250
    edge_max_evaluations: int = 50
    coordinate_tolerance: float = 1.0e-5
    objective_tolerance: float = 1.0e-10
    parameter_change_tolerance: float = 0.005
    loss_change_absolute: float = 1.0e-10
    loss_change_relative: float = 0.01

    def validate(self) -> None:
        import math

        if self.grid_size < 3:
            raise ValueError("inverse.grid_size must be at least 3.")
        if min(self.pilot_cases, self.sweep_cases, self.batch_size, self.max_iterations,
               self.max_evaluations, self.edge_max_evaluations) < 1:
            raise ValueError("inverse counts and optimization budgets must be positive.")
        if min(self.selection_seed, self.start_seed) < 0:
            raise ValueError("inverse seeds must be nonnegative.")
        for name in ("coordinate_tolerance", "objective_tolerance",
                     "parameter_change_tolerance", "loss_change_absolute",
                     "loss_change_relative"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"inverse.{name} must be finite and positive.")


@dataclass(frozen=True)
class SeedConfig:
    sampling: int = 1
    split: int = 2
    pairs: int = 3
    training: int = 4
    refinement: int = 5


@dataclass(frozen=True)
class FoxConfig:
    domain: DomainConfig = field(default_factory=DomainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    pairs: PairConfig = field(default_factory=PairConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    inverse: InverseConfig = field(default_factory=InverseConfig)
    seeds: SeedConfig = field(default_factory=SeedConfig)

    def validate(self) -> None:
        self.domain.validate()
        self.solver.validate()
        self.sampling.validate(self.domain)
        self.splits.validate()
        self.refinement.validate()
        self.pairs.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()
        self.inverse.validate()

    @property
    def solver_cells(self) -> int:
        return self.domain.learning_cells * self.solver.refinement_factor

    def as_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))

    @property
    def sha256(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _build(cls: type, payload: dict[str, Any] | None, path: str) -> Any:
    payload = {} if payload is None else dict(payload)
    known = {item.name: item for item in fields(cls)}
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ValueError(f"Unknown configuration keys under {path or 'root'}: {unknown}")
    defaults = cls()
    values: dict[str, Any] = {}
    for name, spec in known.items():
        default = getattr(defaults, name)
        if name not in payload:
            values[name] = default
            continue
        raw = payload[name]
        key = f"{path}.{name}" if path else name
        if is_dataclass(default):
            values[name] = _build(type(default), raw, key)
        elif spec.type == "Range":
            values[name] = _range(raw, key)
        elif isinstance(default, tuple):
            values[name] = tuple(float(item) for item in raw)
        elif isinstance(default, bool):
            values[name] = bool(raw)
        elif isinstance(default, int):
            if float(raw) != int(raw):
                raise ValueError(f"{key} must be an integer.")
            values[name] = int(raw)
        elif isinstance(default, float):
            values[name] = float(raw)
        else:
            values[name] = raw
    return cls(**values)


def merge_dicts(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def config_from_dict(
    payload: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> FoxConfig:
    data = merge_dicts(payload, overrides or {})
    config = _build(FoxConfig, data, "")
    config.validate()
    return config


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> FoxConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return config_from_dict(payload, overrides)
