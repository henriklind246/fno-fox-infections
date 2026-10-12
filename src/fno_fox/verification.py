"""Reference-solver verification for the fox-rabies model.

Adapted from fno-tps-sizing ``src/fno_tps/verification.py`` and
``tests/test_physics_nonlinear.py``: manufactured solutions, observed
convergence orders, balance checks, Jacobian checks, and failure tests, with
the thermal problems replaced by the coupled S/I system. Also hosts the
pre-generation refinement study on pilot trajectories.

    python -m fno_fox.verification --config conf/default.yaml \
        --output runs/verification/report.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any
import argparse
import json
import time

import numpy as np
from scipy.integrate import solve_ivp

from fno_fox.config import FoxConfig, NewtonConfig, load_config
from fno_fox.physics import (
    Grid,
    NonlinearSolveError,
    RabiesSolver,
    acceptance_report,
    simulate,
)
from fno_fox.sampling import (
    TrajectoryParameters,
    cosine_cell_average,
    initial_fields,
    sample_parameters,
)


MINIMUM_ORDER = 1.8


@dataclass(frozen=True)
class ConvergenceResult:
    resolutions: tuple[float, ...]
    errors: tuple[float, ...]
    orders: tuple[float, ...]
    minimum_order: float

    @property
    def passed(self) -> bool:
        return self.minimum_order >= MINIMUM_ORDER

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def _orders(scales: list[float], errors: list[float]) -> tuple[float, ...]:
    return tuple(
        float(np.log(errors[i] / errors[i + 1]) / np.log(scales[i] / scales[i + 1]))
        for i in range(len(errors) - 1)
    )


def _convergence(scales: list[float], errors: list[float]) -> ConvergenceResult:
    orders = _orders(scales, errors)
    return ConvergenceResult(tuple(scales), tuple(errors), orders, min(orders))


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


# ---------------------------------------------------------------- diffusion


def diffusion_only_convergence(
    config: FoxConfig,
    levels: tuple[int, ...] = (16, 32, 64),
    *,
    diffusivity: float = 0.08,
    final_time: float = 2.0,
    dt: float = 1.0e-3,
) -> ConvergenceResult:
    """beta = 0: I = e^{-mu t} sum_k a_k e^{-D (k pi / L)^2 t} cos(k pi x / L).

    Cosine modes are exact zero-flux eigenfunctions, so exact cell averages
    give an analytic reference at every resolution.
    """
    length = config.domain.length
    mortality = config.domain.mortality
    modes = (0, 2, 4)
    amplitudes = (1.0, 0.5, 0.25)

    def exact_average(faces: np.ndarray, t: float) -> np.ndarray:
        values = np.zeros(len(faces) - 1)
        for mode, amplitude in zip(modes, amplitudes):
            rate = diffusivity * (mode * np.pi / length) ** 2
            coefficients = np.zeros(max(modes))
            if mode == 0:
                values += amplitude * np.exp(-rate * t)
                continue
            coefficients[mode - 1] = 1.0
            values += amplitude * np.exp(-rate * t) * (
                cosine_cell_average(faces, length, 1.0, coefficients) - 1.0
            )
        return np.exp(-mortality * t) * values

    errors, scales = [], []
    for cells in levels:
        grid = Grid(length, cells)
        solver = RabiesSolver(grid, diffusivity, 0.0, mortality, dt)
        susceptible = cosine_cell_average(grid.faces, length, 0.1, (1.0,))
        trajectory = solver.solve(
            susceptible,
            exact_average(grid.faces, 0.0),
            [0.0, final_time],
        )
        errors.append(_rms(trajectory.infectious[-1] - exact_average(grid.faces, final_time)))
        scales.append(grid.h)
        if np.max(np.abs(trajectory.susceptible[-1] - susceptible)) != 0.0:
            raise RuntimeError("S changed although beta = 0.")
    return _convergence(scales, errors)


# ----------------------------------------------------------------- reaction


def uniform_reaction_convergence(
    config: FoxConfig,
    dt_levels: tuple[float, ...] = (0.1, 0.05, 0.025),
    *,
    diffusivity: float = 0.05,
    transmission: float = 1.2,
    initial_infectious: float = 0.05,
) -> ConvergenceResult:
    """Spatially uniform fields reduce to the reaction ODE; compare to DOP853."""
    mortality = config.domain.mortality
    horizon = config.domain.horizon

    def rhs(_: float, state: np.ndarray) -> np.ndarray:
        s, i = state
        return np.array([-transmission * s * i, transmission * s * i - mortality * i])

    reference = solve_ivp(
        rhs,
        (0.0, horizon),
        [1.0, initial_infectious],
        method="DOP853",
        rtol=1.0e-13,
        atol=1.0e-15,
    ).y[:, -1]
    grid = Grid(config.domain.length, 8)
    errors = []
    for dt in dt_levels:
        solver = RabiesSolver(grid, diffusivity, transmission, mortality, dt)
        trajectory = solver.solve(
            np.ones(grid.n_cells),
            np.full(grid.n_cells, initial_infectious),
            [0.0, horizon],
        )
        if np.ptp(trajectory.susceptible[-1]) or np.ptp(trajectory.infectious[-1]):
            raise RuntimeError("Uniform fields did not remain uniform.")
        errors.append(
            float(
                max(
                    abs(trajectory.susceptible[-1, 0] - reference[0]),
                    abs(trajectory.infectious[-1, 0] - reference[1]),
                )
            )
        )
    return _convergence(list(dt_levels), errors)


def zero_infection_equilibrium(config: FoxConfig) -> dict[str, Any]:
    """I = 0 is an exact fixed point for any positive S0."""
    grid = Grid(config.domain.length, config.domain.learning_cells)
    susceptible = cosine_cell_average(
        grid.faces, config.domain.length, 0.2, (0.4, -0.3, 0.2, 0.1)
    )
    solver = RabiesSolver(grid, 0.08, 1.2, config.domain.mortality, config.solver.dt)
    trajectory = solver.solve(
        susceptible,
        np.zeros(grid.n_cells),
        config.domain.save_times,
    )
    s_drift = float(np.max(np.abs(trajectory.susceptible - susceptible)))
    i_max = float(np.max(np.abs(trajectory.infectious)))
    return {
        "susceptible_drift": s_drift,
        "infectious_max": i_max,
        "passed": s_drift == 0.0 and i_max == 0.0,
    }


# --------------------------------------------------- coupled manufactured MMS


class CoupledManufacturedSolution:
    """Smooth zero-flux S/I fields with matching volumetric sources."""

    def __init__(self, length: float, diffusivity: float, transmission: float, mortality: float):
        self.length = length
        self.diffusivity = diffusivity
        self.transmission = transmission
        self.mortality = mortality
        self.k_s = 2.0 * np.pi / length
        self.k_i = 3.0 * np.pi / length

    def exact(self, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        susceptible = 0.8 + 0.15 * np.cos(self.k_s * x) * np.exp(-0.3 * t)
        infectious = 0.1 + 0.05 * np.cos(self.k_i * x) * (1.0 + 0.5 * np.sin(t))
        return susceptible, infectious

    def source(self, x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
        susceptible, infectious = self.exact(x, t)
        ds_dt = -0.3 * 0.15 * np.cos(self.k_s * x) * np.exp(-0.3 * t)
        di_dt = 0.05 * np.cos(self.k_i * x) * 0.5 * np.cos(t)
        d2i_dx2 = -(self.k_i**2) * 0.05 * np.cos(self.k_i * x) * (1.0 + 0.5 * np.sin(t))
        rate = self.transmission * susceptible * infectious
        return (
            ds_dt + rate,
            di_dt - self.diffusivity * d2i_dx2 - rate + self.mortality * infectious,
        )

    def solver(self, cells: int, dt: float, newton: NewtonConfig | None = None) -> RabiesSolver:
        return RabiesSolver(
            Grid(self.length, cells),
            self.diffusivity,
            self.transmission,
            self.mortality,
            dt,
            newton,
            source=self.source,
        )


def _manufactured(config: FoxConfig) -> CoupledManufacturedSolution:
    return CoupledManufacturedSolution(config.domain.length, 0.08, 1.0, config.domain.mortality)


def coupled_spatial_convergence(
    config: FoxConfig,
    levels: tuple[int, ...] = (20, 40, 80),
    *,
    final_time: float = 1.0,
    dt: float = 1.0e-3,
) -> ConvergenceResult:
    mms = _manufactured(config)
    errors, scales = [], []
    for cells in levels:
        solver = mms.solver(cells, dt)
        initial = mms.exact(solver.x, 0.0)
        trajectory = solver.solve(*initial, [0.0, final_time])
        exact = mms.exact(solver.x, final_time)
        errors.append(
            max(
                _rms(trajectory.susceptible[-1] - exact[0]),
                _rms(trajectory.infectious[-1] - exact[1]),
            )
        )
        scales.append(solver.grid.h)
    return _convergence(scales, errors)


def coupled_temporal_convergence(
    config: FoxConfig,
    dt_levels: tuple[float, ...] = (0.2, 0.1, 0.05),
    *,
    cells: int = 40,
    final_time: float = 2.0,
) -> ConvergenceResult:
    """Temporal order against a dt/8 reference on the same mesh."""
    mms = _manufactured(config)

    def final_state(dt: float) -> np.ndarray:
        solver = mms.solver(cells, dt)
        trajectory = solver.solve(*mms.exact(solver.x, 0.0), [0.0, final_time])
        return np.concatenate((trajectory.susceptible[-1], trajectory.infectious[-1]))

    reference = final_state(min(dt_levels) / 8.0)
    errors = [_rms(final_state(dt) - reference) for dt in dt_levels]
    return _convergence(list(dt_levels), errors)


def manufactured_balance(config: FoxConfig) -> dict[str, Any]:
    """Balance closes with sources included, on the coupled MMS."""
    mms = _manufactured(config)
    solver = mms.solver(40, 0.05)
    trajectory = solver.solve(*mms.exact(solver.x, 0.0), [0.0, 2.0])
    residual = trajectory.relative_balance_residual
    return {"relative_balance_residual": residual, "passed": residual < 1.0e-12}


# ------------------------------------------------------------- Jacobian


def jacobian_finite_difference(
    *,
    cells: int = 12,
    dt: float = 0.05,
    epsilon: float = 1.0e-6,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    solver = RabiesSolver(Grid(10.0, cells), 0.06, 0.9, 0.3, dt)
    old_s = 0.5 + rng.random(cells)
    old_i = 0.05 + 0.3 * rng.random(cells)
    state = np.concatenate((old_s + 0.01 * rng.standard_normal(cells), old_i + 0.01 * rng.random(cells)))

    def residual(vector: np.ndarray) -> np.ndarray:
        return solver.residual(vector[:cells], vector[cells:], old_s, old_i, dt)

    analytic = solver.jacobian(state[:cells], state[cells:], dt).toarray()
    numeric = np.empty_like(analytic)
    for column in range(2 * cells):
        bump = np.zeros(2 * cells)
        bump[column] = epsilon
        numeric[:, column] = (residual(state + bump) - residual(state - bump)) / (2.0 * epsilon)
    relative = float(np.max(np.abs(analytic - numeric)) / np.max(np.abs(analytic)))
    return {"relative_max_difference": relative, "passed": relative < 1.0e-8}


# ------------------------------------------------- production-like checks


def corner_pilots(config: FoxConfig) -> list[TrajectoryParameters]:
    """Deterministic pilots at the extremes of the sampling ranges."""
    sampling = config.sampling
    d_low, d_high = sampling.diffusivity
    b_low, b_high = sampling.transmission
    a_low, a_high = sampling.infection_amplitude
    w_low, w_high = sampling.infection_width
    center = 0.5 * sum(sampling.infection_center)
    edge_low, edge_high = sampling.infection_center
    perturbation = sampling.susceptible_perturbation_amplitude[1]
    modes = sampling.susceptible_modes
    highest = tuple(1.0 if k == modes - 1 else 0.0 for k in range(modes))
    lowest = tuple(1.0 if k == 0 else 0.0 for k in range(modes))
    flat = tuple(0.0 for _ in range(modes))
    return [
        TrajectoryParameters(d_high, b_high, a_high, edge_high, w_low, "cosine", perturbation, highest),
        TrajectoryParameters(d_low, b_high, a_high, center, w_low, "constant", 0.0, flat),
        TrajectoryParameters(d_high, b_low, a_low, edge_low, w_high, "constant", 0.0, flat),
        TrajectoryParameters(d_low, b_low, a_low, center, w_low, "cosine", perturbation, lowest),
    ]


def balance_and_positivity(config: FoxConfig) -> dict[str, Any]:
    """Acceptance checks on the corner pilots at the production resolution."""
    reports = []
    for parameters in corner_pilots(config):
        _, _, trajectory = simulate(config, parameters)
        reports.append(
            acceptance_report(
                trajectory,
                config.solver.positivity_tolerance,
                config.solver.balance_tolerance,
            )
        )
    return {
        "maximum_relative_balance_residual": max(r["relative_balance_residual"] for r in reports),
        "minimum_susceptible": min(r["minimum_susceptible"] for r in reports),
        "minimum_infectious": min(r["minimum_infectious"] for r in reports),
        "maximum_newton_iterations": max(r["newton_iterations_max"] for r in reports),
        "pilots": reports,
        "passed": all(r["passed"] for r in reports),
    }


class _CorruptedJacobianSolver(RabiesSolver):
    """Flips the Jacobian sign so every Newton direction is an ascent direction."""

    def jacobian(self, susceptible, infectious, dt):  # type: ignore[override]
        return -super().jacobian(susceptible, infectious, dt)


def failure_modes(config: FoxConfig) -> dict[str, Any]:
    """The solver must raise informative errors rather than return bad states."""
    grid = Grid(config.domain.length, 64)
    parameters = corner_pilots(config)[0]
    susceptible, infectious = initial_fields(parameters, grid.faces, config.domain.length)
    results: dict[str, Any] = {}

    def expect(name: str, error_type: type[Exception], fragment: str, action) -> None:
        try:
            action()
        except error_type as error:
            message = str(error)
            results[name] = {"raised": True, "message": message, "passed": fragment in message}
            return
        results[name] = {"raised": False, "message": "", "passed": False}

    one_iteration = replace(config.solver.newton, max_iterations=1)
    expect(
        "iteration_limit",
        NonlinearSolveError,
        "did not converge",
        lambda: RabiesSolver(
            grid, parameters.diffusivity, parameters.transmission,
            config.domain.mortality, config.solver.dt, one_iteration,
        ).solve(susceptible, infectious, [0.0, 0.1]),
    )
    expect(
        "backtracking",
        NonlinearSolveError,
        "Backtracking line search failed",
        lambda: _CorruptedJacobianSolver(
            grid, parameters.diffusivity, parameters.transmission,
            config.domain.mortality, config.solver.dt, config.solver.newton,
        ).solve(susceptible, infectious, [0.0, 0.1]),
    )
    bad = infectious.copy()
    bad[3] = np.nan
    expect(
        "non_finite_input",
        ValueError,
        "finite",
        lambda: RabiesSolver(
            grid, parameters.diffusivity, parameters.transmission,
            config.domain.mortality, config.solver.dt,
        ).solve(susceptible, bad, [0.0, 0.1]),
    )
    return {**results, "passed": all(item["passed"] for item in results.values())}


# --------------------------------------------------------- refinement study


def refinement_pilots(config: FoxConfig) -> list[TrajectoryParameters]:
    pilots = corner_pilots(config)
    if config.refinement.random_pilots:
        sampling = replace(config.sampling, n_trajectories=max(3, config.refinement.random_pilots))
        rng = np.random.default_rng(config.seeds.refinement)
        pilots.extend(sample_parameters(sampling, rng)[: config.refinement.random_pilots])
    return pilots


def refinement_study(config: FoxConfig) -> dict[str, Any]:
    """Separate spatial (2x cells) and temporal (dt/2) refinement of pilots.

    The change is the learning-grid RMS difference of each pilot, per channel,
    divided by that channel's standard deviation pooled over all baseline
    pilot snapshots (the scale the model sees after normalization).
    """
    factor = config.solver.refinement_factor
    dt = config.solver.dt
    pilots = refinement_pilots(config)
    baseline, spatial, temporal = [], [], []
    start = time.perf_counter()
    for parameters in pilots:
        baseline.append(simulate(config, parameters)[:2])
        spatial.append(simulate(config, parameters, refinement_factor=2 * factor)[:2])
        temporal.append(simulate(config, parameters, dt=0.5 * dt)[:2])
    scales = [
        float(np.std(np.stack([fields[channel] for fields in baseline])))
        for channel in (0, 1)
    ]
    report: dict[str, Any] = {
        "baseline": {"solver_cells": config.solver_cells, "dt": dt},
        "spatial_refinement": {"solver_cells": 2 * config.solver_cells, "dt": dt},
        "temporal_refinement": {"solver_cells": config.solver_cells, "dt": 0.5 * dt},
        "normalization_std": {"S": scales[0], "I": scales[1]},
        "threshold": config.refinement.threshold,
        "pilots": [parameters.as_dict() for parameters in pilots],
    }
    passed = True
    for name, refined in (("spatial", spatial), ("temporal", temporal)):
        changes = {
            channel_name: [
                _rms(fine[channel] - coarse[channel]) / scales[channel]
                for coarse, fine in zip(baseline, refined)
            ]
            for channel, channel_name in ((0, "S"), (1, "I"))
        }
        worst = {key: max(values) for key, values in changes.items()}
        report[f"{name}_normalized_rms_change"] = changes
        report[f"{name}_max_normalized_rms_change"] = worst
        passed &= all(value < config.refinement.threshold for value in worst.values())
    report["wall_seconds"] = time.perf_counter() - start
    report["passed"] = bool(passed)
    return report


# ------------------------------------------------------------------- report


def run_verification(
    config: FoxConfig,
    output_path: str | Path | None = None,
    *,
    include_refinement: bool = True,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "diffusion_only_spatial": diffusion_only_convergence(config).as_dict(),
        "uniform_reaction_temporal": uniform_reaction_convergence(config).as_dict(),
        "zero_infection_equilibrium": zero_infection_equilibrium(config),
        "coupled_mms_spatial": coupled_spatial_convergence(config).as_dict(),
        "coupled_mms_temporal": coupled_temporal_convergence(config).as_dict(),
        "manufactured_balance": manufactured_balance(config),
        "jacobian_finite_difference": jacobian_finite_difference(),
        "balance_and_positivity": balance_and_positivity(config),
        "failure_modes": failure_modes(config),
    }
    if include_refinement:
        checks["refinement"] = refinement_study(config)
    report = {
        "config_sha256": config.sha256,
        "minimum_required_order": MINIMUM_ORDER,
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks.values()),
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-refinement", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    report = run_verification(
        config,
        args.output,
        include_refinement=not args.skip_refinement,
    )
    for name, check in report["checks"].items():
        detail = ""
        if "minimum_order" in check:
            detail = f" (min order {check['minimum_order']:.3f})"
        elif name == "refinement":
            detail = (
                f" (spatial {check['spatial_max_normalized_rms_change']}, "
                f"temporal {check['temporal_max_normalized_rms_change']})"
            )
        print(f"{'PASS' if check['passed'] else 'FAIL'}  {name}{detail}")
    print(f"Report written to {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
