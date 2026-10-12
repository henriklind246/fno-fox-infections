from __future__ import annotations

import numpy as np
import pytest

from fno_fox.config import NewtonConfig
from fno_fox.physics import (
    Grid,
    NonlinearSolveError,
    RabiesSolver,
    acceptance_report,
    conservative_average,
    interval_step_counts,
    neumann_laplacian,
    simulate,
)
from fno_fox.sampling import initial_fields
from fno_fox.verification import (
    _CorruptedJacobianSolver,
    corner_pilots,
    coupled_spatial_convergence,
    coupled_temporal_convergence,
    diffusion_only_convergence,
    failure_modes,
    jacobian_finite_difference,
    manufactured_balance,
    refinement_study,
    uniform_reaction_convergence,
    zero_infection_equilibrium,
)


def _pilot_solver(config, cells=64, newton=None, cls=RabiesSolver):
    parameters = corner_pilots(config)[0]
    grid = Grid(config.domain.length, cells)
    solver = cls(
        grid,
        parameters.diffusivity,
        parameters.transmission,
        config.domain.mortality,
        config.solver.dt,
        newton,
    )
    return solver, initial_fields(parameters, grid.faces, config.domain.length)


def test_laplacian_is_conservative_and_symmetric() -> None:
    lap = neumann_laplacian(10, 0.5)
    np.testing.assert_allclose(np.asarray(lap.sum(axis=1)).ravel(), 0.0, atol=1e-14)
    np.testing.assert_allclose(lap.toarray(), lap.toarray().T)


def test_diffusion_only_matches_analytic_solution(default_config) -> None:
    result = diffusion_only_convergence(default_config)
    assert result.errors[-1] < 1.0e-4
    assert result.minimum_order >= 1.8


def test_uniform_reaction_matches_high_accuracy_ode(default_config) -> None:
    result = uniform_reaction_convergence(default_config)
    assert result.errors[-1] < 1.0e-5
    assert result.minimum_order >= 1.8


def test_zero_infection_is_exact_equilibrium(default_config) -> None:
    assert zero_infection_equilibrium(default_config)["passed"]


def test_coupled_manufactured_solution_is_second_order(default_config) -> None:
    assert coupled_spatial_convergence(default_config).minimum_order >= 1.8
    assert coupled_temporal_convergence(default_config).minimum_order >= 1.8


def test_balance_closes_with_manufactured_sources(default_config) -> None:
    assert manufactured_balance(default_config)["passed"]


def test_analytic_jacobian_matches_finite_difference() -> None:
    assert jacobian_finite_difference()["relative_max_difference"] < 1.0e-8


def test_production_trajectory_is_positive_and_balanced(default_config) -> None:
    for parameters in corner_pilots(default_config)[:2]:
        _, _, trajectory = simulate(default_config, parameters)
        report = acceptance_report(trajectory, 1.0e-10, 1.0e-10)
        assert report["passed"], report["reasons"]
        assert trajectory.relative_balance_residual < 1.0e-12
        assert trajectory.minimum_infectious > -1.0e-12
        # Mortality must actually remove population, not merely balance.
        assert trajectory.population[-1] < trajectory.population[0]


def test_balance_uses_trapezoidal_mortality(default_config) -> None:
    solver, (s0, i0) = _pilot_solver(default_config)
    trajectory = solver.solve(s0, i0, [0.0, 0.05])
    h = solver.grid.h
    # Recompute the first step's expected population independently.
    one_step = solver.solve(s0, i0, [0.0, solver.dt])
    expected = h * np.sum(s0 + i0) - 0.5 * solver.dt * solver.mortality * h * np.sum(
        i0 + one_step.infectious[-1]
    )
    assert one_step.population[-1] == pytest.approx(expected, rel=1e-13)
    assert trajectory.relative_balance_residual < 1.0e-13


def test_save_times_are_hit_exactly(default_config) -> None:
    times = [0.0, 0.125, 0.25, 0.3]
    counts = interval_step_counts(np.asarray(times), 0.01)
    assert counts == [13, 13, 5]
    solver, (s0, i0) = _pilot_solver(default_config)
    trajectory = solver.solve(s0, i0, times)
    np.testing.assert_array_equal(trajectory.times, times)
    assert trajectory.step_times[-1] == 0.3
    assert len(trajectory.step_times) == sum(counts) + 1


def test_conservative_average_preserves_integral() -> None:
    fine = np.random.default_rng(0).random((3, 512))
    coarse = conservative_average(fine, 2)
    assert coarse.shape == (3, 256)
    np.testing.assert_allclose(coarse.sum(axis=-1) * 2.0, fine.sum(axis=-1))


def test_initial_fields_are_grid_consistent(default_config) -> None:
    parameters = corner_pilots(default_config)[0]
    length = default_config.domain.length
    coarse = initial_fields(parameters, Grid(length, 256).faces, length)
    fine = initial_fields(parameters, Grid(length, 1024).faces, length)
    for coarse_field, fine_field in zip(coarse, fine):
        np.testing.assert_allclose(conservative_average(fine_field, 4), coarse_field, atol=1e-14)


def test_iteration_limit_raises_informative_error(default_config) -> None:
    newton = NewtonConfig(max_iterations=1)
    solver, fields = _pilot_solver(default_config, newton=newton)
    with pytest.raises(NonlinearSolveError, match=r"did not converge: step=0.*worst_cell"):
        solver.solve(*fields, [0.0, 0.1])


def test_failed_backtracking_raises(default_config) -> None:
    solver, fields = _pilot_solver(default_config, cls=_CorruptedJacobianSolver)
    with pytest.raises(NonlinearSolveError, match="Backtracking line search failed"):
        solver.solve(*fields, [0.0, 0.1])


def test_invalid_inputs_raise(default_config) -> None:
    solver, (s0, i0) = _pilot_solver(default_config)
    bad = i0.copy()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        solver.solve(s0, bad, [0.0, 0.1])
    with pytest.raises(ValueError, match="strictly increasing"):
        solver.solve(s0, i0, [0.0, 0.2, 0.1])
    with pytest.raises(ValueError, match="starting at 0"):
        solver.solve(s0, i0, [0.1, 0.2])


def test_acceptance_report_flags_negative_and_unbalanced(default_config) -> None:
    solver, fields = _pilot_solver(default_config)
    trajectory = solver.solve(*fields, [0.0, 0.05])
    trajectory.minimum_infectious = -1.0e-6
    trajectory.expected_population = trajectory.expected_population * (1.0 + 1e-6)
    report = acceptance_report(trajectory, 1.0e-10, 1.0e-10)
    assert not report["passed"]
    assert any("I minimum" in reason for reason in report["reasons"])
    assert any("balance" in reason for reason in report["reasons"])


def test_failure_mode_report(default_config) -> None:
    assert failure_modes(default_config)["passed"]


@pytest.mark.slow
def test_refinement_study_meets_threshold(default_config) -> None:
    report = refinement_study(default_config)
    assert report["passed"]
    for name in ("spatial", "temporal"):
        assert max(report[f"{name}_max_normalized_rms_change"].values()) < 0.005
