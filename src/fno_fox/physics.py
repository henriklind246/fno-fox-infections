"""Coupled Crank–Nicolson finite-volume solver for the 1D fox-rabies model.

    S_t = -beta S I
    I_t = D I_xx + beta S I - mu I,     zero infectious flux at x = 0 and x = L.

Adapted from fno-tps-sizing ``src/fno_tps/physics.py``: conservative
cell-centred discretisation, a fixed sparse pattern whose data slots are
updated in place, damped Newton iterations with Armijo backtracking, and
per-step nonlinear diagnostics. The thermal residual and Jacobian are replaced
by the coupled S/I equations. Unlike the source solver, an exhausted line
search raises instead of accepting the last damped step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import math

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import splu

from fno_fox.config import FoxConfig, NewtonConfig, SolverConfig
from fno_fox.sampling import TrajectoryParameters, initial_fields


Array = np.ndarray
# source(x_centers, t) -> (f_S, f_I); used only by manufactured-solution checks.
SourceFunction = Callable[[Array, float], tuple[Array, Array]]


@dataclass(frozen=True)
class Grid:
    length: float
    n_cells: int

    def __post_init__(self) -> None:
        if self.length <= 0.0 or self.n_cells < 2:
            raise ValueError("Grid needs a positive length and at least two cells.")

    @property
    def h(self) -> float:
        return self.length / self.n_cells

    @property
    def faces(self) -> Array:
        return np.linspace(0.0, self.length, self.n_cells + 1)

    @property
    def centers(self) -> Array:
        faces = self.faces
        return 0.5 * (faces[:-1] + faces[1:])


def neumann_laplacian(n_cells: int, h: float) -> sparse.csr_matrix:
    """Cell-centred second difference with zero boundary flux.

    Every row sums to zero, so ``sum(h * L @ u) == 0`` and diffusion conserves
    the total infectious population exactly.
    """
    main = np.full(n_cells, -2.0)
    main[0] = main[-1] = -1.0
    off = np.ones(n_cells - 1)
    return (
        sparse.diags([off, main, off], [-1, 0, 1], format="csr") / (h * h)
    )


def conservative_average(values: Array, factor: int) -> Array:
    """Average blocks of ``factor`` fine cells onto the coarse grid."""
    array = np.asarray(values)
    if factor == 1:
        return array.copy()
    if array.shape[-1] % factor:
        raise ValueError("Fine-grid size must be a multiple of the averaging factor.")
    return array.reshape(*array.shape[:-1], array.shape[-1] // factor, factor).mean(
        axis=-1
    )


def interval_step_counts(save_times: Array, dt: float) -> list[int]:
    """Steps per save interval so every interval lands exactly on its end time."""
    counts = []
    for start, end in zip(save_times[:-1], save_times[1:]):
        span = float(end - start)
        counts.append(max(1, math.ceil(span / dt - 1.0e-9)))
    return counts


@dataclass
class Trajectory:
    times: Array
    x_centers: Array
    susceptible: Array
    infectious: Array
    step_times: Array
    population: Array
    expected_population: Array
    newton_iterations: Array
    backtracks: Array
    final_residual_norms: Array
    minimum_susceptible: float
    minimum_infectious: float

    @property
    def balance_residual(self) -> Array:
        return self.population - self.expected_population

    @property
    def relative_balance_residual(self) -> float:
        scale = max(abs(float(self.population[0])), 1.0e-300)
        return float(np.max(np.abs(self.balance_residual)) / scale)

    def averaged(self, factor: int) -> tuple[Array, Array]:
        return (
            conservative_average(self.susceptible, factor),
            conservative_average(self.infectious, factor),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "solver_steps": int(len(self.newton_iterations)),
            "newton_iterations_max": int(np.max(self.newton_iterations, initial=0)),
            "newton_iterations_mean": float(np.mean(self.newton_iterations))
            if len(self.newton_iterations)
            else 0.0,
            "backtracks_total": int(np.sum(self.backtracks)),
            "final_residual_norm_max": float(
                np.max(self.final_residual_norms, initial=0.0)
            ),
            "minimum_susceptible": self.minimum_susceptible,
            "minimum_infectious": self.minimum_infectious,
            "relative_balance_residual": self.relative_balance_residual,
            "finite": bool(
                np.isfinite(self.susceptible).all()
                and np.isfinite(self.infectious).all()
            ),
        }


def acceptance_report(
    trajectory: Trajectory,
    positivity_tolerance: float,
    balance_tolerance: float,
) -> dict[str, Any]:
    """Finite values, nonnegativity within tolerance, and discrete balance."""
    diagnostics = trajectory.diagnostics()
    reasons = []
    if not diagnostics["finite"]:
        reasons.append("non-finite density")
    if diagnostics["minimum_susceptible"] < -positivity_tolerance:
        reasons.append(
            f"S minimum {diagnostics['minimum_susceptible']:.3e} "
            f"< -{positivity_tolerance:.1e}"
        )
    if diagnostics["minimum_infectious"] < -positivity_tolerance:
        reasons.append(
            f"I minimum {diagnostics['minimum_infectious']:.3e} "
            f"< -{positivity_tolerance:.1e}"
        )
    if not diagnostics["relative_balance_residual"] <= balance_tolerance:
        reasons.append(
            "population balance residual "
            f"{diagnostics['relative_balance_residual']:.3e} "
            f"> {balance_tolerance:.1e}"
        )
    return {**diagnostics, "passed": not reasons, "reasons": reasons}


class NonlinearSolveError(RuntimeError):
    """Raised when a Crank–Nicolson step cannot be solved to tolerance."""


class RabiesSolver:
    """Coupled CN stepping for (S, I) on a uniform cell-centred grid.

    Unknowns are ordered in blocks ``u = [S; I]``. Each step solves

        R_S = S1 - S0 + dt/2 (beta S1 I1 + beta S0 I0) - q_S = 0
        R_I = I1 - I0 - dt/2 (D L I1 + D L I0)
                      - dt/2 (beta S1 I1 - mu I1 + beta S0 I0 - mu I0) - q_I = 0

    with the analytic Jacobian. ``q`` is the trapezoidal source increment and
    is zero outside manufactured-solution checks.
    """

    def __init__(
        self,
        grid: Grid,
        diffusivity: float,
        transmission: float,
        mortality: float,
        dt: float,
        newton: NewtonConfig | None = None,
        source: SourceFunction | None = None,
    ):
        if diffusivity < 0.0 or transmission < 0.0 or mortality < 0.0:
            raise ValueError("D, beta, and mu must be nonnegative.")
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        self.grid = grid
        self.diffusivity = float(diffusivity)
        self.transmission = float(transmission)
        self.mortality = float(mortality)
        self.dt = float(dt)
        self.newton = NewtonConfig() if newton is None else newton
        self.source = source
        self.x = grid.centers
        self.laplacian = neumann_laplacian(grid.n_cells, grid.h)
        self.factorization_count = 0
        self._initialize_sparse_pattern()
        self._base_cache: dict[float, Array] = {}

    @classmethod
    def from_config(
        cls,
        config: SolverConfig,
        grid: Grid,
        diffusivity: float,
        transmission: float,
        mortality: float,
        *,
        source: SourceFunction | None = None,
    ) -> "RabiesSolver":
        return cls(
            grid,
            diffusivity,
            transmission,
            mortality,
            config.dt,
            config.newton,
            source=source,
        )

    def _initialize_sparse_pattern(self) -> None:
        n = self.grid.n_cells
        size = 2 * n
        cells = np.arange(n, dtype=np.int64)
        lap = self.laplacian.tocoo()
        rows = np.concatenate((cells, cells, n + cells, n + lap.row, n + cells))
        cols = np.concatenate((cells, n + cells, cells, n + lap.col, n + cells))
        pattern = sparse.csc_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(size, size),
        )
        pattern.sum_duplicates()
        pattern.sort_indices()
        csc_cols = np.repeat(np.arange(size, dtype=np.int64), np.diff(pattern.indptr))
        csc_keys = csc_cols * size + pattern.indices
        keys = cols * size + rows
        slots = np.searchsorted(csc_keys, keys)
        if not np.array_equal(csc_keys[slots], keys):
            raise RuntimeError("Jacobian triplet does not map to the CSC pattern.")
        self._indptr = pattern.indptr.copy()
        self._indices = pattern.indices.copy()
        self._nnz = len(csc_keys)
        self._slot_ss = slots[:n]
        self._slot_si = slots[n : 2 * n]
        self._slot_is = slots[2 * n : 3 * n]
        self._slot_lap = slots[3 * n : 3 * n + lap.nnz]
        self._slot_ii = slots[3 * n + lap.nnz :]
        self._lap_values = lap.data.copy()

    def _base_data(self, dt: float) -> Array:
        key = float(dt)
        cached = self._base_cache.get(key)
        if cached is None:
            half = 0.5 * dt
            cached = np.zeros(self._nnz, dtype=np.float64)
            cached[self._slot_ss] += 1.0
            cached[self._slot_lap] -= half * self.diffusivity * self._lap_values
            cached[self._slot_ii] += 1.0 + half * self.mortality
            self._base_cache[key] = cached
        return cached

    def jacobian(self, susceptible: Array, infectious: Array, dt: float) -> sparse.csc_matrix:
        """Analytic Jacobian of the CN residual with respect to (S1, I1)."""
        half = 0.5 * dt
        beta = self.transmission
        data = self._base_data(dt).copy()
        data[self._slot_ss] += half * beta * infectious
        data[self._slot_si] += half * beta * susceptible
        data[self._slot_is] -= half * beta * infectious
        data[self._slot_ii] -= half * beta * susceptible
        size = 2 * self.grid.n_cells
        return sparse.csc_matrix(
            (data, self._indices.copy(), self._indptr.copy()),
            shape=(size, size),
        )

    def _source_increment(self, t0: float, t1: float) -> tuple[Array, Array]:
        n = self.grid.n_cells
        if self.source is None:
            return np.zeros(n), np.zeros(n)
        s0, i0 = self.source(self.x, t0)
        s1, i1 = self.source(self.x, t1)
        half = 0.5 * (t1 - t0)
        increment_s = half * (np.asarray(s0, dtype=np.float64) + s1)
        increment_i = half * (np.asarray(i0, dtype=np.float64) + i1)
        if increment_s.shape != (n,) or increment_i.shape != (n,):
            raise ValueError("source must return two arrays of shape (n_cells,).")
        return increment_s, increment_i

    def residual(
        self,
        susceptible: Array,
        infectious: Array,
        old_susceptible: Array,
        old_infectious: Array,
        dt: float,
        source_increment: tuple[Array, Array] | None = None,
    ) -> Array:
        """CN residual for a candidate (S1, I1); used directly by tests."""
        half = 0.5 * dt
        beta = self.transmission
        old_rate = beta * old_susceptible * old_infectious
        old_diffusion = self.diffusivity * (self.laplacian @ old_infectious)
        rate = beta * susceptible * infectious
        residual_s = susceptible - old_susceptible + half * (rate + old_rate)
        residual_i = (
            infectious
            - old_infectious
            - half * (self.diffusivity * (self.laplacian @ infectious) + old_diffusion)
            - half * (rate + old_rate)
            + half * self.mortality * (infectious + old_infectious)
        )
        if source_increment is not None:
            residual_s = residual_s - source_increment[0]
            residual_i = residual_i - source_increment[1]
        return np.concatenate((residual_s, residual_i))

    def step(
        self,
        susceptible: Array,
        infectious: Array,
        dt: float,
        *,
        t0: float = 0.0,
        guess: tuple[Array, Array] | None = None,
        step_index: int = -1,
    ) -> tuple[Array, Array, dict[str, Any]]:
        """Advance one CN step with damped Newton iterations."""
        n = self.grid.n_cells
        controls = self.newton
        t1 = t0 + dt
        increment = self._source_increment(t0, t1)

        def evaluate(state: Array) -> Array:
            return self.residual(
                state[:n], state[n:], susceptible, infectious, dt, increment
            )

        if guess is None:
            state = np.concatenate((susceptible, infectious))
        else:
            state = np.concatenate(guess)
        residual = evaluate(state)
        residual_norm = float(np.max(np.abs(residual)))
        if not np.isfinite(residual_norm):
            raise NonlinearSolveError(
                f"Non-finite residual at the initial guess: step={step_index}, "
                f"t={t1:.9g}."
            )
        damping_history: list[float] = []
        backtracks = 0
        update_norm = float("inf")
        for iteration in range(1, controls.max_iterations + 1):
            jacobian = self.jacobian(state[:n], state[n:], dt)
            delta = splu(jacobian).solve(-residual)
            self.factorization_count += 1
            if not np.isfinite(delta).all():
                raise NonlinearSolveError(
                    f"Non-finite Newton update: step={step_index}, t={t1:.9g}, "
                    f"iteration={iteration}."
                )
            damping = 1.0
            for attempt in range(controls.max_backtracks + 1):
                trial = state + damping * delta
                trial_residual = evaluate(trial)
                trial_norm = float(np.max(np.abs(trial_residual)))
                sufficient = trial_norm <= (1.0 - controls.armijo_c * damping) * residual_norm
                # At the roundoff floor the step only polishes the iterate; an
                # Armijo decrease is neither expected nor required there.
                if np.isfinite(trial_norm) and (
                    sufficient or residual_norm <= controls.residual_tolerance
                ):
                    break
                if attempt == controls.max_backtracks:
                    raise NonlinearSolveError(
                        "Backtracking line search failed: "
                        f"step={step_index}, t={t1:.9g}, iteration={iteration}, "
                        f"||R||_inf={residual_norm:.6e}, "
                        f"last trial ||R||_inf={trial_norm:.6e}, "
                        f"damping={damping:.3e} after {attempt} halvings."
                    )
                damping *= 0.5
                backtracks += 1
            damping_history.append(damping)
            update_norm = damping * float(np.max(np.abs(delta)))
            state = trial
            residual = trial_residual
            residual_norm = trial_norm
            update_limit = (
                controls.update_tolerance
                + controls.update_relative_tolerance * float(np.max(np.abs(state)))
            )
            if residual_norm <= controls.residual_tolerance and update_norm <= update_limit:
                info = {
                    "iterations": iteration,
                    "backtracks": backtracks,
                    "residual_norm": residual_norm,
                }
                return state[:n].copy(), state[n:].copy(), info
        worst = int(np.argmax(np.abs(residual)))
        field = "S" if worst < n else "I"
        raise NonlinearSolveError(
            "Newton iteration did not converge: "
            f"step={step_index}, t={t1:.9g}, "
            f"iterations={controls.max_iterations}, "
            f"||R||_inf={residual_norm:.6e} (tol {controls.residual_tolerance:.1e}), "
            f"||delta||_inf={update_norm:.6e}, "
            f"damping_history={damping_history}, "
            f"worst_cell=({field}, {worst % n})."
        )

    def solve(
        self,
        initial_susceptible: Array,
        initial_infectious: Array,
        save_times: Array | list[float],
    ) -> Trajectory:
        n = self.grid.n_cells
        susceptible = np.asarray(initial_susceptible, dtype=np.float64).copy()
        infectious = np.asarray(initial_infectious, dtype=np.float64).copy()
        if susceptible.shape != (n,) or infectious.shape != (n,):
            raise ValueError(f"Initial fields must have shape ({n},).")
        if not (np.isfinite(susceptible).all() and np.isfinite(infectious).all()):
            raise ValueError("Initial fields must be finite.")
        times = np.asarray(save_times, dtype=np.float64)
        if times.ndim != 1 or len(times) < 1 or times[0] != 0.0:
            raise ValueError("save_times must be a 1D sequence starting at 0.")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("save_times must be strictly increasing.")

        counts = interval_step_counts(times, self.dt)
        total_steps = int(sum(counts))
        saved_s = np.empty((len(times), n), dtype=np.float64)
        saved_i = np.empty_like(saved_s)
        saved_s[0] = susceptible
        saved_i[0] = infectious
        h = self.grid.h
        step_times = np.empty(total_steps + 1, dtype=np.float64)
        population = np.empty(total_steps + 1, dtype=np.float64)
        expected = np.empty(total_steps + 1, dtype=np.float64)
        iterations = np.zeros(total_steps, dtype=np.int32)
        backtracks = np.zeros(total_steps, dtype=np.int32)
        residual_norms = np.zeros(total_steps, dtype=np.float64)
        step_times[0] = 0.0
        population[0] = h * float(np.sum(susceptible + infectious))
        expected[0] = population[0]
        minimum_s = float(np.min(susceptible))
        minimum_i = float(np.min(infectious))

        previous: tuple[Array, Array, float] | None = None
        step = 0
        for interval, count in enumerate(counts):
            start, end = float(times[interval]), float(times[interval + 1])
            dt = (end - start) / count
            for local in range(count):
                t0 = start + local * dt
                guess = None
                if previous is not None:
                    ratio = dt / previous[2]
                    guess = (
                        susceptible + ratio * (susceptible - previous[0]),
                        infectious + ratio * (infectious - previous[1]),
                    )
                new_s, new_i, info = self.step(
                    susceptible,
                    infectious,
                    dt,
                    t0=t0,
                    guess=guess,
                    step_index=step,
                )
                source_s, source_i = self._source_increment(t0, t0 + dt)
                # Same trapezoidal mortality term as the residual: the discrete
                # balance closes to the Newton tolerance.
                expected[step + 1] = (
                    expected[step]
                    - 0.5 * dt * self.mortality * h * float(np.sum(infectious + new_i))
                    + h * float(np.sum(source_s + source_i))
                )
                previous = (susceptible, infectious, dt)
                susceptible, infectious = new_s, new_i
                population[step + 1] = h * float(np.sum(susceptible + infectious))
                step_times[step + 1] = end if local == count - 1 else t0 + dt
                iterations[step] = info["iterations"]
                backtracks[step] = info["backtracks"]
                residual_norms[step] = info["residual_norm"]
                minimum_s = min(minimum_s, float(np.min(susceptible)))
                minimum_i = min(minimum_i, float(np.min(infectious)))
                step += 1
            saved_s[interval + 1] = susceptible
            saved_i[interval + 1] = infectious

        return Trajectory(
            times=times,
            x_centers=self.x.copy(),
            susceptible=saved_s,
            infectious=saved_i,
            step_times=step_times,
            population=population,
            expected_population=expected,
            newton_iterations=iterations,
            backtracks=backtracks,
            final_residual_norms=residual_norms,
            minimum_susceptible=minimum_s,
            minimum_infectious=minimum_i,
        )


def simulate(
    config: FoxConfig,
    parameters: TrajectoryParameters,
    save_times: Array | list[float] | None = None,
    *,
    refinement_factor: int | None = None,
    dt: float | None = None,
) -> tuple[Array, Array, Trajectory]:
    """Solve one trajectory and average it onto the learning grid."""
    factor = config.solver.refinement_factor if refinement_factor is None else int(refinement_factor)
    grid = Grid(config.domain.length, config.domain.learning_cells * factor)
    susceptible, infectious = initial_fields(parameters, grid.faces, config.domain.length)
    solver = RabiesSolver(
        grid,
        parameters.diffusivity,
        parameters.transmission,
        config.domain.mortality,
        config.solver.dt if dt is None else float(dt),
        config.solver.newton,
    )
    trajectory = solver.solve(
        susceptible,
        infectious,
        config.domain.save_times if save_times is None else save_times,
    )
    learning_s, learning_i = trajectory.averaged(factor)
    return learning_s, learning_i, trajectory
