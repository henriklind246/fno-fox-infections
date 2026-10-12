"""Deterministic two-parameter recovery and local sensitivity diagnostics.

Adapted conceptually from no-tps-ihcp scripts/invert.py: multistart recovery,
full-box loss surfaces, and Jacobian SVD. No likelihood or error covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import torch
from scipy.optimize import OptimizeResult, minimize, minimize_scalar
from scipy.stats import qmc

from fno_fox.config import InverseConfig
from fno_fox.data import DatasetBundle, FieldEncoder


class _EvaluationBudgetReached(Exception):
    pass


@dataclass(frozen=True)
class FullFieldObservations:
    source: np.ndarray
    targets: np.ndarray
    physical_targets: np.ndarray
    times: np.ndarray


def observations(bundle: DatasetBundle, encoder: FieldEncoder, sim: int) -> FullFieldObservations:
    return FullFieldObservations(
        source=encoder.spatial(bundle.susceptible[sim, 0], bundle.infectious[sim, 0], bundle.x_centers),
        targets=encoder.target(bundle.susceptible[sim, 1:], bundle.infectious[sim, 1:]),
        physical_targets=np.stack((bundle.susceptible[sim, 1:], bundle.infectious[sim, 1:]), axis=-1),
        times=np.asarray(bundle.times[1:] - bundle.times[0], dtype=np.float64),
    )


class FieldObjective:
    """Truth-free inverse interface; parameter truth is never stored here."""

    def __init__(self, model, encoder: FieldEncoder, obs: FullFieldObservations,
                 device: torch.device, batch_size: int, sensor_indices=None,
                 prediction_cache=None, prediction_stats=None):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.encoder, self.obs = encoder, obs
        self.device, self.batch_size = device, batch_size
        self.prediction_cache, self.prediction_stats = prediction_cache, prediction_stats
        self.bounds = np.asarray((encoder.diffusivity_range, encoder.transmission_range), dtype=np.float64)
        self.low = self.bounds[:, 0]
        self.ranges = np.diff(self.bounds, axis=1).ravel()
        self.source = torch.as_tensor(obs.source, device=device)
        self.times = torch.as_tensor(obs.times / encoder.horizon, dtype=torch.float32, device=device)
        nx = obs.targets.shape[1]
        indices = np.arange(nx) if sensor_indices is None else np.asarray(sensor_indices)
        if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
                or np.any(indices < 0) or np.any(indices >= nx)
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Sensor indices must be distinct integer learning-grid cell indices.")
        self.sensor_indices = indices.astype(np.int64)

    def observe(self, fields: np.ndarray) -> np.ndarray:
        return np.take(fields, self.sensor_indices, axis=-2)

    def physical_parameters(self, z: np.ndarray) -> np.ndarray:
        return self.low + np.asarray(z) * self.ranges

    def unit_parameters(self, theta: np.ndarray) -> np.ndarray:
        return (np.asarray(theta) - self.low) / self.ranges

    def _forward(self, z: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        cond = torch.cat((z.expand(len(time), 2), time[:, None]), dim=1)
        return self.model(self.source.expand(len(time), -1, -1), cond)

    @torch.no_grad()
    def predict_many(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        key = tuple(points[0]) if self.prediction_cache is not None and len(points) == 1 else None
        if key is not None and key in self.prediction_cache:
            self.prediction_cache.move_to_end(key)
            if self.prediction_stats is not None:
                self.prediction_stats["cache_hits"] += 1
            return self.prediction_cache[key].copy()
        nt = len(self.times)
        # Batch field queries, not parameter pairs, so memory stays bounded as Nt grows.
        result = np.empty((len(points) * nt, *self.obs.targets.shape[1:]), dtype=np.float32)
        for first in range(0, len(result), self.batch_size):
            idx = np.arange(first, min(first + self.batch_size, len(result)))
            z = torch.as_tensor(points[idx // nt], dtype=torch.float32, device=self.device)
            time = self.times[torch.as_tensor(idx % nt, device=self.device)]
            cond = torch.cat((z, time[:, None]), dim=1)
            prediction = self.model(self.source.expand(len(idx), -1, -1), cond)
            if self.prediction_stats is not None:
                self.prediction_stats["field_queries"] += len(idx)
            values = prediction.cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("Non-finite FNO prediction during inverse evaluation.")
            result[first:first + len(idx)] = values
        result = result.reshape(len(points), *self.obs.targets.shape)
        if self.prediction_stats is not None:
            self.prediction_stats["parameter_predictions"] += len(points)
        if key is not None:
            self.prediction_cache[key] = result.copy()
            # A per-trajectory LRU bounds storage during thousands of optimizer queries.
            if len(self.prediction_cache) > 64:
                self.prediction_cache.popitem(last=False)
        return result

    def losses(self, points: np.ndarray) -> np.ndarray:
        # Keep the full 625 x Nt x Nx x 2 grid out of memory on larger datasets.
        count = max(1, self.batch_size // len(self.times))
        result = []
        for first in range(0, len(points), count):
            values = self.predict_many(points[first:first + count]).astype(np.float64)
            residual = self.observe(values) - self.observe(self.obs.targets)
            result.extend(0.5 * np.mean(residual ** 2, axis=(1, 2, 3)))
        return np.asarray(result)

    def __call__(self, z: np.ndarray) -> float:
        return float(self.losses(np.asarray(z).reshape(1, 2))[0])

    def physical_prediction(self, z: np.ndarray) -> np.ndarray:
        pred = self.predict_many(np.asarray(z).reshape(1, 2))[0]
        return np.stack(self.encoder.decode(pred), axis=-1)

    def jacobians(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        physical, scaled = self.full_jacobians(z)
        return select_jacobians(physical, scaled, self.sensor_indices)

    def full_jacobians(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Physical J[..., output_channel, parameter] and dimensionless flattened Js."""
        point = torch.tensor(z, dtype=torch.float32, device=self.device)
        chunks = []
        for first in range(0, len(self.times), self.batch_size):
            time = self.times[first:first + self.batch_size]
            columns = []
            for direction in torch.eye(2, device=self.device):
                _, tangent = torch.autograd.functional.jvp(
                    lambda p: self._forward(p, time), point, direction, strict=False,
                )
                columns.append(tangent.detach().cpu().numpy())
            chunks.append(np.stack(columns, axis=-1))
        normalized = np.concatenate(chunks).astype(np.float64)
        if not np.isfinite(normalized).all():
            raise ValueError("Non-finite FNO parameter sensitivity.")
        std = np.asarray((self.encoder.s_std, self.encoder.i_std))
        physical = normalized * std[None, None, :, None] / self.ranges
        scaled = normalized.reshape(-1, 2) / np.sqrt(normalized.size / 2)
        return physical, scaled


def select_jacobians(physical: np.ndarray, scaled: np.ndarray, indices: np.ndarray):
    selected = np.take(physical, indices, axis=1)
    selected_scaled = np.take(scaled.reshape(physical.shape), indices, axis=1).reshape(-1, 2)
    selected_scaled *= np.sqrt(len(scaled) / len(selected_scaled))
    return selected, selected_scaled


def sensitivity_summary(jacobian: np.ndarray) -> dict:
    matrix = np.asarray(jacobian, dtype=np.float64).reshape(-1, 2)
    norms = np.linalg.norm(matrix, axis=0)
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    weak = vt[-1].copy()
    # Fix the arbitrary SVD sign to make saved directions comparable across runs.
    if weak[np.argmax(np.abs(weak))] < 0:
        weak *= -1
    return {
        "observed_scalars": len(matrix),
        "column_norms": norms.tolist(),
        "column_cosine": float(np.clip(matrix[:, 0] @ matrix[:, 1] / np.prod(norms), -1, 1))
        if np.all(norms > 0) else None,
        "singular_values": singular.tolist(),
        "condition_number": float(singular[0] / singular[-1]) if rank == 2 else None,
        "numerical_rank": rank, "rank_deficient": rank < 2,
        "weakest_direction": weak.tolist() if singular[0] > 0 else None,
        "weakest_direction_unique": bool(singular[0] - singular[-1] > tolerance),
        "zero_columns": (norms == 0).tolist(),
        "rank_tolerance": float(tolerance),
    }


def joint_grid(losses: Callable, size: int) -> dict:
    axis = np.linspace(0, 1, size)
    d, beta = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((d.ravel(), beta.ravel()), axis=-1)
    start = perf_counter()
    loss = np.asarray(losses(points), dtype=np.float64).reshape(size, size)
    if not np.isfinite(loss).all():
        raise ValueError("Non-finite joint-grid loss.")
    return {"axis": axis, "loss": loss, "best_z": points[np.argmin(loss)],
            "evaluations": len(points), "wall_seconds": perf_counter() - start}


def shared_joint_grids(objective: FieldObjective, layouts: dict, size: int) -> tuple[dict, dict]:
    """One full-grid forward per parameter pair, reduced for every sensor layout."""
    axis = np.linspace(0, 1, size)
    d, beta = np.meshgrid(axis, axis, indexing="ij")
    points = np.stack((d.ravel(), beta.ravel()), axis=-1)
    losses = {name: np.empty(len(points)) for name in layouts}
    batch = max(1, objective.batch_size // len(objective.times))
    begin = perf_counter()
    for first in range(0, len(points), batch):
        values = objective.predict_many(points[first:first + batch]).astype(np.float64)
        residual = values - objective.obs.targets
        for name, indices in layouts.items():
            selected = np.take(residual, indices, axis=2)
            losses[name][first:first + len(values)] = .5 * np.mean(selected ** 2, axis=(1, 2, 3))
    grids = {}
    for name, values in losses.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite joint-grid loss for {name}.")
        grids[name] = {"axis": axis, "loss": values.reshape(size, size),
                       "best_z": points[np.argmin(values)], "evaluations": 0,
                       "wall_seconds": 0., "shared_grid": True,
                       "logical_grid_evaluations": len(points)}
    cost = {"parameter_predictions": len(points), "field_queries": len(points) * len(objective.times),
            "wall_seconds": perf_counter() - begin}
    return grids, cost


def recover(objective: Callable, grid: dict, config: InverseConfig, seed: int) -> dict:
    """Eight starts run once; four/eight summaries share the grid and edge candidates."""
    starts = np.vstack((grid["best_z"], qmc.LatinHypercube(d=2, seed=seed).random(7)))
    traces, runs, shared = [], [], []
    axis, loss = grid["axis"], grid["loss"]
    best_idx = np.unravel_index(np.argmin(loss), loss.shape)
    shared.append({"z": np.asarray(grid["best_z"]), "loss": float(loss[best_idx]),
                   "source": "grid", "converged": None})
    for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
        shared.append({"z": np.array([axis[i], axis[j]]), "loss": float(loss[i, j]),
                       "source": "corner", "converged": None})

    def traced(z, phase, index):
        value = float(objective(np.asarray(z)))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite loss in {phase} {index}.")
        traces.append({"phase": phase, "index": index, "z_D": float(z[0]),
                       "z_beta": float(z[1]), "loss": value})
        return value

    for index, start in enumerate(starts):
        begin = perf_counter()
        # Explicit nondegenerate initial simplex, including starts exactly on an edge.
        simplex = np.tile(start, (3, 1))
        for k in range(2):
            simplex[k + 1, k] += 0.05 if start[k] <= 0.95 else -0.05
        result = minimize(lambda z: traced(z, "start", index), start, method="Nelder-Mead",
                          bounds=[(0, 1)] * 2,
                          options={"maxiter": config.max_iterations,
                                   "maxfev": config.max_evaluations,
                                   "xatol": config.coordinate_tolerance,
                                   "fatol": config.objective_tolerance,
                                   "initial_simplex": simplex})
        runs.append({"start": start, "z": result.x, "loss": float(result.fun),
                     "source": f"start_{index}", "converged": bool(result.success),
                     "status": int(result.status), "message": str(result.message),
                     "iterations": int(result.nit), "evaluations": int(result.nfev),
                     "budget_exhausted": bool(not result.success and result.status in (1, 2)),
                     "wall_seconds": perf_counter() - begin})

    edges = []
    for fixed in range(2):
        for side in (0, 1):
            values = loss[side * (len(axis) - 1), :] if fixed == 0 else loss[:, side * (len(axis) - 1)]
            idx = int(np.argmin(values))
            bracket = (axis[max(0, idx - 1)], axis[min(len(axis) - 1, idx + 1)])
            point = np.empty(2)
            point[fixed] = side
            point[1 - fixed] = axis[idx]
            shared.append({"z": point.copy(), "loss": float(values[idx]),
                           "source": f"edge_{fixed}_{side}_grid", "converged": None})
            edge_evaluations = []

            def edge_objective(free):
                # SciPy's scalar solver can exceed maxiter=1 during initialization.
                if len(edge_evaluations) >= config.edge_max_evaluations:
                    raise _EvaluationBudgetReached
                point[1 - fixed] = free
                value = traced(point, "edge", 2 * fixed + side)
                edge_evaluations.append((free, value))
                return value

            begin = perf_counter()
            try:
                result = minimize_scalar(edge_objective, bounds=bracket, method="bounded",
                                         options={"maxiter": config.edge_max_evaluations,
                                                  "xatol": config.coordinate_tolerance})
            except _EvaluationBudgetReached:
                x, value = min(edge_evaluations, key=lambda pair: pair[1])
                result = OptimizeResult(x=x, fun=value, success=False, status=1,
                                        message="Edge evaluation budget exhausted.",
                                        nfev=len(edge_evaluations))
            point[1 - fixed] = result.x
            record = {"z": point.copy(), "loss": float(result.fun),
                      "source": f"edge_{fixed}_{side}", "converged": bool(result.success),
                      "status": int(result.status), "message": str(result.message),
                      "evaluations": int(result.nfev),
                      "budget_exhausted": bool(not result.success and result.nfev >= config.edge_max_evaluations),
                      "wall_seconds": perf_counter() - begin}
            edges.append(record)
            shared.append(record)

    arms = {}
    for count in (4, 8):
        selected = min([*runs[:count], *shared], key=lambda r: r["loss"])
        arms[str(count)] = {
            **{key: selected[key] for key in ("z", "loss", "source", "converged")},
            "on_boundary": bool(np.any((selected["z"] <= config.coordinate_tolerance)
                                       | (selected["z"] >= 1 - config.coordinate_tolerance))),
            "evaluations": grid["evaluations"] + sum(r["evaluations"] for r in runs[:count] + edges),
            "wall_seconds": grid["wall_seconds"] + sum(r["wall_seconds"] for r in runs[:count] + edges),
            "nonconverged_starts": sum(not r["converged"] for r in runs[:count]),
            "nonconverged_edges": sum(not r["converged"] for r in edges),
        }
    change = np.abs(arms["8"]["z"] - arms["4"]["z"])
    improvement = arms["4"]["loss"] - arms["8"]["loss"]
    return {"seed": seed, "arms": arms, "starts": runs, "edges": edges, "trace": traces,
            "parameter_change_fraction": change,
            "loss_improvement": improvement,
            "parameter_change_flag": bool(np.any(change > config.parameter_change_tolerance)),
            "loss_change_flag": bool(improvement > max(config.loss_change_absolute,
                                                       config.loss_change_relative * arms["4"]["loss"]))}
