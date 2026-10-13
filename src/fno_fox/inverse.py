"""Noiseless, full-field parameter recovery with deterministic identifiability diagnostics.

    python -m fno_fox.inverse --config conf/default.yaml --data data/fox_rabies \
        --checkpoint runs/default/best.pt --output runs/default/inverse
"""
from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch

from fno_fox.config import FoxConfig, config_from_dict, load_config
from fno_fox.data import DatasetBundle, FieldEncoder, capture_git_state, file_sha256, load_dataset
from fno_fox.evaluate import trajectory_parameters
from fno_fox.inverse_numerics import (
    FieldObjective, joint_grid, observations, recover, select_jacobians, sensitivity_summary,
)
from fno_fox.physics import acceptance_report, simulate
from fno_fox.train import check_dataset_compatibility, load_checkpoint, resolve_device


PARAMETERS = ("D", "beta")
CHANNELS = ("S", "I")
METRIC_SCOPE_LABELS = {
    "sensor": "Fitting observations: selected sensors at selected positive times.",
    "full_grid_selected_times": "Evaluation against ground truth: all cells at selected times, including spatially withheld values.",
    "full_grid": "Evaluation against ground truth: all cells at all saved positive times, including spatially and temporally withheld values.",
}
CHANNEL_SCOPE_LABELS = {
    **METRIC_SCOPE_LABELS,
    "sensor": "Fitting observations: selected channels at selected sensors and selected positive times.",
    "full_grid_selected_times": "Evaluation against ground truth: both S and I, all cells at selected times, including withheld channels and cells.",
    "full_grid": "Evaluation against ground truth: both S and I, all cells at all saved positive times, including withheld channels, cells, and times.",
}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    # Rows from configurations observing different channels carry different metric columns.
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(json_safe(rows))


def select_cases(bundle: DatasetBundle, config, sim_ids=None, all_test=False) -> np.ndarray:
    available = np.sort(bundle.splits["test"])
    if sim_ids is not None:
        ids = np.asarray(sim_ids, dtype=np.int64)
        if all_test or ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids):
            raise ValueError("Specify unique test sim IDs or --all-test, not both.")
        if not np.isin(ids, available).all():
            raise ValueError("Inverse recovery sim IDs must belong to the test split.")
        return np.sort(ids)
    if all_test:
        return available
    count = min(config.pilot_cases, len(available))
    rng = np.random.default_rng(config.selection_seed)
    constant = bundle.parameters["susceptible_constant"][available].astype(bool)
    groups = (rng.permutation(available[constant]), rng.permutation(available[~constant]))
    chosen = list(groups[0][:count // 2]) + list(groups[1][:count - count // 2])
    remaining = np.setdiff1d(available, chosen)
    chosen.extend(rng.permutation(remaining)[:count - len(chosen)])
    return np.sort(np.asarray(chosen, dtype=np.int64))


def validate_inputs(config: FoxConfig, bundle: DatasetBundle, payload: dict) -> FieldEncoder:
    config.validate()
    check_dataset_compatibility(config, bundle)
    checkpoint_config = config_from_dict(payload["config"])
    provenance_bundle = bundle
    if "test_extension" in bundle.manifest:
        from fno_fox.extend_test import validate_test_extension
        provenance_bundle = validate_test_extension(bundle)
    check_dataset_compatibility(checkpoint_config, provenance_bundle)
    for key in ("dataset_sha256", "config_sha256"):
        if payload.get("dataset", {}).get(key) != provenance_bundle.manifest[key]:
            raise ValueError(f"Checkpoint {key} does not match dataset provenance.")
    encoder = FieldEncoder.from_dict(payload["encoder"])
    if encoder != FieldEncoder.from_bundle(bundle):
        raise ValueError("Checkpoint normalization/conditioning does not match dataset.")
    if min(encoder.s_std, encoder.i_std) <= 0:
        raise ValueError("Inverse observations require positive normalization scales.")
    shape = (bundle.n_trajectories, config.domain.n_snapshots, config.domain.learning_cells)
    if bundle.susceptible.shape != shape or bundle.infectious.shape != shape:
        raise ValueError("Density array shape disagrees with domain configuration.")
    if not np.isfinite(bundle.susceptible).all() or not np.isfinite(bundle.infectious).all():
        raise ValueError("Inverse dataset contains non-finite observations.")
    if not np.allclose(bundle.times, config.domain.save_times, rtol=1e-7, atol=1e-8):
        raise ValueError("Observation times disagree with the dataset configuration.")
    expected_x = (np.arange(config.domain.learning_cells) + 0.5) * config.domain.length / config.domain.learning_cells
    if not np.allclose(bundle.x_centers, expected_x, rtol=1e-7, atol=1e-8):
        raise ValueError("Observation grid disagrees with the dataset configuration.")
    ids = np.concatenate(list(bundle.splits.values()))
    if len(ids) != bundle.n_trajectories or not np.array_equal(np.sort(ids), np.arange(bundle.n_trajectories)):
        raise ValueError("Dataset splits must partition the trajectories exactly.")
    if not len(bundle.splits["val"]) or not len(bundle.splits["test"]):
        raise ValueError("Validation and test splits must be nonempty.")
    return encoder


def field_metrics(prediction: np.ndarray, truth: np.ndarray, times: np.ndarray, channels=CHANNELS) -> dict:
    """Metrics per named channel; ``channels`` names the last axis of both arrays."""
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    result = {}
    for ch, name in enumerate(channels):
        err, ref = error[..., ch], truth[..., ch]
        squared, reference = np.mean(err ** 2, axis=-1), np.mean(ref ** 2, axis=-1)
        rms_ref = np.sqrt(np.mean(ref ** 2))
        result[name] = {
            "bias": float(err.mean()), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "relative_l2": float(np.sqrt(np.mean(err ** 2)) / rms_ref) if rms_ref > 1e-12 else None,
            "by_time": [{"time": float(t), "bias": float(err[k].mean()),
                         "rmse": float(np.sqrt(squared[k])),
                         "relative_l2": float(np.sqrt(squared[k] / reference[k]))
                         if np.sqrt(reference[k]) > 1e-12 else None}
                        for k, t in enumerate(times)],
        }
    return result


def residual_summary(residuals: np.ndarray, channels=CHANNELS) -> tuple[dict, dict]:
    residuals = np.asarray(residuals, dtype=np.float64)
    arrays = {
        "signed_mean": residuals.mean(axis=0),
        "rmse": np.sqrt(np.mean(residuals ** 2, axis=0)),
        "std_across_trajectories": residuals.std(axis=0),
        "per_trajectory_bias": residuals.mean(axis=(1, 2)),
        "per_trajectory_rmse": np.sqrt(np.mean(residuals ** 2, axis=(1, 2))),
        "bias_by_time": residuals.mean(axis=(0, 2)),
        "rmse_by_time": np.sqrt(np.mean(residuals ** 2, axis=(0, 2))),
        "bias_by_cell": residuals.mean(axis=(0, 1)),
        "rmse_by_cell": np.sqrt(np.mean(residuals ** 2, axis=(0, 1))),
    }
    report = {name: {"bias": float(residuals[..., k].mean()),
                     "rmse": float(np.sqrt(np.mean(residuals[..., k] ** 2)))}
              for k, name in enumerate(channels)}
    return report, arrays


def validation_residuals(bundle, model, encoder, device, config, destination):
    residuals = []
    for sim in bundle.splits["val"]:
        obj = FieldObjective(model, encoder, observations(bundle, encoder, int(sim)), device, config.batch_size)
        theta = np.array([bundle.parameters["diffusivity"][sim], bundle.parameters["transmission"][sim]])
        residuals.append(obj.physical_prediction(obj.unit_parameters(theta)) - obj.obs.physical_targets)
    residuals = np.asarray(residuals)
    physical, arrays = residual_summary(residuals)
    std = np.array([encoder.s_std, encoder.i_std])
    normalized, _ = residual_summary(residuals / std)
    np.savez_compressed(destination / "validation_residuals.npz", residuals=residuals,
                        simulation_ids=bundle.splits["val"], times=bundle.times[1:],
                        x_centers=bundle.x_centers, **arrays)
    report = {"simulation_ids": bundle.splits["val"], "physical": physical,
              "normalized": normalized, "purpose": "Descriptive surrogate discrepancy; no likelihood or bias correction.",
              "selection_note": "Validation also selected the training checkpoint; this is not independent uncertainty calibration."}
    write_json(destination / "validation_residuals.json", report)
    return report


def fv_replay(bundle, sim, theta):
    initial = trajectory_parameters(bundle, sim)
    parameters = replace(initial, diffusivity=float(theta[0]), transmission=float(theta[1]))
    s, i, trajectory = simulate(bundle.config, parameters, save_times=bundle.times)
    report = acceptance_report(trajectory, bundle.config.solver.positivity_tolerance,
                               bundle.config.solver.balance_tolerance)
    if not report["passed"]:
        raise ValueError(f"FV replay failed acceptance for sim {sim}: {report}")
    return np.stack((s[1:], i[1:]), axis=-1)


def evaluate_case(bundle, sim, model, encoder, device, config, directory, *,
                  sensor_indices=None, time_indices=None, channel_indices=None, grid=None, shared=None,
                  configuration=None):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("failure.json", "result.json", "metrics.json", "sensitivities.json",
                 "grid.npz", "optimizer.json", "optimizer_trace.csv", "fields.npz", "jacobians.npz",
                 "loss_surface.png", "reconstruction.png"):
        (directory / name).unlink(missing_ok=True)
    begin = perf_counter()
    obs = observations(bundle, encoder, sim)
    cache_enabled = shared is not None
    shared = {} if shared is None else shared
    prediction_cache = shared.setdefault("objective_predictions", OrderedDict()) if cache_enabled else None
    prediction_stats = shared.setdefault("forward_queries", {"parameter_predictions": 0, "field_queries": 0,
                                                            "cache_hits": 0}) if cache_enabled else None
    objective = FieldObjective(model, encoder, obs, device, config.batch_size, sensor_indices,
                               prediction_cache, prediction_stats, time_indices=time_indices,
                               channel_indices=channel_indices)
    selected_observations = sensor_indices is not None or time_indices is not None or channel_indices is not None
    # Callers that declare channels also record channel metadata; older callers keep their schema.
    channel_declared = channel_indices is not None
    observed_channels = tuple(CHANNELS[k] for k in objective.channel_indices)
    if grid is None:
        grid = joint_grid(objective.losses, config.grid_size)
    full_jacobians = shared.setdefault("jacobians", {})
    full_predictions = shared.setdefault("predictions", {})
    fv_predictions = shared.setdefault("fv", {})
    shared_cost = shared.setdefault("cost", {"jacobian_wall_seconds": 0., "prediction_wall_seconds": 0.,
                                            "fv_wall_seconds": 0.})
    np.savez_compressed(directory / "grid.npz", unit_axis=grid["axis"],
                        D_axis=objective.low[0] + grid["axis"] * objective.ranges[0],
                        beta_axis=objective.low[1] + grid["axis"] * objective.ranges[1],
                        loss=grid["loss"])
    theta_true = np.array([bundle.parameters["diffusivity"][sim], bundle.parameters["transmission"][sim]])
    sensitivities, sensitivity_arrays, cache = {}, {}, {}

    def sensitivity(label, point):
        key = tuple(np.asarray(point, dtype=float))
        if key not in cache:
            if key not in full_jacobians:
                start = perf_counter()
                full_jacobians[key] = objective.full_jacobians(point)
                shared_cost["jacobian_wall_seconds"] += perf_counter() - start
            physical, scaled = select_jacobians(*full_jacobians[key], objective.sensor_indices,
                                               objective.time_indices, channel_indices=objective.channel_indices)
            summary = sensitivity_summary(scaled)
            if channel_declared:
                largest, smallest = summary["singular_values"]
                summary.update(singular_value_ratio=smallest / largest if largest > 0 else None,
                               observed_channels=list(observed_channels))
            cache[key] = physical, scaled, summary
        physical, scaled, report = cache[key]
        sensitivities[label] = {"theta": objective.physical_parameters(point), **report}
        sensitivity_arrays[f"{label}_physical"] = physical
        sensitivity_arrays[f"{label}_scaled"] = scaled

    sensitivity("grid", grid["best_z"])
    sensitivity("truth", objective.unit_parameters(theta_true))
    result = recover(objective, grid, config, config.start_seed + sim)
    write_csv(directory / "optimizer_trace.csv", result.pop("trace"))
    write_json(directory / "optimizer.json", result)
    predictions, replay_cache, metrics, rows = {}, {}, {}, []
    for arm, estimate in result["arms"].items():
        z = estimate["z"]
        theta = objective.physical_parameters(z)
        sensitivity(f"starts_{arm}", z)
        key = tuple(np.asarray(z, dtype=float))
        if key not in replay_cache:
            if key not in full_predictions:
                start = perf_counter()
                full_predictions[key] = objective.physical_prediction(z)
                shared_cost["prediction_wall_seconds"] += perf_counter() - start
            if key not in fv_predictions:
                start = perf_counter()
                fv_predictions[key] = fv_replay(bundle, sim, theta)
                shared_cost["fv_wall_seconds"] += perf_counter() - start
            replay_cache[key] = full_predictions[key], fv_predictions[key]
        fno, fv = replay_cache[key]
        predictions[f"fno_{arm}"] = fno
        predictions[f"fv_{arm}"] = fv
        comparisons = {"fno_vs_observation": (fno, obs.physical_targets),
                       "fv_vs_observation": (fv, obs.physical_targets), "fno_vs_fv": (fno, fv)}
        full_metrics = {name: field_metrics(pred, truth, obs.times)
                        for name, (pred, truth) in comparisons.items()}
        metrics[arm] = full_metrics if not selected_observations else {
            "full_grid": full_metrics,
            "sensor": {name: field_metrics(objective.observe(pred), objective.observe(truth), objective.observed_times,
                                           observed_channels)
                       for name, (pred, truth) in comparisons.items()},
        }
        if time_indices is not None:
            metrics[arm]["full_grid_selected_times"] = {
                name: field_metrics(pred[objective.time_indices], truth[objective.time_indices], objective.observed_times)
                for name, (pred, truth) in comparisons.items()}
        row = {"sim_id": sim, "starts": int(arm),
               "susceptible_constant": bool(bundle.parameters["susceptible_constant"][sim]),
               **{k: estimate[k] for k in ("loss", "source", "converged", "on_boundary", "evaluations",
                                          "wall_seconds", "nonconverged_starts", "nonconverged_edges")},
               "parameter_change_flag": result["parameter_change_flag"],
               "loss_change_flag": result["loss_change_flag"], "loss_improvement": result["loss_improvement"]}
        if configuration is not None:
            row.update(configuration=configuration, sensor_count=len(objective.sensor_indices),
                       observed_scalars=len(objective.time_indices) * len(objective.sensor_indices)
                       * len(objective.channel_indices))
            if time_indices is not None:
                row["time_count"] = len(objective.time_indices)
        if channel_declared:
            row["observed_channels"] = "/".join(observed_channels)
        for k, name in enumerate(PARAMETERS):
            error = float(theta[k] - theta_true[k])
            row.update({f"{name}_true": float(theta_true[k]), f"{name}_hat": float(theta[k]),
                        f"{name}_error": error, f"{name}_abs_error": abs(error),
                        f"{name}_relative_error_pct": 100 * abs(error / theta_true[k]) if theta_true[k] != 0 else None,
                        f"{name}_start_change_fraction": float(result["parameter_change_fraction"][k])})
        scopes = {"": metrics[arm]} if not selected_observations else metrics[arm]
        for scope, comparisons_ in scopes.items():
            for comparison, channels in comparisons_.items():
                for channel, values in channels.items():
                    for metric in ("rmse", "relative_l2", "bias"):
                        prefix = f"{comparison}_{scope}" if scope else comparison
                        row[f"{prefix}_{channel}_{metric}"] = values[metric]
        report = sensitivities[f"starts_{arm}"]
        for k, name in enumerate(PARAMETERS):
            row[f"sensitivity_{name}"] = report["column_norms"][k]
            row[f"singular_value_{k}"] = report["singular_values"][k]
        row.update({k: report[k] for k in ("condition_number", "column_cosine", "numerical_rank")})
        if channel_declared:
            weakest = report["weakest_direction"] or [None] * len(PARAMETERS)
            row["singular_value_ratio"] = report["singular_value_ratio"]
            row.update({f"weakest_direction_{name}": weakest[k] for k, name in enumerate(PARAMETERS)})
        rows.append(row)
    channel_arrays = {"channel_indices": objective.channel_indices} if channel_declared else {}
    np.savez_compressed(directory / "jacobians.npz", times=objective.observed_times,
                        time_indices=objective.time_indices,
                        x_centers=bundle.x_centers[objective.sensor_indices], sensor_indices=objective.sensor_indices,
                        parameter_names=np.array(PARAMETERS), channel_names=np.array(observed_channels),
                        **channel_arrays, **sensitivity_arrays)
    # Fields always keep both channels; the observed-channel mask is stored beside them.
    field_channels = ({"observed_channel_indices": objective.channel_indices,
                       "field_channel_names": np.array(CHANNELS)} if channel_declared else {})
    np.savez_compressed(directory / "fields.npz", times=obs.times, x_centers=bundle.x_centers,
                        sensor_indices=objective.sensor_indices, time_indices=objective.time_indices,
                        observed=obs.physical_targets, **field_channels,
                        initial=np.stack((bundle.susceptible[sim, 0], bundle.infectious[sim, 0]), axis=-1), **predictions)
    if channel_declared:
        metrics["scope_labels"] = {scope: CHANNEL_SCOPE_LABELS[scope] for scope in metrics["8"]}
        metrics["channel_roles"] = {name: "fitted" if name in observed_channels else "evaluation_only"
                                    for name in CHANNELS}
    elif time_indices is not None:
        metrics["scope_labels"] = METRIC_SCOPE_LABELS
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "sensitivities.json", sensitivities)
    report = {"sim_id": sim, "theta_true": theta_true, "rows": rows,
              "grid_evaluations": grid["evaluations"], "grid_wall_seconds": grid["wall_seconds"],
              "shared_grid": grid.get("shared_grid", False),
              "logical_grid_evaluations": grid.get("logical_grid_evaluations", grid["evaluations"]),
              "unique_jacobian_points": len(cache), "unique_fv_replays": len(replay_cache),
              "wall_seconds": perf_counter() - begin}
    write_json(directory / "result.json", report)
    return rows


def distribution(values) -> dict:
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(values):
        return {"count": 0, "median": None, "q1": None, "q3": None, "iqr": None}
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return {"count": len(values), "median": median, "q1": q1, "q3": q3, "iqr": q3 - q1}


def aggregate(rows: list[dict], failures: list[dict]) -> dict:
    report = {"completed_cases": len(rows) // 2, "failed_cases": failures, "arms": {},
              "interpretation": "Local FNO sensitivity and deterministic loss geometry; no confidence intervals or structural PDE identifiability claim."}
    for arm in (4, 8):
        selected = [r for r in rows if r["starts"] == arm]
        params = {}
        for name in PARAMETERS:
            params[name] = {"rmse": float(np.sqrt(np.mean([r[f"{name}_error"] ** 2 for r in selected]))) if selected else None,
                            **{metric: distribution([r[f"{name}_{metric}"] for r in selected])
                               for metric in ("error", "abs_error", "relative_error_pct")}}
        fields = {key: distribution([r[key] for r in selected]) for key in (selected[0] if selected else [])
                  if any(key.startswith(prefix) for prefix in ("fno_vs_", "fv_vs_"))}
        report["arms"][str(arm)] = {"parameters": params, "fields": fields,
                                    "boundary_cases": sum(r["on_boundary"] for r in selected),
                                    "nonconverged_starts": sum(r["nonconverged_starts"] for r in selected),
                                    "nonconverged_edges": sum(r["nonconverged_edges"] for r in selected),
                                    "selected_candidate_not_converged": sum(r["converged"] is False for r in selected),
                                    "selected_candidate_without_convergence_test": sum(r["converged"] is None for r in selected),
                                    "objective_evaluations": sum(r["evaluations"] for r in selected),
                                    "optimization_wall_seconds": sum(r["wall_seconds"] for r in selected)}
    selected = [r for r in rows if r["starts"] == 8]
    report["start_comparison"] = {
        "parameter_change_cases": sum(r["parameter_change_flag"] for r in selected),
        "loss_change_cases": sum(r["loss_change_flag"] for r in selected),
        "loss_improvement": distribution([r["loss_improvement"] for r in selected]),
        **{f"{name}_change_fraction": distribution([r[f"{name}_start_change_fraction"] for r in selected]) for name in PARAMETERS},
        "cost_note": "Arm totals each include the shared grid and edges; do not sum arms. Eight-start cost is the actual optimization work.",
    }
    return report


def inverse(config: FoxConfig, data_dir, checkpoint, output_dir, *, device=None,
            sim_ids=None, all_test=False, overwrite=False, figures=True, verbose=True) -> dict:
    destination = Path(output_dir).resolve()
    for source in (Path(data_dir).resolve(), Path(checkpoint).resolve()):
        if source == destination or destination in source.parents:
            raise ValueError("Inverse output cannot contain or replace an input dataset/checkpoint.")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Inverse output already exists: {destination}; use --overwrite.")
        marker = destination / "manifest.json"
        if not marker.exists() or json.loads(marker.read_text()).get("pipeline") != "fno_fox.inverse":
            raise ValueError("--overwrite requires an existing fno_fox.inverse output directory.")
    bundle = load_dataset(data_dir)
    resolved = resolve_device(device or config.training.device)
    model, payload = load_checkpoint(checkpoint, resolved)
    encoder = validate_inputs(config, bundle, payload)
    ids = select_cases(bundle, config.inverse, sim_ids, all_test)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "pipeline": "fno_fox.inverse", "status": "running",
                "config": config.as_dict(), "config_sha256": config.sha256,
                "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_epoch": payload["epoch"], "data": str(bundle.root.resolve()),
                "dataset_sha256": bundle.manifest["dataset_sha256"],
                "dataset_config_sha256": bundle.manifest["config_sha256"],
                "checkpoint_training_dataset": payload.get("dataset"),
                "test_extension": bundle.manifest.get("test_extension"),
                "selected_ids": ids, "validation_ids": bundle.splits["val"],
                "encoder": encoder.as_dict(), "parameter_bounds": [encoder.diffusivity_range, encoder.transmission_range],
                "observation_times": bundle.times[1:], "channels": CHANNELS,
                "normalization": bundle.normalization, "source_time": float(bundle.times[0]),
                "device": str(resolved), "code_provenance": capture_git_state(),
                "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
                "interpretation": "Noiseless full fields; deterministic diagnostics only. No probabilistic error model.",
                "overwrite_note": "Only selected IDs in this manifest belong to this run; older case directories may remain."}
    write_json(destination / "manifest.json", manifest)
    for name in ("cases.csv", "summary.json", "recovery.png", "start_comparison.png", "sensitivities.png"):
        (destination / name).unlink(missing_ok=True)
    rows, failures = [], []
    begin = perf_counter()
    try:
        if verbose:
            print(f"Validation residuals: {len(bundle.splits['val'])} trajectories; recovery: {len(ids)} test cases.", flush=True)
        validation_residuals(bundle, model, encoder, resolved, config.inverse, destination)
        for sim in ids:
            case_directory = destination / "cases" / str(int(sim))
            case_directory.mkdir(parents=True, exist_ok=True)
            if verbose:
                print(f"Inverse case {int(sim)}: grid, sensitivities, nested starts, FV replay", flush=True)
            try:
                rows.extend(evaluate_case(bundle, int(sim), model, encoder, resolved, config.inverse,
                                          case_directory))
            except (ValueError, RuntimeError) as exc:
                failure = {"sim_id": int(sim), "error_type": type(exc).__name__, "message": str(exc)}
                failures.append(failure)
                write_json(case_directory / "failure.json", failure)
                if verbose:
                    print(f"Case {int(sim)} failed: {exc}", flush=True)
            write_csv(destination / "cases.csv", rows)
            write_json(destination / "summary.json", aggregate(rows, failures))
        report = aggregate(rows, failures)
        report["wall_seconds"] = perf_counter() - begin
        write_json(destination / "summary.json", report)
        if figures:
            from fno_fox.inverse_plots import plot_inverse
            plot_inverse(destination, rows)
        manifest["status"] = "completed_with_failures" if failures else "completed"
        manifest["completed_ids"] = [r["sim_id"] for r in rows if r["starts"] == 8]
        manifest["failed_ids"] = [r["sim_id"] for r in failures]
        write_json(destination / "manifest.json", manifest)
        return json_safe(report)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "manifest.json", manifest)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sim-ids", nargs="+", type=int)
    selection.add_argument("--all-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = inverse(load_config(args.config), args.data, args.checkpoint, args.output,
                     device=args.device, sim_ids=args.sim_ids, all_test=args.all_test, overwrite=args.overwrite)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
