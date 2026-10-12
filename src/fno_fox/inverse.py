"""Noiseless, full-field parameter recovery with deterministic identifiability diagnostics.

    python -m fno_fox.inverse --config conf/default.yaml --data data/fox_rabies \
        --checkpoint runs/default/best.pt --output runs/default/inverse
"""
from __future__ import annotations

import argparse
import csv
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
from fno_fox.inverse_numerics import FieldObjective, joint_grid, observations, recover, sensitivity_summary
from fno_fox.physics import acceptance_report, simulate
from fno_fox.train import check_dataset_compatibility, load_checkpoint, resolve_device


PARAMETERS = ("D", "beta")
CHANNELS = ("S", "I")


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
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
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
    check_dataset_compatibility(checkpoint_config, bundle)
    for key in ("dataset_sha256", "config_sha256"):
        if payload.get("dataset", {}).get(key) != bundle.manifest[key]:
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


def field_metrics(prediction: np.ndarray, truth: np.ndarray, times: np.ndarray) -> dict:
    prediction, truth = np.asarray(prediction, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    error = prediction - truth
    result = {}
    for ch, name in enumerate(CHANNELS):
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


def residual_summary(residuals: np.ndarray) -> tuple[dict, dict]:
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
              for k, name in enumerate(CHANNELS)}
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


def evaluate_case(bundle, sim, model, encoder, device, config, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("failure.json", "result.json", "metrics.json", "sensitivities.json",
                 "grid.npz", "optimizer.json", "optimizer_trace.csv", "fields.npz", "jacobians.npz",
                 "loss_surface.png", "reconstruction.png"):
        (directory / name).unlink(missing_ok=True)
    begin = perf_counter()
    obs = observations(bundle, encoder, sim)
    objective = FieldObjective(model, encoder, obs, device, config.batch_size)
    grid = joint_grid(objective.losses, config.grid_size)
    np.savez_compressed(directory / "grid.npz", unit_axis=grid["axis"],
                        D_axis=objective.low[0] + grid["axis"] * objective.ranges[0],
                        beta_axis=objective.low[1] + grid["axis"] * objective.ranges[1],
                        loss=grid["loss"])
    theta_true = np.array([bundle.parameters["diffusivity"][sim], bundle.parameters["transmission"][sim]])
    sensitivities, sensitivity_arrays, cache = {}, {}, {}

    def sensitivity(label, point):
        key = tuple(np.asarray(point, dtype=float))
        if key not in cache:
            physical, scaled = objective.jacobians(point)
            cache[key] = physical, scaled, sensitivity_summary(scaled)
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
            replay_cache[key] = objective.physical_prediction(z), fv_replay(bundle, sim, theta)
        fno, fv = replay_cache[key]
        predictions[f"fno_{arm}"] = fno
        predictions[f"fv_{arm}"] = fv
        metrics[arm] = {
            "fno_vs_observation": field_metrics(fno, obs.physical_targets, obs.times),
            "fv_vs_observation": field_metrics(fv, obs.physical_targets, obs.times),
            "fno_vs_fv": field_metrics(fno, fv, obs.times),
        }
        row = {"sim_id": sim, "starts": int(arm),
               "susceptible_constant": bool(bundle.parameters["susceptible_constant"][sim]),
               **{k: estimate[k] for k in ("loss", "source", "converged", "on_boundary", "evaluations",
                                          "wall_seconds", "nonconverged_starts", "nonconverged_edges")},
               "parameter_change_flag": result["parameter_change_flag"],
               "loss_change_flag": result["loss_change_flag"], "loss_improvement": result["loss_improvement"]}
        for k, name in enumerate(PARAMETERS):
            error = float(theta[k] - theta_true[k])
            row.update({f"{name}_true": float(theta_true[k]), f"{name}_hat": float(theta[k]),
                        f"{name}_error": error, f"{name}_abs_error": abs(error),
                        f"{name}_relative_error_pct": 100 * abs(error / theta_true[k]) if theta_true[k] != 0 else None,
                        f"{name}_start_change_fraction": float(result["parameter_change_fraction"][k])})
        for comparison, channels in metrics[arm].items():
            for channel, values in channels.items():
                for metric in ("rmse", "relative_l2", "bias"):
                    row[f"{comparison}_{channel}_{metric}"] = values[metric]
        report = sensitivities[f"starts_{arm}"]
        for k, name in enumerate(PARAMETERS):
            row[f"sensitivity_{name}"] = report["column_norms"][k]
            row[f"singular_value_{k}"] = report["singular_values"][k]
        row.update({k: report[k] for k in ("condition_number", "column_cosine", "numerical_rank")})
        rows.append(row)
    np.savez_compressed(directory / "jacobians.npz", times=obs.times, x_centers=bundle.x_centers,
                        parameter_names=np.array(PARAMETERS), channel_names=np.array(CHANNELS), **sensitivity_arrays)
    np.savez_compressed(directory / "fields.npz", times=obs.times, x_centers=bundle.x_centers,
                        observed=obs.physical_targets,
                        initial=np.stack((bundle.susceptible[sim, 0], bundle.infectious[sim, 0]), axis=-1), **predictions)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "sensitivities.json", sensitivities)
    report = {"sim_id": sim, "theta_true": theta_true, "rows": rows,
              "grid_evaluations": grid["evaluations"], "grid_wall_seconds": grid["wall_seconds"],
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
