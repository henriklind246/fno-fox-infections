"""Paired, noiseless spatial-sensor sweep for inverse recovery of D and beta.

    python -m fno_fox.inverse_sensor_sweep --config conf/default.yaml \
        --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/sensor_sweep
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import torch

from fno_fox.config import FoxConfig, load_config
from fno_fox.data import capture_git_state, file_sha256, load_dataset
from fno_fox.inverse import (
    aggregate, distribution, evaluate_case, json_safe, residual_summary, select_cases,
    validate_inputs, validation_residuals, write_csv, write_json,
)
from fno_fox.inverse_numerics import FieldObjective, observations, shared_joint_grids
from fno_fox.train import load_checkpoint, resolve_device


DESIGNS = (("baseline", "Baseline", None), ("sparse_a", "Sparse A", 32),
           ("sparse_b", "Sparse B", 16), ("sparse_c", "Sparse C", 8), ("sparse_d", "Sparse D", 4))


def sensor_layouts(x_centers: np.ndarray) -> dict:
    nx = len(x_centers)
    if nx < 32:
        raise ValueError("Spatial sensor sweep requires at least 32 learning-grid cells.")
    layouts = {}
    for key, label, count in DESIGNS:
        count = nx if count is None else count
        indices = np.rint(np.linspace(0, nx - 1, count)).astype(np.int64)
        layouts[key] = {"label": label, "sensor_count": count, "indices": indices,
                        "x_centers": np.asarray(x_centers)[indices]}
    return layouts


def select_sweep_cases(bundle, config, sim_ids=None, all_test=False) -> np.ndarray:
    required = config.sweep_cases
    available = len(bundle.splits["test"])
    if available < required:
        raise ValueError(f"Spatial sensor sweep requires at least {required} unique test trajectories; "
                         f"dataset has {available}. Supply a matching dataset/checkpoint with enough test cases.")
    if sim_ids is not None and len(sim_ids) != required:
        raise ValueError(f"--sim-ids must contain exactly {required} unique test IDs.")
    return select_cases(bundle, replace(config, pilot_cases=required), sim_ids, all_test)


def validation_by_layout(root: Path, layouts: dict) -> None:
    with np.load(root / "validation_residuals.npz") as archive:
        residuals = archive["residuals"]
        times, ids = archive["times"], archive["simulation_ids"]
    manifest = json.loads((root / "manifest.json").read_text())
    std = np.array([manifest["encoder"]["s_std"], manifest["encoder"]["i_std"]])
    for key, layout in layouts.items():
        directory = root / "configurations" / key
        directory.mkdir(parents=True, exist_ok=True)
        time_indices = layout.get("time_indices", np.arange(len(times)))
        selected = np.take(np.take(residuals, time_indices, axis=1), layout["indices"], axis=2)
        physical, arrays = residual_summary(selected)
        normalized, _ = residual_summary(selected / std)
        np.savez_compressed(directory / "validation_residuals.npz", residuals=selected,
                            simulation_ids=ids, times=times[time_indices], time_indices=time_indices,
                            sensor_indices=layout["indices"],
                            x_centers=layout["x_centers"], **arrays)
        write_json(directory / "validation_residuals.json", {
            "simulation_ids": ids, "sensor_count": layout["sensor_count"],
            "time_count": len(time_indices), "times": times[time_indices],
            "physical": physical, "normalized": normalized,
            "purpose": "Descriptive signed surrogate discrepancies; no likelihood or bias correction.",
            "selection_note": "Validation also selected the training checkpoint.",
        })


def configuration_summary(rows, failures, layout) -> dict:
    summary = aggregate(rows, failures)
    summary.update(label=layout["label"], sensor_count=layout["sensor_count"])
    if "time_count" in layout:
        summary["time_count"] = layout["time_count"]
    summary["start_comparison"]["cost_note"] = (
        "Grid forward work is shared and excluded from configuration arm totals. "
        "Each arm includes its boundary searches; eight-start totals count actual optimization work."
    )
    for arm in (4, 8):
        subset = [r for r in rows if r["starts"] == arm]
        summary["arms"][str(arm)]["sensitivities"] = {
            metric: distribution([r[metric] for r in subset])
            for metric in ("sensitivity_D", "sensitivity_beta", "singular_value_0", "singular_value_1",
                           "condition_number", "column_cosine")}
        summary["arms"][str(arm)]["rank_deficient_cases"] = sum(r["numerical_rank"] < 2 for r in subset)
    return summary


def sweep_summary(rows: list[dict], failures: list[dict], selected_ids, layouts) -> tuple[dict, list[dict]]:
    report = {"selected_ids": list(selected_ids), "selected_count": len(selected_ids),
              "failed_cases": failures, "configurations": {}, "paired_changes": {},
              "interpretation": "Noiseless spatial sparsity comparison; deterministic local FNO diagnostics."}
    indexed = {(r["configuration"], r["sim_id"], r["starts"]): r for r in rows}
    paired_rows = []
    for key, layout in layouts.items():
        selected = [r for r in rows if r["configuration"] == key]
        failed = [f for f in failures if f["configuration"] == key]
        summary = configuration_summary(selected, failed, layout)
        report["configurations"][key] = summary
        if key == "baseline":
            continue
        comparisons = {}
        for arm in (4, 8):
            pairs = []
            for sim in selected_ids:
                baseline = indexed.get(("baseline", int(sim), arm))
                sparse = indexed.get((key, int(sim), arm))
                if baseline is None or sparse is None:
                    continue
                metrics = ["D_abs_error", "beta_abs_error", "singular_value_1", "condition_number", "column_cosine"]
                metrics += [name for name in sparse if name.startswith(("fno_vs_", "fv_vs_"))]
                change = {"configuration": key, "sensor_count": layout["sensor_count"],
                          "sim_id": int(sim), "starts": arm}
                for metric in metrics:
                    a, b = sparse[metric], baseline[metric]
                    change[f"{metric}_change"] = a - b if a is not None and b is not None else None
                pairs.append(change)
                paired_rows.append(change)
            comparisons[str(arm)] = {
                "paired_count": len(pairs), "unpaired_count": len(selected_ids) - len(pairs),
                "paired_ids": [p["sim_id"] for p in pairs],
                "changes": {metric: distribution([p[metric] for p in pairs])
                            for metric in (pairs[0] if pairs else []) if metric.endswith("_change")},
            }
        report["paired_changes"][key] = comparisons
    report["change_convention"] = "Sparse minus baseline; each comparison uses matching successful IDs and start count."
    return report, paired_rows


def sensor_sweep(config: FoxConfig, data_dir, checkpoint, output_dir, *, device=None,
                 sim_ids=None, all_test=False, overwrite=False, figures=True, verbose=True) -> dict:
    config.validate()
    bundle = load_dataset(data_dir)
    # Cardinality is checked before loading the model or creating any output.
    ids = select_sweep_cases(bundle, config.inverse, sim_ids, all_test)
    layouts = sensor_layouts(bundle.x_centers)
    return _run_sweep(config, bundle, checkpoint, output_dir, ids, layouts, device=device,
                      overwrite=overwrite, figures=figures, verbose=verbose)


def _run_sweep(config, bundle, checkpoint, output_dir, ids, layouts, *, device=None,
               overwrite=False, figures=True, verbose=True,
               pipeline="fno_fox.inverse_sensor_sweep", summary_function=sweep_summary,
               comparison_plotter=None, manifest_extra=None):
    """Common execution after each entry point validates its cases and observation layouts."""
    destination = Path(output_dir).resolve()
    for source in (bundle.root.resolve(), Path(checkpoint).resolve()):
        if source == destination or destination in source.parents:
            raise ValueError("Sensor sweep output cannot contain or replace an input dataset/checkpoint.")
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Sensor sweep output already exists: {destination}; use --overwrite.")
        marker = destination / "manifest.json"
        if not marker.exists() or json.loads(marker.read_text()).get("pipeline") != pipeline:
            raise ValueError(f"--overwrite requires an existing {pipeline} output directory.")
    resolved = resolve_device(device or config.training.device)
    model, payload = load_checkpoint(checkpoint, resolved)
    encoder = validate_inputs(config, bundle, payload)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "pipeline": pipeline, "status": "running",
        "config": config.as_dict(), "config_sha256": config.sha256,
        "checkpoint": str(Path(checkpoint).resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_epoch": payload["epoch"], "data": str(bundle.root.resolve()),
        "dataset_sha256": bundle.manifest["dataset_sha256"], "dataset_config_sha256": bundle.manifest["config_sha256"],
        "selected_ids": ids, "selected_count": len(ids), "validation_ids": bundle.splits["val"],
        "sensor_layouts": layouts, "observation_times": bundle.times[1:], "channels": ["S", "I"],
        "sensor_selection": "rint(linspace(0, Nx-1, count)); grid cells; identical for both channels and all times",
        "encoder": encoder.as_dict(), "normalization": bundle.normalization,
        "parameter_bounds": [encoder.diffusivity_range, encoder.transmission_range],
        "device": str(resolved), "code_provenance": capture_git_state(),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "interpretation": "Noiseless observations; deterministic diagnostics only. Full initial fields remain known.",
        "cost_note": "Shared grids/diagnostic predictions/Jacobians/FV are counted once; configuration optimizer totals exclude shared grids.",
        "overwrite_note": "Only selected IDs and configurations in this manifest belong to this run.",
        **(manifest_extra or {}),
    }
    write_json(destination / "manifest.json", manifest)
    for name in ("cases.csv", "summary.json", "paired_changes.csv", "shared_costs.json",
                 "sensor_recovery.png", "sensor_replay_errors.png", "sensor_sensitivities.png", "sensor_start_stability.png"):
        (destination / name).unlink(missing_ok=True)
    for name in ("sensor_time_recovery.png", "sensor_time_replay_errors_sensor.png",
                 "sensor_time_replay_errors_full_grid_selected_times.png", "sensor_time_replay_errors_full_grid.png",
                 "sensor_time_sensitivities.png",
                 "sensor_time_start_stability.png"):
        (destination / name).unlink(missing_ok=True)
    for key in layouts:
        directory = destination / "configurations" / key
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("cases.csv", "summary.json", "recovery.png", "start_comparison.png", "sensitivities.png",
                     "validation_residuals.png"):
            (directory / name).unlink(missing_ok=True)
    rows, failures, shared_costs = [], [], {"cases": {}}
    begin = perf_counter()

    def save_reports():
        report, paired = summary_function(rows, failures, ids, layouts)
        write_csv(destination / "cases.csv", rows)
        write_csv(destination / "paired_changes.csv", paired)
        write_json(destination / "summary.json", report)
        write_json(destination / "shared_costs.json", shared_costs)
        for key in layouts:
            directory = destination / "configurations" / key
            write_csv(directory / "cases.csv", [r for r in rows if r["configuration"] == key])
            write_json(directory / "summary.json", report["configurations"][key])
        return report

    def failure(key, sim, exc):
        record = {"configuration": key, "sim_id": int(sim),
                  "error_type": type(exc).__name__, "message": str(exc)}
        failures.append(record)
        directory = destination / "configurations" / key / "cases" / str(int(sim))
        directory.mkdir(parents=True, exist_ok=True)
        # Remove a previous success marker so overwrite never mislabels a failed case.
        (directory / "result.json").unlink(missing_ok=True)
        write_json(directory / "failure.json", record)
        if verbose:
            print(f"{key}, case {int(sim)} failed: {exc}", flush=True)

    try:
        start = perf_counter()
        validation_residuals(bundle, model, encoder, resolved, config.inverse, destination)
        shared_costs["validation"] = {"parameter_predictions": len(bundle.splits["val"]),
                                       "field_queries": len(bundle.splits["val"]) * (bundle.n_snapshots - 1),
                                       "wall_seconds": perf_counter() - start}
        validation_by_layout(destination, layouts)
        for sim in ids:
            if verbose:
                print(f"Sweep case {int(sim)}: shared grid followed by {len(layouts)} configurations", flush=True)
            shared = {}
            base = FieldObjective(model, encoder, observations(bundle, encoder, int(sim)), resolved, config.inverse.batch_size)
            try:
                grids, cost = shared_joint_grids(base, layouts, config.inverse.grid_size)
            except (ValueError, RuntimeError) as exc:
                for key in layouts:
                    failure(key, sim, exc)
                save_reports()
                continue
            shared_costs["cases"][str(int(sim))] = {"grid": cost}
            case_begin = perf_counter()
            for key, layout in layouts.items():
                directory = destination / "configurations" / key / "cases" / str(int(sim))
                try:
                    rows.extend(evaluate_case(bundle, int(sim), model, encoder, resolved, config.inverse, directory,
                                              sensor_indices=layout["indices"], grid=grids[key], shared=shared,
                                              time_indices=layout.get("time_indices"),
                                              configuration=key))
                except (ValueError, RuntimeError) as exc:
                    failure(key, sim, exc)
                shared_costs["cases"][str(int(sim))]["diagnostics"] = {
                    **shared.get("cost", {}), "unique_jacobian_points": len(shared.get("jacobians", {})),
                    "unique_prediction_points": len(shared.get("predictions", {})),
                    "unique_fv_replays": len(shared.get("fv", {})),
                }
                shared_costs["cases"][str(int(sim))]["forward_queries_excluding_grid"] = shared.get("forward_queries", {})
                shared_costs["cases"][str(int(sim))]["wall_seconds_including_grid"] = cost["wall_seconds"] + perf_counter() - case_begin
                save_reports()
        report = save_reports()
        if figures:
            from fno_fox.inverse_plots import plot_inverse, plot_sensor_sweep
            for key in layouts:
                plot_inverse(destination / "configurations" / key, [r for r in rows if r["configuration"] == key])
            (comparison_plotter or plot_sensor_sweep)(destination, report, layouts)
        report["wall_seconds"] = perf_counter() - begin
        write_json(destination / "summary.json", report)
        manifest.update(status="completed_with_failures" if failures else "completed",
                        completed_configuration_cases=len(rows) // 2, failed_configuration_cases=len(failures))
        write_json(destination / "manifest.json", manifest)
        return json_safe(report)
    except Exception as exc:
        manifest.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(destination / "manifest.json", manifest)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sim-ids", nargs="+", type=int)
    selection.add_argument("--all-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    report = sensor_sweep(load_config(args.config), args.data, args.checkpoint, args.output,
                          device=args.device, sim_ids=args.sim_ids, all_test=args.all_test, overwrite=args.overwrite)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
