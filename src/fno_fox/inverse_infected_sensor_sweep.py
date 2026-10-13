"""Paired, noiseless five-time inverse sweep comparing I-only and S/I observations for D and beta.

    python -m fno_fox.inverse_infected_sensor_sweep --config conf/default.yaml \
        --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/infected_sensor_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fno_fox.config import FoxConfig, load_config
from fno_fox.data import load_dataset
from fno_fox.inverse import CHANNEL_SCOPE_LABELS, CHANNELS, PARAMETERS, distribution
from fno_fox.inverse_sensor_sweep import _run_sweep, select_sweep_cases
from fno_fox.inverse_sensor_sweep import configuration_summary as base_configuration_summary
from fno_fox.inverse_sensor_time_sweep import nested_time_layouts


TIME_COUNT = 5
# Table order; each trajectory schedules its configurations in this order.
CONFIGURATIONS = ((32, ("I",)), (16, ("I",)), (8, ("I",)), (16, ("S", "I")), (8, ("S", "I")))


def layout_key(sensors: int, channels) -> str:
    return f"sensors_{sensors}_times_{TIME_COUNT}_channels_{''.join(channels)}"


# (kind, current, reference); every paired change is current minus reference.
COMPARISONS = (
    ("channel", layout_key(16, ("I",)), layout_key(16, ("S", "I"))),
    ("channel", layout_key(8, ("I",)), layout_key(8, ("S", "I"))),
    ("sensor_count", layout_key(16, ("I",)), layout_key(32, ("I",))),
    ("sensor_count", layout_key(8, ("I",)), layout_key(32, ("I",))),
    ("equal_scalar_budget", layout_key(32, ("I",)), layout_key(16, ("S", "I"))),
)
# Losses are excluded: I-only and S/I losses average differently normalized observation sets.
PAIRED_METRICS = (
    *(f"{name}_{metric}" for name in PARAMETERS
      for metric in ("error", "abs_error", "relative_error_pct", "start_change_fraction")),
    "evaluations", "nonconverged_starts", "nonconverged_edges",
    "sensitivity_D", "sensitivity_beta", "singular_value_0", "singular_value_1", "singular_value_ratio",
    "condition_number", "column_cosine", "numerical_rank",
)
NOTES = {
    "change_convention": "Current minus reference, matching completed trajectory ID and start count. Channel "
                         "comparisons are I-only minus S/I at equal sensor count; sensor-count comparisons are "
                         "I-only minus 32-sensor I-only; the equal-scalar-budget comparison is 32-sensor I-only "
                         "minus 16-sensor S/I (160 observed scalars each).",
    "loss_note": "Each loss is 0.5 * mean(normalized residual^2) over its own observed scalars, with each channel "
                 "divided by the checkpoint's own scale. I-only and S/I losses are not directly comparable and "
                 "are not paired.",
    "replay_note": "Paired replay changes use only metrics present in both configurations: I at fitted sensors "
                   "and times, and full-grid S/I evaluation. Sensor-scope S metrics exist only for S/I "
                   "configurations. Sensor-scope metrics in sensor-count and equal-budget comparisons cover "
                   "different cells.",
    "layout_note": "Spatial layouts are independently uniform for each count and not nested, so sensor-count "
                   "comparisons change both count and placement. All configurations share the same five times.",
    "sensitivity_note": "Local FNO sensitivity diagnostics from J_scaled = diag(1/channel_scale) J_physical "
                        "diag(parameter_ranges) / sqrt(m). singular_value_1 is the smallest singular value and "
                        "singular_value_ratio is smallest/largest. Not a structural identifiability claim.",
}


def observation_layouts(x_centers, positive_times) -> dict:
    x = np.asarray(x_centers)
    largest = max(sensors for sensors, _ in CONFIGURATIONS)
    if len(x) < largest:
        raise ValueError(f"Infected-channel sweep requires at least {largest} learning-grid cells; grid has {len(x)}.")
    # Exactly five times, independent of inverse.sweep_time_counts.
    time_indices = nested_time_layouts(positive_times, (TIME_COUNT,))[TIME_COUNT]
    times = np.asarray(positive_times, dtype=np.float64)[time_indices]
    layouts = {}
    for sensors, channels in CONFIGURATIONS:
        indices = np.rint(np.linspace(0, len(x) - 1, sensors)).astype(np.int64)
        label = "/".join(channels)
        layouts[layout_key(sensors, channels)] = {
            "label": f"{sensors} sensors, {TIME_COUNT} times, {label}", "sensor_count": sensors,
            "indices": indices, "x_centers": x[indices], "time_count": TIME_COUNT,
            "time_indices": time_indices, "times": times,
            "channel_indices": np.array([CHANNELS.index(name) for name in channels], dtype=np.int64),
            "observed_channels": list(channels), "channel_label": label, "full_field_channels": list(CHANNELS),
            "channel_roles": {name: "fitted" if name in channels else "evaluation_only" for name in CHANNELS},
            "observed_scalars": sensors * TIME_COUNT * len(channels),
        }
    return layouts


def configuration_summary(rows, failures, layout) -> dict:
    summary = base_configuration_summary(rows, failures, layout)
    summary.update(observed_channels=layout["observed_channels"], channel_roles=layout["channel_roles"],
                   observed_scalars=layout["observed_scalars"])
    for arm in (4, 8):
        subset = [r for r in rows if r["starts"] == arm]
        summary["arms"][str(arm)]["sensitivities"].update({
            metric: distribution([r[metric] for r in subset])
            for metric in ("singular_value_ratio", "weakest_direction_D", "weakest_direction_beta")})
    return summary


def direction_angle(current: dict, reference: dict) -> float | None:
    """Sign-invariant angle in degrees between two weakest unit-box parameter directions."""
    a = [current[f"weakest_direction_{name}"] for name in PARAMETERS]
    b = [reference[f"weakest_direction_{name}"] for name in PARAMETERS]
    if any(v is None for v in (*a, *b)):
        return None
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cosine = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cosine, 0, 1))))


def optimizer_counts(rows) -> dict:
    return {"selected_candidate_not_converged": sum(r["converged"] is False for r in rows),
            "selected_candidate_without_convergence_test": sum(r["converged"] is None for r in rows),
            "nonconverged_starts": sum(r["nonconverged_starts"] for r in rows),
            "nonconverged_edges": sum(r["nonconverged_edges"] for r in rows),
            "parameter_change_flags": sum(r["parameter_change_flag"] for r in rows)}


def sweep_summary(rows, failures, selected_ids, layouts) -> tuple[dict, list[dict]]:
    report = {"selected_ids": list(selected_ids), "selected_count": len(selected_ids),
              "completed_configuration_cases": len(rows) // 2, "failed_cases": failures,
              "failed_configuration_cases": len(failures), "configurations": {},
              "paired_changes": {kind: {} for kind, _, _ in COMPARISONS}, "metric_scopes": CHANNEL_SCOPE_LABELS,
              **NOTES,
              "interpretation": "Noiseless I-only versus S/I observations at five fixed times; deterministic local FNO diagnostics."}
    for key, layout in layouts.items():
        report["configurations"][key] = configuration_summary(
            [r for r in rows if r["configuration"] == key],
            [f for f in failures if f["configuration"] == key], layout)
    indexed = {(r["configuration"], r["sim_id"], r["starts"]): r for r in rows}
    failed = {(f["configuration"], f["sim_id"]) for f in failures}
    paired_rows = []
    for kind, key, reference in COMPARISONS:
        comparisons = {}
        for arm in (4, 8):
            pairs, currents, references, unpaired = [], [], [], []
            for sim in map(int, selected_ids):
                current, baseline = indexed.get((key, sim, arm)), indexed.get((reference, sim, arm))
                if current is None or baseline is None:
                    unpaired.append(sim)
                    continue
                replay = [name for name in current if name.startswith(("fno_vs_", "fv_vs_")) and name in baseline]
                change = {"comparison": kind, "configuration": key, "reference_configuration": reference,
                          "sensor_count": current["sensor_count"], "reference_sensor_count": baseline["sensor_count"],
                          "observed_channels": current["observed_channels"],
                          "reference_observed_channels": baseline["observed_channels"],
                          "observed_scalars": current["observed_scalars"],
                          "reference_observed_scalars": baseline["observed_scalars"],
                          "sim_id": sim, "starts": arm, "converged": current["converged"],
                          "reference_converged": baseline["converged"]}
                for metric in (*PAIRED_METRICS, *replay):
                    a, b = current[metric], baseline[metric]
                    change[f"{metric}_change"] = a - b if a is not None and b is not None else None
                change["weakest_direction_angle_deg"] = direction_angle(current, baseline)
                pairs.append(change)
                currents.append(current)
                references.append(baseline)
            paired_rows.extend(pairs)
            comparisons[str(arm)] = {
                "comparison": kind, "reference_configuration": reference,
                "paired_count": len(pairs), "unpaired_count": len(unpaired),
                "paired_ids": [p["sim_id"] for p in pairs], "unpaired_ids": unpaired,
                "failed_unpaired_ids": [sim for sim in unpaired if (key, sim) in failed or (reference, sim) in failed],
                "optimizer": {"current": optimizer_counts(currents), "reference": optimizer_counts(references)},
                "changes": {metric: distribution([p[metric] for p in pairs])
                            for metric in (pairs[0] if pairs else [])
                            if metric.endswith("_change") or metric == "weakest_direction_angle_deg"},
            }
        report["paired_changes"][kind][key] = comparisons
    return report, paired_rows


def infected_sensor_sweep(config: FoxConfig, data_dir, checkpoint, output_dir, *, device=None,
                          sim_ids=None, all_test=False, overwrite=False, resume=False,
                          max_recovery_problems=None, figures=True, verbose=True) -> dict:
    config.validate()
    bundle = load_dataset(data_dir)
    # Cohort, grid, and snapshot checks all run before loading the model or writing output.
    ids = select_sweep_cases(bundle, config.inverse, sim_ids, all_test)
    layouts = observation_layouts(bundle.x_centers, bundle.times[1:] - bundle.times[0])
    plotter = None
    if figures:
        from fno_fox.inverse_plots import plot_infected_sensor_sweep
        plotter = plot_infected_sensor_sweep
    return _run_sweep(config, bundle, checkpoint, output_dir, ids, layouts, device=device,
                      overwrite=overwrite, resume=resume, max_recovery_problems=max_recovery_problems,
                      figures=figures, verbose=verbose,
                      pipeline="fno_fox.inverse_infected_sensor_sweep", summary_function=sweep_summary,
                      comparison_plotter=plotter, manifest_extra={
                          "observation_layouts": layouts, "metric_scopes": CHANNEL_SCOPE_LABELS,
                          "time_count": TIME_COUNT,
                          "time_selection": "nested_time_layouts(positive_times, (5,)), i.e. rint(linspace(0, N_positive-1, 5)); independent of inverse.sweep_time_counts.",
                          "time_index_origin": "Zero-based index into positive saved times, excluding the initial snapshot.",
                          "sensor_selection": "rint(linspace(0, Nx-1, count)); grid cells shared by every observed channel and selected time. Layouts are independently uniform per count, not nested.",
                          "full_field_channels": list(CHANNELS),
                          "observed_channels_by_configuration": {k: v["observed_channels"] for k, v in layouts.items()},
                          "observed_scalars_by_configuration": {k: v["observed_scalars"] for k, v in layouts.items()},
                          "comparisons": [{"kind": kind, "configuration": key, "reference_configuration": reference}
                                          for kind, key, reference in COMPARISONS],
                          "objective": "L = 0.5 * mean(normalized residual^2) over selected observations; each observed channel uses the checkpoint's own scale (S_std or I_std).",
                          "jacobian_convention": "*_physical: d(physical observed density)/d(D, beta), shaped [time, sensor, observed channel, parameter]. *_scaled: diag(1/channel_scale) J_physical diag(parameter_ranges) / sqrt(m), shaped [m, 2] with m = observed_scalars.",
                          **NOTES,
                          "interpretation": "Noiseless observations; full initial S/I fields known. Unobserved channels, cells, and times are evaluation ground truth, not fitting measurements; S is evaluation-only for I-only configurations.",
                      })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default=None)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sim-ids", nargs="+", type=int)
    selection.add_argument("--all-test", action="store_true")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--overwrite", action="store_true")
    output_mode.add_argument("--resume", action="store_true")
    parser.add_argument("--max-recovery-problems", type=int,
                        help="Cap case/layout recoveries in trajectory then configuration order; both start arms are retained.")
    args = parser.parse_args(argv)
    report = infected_sensor_sweep(load_config(args.config), args.data, args.checkpoint, args.output,
                                   device=args.device, sim_ids=args.sim_ids, all_test=args.all_test,
                                   overwrite=args.overwrite, resume=args.resume,
                                   max_recovery_problems=args.max_recovery_problems)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
