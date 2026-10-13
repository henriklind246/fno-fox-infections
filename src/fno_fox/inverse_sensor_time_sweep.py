"""Paired, noiseless spatial-and-temporal inverse sweep for D and beta.

    python -m fno_fox.inverse_sensor_time_sweep --config conf/default.yaml \
        --data data/fox_rabies --checkpoint runs/default/best.pt --output runs/default/sensor_time_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fno_fox.config import FoxConfig, InverseConfig, load_config
from fno_fox.data import load_dataset
from fno_fox.inverse import METRIC_SCOPE_LABELS, distribution
from fno_fox.inverse_sensor_sweep import _run_sweep, configuration_summary, select_sweep_cases


SENSOR_COUNTS = (16, 8, 4)


def nested_time_layouts(positive_times, counts) -> dict[int, np.ndarray]:
    """Seed uniform indices, then split the largest available interval at its midpoint."""
    InverseConfig(sweep_time_counts=tuple(counts)).validate()
    times = np.asarray(positive_times, dtype=np.float64)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] <= 0 or np.any(np.diff(times) <= 0)):
        raise ValueError("Temporal sweep requires finite, strictly increasing positive saved times.")
    if max(counts) > len(times):
        raise ValueError(f"Temporal sweep requires at least {max(counts)} positive snapshots; dataset has {len(times)}.")
    requested = sorted(counts)
    selected = list(np.rint(np.linspace(0, len(times) - 1, requested[0])).astype(int))
    result = {requested[0]: np.asarray(selected, dtype=np.int64)}
    tolerance = 32 * np.finfo(float).eps * times[-1]
    for count in requested[1:]:
        while len(selected) < count:
            intervals = [(left, right) for left, right in zip(selected[:-1], selected[1:]) if right - left > 1]
            # Strict ordering and the cardinality check guarantee a splittable interval.
            widths = np.array([times[right] - times[left] for left, right in intervals])
            left, right = intervals[int(np.flatnonzero(widths >= widths.max() - tolerance)[0])]
            candidates = np.arange(left + 1, right)
            midpoint = times[left] + (times[right] - times[left]) / 2
            distance = np.abs(times[candidates] - midpoint)
            index = int(candidates[np.flatnonzero(distance <= distance.min() + tolerance)[0]])
            selected.append(index)
            selected.sort()
        result[count] = np.asarray(selected, dtype=np.int64)
    return result


def observation_layouts(x_centers, positive_times, counts) -> dict:
    x = np.asarray(x_centers)
    if len(x) < max(SENSOR_COUNTS):
        raise ValueError("Spatial-and-temporal sweep requires at least 16 learning-grid cells.")
    time_layouts = nested_time_layouts(positive_times, counts)
    times = np.asarray(positive_times)
    layouts = {}
    for sensors in SENSOR_COUNTS:
        indices = np.rint(np.linspace(0, len(x) - 1, sensors)).astype(np.int64)
        for count in sorted(time_layouts, reverse=True):
            key = f"sensors_{sensors}_times_{count}"
            layouts[key] = {"label": f"{sensors} sensors, {count} times", "sensor_count": sensors,
                            "indices": indices, "x_centers": x[indices], "time_count": count,
                            "time_indices": time_layouts[count], "times": times[time_layouts[count]],
                            "observed_scalars": 2 * sensors * count}
    return layouts


def sweep_summary(rows, failures, selected_ids, layouts) -> tuple[dict, list[dict]]:
    report = {"selected_ids": list(selected_ids), "selected_count": len(selected_ids),
              "completed_configuration_cases": len(rows) // 2, "failed_cases": failures,
              "failed_configuration_cases": len(failures), "configurations": {},
              "paired_changes": {"temporal": {}, "spatial": {}}, "metric_scopes": METRIC_SCOPE_LABELS,
              "interpretation": "Noiseless spatial/temporal sparsity; deterministic local FNO diagnostics."}
    indexed = {(r["configuration"], r["sim_id"], r["starts"]): r for r in rows}
    largest = max(v["time_count"] for v in layouts.values())
    paired_rows = []
    for key, layout in layouts.items():
        report["configurations"][key] = configuration_summary(
            [r for r in rows if r["configuration"] == key],
            [f for f in failures if f["configuration"] == key], layout)
        references = {"temporal": f"sensors_{layout['sensor_count']}_times_{largest}",
                      "spatial": f"sensors_16_times_{layout['time_count']}"}
        for axis, reference in references.items():
            if key == reference:
                continue
            comparisons = {}
            for arm in (4, 8):
                pairs = []
                for sim in selected_ids:
                    current = indexed.get((key, int(sim), arm))
                    baseline = indexed.get((reference, int(sim), arm))
                    if current is None or baseline is None:
                        continue
                    metrics = [f"{name}_{metric}" for name in ("D", "beta")
                               for metric in ("error", "abs_error", "relative_error_pct")]
                    metrics += ["singular_value_1", "condition_number", "column_cosine"]
                    metrics += [name for name in current if name.startswith(("fno_vs_", "fv_vs_"))]
                    change = {"axis": axis, "configuration": key, "reference_configuration": reference,
                              "sensor_count": layout["sensor_count"], "time_count": layout["time_count"],
                              "sim_id": int(sim), "starts": arm}
                    for metric in metrics:
                        a, b = current[metric], baseline[metric]
                        change[f"{metric}_change"] = a - b if a is not None and b is not None else None
                    pairs.append(change)
                paired_rows.extend(pairs)
                comparisons[str(arm)] = {
                    "reference_configuration": reference, "paired_count": len(pairs),
                    "unpaired_count": len(selected_ids) - len(pairs), "paired_ids": [p["sim_id"] for p in pairs],
                    "changes": {metric: distribution([p[metric] for p in pairs])
                                for metric in (pairs[0] if pairs else []) if metric.endswith("_change")},
                }
            report["paired_changes"][axis][key] = comparisons
    report["change_convention"] = "Current minus reference, matching completed trajectory ID and start count; nonconvergence is reported separately."
    return report, paired_rows


def sensor_time_sweep(config: FoxConfig, data_dir, checkpoint, output_dir, *, device=None,
                      sim_ids=None, all_test=False, overwrite=False, resume=False,
                      max_recovery_problems=None, figures=True, verbose=True) -> dict:
    config.validate()
    bundle = load_dataset(data_dir)
    ids = select_sweep_cases(bundle, config.inverse, sim_ids, all_test)
    layouts = observation_layouts(bundle.x_centers, bundle.times[1:] - bundle.times[0], config.inverse.sweep_time_counts)
    plotter = None
    if figures:
        from fno_fox.inverse_plots import plot_sensor_time_sweep
        plotter = plot_sensor_time_sweep
    return _run_sweep(config, bundle, checkpoint, output_dir, ids, layouts, device=device,
                      overwrite=overwrite, resume=resume, max_recovery_problems=max_recovery_problems,
                      figures=figures, verbose=verbose,
                      pipeline="fno_fox.inverse_sensor_time_sweep", summary_function=sweep_summary,
                      comparison_plotter=plotter, manifest_extra={
                          "observation_layouts": layouts, "metric_scopes": METRIC_SCOPE_LABELS,
                          "time_counts": sorted(config.inverse.sweep_time_counts, reverse=True),
                          "time_selection": "Nested: uniform seed; split largest available time interval at nearest saved midpoint; ties choose earlier.",
                          "time_index_origin": "Zero-based index into positive saved times, excluding the initial snapshot.",
                          "interpretation": "Noiseless S/I observations; full initial fields known. Full-grid targets are evaluation ground truth, not fitting measurements.",
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
    report = sensor_time_sweep(load_config(args.config), args.data, args.checkpoint, args.output,
                               device=args.device, sim_ids=args.sim_ids, all_test=args.all_test,
                               overwrite=args.overwrite, resume=args.resume,
                               max_recovery_problems=args.max_recovery_problems)
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
