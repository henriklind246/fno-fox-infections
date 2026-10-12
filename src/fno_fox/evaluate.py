"""Held-out evaluation of the fox-rabies operator.

    python -m fno_fox.evaluate --config conf/default.yaml --data data/fox_rabies \
        --checkpoint runs/default/best.pt --output runs/default/evaluation

Adapted from fno-tps-sizing ``src/fno_tps/evaluation.py`` (batched inference
and physical-unit error accumulation); the thermal metrics are replaced by
S/I field errors, a source-persistence baseline, epidemic-front diagnostics,
and a continuous lead-time interpolation check. Every admissible pair of every
held-out trajectory is queried directly; nothing is rolled out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import csv
import json

import numpy as np
import torch

from fno_fox.config import FoxConfig, load_config
from fno_fox.data import (
    DatasetBundle,
    FieldEncoder,
    admissible_pairs,
    load_dataset,
)
from fno_fox.model import FNO1d
from fno_fox.physics import simulate
from fno_fox.sampling import TrajectoryParameters
from fno_fox.train import load_checkpoint, resolve_device


ZERO_NORM = 1.0e-12
CATEGORIES = ("initial", "later")


# ----------------------------------------------------------------- inference


@torch.no_grad()
def predict(
    model: FNO1d,
    encoder: FieldEncoder,
    source_s: np.ndarray,
    source_i: np.ndarray,
    x_centers: np.ndarray,
    diffusivity: np.ndarray,
    transmission: np.ndarray,
    lead_time: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched model queries in physical units; predictions are never clipped."""
    model.eval()
    outputs_s, outputs_i = [], []
    for start in range(0, len(source_s), batch_size):
        part = slice(start, start + batch_size)
        spatial = torch.from_numpy(encoder.spatial(source_s[part], source_i[part], x_centers))
        cond = torch.from_numpy(
            encoder.condition(diffusivity[part], transmission[part], lead_time[part])
        )
        output = model(spatial.to(device), cond.to(device)).cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError(f"Non-finite model prediction in query batch starting at {start}.")
        s, i = encoder.decode(output)
        if not (np.isfinite(s).all() and np.isfinite(i).all()):
            raise ValueError(f"Non-finite decoded prediction in query batch starting at {start}.")
        outputs_s.append(s)
        outputs_i.append(i)
    return np.concatenate(outputs_s), np.concatenate(outputs_i)


def predict_pairs(
    model: FNO1d,
    encoder: FieldEncoder,
    bundle: DatasetBundle,
    pairs: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    sims, sources, targets = pairs.T
    return predict(
        model,
        encoder,
        bundle.susceptible[sims, sources],
        bundle.infectious[sims, sources],
        bundle.x_centers,
        bundle.parameters["diffusivity"][sims],
        bundle.parameters["transmission"][sims],
        bundle.times[targets] - bundle.times[sources],
        batch_size=batch_size,
        device=device,
    )


# ------------------------------------------------------------------ metrics


def field_errors(prediction: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query RMSE and relative L2 (NaN where the reference norm is zero)."""
    error = np.asarray(prediction, dtype=np.float64) - truth
    squared = np.mean(error**2, axis=-1)
    reference = np.mean(np.asarray(truth, dtype=np.float64) ** 2, axis=-1)
    defined = np.sqrt(reference) > ZERO_NORM
    relative = np.full(squared.shape, np.nan)
    relative[defined] = np.sqrt(squared[defined] / reference[defined])
    return {"squared": squared, "reference": reference, "rmse": np.sqrt(squared), "relative": relative}


def summarize(errors: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    if not mask.any():
        return {"count": 0}
    relative = errors["relative"][mask]
    defined = np.isfinite(relative)
    reference = float(np.sum(errors["reference"][mask]))
    return {
        "count": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(errors["squared"][mask]))),
        "relative_l2_pooled": float(np.sqrt(np.sum(errors["squared"][mask]) / reference))
        if reference > 0.0
        else None,
        "relative_l2_mean": float(np.mean(relative[defined])) if defined.any() else None,
        "relative_l2_median": float(np.median(relative[defined])) if defined.any() else None,
        "relative_l2_undefined_count": int((~defined).sum()),
    }


def negative_summary(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = values[mask]
    negative = selected < 0.0
    return {
        "negative_fraction": float(negative.mean()) if selected.size else 0.0,
        "negative_count": int(negative.sum()),
        "minimum": float(selected.min()) if selected.size else None,
    }


# -------------------------------------------------------------------- fronts


def right_front(infectious: np.ndarray, x_centers: np.ndarray, threshold: float) -> tuple[float, str]:
    """Outermost downward crossing of ``I = threshold``, linearly interpolated.

    Returns ``(nan, "absent")`` without a crossing and ``(nan, "boundary")``
    when the infection still exceeds the threshold in the last cell.
    """
    above = np.flatnonzero(infectious >= threshold)
    if above.size == 0:
        return float("nan"), "absent"
    last = int(above[-1])
    if last == len(infectious) - 1:
        return float("nan"), "boundary"
    left, right = float(infectious[last]), float(infectious[last + 1])
    fraction = (left - threshold) / (left - right)
    return float(x_centers[last] + fraction * (x_centers[last + 1] - x_centers[last])), "valid"


def front_track(
    fields: np.ndarray,
    x_centers: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, list[str]]:
    positions, statuses = [], []
    for profile in fields:
        position, status = right_front(profile, x_centers, threshold)
        positions.append(position)
        statuses.append(status)
    return np.asarray(positions), statuses


def fitted_speed(
    times: np.ndarray,
    positions: np.ndarray,
    window: tuple[float, float],
    min_points: int,
) -> float:
    """Least-squares front speed over ``window`` from valid crossings only."""
    inside = (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9) & np.isfinite(positions)
    if inside.sum() < min_points:
        return float("nan")
    return float(np.polyfit(times[inside], positions[inside], 1)[0])


def asymptotic_speed(diffusivity: float, transmission: float, mortality: float, background: float = 1.0) -> float:
    growth = transmission * background - mortality
    return float(2.0 * np.sqrt(diffusivity * growth)) if growth > 0.0 else float("nan")


def front_diagnostics(
    config: FoxConfig,
    bundle: DatasetBundle,
    sims: np.ndarray,
    initial_prediction: dict[int, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, np.ndarray]]]:
    """Front positions and speeds from initial-source predictions vs truth."""
    threshold = config.evaluation.front_threshold
    window = config.evaluation.speed_window
    min_points = config.evaluation.min_front_points
    mortality = bundle.config.domain.mortality
    times = bundle.times
    rows: list[dict[str, Any]] = []
    tracks: dict[int, dict[str, np.ndarray]] = {}
    position_errors, speed_rows = [], []
    availability_mismatch = 0
    for sim in sims:
        truth, truth_status = front_track(bundle.infectious[sim].astype(np.float64), bundle.x_centers, threshold)
        # t = 0 is the known initial field; predictions cover every later time.
        predicted_fields = np.concatenate((bundle.infectious[sim, :1], initial_prediction[int(sim)]))
        prediction, prediction_status = front_track(predicted_fields, bundle.x_centers, threshold)
        tracks[int(sim)] = {"truth": truth, "prediction": prediction}
        for index in range(1, len(times)):
            both = np.isfinite(truth[index]) and np.isfinite(prediction[index])
            availability_mismatch += int(truth_status[index] != prediction_status[index])
            if both:
                position_errors.append(abs(prediction[index] - truth[index]))
            rows.append(
                {
                    "sim_id": int(sim),
                    "time": float(times[index]),
                    "true_front": truth[index],
                    "predicted_front": prediction[index],
                    "true_status": truth_status[index],
                    "predicted_status": prediction_status[index],
                }
            )
        constant = bool(bundle.parameters["susceptible_constant"][sim])
        speed_rows.append(
            {
                "sim_id": int(sim),
                "true_speed": fitted_speed(times, truth, window, min_points),
                "predicted_speed": fitted_speed(times, prediction, window, min_points),
                "uniform_background": constant,
                "asymptotic_speed": asymptotic_speed(
                    float(bundle.parameters["diffusivity"][sim]),
                    float(bundle.parameters["transmission"][sim]),
                    mortality,
                )
                if constant
                else float("nan"),
            }
        )
    true_speed = np.array([row["true_speed"] for row in speed_rows])
    predicted_speed = np.array([row["predicted_speed"] for row in speed_rows])
    both = np.isfinite(true_speed) & np.isfinite(predicted_speed)
    uniform = np.array([row["uniform_background"] for row in speed_rows])
    asymptotic = np.array([row["asymptotic_speed"] for row in speed_rows])
    with_asymptotic = uniform & np.isfinite(true_speed) & np.isfinite(asymptotic)
    summary = {
        "threshold": threshold,
        "definition": "outermost downward crossing of I = threshold, linear interpolation; "
        "absent or boundary-reaching fronts are unavailable",
        "position_abs_error_mean": float(np.mean(position_errors)) if position_errors else None,
        "position_abs_error_p95": float(np.percentile(position_errors, 95)) if position_errors else None,
        "position_pairs_compared": len(position_errors),
        "availability_mismatches": availability_mismatch,
        "speed_window": list(window),
        "speed_min_points": min_points,
        "speed_trajectories_compared": int(both.sum()),
        "speed_abs_error_mean": float(np.mean(np.abs(predicted_speed[both] - true_speed[both])))
        if both.any()
        else None,
        "speed_relative_error_mean": float(
            np.mean(np.abs(predicted_speed[both] - true_speed[both]) / np.abs(true_speed[both]))
        )
        if both.any()
        else None,
        "uniform_background_true_over_asymptotic_mean": float(
            np.mean(true_speed[with_asymptotic] / asymptotic[with_asymptotic])
        )
        if with_asymptotic.any()
        else None,
        "uniform_background_count": int(with_asymptotic.sum()),
        "speeds": speed_rows,
    }
    return summary, rows, tracks


# ------------------------------------------------------- interpolation test


def trajectory_parameters(bundle: DatasetBundle, sim: int) -> TrajectoryParameters:
    p = bundle.parameters
    constant = bool(p["susceptible_constant"][sim])
    return TrajectoryParameters(
        diffusivity=float(p["diffusivity"][sim]),
        transmission=float(p["transmission"][sim]),
        infection_amplitude=float(p["infection_amplitude"][sim]),
        infection_center=float(p["infection_center"][sim]),
        infection_width=float(p["infection_width"][sim]),
        susceptible_family="constant" if constant else "cosine",
        susceptible_amplitude=float(p["susceptible_amplitude"][sim]),
        susceptible_coefficients=tuple(float(v) for v in p["susceptible_coefficients"][sim]),
    )


def _interpolation_sources(config: FoxConfig, bundle: DatasetBundle) -> list[int]:
    """Validate against the actual dataset before running any model or solver queries."""
    if config.evaluation.interpolation_trajectories == 0:
        return []
    sources = []
    for source_time in config.evaluation.interpolation_source_times:
        matches = np.flatnonzero(np.isclose(bundle.times, source_time, rtol=0.0, atol=1e-9))
        if not np.isfinite(source_time) or matches.size != 1:
            raise ValueError(f"Interpolation source time {source_time} is not a saved snapshot.")
        source = int(matches[0])
        if source == len(bundle.times) - 1:
            raise ValueError(
                f"Interpolation source time {source_time} must precede the final snapshot "
                "so future targets exist."
            )
        sources.append(source)
    return sources


def interpolation_check(
    config: FoxConfig,
    bundle: DatasetBundle,
    model: FNO1d,
    encoder: FieldEncoder,
    device: torch.device,
    *,
    split: str = "test",
) -> dict[str, Any]:
    """Query lead times halfway between saved snapshots on the requested split."""
    count = config.evaluation.interpolation_trajectories
    if count == 0:
        return {"skipped": True}
    sources = _interpolation_sources(config, bundle)
    physics = bundle.config
    sims = np.sort(bundle.splits[split])[:count]
    times = bundle.times.astype(np.float64)
    midpoints = 0.5 * (times[:-1] + times[1:])
    schedule = np.sort(np.concatenate((times, midpoints)))
    saved_rows = np.searchsorted(schedule, times)
    mid_rows = np.searchsorted(schedule, midpoints)
    regenerated: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    consistency = 0.0
    for sim in sims:
        s, i, _ = simulate(physics, trajectory_parameters(bundle, int(sim)), schedule)
        regenerated[int(sim)] = (s, i)
        consistency = max(
            consistency,
            float(np.max(np.abs(s[saved_rows] - bundle.susceptible[sim]))),
            float(np.max(np.abs(i[saved_rows] - bundle.infectious[sim]))),
        )
    result: dict[str, Any] = {
        "split": split,
        "trajectories": sims.tolist(),
        "regeneration_max_abs_difference_at_saved_times": consistency,
        "by_source_time": {},
    }
    for source in sources:
        groups: dict[str, Any] = {}
        for name, target_times, target_s, target_i in (
            (
                "midpoint_targets",
                midpoints[midpoints > times[source]],
                lambda sim: regenerated[sim][0][mid_rows][midpoints > times[source]],
                lambda sim: regenerated[sim][1][mid_rows][midpoints > times[source]],
            ),
            (
                "saved_targets",
                times[source + 1 :],
                lambda sim: bundle.susceptible[sim, source + 1 :],
                lambda sim: bundle.infectious[sim, source + 1 :],
            ),
        ):
            k = len(target_times)
            sim_list = np.repeat(sims, k)
            source_s = np.repeat(bundle.susceptible[sims, source], k, axis=0)
            source_i = np.repeat(bundle.infectious[sims, source], k, axis=0)
            truth_s = np.concatenate([target_s(int(sim)) for sim in sims])
            truth_i = np.concatenate([target_i(int(sim)) for sim in sims])
            leads = np.tile(target_times - times[source], len(sims))
            pred_s, pred_i = predict(
                model,
                encoder,
                source_s,
                source_i,
                bundle.x_centers,
                bundle.parameters["diffusivity"][sim_list],
                bundle.parameters["transmission"][sim_list],
                leads,
                batch_size=config.evaluation.batch_size,
                device=device,
            )
            everything = np.ones(len(leads), dtype=bool)
            groups[name] = {
                "lead_times": sorted(set(np.round(leads, 9).tolist())),
                "model": {
                    "S": summarize(field_errors(pred_s, truth_s), everything),
                    "I": summarize(field_errors(pred_i, truth_i), everything),
                },
                "persistence": {
                    "S": summarize(field_errors(source_s, truth_s), everything),
                    "I": summarize(field_errors(source_i, truth_i), everything),
                },
            }
        result["by_source_time"][f"{times[source]:g}"] = groups
    return result


# ------------------------------------------------------------------- figures


INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # reference categorical slots 1-3
SURFACE = "#fcfcfb"


def _style_axes(axis) -> None:
    axis.set_facecolor(SURFACE)
    axis.grid(True, color=INK["grid"], linewidth=0.75, linestyle="-")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(INK["axis"])
    axis.tick_params(colors=INK["secondary"], labelsize=8)
    axis.xaxis.label.set_color(INK["secondary"])
    axis.yaxis.label.set_color(INK["secondary"])
    axis.title.set_color(INK["primary"])


def _figure(rows: int, cols: int, size: tuple[float, float]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, facecolor=SURFACE)
    for axis in axes.flat:
        _style_axes(axis)
    return figure, axes, plt


def plot_profiles(path: Path, bundle: DatasetBundle, representatives: list[tuple[int, str]], predictions: dict[int, tuple[np.ndarray, np.ndarray]]) -> None:
    n = len(bundle.times)
    picks = sorted({max(1, n // 4), n // 2, n - 1})
    figure, axes, plt = _figure(len(representatives), 4, (13.0, 2.9 * len(representatives)))
    x = bundle.x_centers
    for row, (sim, label) in enumerate(representatives):
        pred_s, pred_i = predictions[sim]
        for color, index in zip(SERIES, picks):
            t = bundle.times[index]
            truth_s, truth_i = bundle.susceptible[sim, index], bundle.infectious[sim, index]
            axes[row, 0].plot(x, truth_s, color=color, lw=1.5, label=f"t = {t:g}")
            axes[row, 0].plot(x, pred_s[index - 1], color=color, lw=1.5, ls="--")
            axes[row, 1].plot(x, truth_i, color=color, lw=1.5)
            axes[row, 1].plot(x, pred_i[index - 1], color=color, lw=1.5, ls="--")
            axes[row, 2].plot(x, pred_s[index - 1] - truth_s, color=color, lw=1.5)
            axes[row, 3].plot(x, pred_i[index - 1] - truth_i, color=color, lw=1.5)
        d = bundle.parameters["diffusivity"][sim]
        b = bundle.parameters["transmission"][sim]
        axes[row, 0].set_ylabel(f"sim {sim} ({label})\nD={d:.3f}, β={b:.2f}", fontsize=8)
        for column, title in enumerate(("S: truth (solid) vs FNO (dashed)", "I: truth (solid) vs FNO (dashed)", "S error (FNO − truth)", "I error (FNO − truth)")):
            if row == 0:
                axes[row, column].set_title(title, fontsize=9, loc="left")
            if row == len(representatives) - 1:
                axes[row, column].set_xlabel("x")
    handles = [plt.Line2D([], [], color=c, lw=1.5) for c in SERIES[: len(picks)]]
    handles += [plt.Line2D([], [], color=INK["secondary"], lw=1.5), plt.Line2D([], [], color=INK["secondary"], lw=1.5, ls="--")]
    labels = [f"t = {bundle.times[i]:g}" for i in picks] + ["reference solver", "FNO from t = 0"]
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8, labelcolor=INK["secondary"])
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_error_vs_lead(path: Path, by_lead: list[dict[str, Any]]) -> None:
    figure, axes, plt = _figure(1, 2, (11.0, 3.8))
    for column, channel in enumerate(("S", "I")):
        axis = axes[0, column]
        for color, category in zip(SERIES, CATEGORIES):
            rows = [row for row in by_lead if row["category"] == category]
            if not rows:
                continue
            leads = [row["lead_time"] for row in rows]
            axis.plot(leads, [row["model"][channel]["rmse"] for row in rows], color=color, lw=1.5, marker="o", ms=4, label=f"FNO, {category} source")
            axis.plot(leads, [row["persistence"][channel]["rmse"] for row in rows], color=color, lw=1.5, ls="--", label=f"persistence, {category} source")
        axis.set_yscale("log")
        axis.set_xlabel("lead time t_j − t_s")
        axis.set_ylabel(f"{channel} RMSE (physical units)")
        axis.set_title(f"{channel}: held-out RMSE by lead time", fontsize=9, loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8, labelcolor=INK["secondary"])
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


def plot_fronts(path: Path, times: np.ndarray, representatives: list[tuple[int, str]], tracks: dict[int, dict[str, np.ndarray]], speeds: list[dict[str, Any]]) -> None:
    figure, axes, plt = _figure(1, 2, (11.0, 4.0))
    axis = axes[0, 0]
    for color, (sim, label) in zip(SERIES, representatives):
        axis.plot(times, tracks[sim]["truth"], color=color, lw=1.5, label=f"sim {sim} ({label})")
        axis.plot(times, tracks[sim]["prediction"], color=color, lw=1.5, ls="--", marker="o", ms=4)
    axis.set_xlabel("t")
    axis.set_ylabel("right front x_f(t), I = threshold")
    axis.set_title("Front position: truth (solid) vs FNO from t = 0 (dashed)", fontsize=9, loc="left")
    axis.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"])
    axis = axes[0, 1]
    for color, uniform, label in ((SERIES[0], True, "uniform S0"), (SERIES[1], False, "perturbed S0")):
        rows = [r for r in speeds if r["uniform_background"] == uniform and np.isfinite(r["true_speed"]) and np.isfinite(r["predicted_speed"])]
        axis.scatter([r["true_speed"] for r in rows], [r["predicted_speed"] for r in rows], s=22, color=color, edgecolor=SURFACE, linewidth=1.0, label=label, zorder=3)
    finite = [v for r in speeds for v in (r["true_speed"], r["predicted_speed"]) if np.isfinite(v)]
    if finite:
        low, high = min(finite), max(finite)
        axis.plot([low, high], [low, high], color=INK["muted"], lw=1.0, zorder=2, label="identity")
    axis.set_xlabel("fitted front speed, reference solver")
    axis.set_ylabel("fitted front speed, FNO")
    axis.set_title("Finite-time front speed over the fit window", fontsize=9, loc="left")
    axis.legend(frameon=False, fontsize=8, labelcolor=INK["secondary"])
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)


# ------------------------------------------------------------------- driver


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    """Represent unavailable numerical diagnostics as JSON null, including nested speeds."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def evaluate(
    config: FoxConfig,
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    split: str = "test",
    device: str | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved = resolve_device(device or config.training.device)
    bundle = load_dataset(data_dir)
    _interpolation_sources(config, bundle)
    model, payload = load_checkpoint(checkpoint, resolved)
    encoder = FieldEncoder.from_dict(payload["encoder"])
    if encoder != FieldEncoder.from_bundle(bundle):
        raise ValueError("Checkpoint normalization/conditioning does not match this dataset.")

    sims = np.sort(bundle.splits[split])
    pairs = admissible_pairs(sims, bundle.n_snapshots)
    pred_s, pred_i = predict_pairs(
        model, encoder, bundle, pairs, batch_size=config.evaluation.batch_size, device=resolved
    )
    sim_index, sources, targets = pairs.T
    truth_s = bundle.susceptible[sim_index, targets]
    truth_i = bundle.infectious[sim_index, targets]
    errors = {
        "model": {"S": field_errors(pred_s, truth_s), "I": field_errors(pred_i, truth_i)},
        "persistence": {
            "S": field_errors(bundle.susceptible[sim_index, sources], truth_s),
            "I": field_errors(bundle.infectious[sim_index, sources], truth_i),
        },
    }
    category_masks = {"initial": sources == 0, "later": sources > 0}
    lead_time = bundle.times[targets] - bundle.times[sources]
    lag = targets - sources

    def block(mask: np.ndarray) -> dict[str, Any]:
        return {
            method: {channel: summarize(errors[method][channel], mask) for channel in ("S", "I")}
            for method in errors
        }

    by_category = {name: block(mask) for name, mask in category_masks.items()}
    by_category["all"] = block(np.ones(len(pairs), dtype=bool))
    by_lead = []
    for name, mask in category_masks.items():
        for step in np.unique(lag[mask]):
            selected = mask & (lag == step)
            by_lead.append(
                {"category": name, "lead_steps": int(step), "lead_time": float(lead_time[selected][0]), **block(selected)}
            )
    negatives = {
        name: {
            "S": negative_summary(pred_s, mask),
            "I": negative_summary(pred_i, mask),
        }
        for name, mask in category_masks.items()
    }

    initial_prediction: dict[int, np.ndarray] = {}
    initial_prediction_s: dict[int, np.ndarray] = {}
    for sim in sims:
        rows = np.flatnonzero((sim_index == sim) & (sources == 0))
        rows = rows[np.argsort(targets[rows])]
        initial_prediction[int(sim)] = pred_i[rows]
        initial_prediction_s[int(sim)] = pred_s[rows]
    fronts, front_rows, tracks = front_diagnostics(config, bundle, sims, initial_prediction)

    # Representative trajectories: low, median, and high initial-source I error.
    per_sim_error = np.array(
        [np.sqrt(np.mean(errors["model"]["I"]["squared"][(sim_index == sim) & (sources == 0)])) for sim in sims]
    )
    order = np.argsort(per_sim_error)
    count = min(config.evaluation.representative_trajectories, len(sims))
    quantiles = np.linspace(0.1, 0.9, count) if count > 1 else np.array([0.5])
    names = {0.1: "low error", 0.5: "median error", 0.9: "high error"}
    representatives = []
    for quantile in quantiles:
        sim = int(sims[order[int(round(quantile * (len(sims) - 1)))]])
        if sim not in [r[0] for r in representatives]:
            representatives.append((sim, names.get(round(float(quantile), 2), f"q{quantile:.2f}")))

    interpolation = interpolation_check(config, bundle, model, encoder, resolved, split=split)
    report = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_epoch": payload["epoch"],
        "dataset": str(Path(data_dir).resolve()),
        "dataset_sha256": bundle.manifest["dataset_sha256"],
        "split": split,
        "trajectories": int(len(sims)),
        "pairs": int(len(pairs)),
        "pairs_per_trajectory": int(len(pairs) // len(sims)),
        "units": "physical (nondimensional) densities on the learning grid",
        "relative_l2_note": f"undefined (excluded) where the reference RMS <= {ZERO_NORM:g}",
        "by_category": by_category,
        "by_lead_time": by_lead,
        "negative_predictions": negatives,
        "fronts": fronts,
        "interpolation": interpolation,
        "representative_trajectories": [{"sim_id": s, "label": l} for s, l in representatives],
    }
    report = _json_safe(report)
    (destination / "metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pair_rows = [
        {
            "sim_id": int(sim_index[k]),
            "source_index": int(sources[k]),
            "target_index": int(targets[k]),
            "lead_time": float(lead_time[k]),
            "category": "initial" if sources[k] == 0 else "later",
            **{
                f"{method}_{metric}_{channel}": float(errors[method][channel][metric][k])
                for method in errors
                for channel in ("S", "I")
                for metric in ("rmse", "relative")
            },
        }
        for k in range(len(pairs))
    ]
    _write_csv(destination / "pair_metrics.csv", pair_rows)
    _write_csv(destination / "fronts.csv", front_rows)
    _write_csv(destination / "front_speeds.csv", fronts["speeds"])
    if figures:
        plot_profiles(
            destination / "profiles.png",
            bundle,
            representatives,
            {sim: (initial_prediction_s[sim], initial_prediction[sim]) for sim, _ in representatives},
        )
        plot_error_vs_lead(destination / "error_vs_lead_time.png", by_lead)
        plot_fronts(destination / "front_position.png", bundle.times, representatives, tracks, fronts["speeds"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    report = evaluate(
        load_config(args.config),
        args.data,
        args.checkpoint,
        args.output,
        split=args.split,
        device=args.device,
    )
    for category in ("initial", "later"):
        block = report["by_category"][category]
        print(
            f"{category:>7} source: "
            + "  ".join(
                f"{channel} RMSE FNO {block['model'][channel]['rmse']:.3e} "
                f"vs persistence {block['persistence'][channel]['rmse']:.3e}"
                for channel in ("S", "I")
            )
        )
    print(f"Wrote {args.output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
