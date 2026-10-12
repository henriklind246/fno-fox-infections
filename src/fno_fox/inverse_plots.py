"""Plots read saved inverse artifacts; no inference or fitting is performed here."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_inverse(root: Path, rows: list[dict]) -> None:
    with np.load(root / "validation_residuals.npz") as data:
        x, times = data["x_centers"], data["times"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
        for k, channel in enumerate(("S", "I")):
            for col, key in enumerate(("signed_mean", "rmse")):
                values = data[key][..., k]
                kwargs = {"cmap": "viridis"}
                if key == "signed_mean":
                    limit = max(float(np.max(np.abs(values))), 1e-15)
                    kwargs = {"cmap": "RdBu_r", "vmin": -limit, "vmax": limit}
                artist = axes[k, col].pcolormesh(x, times, values, shading="auto", **kwargs)
                axes[k, col].set(title=f"Validation {channel}: {key.replace('_', ' ')}", xlabel="x", ylabel="Time")
                axes[k, col].set_ylim(times[0], times[-1])
                fig.colorbar(artist, ax=axes[k, col], label="Density error")
        _save(fig, root / "validation_residuals.png")
    if not rows:
        return
    arms = {n: sorted([r for r in rows if r["starts"] == n], key=lambda r: r["sim_id"]) for n in (4, 8)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for k, name in enumerate(("D", "beta")):
        for count, marker in ((4, "o"), (8, "x")):
            truth = np.array([r[f"{name}_true"] for r in arms[count]])
            estimate = np.array([r[f"{name}_hat"] for r in arms[count]])
            axes[k].scatter(truth, estimate, label=f"{count} starts", marker=marker, alpha=0.8)
        limits = np.concatenate((truth, [r[f"{name}_hat"] for r in rows]))
        lo, hi = limits.min(), limits.max()
        axes[k].plot([lo, hi], [lo, hi], "k--", linewidth=1)
        axes[k].set(xlabel=f"True {name}", ylabel=f"Recovered {name}", title=f"{name} recovery")
        axes[k].legend()
    _save(fig, root / "recovery.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ids = [r["sim_id"] for r in arms[8]]
    index = np.arange(len(ids))
    for name in ("D", "beta"):
        axes[0].plot(index, [r[f"{name}_start_change_fraction"] for r in arms[8]], "o-", label=name)
    axes[0].set(title="Four → eight starts", ylabel="Absolute change / parameter range")
    axes[0].legend()
    axes[1].bar(index, [r["loss_improvement"] for r in arms[8]])
    axes[1].set(title="Objective improvement", ylabel="Loss (4 starts) − loss (8 starts)")
    for count in (4, 8):
        axes[2].plot(index, [r["evaluations"] for r in arms[count]], "o-", label=f"{count} starts")
    axes[2].set(title="Optimization cost", ylabel="Objective evaluations incl. grid/edges")
    axes[2].legend()
    for ax in axes:
        ax.set_xticks(index, ids, rotation=45)
        ax.set_xlabel("Test simulation ID")
    _save(fig, root / "start_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for k in (0, 1):
        axes[0].plot(index, [r[f"singular_value_{k}"] for r in arms[8]], "o-", label=f"s{k + 1}")
    axes[0].set(title="Scaled Jacobian singular values", ylabel="Singular value")
    axes[0].legend()
    for name in ("D", "beta"):
        axes[1].plot(index, [r[f"sensitivity_{name}"] for r in arms[8]], "o-", label=name)
    axes[1].set(title="Range-scaled sensitivity", ylabel="Column norm")
    axes[1].legend()
    axes[2].plot(index, [np.nan if r["column_cosine"] is None else r["column_cosine"] for r in arms[8]], "o")
    axes[2].set(title="Column alignment", ylabel="Signed cosine (undefined if zero column)", ylim=(-1.05, 1.05))
    for ax in axes[:2]:
        ax.set_ylim(bottom=0)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        ax.set_xticks(index, ids, rotation=45)
        ax.set_xlabel("Test simulation ID")
    fig.suptitle("Local FNO diagnostics at eight-start estimates")
    _save(fig, root / "sensitivities.png")

    for four, eight in zip(arms[4], arms[8]):
        directory = root / "cases" / str(eight["sim_id"])
        with np.load(directory / "grid.npz") as data:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            surface = axes[0].pcolormesh(data["D_axis"], data["beta_axis"], data["loss"].T, shading="auto")
            fig.colorbar(surface, ax=axes[0], label="Half normalized mean squared residual")
            axes[0].plot(eight["D_true"], eight["beta_true"], "w*", markeredgecolor="k", markersize=12, label="Truth")
            for row, marker in ((four, "o"), (eight, "x")):
                axes[0].plot(row["D_hat"], row["beta_hat"], marker, color="red", label=f"{row['starts']} starts")
            axes[0].set(xlabel="D", ylabel="beta", title=f"Case {eight['sim_id']}: joint loss")
            axes[0].set_xlim(data["D_axis"][[0, -1]])
            axes[0].set_ylim(data["beta_axis"][[0, -1]])
            axes[0].legend()
            axes[1].plot(data["unit_axis"], data["loss"].min(axis=1), label="D")
            axes[1].plot(data["unit_axis"], data["loss"].min(axis=0), label="beta")
            axes[1].set(xlabel="Fraction of parameter range", ylabel="Minimum sampled loss", title="Coarse grid minima")
            axes[1].legend()
            _save(fig, directory / "loss_surface.png")
        with np.load(directory / "fields.npz") as data:
            fig, axes = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
            indices = np.linspace(0, len(data["times"]) - 1, 3).astype(int)
            for channel, name in enumerate(("S", "I")):
                for col, idx in enumerate(indices):
                    for field, label, style in (("observed", "Observed", "-"), ("fno_8", "FNO", "--"), ("fv_8", "FV replay", ":")):
                        axes[channel, col].plot(data["x_centers"], data[field][idx, :, channel], style, label=label)
                    sensors = data["sensor_indices"] if "sensor_indices" in data else np.arange(len(data["x_centers"]))
                    if len(sensors) < len(data["x_centers"]):
                        axes[channel, col].plot(data["x_centers"][sensors], data["observed"][idx, sensors, channel],
                                               "ko", markerfacecolor="none", markersize=4, label="Fitted sensors")
                    axes[channel, col].set(xlabel="x", ylabel=f"{name} density", title=f"t = {data['times'][idx]:g}")
                    axes[channel, col].legend()
            fig.suptitle(f"Case {eight['sim_id']}: eight-start estimate")
            _save(fig, directory / "reconstruction.png")


def plot_sensor_sweep(root: Path, report: dict, layouts: dict) -> None:
    ordered = sorted(layouts, key=lambda key: layouts[key]["sensor_count"])
    counts = [layouts[key]["sensor_count"] for key in ordered]

    def curve(ax, distributions, label):
        def values(key):
            return np.array([np.nan if d[key] is None else d[key] for d in distributions])
        median, low, high = values("median"), values("q1"), values("q3")
        ax.errorbar(counts, median, yerr=np.stack((median - low, high - median)),
                    marker="o", capsize=3, label=label)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(set(counts)), sorted(set(counts)))
        ax.set_xlabel("Spatial sensors")
        ax.grid(alpha=.2)

    def arm(key, n=8):
        return report["configurations"][key]["arms"][str(n)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ("D", "beta")):
        curve(ax, [arm(key)["parameters"][name]["abs_error"] for key in ordered], name)
        ax.set(title=f"{name} recovery", ylabel="Absolute parameter error")
        ax.set_ylim(bottom=0)
    fig.suptitle("Eight starts: median and interquartile range across successful cases")
    _save(fig, root / "sensor_recovery.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for row, scope in enumerate(("sensor", "full_grid")):
        for col, channel in enumerate(("S", "I")):
            ax = axes[row, col]
            for comparison in ("fno_vs_observation", "fv_vs_observation", "fno_vs_fv"):
                metric = f"{comparison}_{scope}_{channel}_rmse"
                curve(ax, [arm(key)["fields"].get(metric, {"median": None, "q1": None, "q3": None}) for key in ordered],
                      comparison.replace("_", " "))
            ax.set(title=f"{channel}: {scope.replace('_', ' ')}", ylabel="Physical density RMSE")
            ax.set_ylim(bottom=0)
            ax.legend()
    fig.suptitle("Eight starts: median and interquartile range across successful cases")
    _save(fig, root / "sensor_replay_errors.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes, ("singular_value_1", "condition_number", "column_cosine"),
                               ("Smallest scaled singular value", "Condition number", "Signed column cosine")):
        curve(ax, [arm(key)["sensitivities"][metric] for key in ordered], title)
        ax.set_title(title)
        if metric == "column_cosine":
            ax.set_ylim(-1.05, 1.05)
        else:
            ax.set_ylim(bottom=0)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Local FNO diagnostics at eight-start estimates; undefined quantities omitted")
    _save(fig, root / "sensor_sensitivities.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name in ("D", "beta"):
        curve(axes[0], [report["configurations"][key]["start_comparison"][f"{name}_change_fraction"]
                        for key in ordered], name)
    axes[0].set(title="Four → eight starts", ylabel="Parameter change / admissible range")
    axes[0].legend()
    curve(axes[1], [report["configurations"][key]["start_comparison"]["loss_improvement"] for key in ordered], "Loss")
    axes[1].set(title="Additional-start improvement", ylabel="Loss (4 starts) − loss (8 starts)")
    for n in (4, 8):
        total = [arm(key, n)["objective_evaluations"] for key in ordered]
        axes[2].plot(counts, total, "o-", label=f"{n} starts")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(sorted(set(counts)), sorted(set(counts)))
    axes[2].set(title="Configuration optimization cost", ylabel="Total objective evaluations", xlabel="Spatial sensors")
    axes[2].legend()
    for ax in axes:
        ax.set_ylim(bottom=0)
    fig.suptitle("Shared grids excluded; per-configuration costs include boundary searches")
    _save(fig, root / "sensor_start_stability.png")
