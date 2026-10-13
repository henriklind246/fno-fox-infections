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
        channels = [str(name) for name in data["channel_names"]] if "channel_names" in data else ["S", "I"]
        fig, axes = plt.subplots(len(channels), 2, figsize=(10, 3.5 * len(channels)), squeeze=False)
        for k, channel in enumerate(channels):
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
    axes[2].set(title="Optimization cost", ylabel="Objective evaluations (shared grids excluded)"
                if "configuration" in rows[0] else "Objective evaluations incl. grid/edges")
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
            temporal = "time_count" in eight
            fitted_times = data["time_indices"] if "time_indices" in data else np.arange(len(data["times"]))
            fitted_channels = data["observed_channel_indices"] if "observed_channel_indices" in data else np.arange(2)
            if temporal:
                withheld = np.setdiff1d(np.arange(len(data["times"])), fitted_times)
                middle = withheld[len(withheld) // 2] if len(withheld) else fitted_times[len(fitted_times) // 2]
                indices = [fitted_times[0], middle, fitted_times[-1]]
            for channel, name in enumerate(("S", "I")):
                fitted_channel = channel in fitted_channels
                for col, idx in enumerate(indices):
                    for field, label, style in (("observed", "Evaluation ground truth" if temporal else "Observed", "-"),
                                                ("fno_8", "FNO", "--"), ("fv_8", "FV replay", ":")):
                        axes[channel, col].plot(data["x_centers"], data[field][idx, :, channel], style, label=label)
                    sensors = data["sensor_indices"] if "sensor_indices" in data else np.arange(len(data["x_centers"]))
                    partial = temporal or len(sensors) < len(data["x_centers"]) or len(fitted_channels) < 2
                    if partial and idx in fitted_times and fitted_channel:
                        axes[channel, col].plot(data["x_centers"][sensors], data["observed"][idx, sensors, channel],
                                               "ko", markerfacecolor="none", markersize=4, label="Fitted sensors")
                    suffix = (" (fitted time)" if idx in fitted_times else " (evaluation-only time)") if temporal else ""
                    ylabel = f"{name} density" + ("" if fitted_channel else " (evaluation only)")
                    axes[channel, col].set(xlabel="x", ylabel=ylabel, title=f"t = {data['times'][idx]:g}{suffix}")
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


def plot_sensor_time_sweep(root: Path, report: dict, layouts: dict) -> None:
    counts = sorted({v["time_count"] for v in layouts.values()})
    sensors = sorted({v["sensor_count"] for v in layouts.values()}, reverse=True)
    empty = {"median": None, "q1": None, "q3": None}

    def configuration(n, t):
        return report["configurations"][f"sensors_{n}_times_{t}"]

    def arm(n, t):
        return configuration(n, t)["arms"]["8"]

    def curve(ax, distributions, label):
        values = lambda key: np.array([np.nan if d[key] is None else d[key] for d in distributions])
        median, low, high = values("median"), values("q1"), values("q3")
        ax.errorbar(counts, median, yerr=np.stack((median - low, high - median)), marker="o", capsize=3, label=label)
        ax.set_xticks(counts)
        ax.set_xlabel("Positive observation times")
        ax.grid(alpha=.2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ("D", "beta")):
        for n in sensors:
            curve(ax, [arm(n, t)["parameters"][name]["abs_error"] for t in counts], f"{n} sensors")
        ax.set(title=f"{name} recovery", ylabel="Absolute parameter error", ylim=(0, None))
        ax.legend()
    fig.suptitle("Eight starts: median and interquartile spread across completed trajectories")
    _save(fig, root / "sensor_time_recovery.png")

    scope_labels = {"sensor": "Fitting observations", "full_grid_selected_times": "Evaluation ground truth: selected times",
                    "full_grid": "Evaluation ground truth: all saved times"}
    for scope, scope_label in scope_labels.items():
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
        for row, channel in enumerate(("S", "I")):
            for col, comparison in enumerate(("fno_vs_observation", "fv_vs_observation", "fno_vs_fv")):
                ax = axes[row, col]
                for n in sensors:
                    metric = f"{comparison}_{scope}_{channel}_rmse"
                    curve(ax, [arm(n, t)["fields"].get(metric, empty) for t in counts], f"{n} sensors")
                ax.set(title=f"{channel}: {comparison.replace('_', ' ')}", ylabel="Physical density RMSE", ylim=(0, None))
                ax.legend()
        fig.suptitle(f"{scope_label}; eight starts, median and interquartile spread")
        _save(fig, root / f"sensor_time_replay_errors_{scope}.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric, title in zip(axes, ("singular_value_1", "condition_number", "column_cosine"),
                               ("Smallest scaled singular value", "Condition number", "Signed column cosine")):
        for n in sensors:
            curve(ax, [arm(n, t)["sensitivities"][metric] for t in counts], f"{n} sensors")
        ax.set_title(title)
        ax.set_ylim((-1.05, 1.05) if metric == "column_cosine" else (0, None))
        ax.legend()
    fig.suptitle("Local FNO diagnostics at eight-start estimates; undefined quantities omitted")
    _save(fig, root / "sensor_time_sensitivities.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for n in sensors:
        for name, style in (("D", "-"), ("beta", "--")):
            values = [configuration(n, t)["start_comparison"][f"{name}_change_fraction"] for t in counts]
            before = len(axes[0].lines)
            curve(axes[0], values, f"{n} sensors: {name}")
            axes[0].lines[before].set_linestyle(style)
        curve(axes[1], [configuration(n, t)["start_comparison"]["loss_improvement"] for t in counts], f"{n} sensors")
        for starts, style in ((4, "--"), (8, "-")):
            values = [configuration(n, t)["arms"][str(starts)]["objective_evaluations"] for t in counts]
            axes[2].plot(counts, values, marker="o", linestyle=style, label=f"{n} sensors: {starts} starts")
    axes[0].set(title="Four → eight starts", ylabel="Parameter change / admissible range")
    axes[1].set(title="Additional-start improvement", ylabel="Loss (4 starts) − loss (8 starts)")
    axes[2].set(title="Configuration optimization cost", ylabel="Objective evaluations")
    for ax in axes:
        ax.set(xticks=counts, xlabel="Positive observation times", ylim=(0, None))
        ax.legend(fontsize=8)
    fig.suptitle("Shared grids excluded; boundary searches included; arms must not be added together")
    _save(fig, root / "sensor_time_start_stability.png")


# Validated categorical slots 1-3; markers carry identity as well as color.
SERIES = (("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"))
REFERENCE_INK = "#52514e"


def plot_infected_sensor_sweep(root: Path, report: dict, layouts: dict) -> None:
    from matplotlib.lines import Line2D

    keys = list(layouts)
    position = np.arange(len(keys))
    configurations = report["configurations"]
    styles = dict(zip(sorted({v["channel_label"] for v in layouts.values()}, key=len), SERIES))
    names = {"I": "I observed", "S/I": "S and I observed"}
    ticks = [f"{layouts[k]['sensor_count']} {layouts[k]['channel_label']}\nn={configurations[k]['completed_cases']}"
             for k in keys]
    empty = {"median": None, "q1": None, "q3": None}

    def arm(key):
        return configurations[key]["arms"]["8"]

    def interval(ax, x, distribution, color, marker):
        median, low, high = (np.nan if distribution.get(k) is None else distribution[k] for k in ("median", "q1", "q3"))
        ax.errorbar([x], [median], yerr=[[median - low], [high - median]], fmt=marker, color=color,
                    markersize=7, capsize=3, linewidth=1.5)

    def categorical(ax, labels):
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_xlim(-.5, len(labels) - .5)
        ax.grid(alpha=.2)

    def by_configuration(ax, distributions):
        for x, key, distribution in zip(position, keys, distributions):
            interval(ax, x, distribution, *styles[layouts[key]["channel_label"]])
        categorical(ax, ticks)

    def channel_legend(ax):
        ax.legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7, label=names[label])
                           for label, (color, marker) in styles.items()], fontsize=8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, name in zip(axes, ("D", "beta")):
        by_configuration(ax, [arm(key)["parameters"][name]["abs_error"] for key in keys])
        ax.set(title=f"{name} recovery", ylabel="Absolute parameter error", ylim=(0, None))
    channel_legend(axes[0])
    fig.suptitle("Eight starts at five shared times: median and interquartile range across completed trajectories")
    _save(fig, root / "infected_sensor_recovery.png")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, metric, title in zip(axes, ("singular_value_1", "singular_value_ratio", "condition_number", "column_cosine"),
                                 ("Smallest scaled singular value", "Smallest / largest singular value",
                                  "Condition number", "Signed column cosine")):
        by_configuration(ax, [arm(key)["sensitivities"][metric] for key in keys])
        ax.set_title(title)
        ax.set_ylim((-1.05, 1.05) if metric == "column_cosine" else (0, None))
        if metric in ("singular_value_1", "condition_number"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    channel_legend(axes[0])
    fig.suptitle("Local FNO sensitivity diagnostics at eight-start estimates (dimensionless Jacobian); undefined values omitted")
    _save(fig, root / "infected_sensor_sensitivities.png")

    comparisons = (("fno_vs_observation", "FNO vs observation"), ("fv_vs_observation", "FV replay vs observation"),
                   ("fno_vs_fv", "FNO vs FV replay"))
    scopes = (("sensor", "fitted sensors and times"), ("full_grid_selected_times", "all cells, five fitted times"),
              ("full_grid", "all cells, all saved times"))
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5), squeeze=False)
    for row, channel in enumerate(("I", "S")):
        for col, (scope, title) in enumerate(scopes):
            ax = axes[row, col]
            for offset, (comparison, _), (color, marker) in zip((-.2, 0., .2), comparisons, SERIES):
                for x, key in zip(position, keys):
                    distribution = arm(key)["fields"].get(f"{comparison}_{scope}_{channel}_rmse", empty)
                    interval(ax, x + offset, distribution, color, marker)
            categorical(ax, ticks)
            ax.set(title=f"{channel}: {title}", ylabel="Physical density RMSE", ylim=(0, None))
            if channel == "S" and scope == "sensor":
                ax.text(.02, .96, "S is evaluation-only for I-only configurations", transform=ax.transAxes,
                        va="top", fontsize=8, color=REFERENCE_INK)
    # The I-only columns of the S fitting panel are empty, so the legend never covers data there.
    axes[1, 0].legend(handles=[Line2D([], [], color=color, marker=marker, linestyle="none", markersize=7, label=label)
                               for (_, label), (color, marker) in zip(comparisons, SERIES)],
                      loc="center left", fontsize=8)
    fig.suptitle("Eight starts: median and interquartile range; full-grid fields are evaluation ground truth, not fitted")
    _save(fig, root / "infected_sensor_replay_errors.png")

    def paired_changes(kinds, filename, title):
        entries = [(kind, key, report["paired_changes"][kind][key]["8"])
                   for kind in kinds for key in report["paired_changes"].get(kind, {})]
        labels = []
        for _, key, comparison in entries:
            reference = comparison["reference_configuration"]
            labels.append(f"{layouts[key]['sensor_count']} {layouts[key]['channel_label']} − "
                          f"{layouts[reference]['sensor_count']} {layouts[reference]['channel_label']}"
                          f"\nn={comparison['paired_count']}")
        metrics = (("D_abs_error_change", "D absolute error"), ("beta_abs_error_change", "beta absolute error"),
                   ("singular_value_1_change", "Smallest scaled singular value"),
                   ("singular_value_ratio_change", "Smallest / largest singular value"),
                   ("fv_vs_observation_full_grid_I_rmse_change", "FV replay I RMSE, all cells and times"),
                   ("fv_vs_observation_full_grid_S_rmse_change", "FV replay S RMSE, all cells and times"))
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
        for ax, (metric, metric_title) in zip(axes.ravel(), metrics):
            ax.axhline(0, color=REFERENCE_INK, linewidth=1)
            for x, (_, _, comparison) in enumerate(entries):
                interval(ax, x, comparison["changes"].get(metric, empty), *SERIES[0])
            categorical(ax, labels)
            ax.set(title=metric_title, ylabel="Paired change (current − reference)")
        fig.suptitle(title)
        _save(fig, root / filename)

    paired_changes(("channel", "equal_scalar_budget"), "infected_sensor_channel_comparison.png",
                   "I-only minus S/I at matched sensors, and 32 I minus 16 S/I at 160 scalars each; "
                   "eight starts, paired median and IQR")
    paired_changes(("sensor_count",), "infected_sensor_count_comparison.png",
                   "I-only sensor count minus the 32-sensor I-only reference; layouts are not nested, "
                   "so count and placement both change")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, name in zip(axes[:2], ("D", "beta")):
        by_configuration(ax, [configurations[key]["start_comparison"][f"{name}_change_fraction"] for key in keys])
        ax.set(title=f"Four → eight starts: {name}", ylabel="Parameter change / admissible range", ylim=(0, None))
    for ax, metric, title in ((axes[2], "nonconverged_starts", "Nonconverged Nelder–Mead starts"),
                              (axes[3], "selected_candidate_not_converged", "Selected estimate not converged")):
        bars = ax.bar(position, [arm(key)[metric] for key in keys], width=.6,
                      color=[styles[layouts[key]["channel_label"]][0] for key in keys])
        # Labels distinguish true zero counts from absent bars.
        ax.bar_label(bars, fontsize=8, padding=2)
        categorical(ax, ticks)
        ax.set(title=title, ylabel="Count, eight-start arm, all cases", ylim=(0, None))
    channel_legend(axes[0])
    fig.suptitle("Start stability and optimizer nonconvergence; losses are not compared across channel sets")
    _save(fig, root / "infected_sensor_start_stability.png")
