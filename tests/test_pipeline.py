from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import torch

from fno_fox.data import FieldEncoder, PairDataset, initial_source_pairs, later_source_pairs, load_dataset
from fno_fox.evaluate import evaluate, fitted_speed, right_front
from fno_fox.model import FNOConfig, build_model
from fno_fox.train import per_sample_loss, train


def test_right_front_definition() -> None:
    x = np.arange(6, dtype=float)
    # Two humps: the outermost downward crossing is between x=4 and x=5.
    profile = np.array([0.0, 0.05, 0.0, 0.02, 0.03, 0.0])
    position, status = right_front(profile, x, 0.01)
    assert status == "valid"
    assert position == pytest.approx(4.0 + (0.03 - 0.01) / 0.03)
    assert right_front(np.full(6, 0.005), x, 0.01) == (pytest.approx(np.nan, nan_ok=True), "absent")
    assert right_front(np.full(6, 0.02), x, 0.01)[1] == "boundary"


def test_fitted_speed_needs_enough_crossings() -> None:
    times = np.linspace(0.0, 5.0, 21)
    positions = 5.0 + 0.4 * times
    assert fitted_speed(times, positions, (2.5, 5.0), 4) == pytest.approx(0.4)
    sparse = positions.copy()
    sparse[times >= 2.5] = np.nan
    sparse[-1] = 7.0
    assert np.isnan(fitted_speed(times, sparse, (2.5, 5.0), 4))


def test_tiny_cpu_generate_train_evaluate(tmp_path, tiny_dataset, tiny_config) -> None:
    run = tmp_path / "run"
    summary = train(tiny_config, tiny_dataset, run, device="cpu", verbose=False)
    for name in ("best.pt", "last.pt", "metrics.csv", "config.json", "normalization.json", "summary.json"):
        assert (run / name).exists(), name
    with (run / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == tiny_config.training.epochs
    for row in rows:
        assert float(row["val_loss"]) == pytest.approx(
            0.5 * (float(row["val_loss_initial"]) + float(row["val_loss_later"]))
        )
    assert summary["train_pairs_per_epoch"] == 14 * (5 + 5)
    with pytest.raises(FileExistsError):
        train(tiny_config, tiny_dataset, run, device="cpu", verbose=False)

    report = evaluate(tiny_config, tiny_dataset, run / "best.pt", tmp_path / "eval", device="cpu")
    assert report["pairs"] == 3 * 15
    assert report["pairs_per_trajectory"] == 15
    for category in ("initial", "later"):
        for method in ("model", "persistence"):
            for channel in ("S", "I"):
                assert np.isfinite(report["by_category"][category][method][channel]["rmse"])
    assert {row["category"] for row in report["by_lead_time"]} == {"initial", "later"}
    assert report["interpolation"]["regeneration_max_abs_difference_at_saved_times"] < 1e-6
    assert set(report["interpolation"]["by_source_time"]) == {"0", "0.6"}
    for name in ("metrics.json", "pair_metrics.csv", "fronts.csv", "profiles.png", "error_vs_lead_time.png", "front_position.png"):
        assert (tmp_path / "eval" / name).exists(), name
    json.loads((tmp_path / "eval" / "metrics.json").read_text())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_single_gpu_smoke(tmp_path, tiny_dataset, tiny_config) -> None:
    summary = train(tiny_config, tiny_dataset, tmp_path / "gpu", epochs=1, device="cuda", verbose=False)
    assert summary["device"].startswith("cuda")
    report = evaluate(tiny_config, tiny_dataset, tmp_path / "gpu" / "best.pt", tmp_path / "eval", device="cuda", figures=False)
    assert np.isfinite(report["by_category"]["later"]["model"]["I"]["rmse"])


def test_small_data_overfit_initial_and_later_sources(tiny_dataset) -> None:
    bundle = load_dataset(tiny_dataset)
    encoder = FieldEncoder.from_bundle(bundle)
    sims = bundle.splits["train"][:2]
    pairs = np.concatenate(
        (
            initial_source_pairs(sims, bundle.n_snapshots),
            later_source_pairs(sims, bundle.n_snapshots, 4, np.random.default_rng(0)),
        )
    )
    dataset = PairDataset(bundle, pairs, encoder)
    batch = {key: torch.stack([dataset[i][key] for i in range(len(dataset))]) for key in dataset[0]}
    initial = batch["source_index"] == 0
    assert initial.any() and (~initial).any()
    torch.manual_seed(0)
    model = build_model(FNOConfig(modes=8, width=16, n_layers=2, cond_hidden=16))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    def losses() -> torch.Tensor:
        return per_sample_loss(model(batch["spatial"], batch["cond"]), batch["target"])

    with torch.no_grad():
        start = losses()
    for _ in range(400):
        optimizer.zero_grad()
        losses().mean().backward()
        optimizer.step()
    with torch.no_grad():
        end = losses()
    for mask in (initial, ~initial):
        assert end[mask].mean() < 0.02 * start[mask].mean()
