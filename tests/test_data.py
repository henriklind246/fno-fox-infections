from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from fno_fox.config import SamplingConfig
from fno_fox.data import (
    FieldEncoder,
    PairDataset,
    admissible_pairs,
    build_datasets,
    compute_normalization,
    initial_source_pairs,
    later_source_pairs,
    load_dataset,
    split_simulation_ids,
    training_pairs,
    validation_pairs,
)
from fno_fox.physics import Grid
from fno_fox.sampling import initial_fields, sample_parameters


N_SNAPSHOTS = 21


# ----------------------------------------------------------------- sampling


def test_sampling_is_reproducible() -> None:
    config = SamplingConfig()
    first = sample_parameters(config, np.random.default_rng(1))
    second = sample_parameters(config, np.random.default_rng(1))
    other = sample_parameters(config, np.random.default_rng(2))
    assert first == second
    assert first != other


def test_parameter_distributions() -> None:
    config = SamplingConfig()
    params = sample_parameters(config, np.random.default_rng(1))
    n = len(params)
    d = np.array([p.diffusivity for p in params])
    b = np.array([p.transmission for p in params])
    for values, (low, high) in ((d, config.diffusivity), (b, config.transmission)):
        assert values.min() >= low and values.max() <= high
        # Latin hypercube: exactly one sample in each of the n strata.
        strata = np.floor((values - low) / (high - low) * n).astype(int)
        assert sorted(strata) == list(range(n))
    assert abs(np.corrcoef(d, b)[0, 1]) < 0.15
    for name in ("infection_amplitude", "infection_center", "infection_width"):
        values = np.array([getattr(p, name) for p in params])
        low, high = getattr(config, name)
        assert values.min() >= low and values.max() <= high
    families = [p.susceptible_family for p in params]
    assert families.count("constant") == families.count("cosine") == n // 2
    for p in params:
        if p.susceptible_family == "cosine":
            assert np.sum(np.abs(p.susceptible_coefficients)) == pytest.approx(1.0)
            low, high = config.susceptible_perturbation_amplitude
            assert low <= p.susceptible_amplitude <= high
        else:
            assert p.susceptible_amplitude == 0.0
            assert not any(p.susceptible_coefficients)


def test_initial_fields_are_positive(default_config) -> None:
    params = sample_parameters(default_config.sampling, np.random.default_rng(1))
    grid = Grid(default_config.domain.length, default_config.domain.learning_cells)
    for p in params:
        s0, i0 = initial_fields(p, grid.faces, default_config.domain.length)
        assert s0.min() >= 1.0 - p.susceptible_amplitude - 1e-12
        assert s0.min() > 0.0
        assert i0.min() >= 0.0
        assert i0.max() <= p.infection_amplitude
        # The Gaussian is resolved: the peak cell average is close to A.
        assert i0.max() > 0.95 * p.infection_amplitude
        if p.susceptible_family == "constant":
            np.testing.assert_array_equal(s0, 1.0)


# ------------------------------------------------------- splits and stats


def test_simulation_splits_are_disjoint() -> None:
    splits = split_simulation_ids(600, (0.7, 0.15, 0.15), seed=2)
    assert [len(splits[k]) for k in ("train", "val", "test")] == [420, 90, 90]
    combined = np.concatenate(list(splits.values()))
    assert sorted(combined) == list(range(600))
    again = split_simulation_ids(600, (0.7, 0.15, 0.15), seed=2)
    for key in splits:
        np.testing.assert_array_equal(splits[key], again[key])


def test_normalization_uses_training_snapshots_only() -> None:
    rng = np.random.default_rng(0)
    s = rng.random((10, 5, 8))
    i = rng.random((10, 5, 8))
    train = np.array([0, 3, 4, 7])
    stats = compute_normalization(s, i, train)
    assert stats["S_mean"] == pytest.approx(s[train].mean())
    assert stats["I_std"] == pytest.approx(i[train].std())
    s_other, i_other = s.copy(), i.copy()
    held_out = np.setdiff1d(np.arange(10), train)
    s_other[held_out] += 100.0
    i_other[held_out] *= 50.0
    assert compute_normalization(s_other, i_other, train) == stats
    with pytest.raises(ValueError):
        compute_normalization(np.ones((2, 3, 4)), i[:2], np.array([0, 1]))


# --------------------------------------------------------------------- pairs


def _check_admissible(pairs: np.ndarray, sims: np.ndarray) -> None:
    assert np.isin(pairs[:, 0], sims).all()
    assert (pairs[:, 1] >= 0).all()
    assert (pairs[:, 1] < pairs[:, 2]).all()
    assert (pairs[:, 2] <= N_SNAPSHOTS - 1).all()


def test_initial_source_pairs_cover_every_target() -> None:
    sims = np.array([3, 8])
    pairs = initial_source_pairs(sims, N_SNAPSHOTS)
    _check_admissible(pairs, sims)
    assert len(pairs) == 2 * 20
    for sim in sims:
        rows = pairs[pairs[:, 0] == sim]
        assert (rows[:, 1] == 0).all()
        assert sorted(rows[:, 2]) == list(range(1, 21))


def test_later_source_pairs_support_and_lag_balance() -> None:
    sims = np.arange(5)
    pairs = later_source_pairs(sims, N_SNAPSHOTS, 40_000, np.random.default_rng(0))
    _check_admissible(pairs, sims)
    assert (pairs[:, 1] >= 1).all()
    lags = pairs[:, 2] - pairs[:, 1]
    counts = np.bincount(lags, minlength=20)[1:]
    assert len(counts) == 19 and counts.min() > 0
    expected = len(pairs) / 19
    assert np.max(np.abs(counts - expected)) < 0.05 * expected
    # Every admissible later-source pair (1 <= s < j <= 20) is in the support.
    observed = {(int(s), int(j)) for s, j in pairs[:, 1:]}
    assert len(observed) == 190
    # Given a lag, the source is uniform on 1 .. 20 - lag.
    lag_five = pairs[lags == 5, 1]
    source_counts = np.bincount(lag_five)[1:]
    assert len(source_counts) == 15
    assert np.max(np.abs(source_counts - source_counts.mean())) < 0.2 * source_counts.mean()


def test_admissible_pairs_enumerate_all_210() -> None:
    pairs = admissible_pairs(np.array([0, 1]), N_SNAPSHOTS)
    _check_admissible(pairs, np.array([0, 1]))
    assert len(pairs) == 420
    assert len({tuple(row) for row in pairs}) == 420


def test_training_budget_and_deterministic_epoch_resampling() -> None:
    sims = np.arange(420)
    epoch0 = training_pairs(sims, N_SNAPSHOTS, 20, seed=3, epoch=0)
    assert len(epoch0) == 16_800
    np.testing.assert_array_equal(epoch0, training_pairs(sims, N_SNAPSHOTS, 20, seed=3, epoch=0))
    epoch1 = training_pairs(sims, N_SNAPSHOTS, 20, seed=3, epoch=1)
    initial = epoch0[:, 1] == 0
    assert initial.sum() == 420 * 20
    np.testing.assert_array_equal(epoch0[initial], epoch1[epoch1[:, 1] == 0])
    assert not np.array_equal(epoch0[~initial], epoch1[epoch1[:, 1] > 0])
    for pairs in (epoch0, epoch1):
        assert (np.bincount(pairs[:, 0]) == 40).all()


def test_validation_pairs_are_frozen_and_structured() -> None:
    sims = np.arange(90)
    first = validation_pairs(sims, N_SNAPSHOTS, 20, seed=3)
    np.testing.assert_array_equal(first, validation_pairs(sims, N_SNAPSHOTS, 20, seed=3))
    assert len(first) == 90 * 40
    assert (first[:, 1] == 0).sum() == 90 * 20
    train_like = training_pairs(sims, N_SNAPSHOTS, 20, seed=3, epoch=0)
    assert not np.array_equal(first, train_like)


# -------------------------------------------------------- generated dataset


def test_generated_manifest_and_bundle(tiny_dataset, tiny_config) -> None:
    bundle = load_dataset(tiny_dataset)
    manifest = json.loads((tiny_dataset / "manifest.json").read_text())
    assert manifest["seeds"] == tiny_config.as_dict()["seeds"]
    assert manifest["config_sha256"] == tiny_config.sha256
    assert manifest["refinement"]["passed"]
    assert bundle.susceptible.shape == (20, 6, 64)
    assert bundle.susceptible.dtype == np.float32
    np.testing.assert_allclose(bundle.times, tiny_config.domain.save_times)
    np.testing.assert_allclose(bundle.x_centers, manifest["grid"]["x_centers"])
    assert bundle.parameters["susceptible_coefficients"].shape == (20, 4)
    assert len(manifest["trajectories"]) == 20
    assert [manifest["split_counts"][k] for k in ("train", "val", "test")] == [14, 3, 3]
    stats = compute_normalization(bundle.susceptible, bundle.infectious, bundle.splits["train"])
    assert stats == pytest.approx(bundle.normalization)
    assert manifest["solver_diagnostics"]["maximum_relative_balance_residual"] < 1e-10
    assert bundle.infectious.min() >= -1e-10


def test_generation_refuses_to_overwrite(tiny_dataset, tiny_config) -> None:
    from fno_fox.generate import generate_dataset

    with pytest.raises(FileExistsError):
        generate_dataset(tiny_config, tiny_dataset, verbose=False)


def test_loading_rejects_modified_held_out_data(tmp_path, tiny_dataset) -> None:
    root = tmp_path / "modified"
    shutil.copytree(tiny_dataset, root)
    with np.load(root / "dataset.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    # Training statistics stay unchanged, so normalization validation alone
    # cannot detect this alteration to a held-out target.
    arrays["infectious"][arrays["test_ids"][0], -1, 0] = 123.0
    np.savez_compressed(root / "dataset.npz", **arrays)
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_dataset(root)


def test_pair_dataset_items(tiny_dataset, tiny_config) -> None:
    bundle = load_dataset(tiny_dataset)
    encoder = FieldEncoder.from_bundle(bundle)
    train, val = build_datasets(tiny_config, bundle, encoder)
    n = bundle.n_snapshots
    assert len(train) == 14 * ((n - 1) + tiny_config.pairs.later_source_per_trajectory)
    assert len(val) == 3 * ((n - 1) + tiny_config.pairs.later_source_per_trajectory)
    assert not np.isin(train.pairs[:, 0], bundle.splits["val"]).any()
    assert np.isin(val.pairs[:, 0], bundle.splits["val"]).all()
    dataset = PairDataset(bundle, np.array([[bundle.splits["test"][0], 2, 5]]), encoder)
    item = dataset[0]
    sim = int(item["sim_id"])
    assert item["spatial"].shape == (64, 3)
    assert item["cond"].shape == (3,)
    assert item["target"].shape == (64, 2)
    assert (int(item["source_index"]), int(item["target_index"])) == (2, 5)
    s, i = encoder.decode(item["target"].numpy())
    np.testing.assert_allclose(s, bundle.susceptible[sim, 5], atol=1e-6)
    np.testing.assert_allclose(i, bundle.infectious[sim, 5], atol=1e-7)
    s_source, i_source = encoder.decode(item["spatial"][..., :2].numpy())
    np.testing.assert_allclose(s_source, bundle.susceptible[sim, 2], atol=1e-6)
    np.testing.assert_allclose(i_source, bundle.infectious[sim, 2], atol=1e-7)
    np.testing.assert_allclose(item["spatial"][:, 2].numpy(), bundle.x_centers / 10.0, rtol=1e-6)
    lead = bundle.times[5] - bundle.times[2]
    assert float(item["cond"][2]) == pytest.approx(lead / tiny_config.domain.horizon)
    d_low, d_high = tiny_config.sampling.diffusivity
    assert float(item["cond"][0]) == pytest.approx(
        (bundle.parameters["diffusivity"][sim] - d_low) / (d_high - d_low), rel=1e-6
    )
    train.set_epoch(4)
    first = train.pairs.copy()
    train.set_epoch(5)
    train.set_epoch(4)
    np.testing.assert_array_equal(first, train.pairs)
    with pytest.raises(ValueError):
        PairDataset(bundle, np.array([[0, 3, 3]]), encoder)
