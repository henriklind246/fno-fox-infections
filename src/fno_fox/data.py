"""Trajectory storage, simulation splits, normalization, and admissible pairs.

Adapted from no-tps-ihcp ``data/dataset.py`` (``SnapshotPairDataset``,
``_build_all_pairs``, ``split_sim_ids``: within-trajectory forward pairing and
simulation-level splitting) and fno-tps-sizing ``src/fno_tps/data.py``
(dataset bundle, training-only normalization, ``set_epoch`` resampling, and
ordinary loaders). Benchmark adapters, forcing encoders, curricula, and
distributed sampling are removed.

A pair ``(sim, s, j)`` always uses snapshots of one trajectory with
``0 <= s < j <= n_snapshots - 1``. Pairs are stored as indices; fields are
looked up at access time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import json
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fno_fox.config import FoxConfig, config_from_dict


SCHEMA_VERSION = 1
DATASET_FILE = "dataset.npz"
MANIFEST_FILE = "manifest.json"
PARAMETER_KEYS = (
    "diffusivity",
    "transmission",
    "infection_amplitude",
    "infection_center",
    "infection_width",
    "susceptible_constant",
    "susceptible_amplitude",
    "susceptible_coefficients",
)
# Seed-sequence stream tags keep training and validation pair draws disjoint.
TRAIN_PAIR_STREAM = 0
VALIDATION_PAIR_STREAM = 1


# ---------------------------------------------------------------- provenance


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_git_state(directory: str | Path | None = None) -> dict[str, str]:
    """Repository revision and working-tree state when available."""
    cwd = Path(__file__).resolve().parent if directory is None else Path(directory)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, capture_output=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"], cwd=cwd, text=True, capture_output=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        commit, status = "unavailable", "unavailable\n"
    return {"git_commit": commit, "git_status": status}


# -------------------------------------------------------------------- bundle


@dataclass
class DatasetBundle:
    root: Path
    x_centers: np.ndarray
    times: np.ndarray
    susceptible: np.ndarray
    infectious: np.ndarray
    parameters: dict[str, np.ndarray]
    splits: dict[str, np.ndarray]
    normalization: dict[str, float]
    manifest: dict[str, Any]

    @property
    def n_trajectories(self) -> int:
        return int(self.susceptible.shape[0])

    @property
    def n_snapshots(self) -> int:
        return int(len(self.times))

    @property
    def config(self) -> FoxConfig:
        """The configuration the data were generated with."""
        return config_from_dict(self.manifest["config"])


def split_simulation_ids(
    n_trajectories: int,
    fractions: tuple[float, float, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Shuffle complete trajectories, then cut into train/val/test."""
    ids = np.arange(n_trajectories, dtype=np.int64)
    np.random.default_rng(seed).shuffle(ids)
    n_train = int(round(n_trajectories * fractions[0]))
    n_val = int(round(n_trajectories * fractions[1]))
    splits = {
        "train": np.sort(ids[:n_train]),
        "val": np.sort(ids[n_train : n_train + n_val]),
        "test": np.sort(ids[n_train + n_val :]),
    }
    if any(len(values) == 0 for values in splits.values()):
        raise ValueError(f"Every split needs at least one trajectory; got {n_trajectories}.")
    return splits


def compute_normalization(
    susceptible: np.ndarray,
    infectious: np.ndarray,
    train_ids: np.ndarray,
) -> dict[str, float]:
    """Per-channel mean/std over every training snapshot and cell.

    Each training snapshot counts once, independent of how often it appears
    in source-target pairs.
    """
    if len(train_ids) == 0:
        raise ValueError("Normalization requires at least one training trajectory.")
    stats: dict[str, float] = {}
    for name, field in (("S", susceptible), ("I", infectious)):
        values = np.asarray(field[np.asarray(train_ids)], dtype=np.float64)
        std = float(values.std())
        if not np.isfinite(std) or std <= 0.0:
            raise ValueError(f"Training {name} snapshots have zero or invalid spread.")
        stats[f"{name}_mean"] = float(values.mean())
        stats[f"{name}_std"] = std
    return stats


def load_dataset(path: str | Path) -> DatasetBundle:
    root = Path(path)
    manifest = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    if int(manifest["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema {manifest['schema_version']}; expected {SCHEMA_VERSION}."
        )
    dataset_path = root / DATASET_FILE
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError(f"Dataset checksum mismatch for {dataset_path}; the file differs from its manifest.")
    with np.load(dataset_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    splits = {name: arrays[f"{name}_ids"].astype(np.int64) for name in ("train", "val", "test")}
    combined = np.concatenate(list(splits.values()))
    if len(np.unique(combined)) != len(combined):
        raise ValueError("Dataset splits overlap.")
    normalization = {key: float(value) for key, value in manifest["normalization"].items()}
    recomputed = compute_normalization(arrays["susceptible"], arrays["infectious"], splits["train"])
    for key, value in recomputed.items():
        if not np.isclose(value, normalization[key], rtol=1e-10, atol=0.0):
            raise ValueError(
                f"Manifest normalization {key}={normalization[key]} does not match "
                f"the training split ({value})."
            )
    return DatasetBundle(
        root=root,
        x_centers=arrays["x_centers"],
        times=arrays["times"],
        susceptible=arrays["susceptible"],
        infectious=arrays["infectious"],
        parameters={key: arrays[key] for key in PARAMETER_KEYS},
        splits=splits,
        normalization=normalization,
        manifest=manifest,
    )


# --------------------------------------------------------------------- pairs


def initial_source_pairs(sim_ids: np.ndarray, n_snapshots: int) -> np.ndarray:
    """All ``(sim, 0, j)`` for ``j = 1 .. n_snapshots - 1``."""
    targets = np.arange(1, n_snapshots, dtype=np.int64)
    sims = np.repeat(np.asarray(sim_ids, dtype=np.int64), len(targets))
    return np.column_stack(
        (sims, np.zeros_like(sims), np.tile(targets, len(sim_ids)))
    )


def later_source_pairs(
    sim_ids: np.ndarray,
    n_snapshots: int,
    per_trajectory: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pairs with ``s >= 1``, sampled with replacement and balanced in lag.

    The lag is uniform on ``1 .. n_snapshots - 2``; given the lag, the source is
    uniform on ``1 .. n_snapshots - 1 - lag``. Every admissible later-source
    pair has positive probability.
    """
    last = n_snapshots - 1
    if last < 2:
        raise ValueError("Later-source pairs need at least three snapshots.")
    sims = np.repeat(np.asarray(sim_ids, dtype=np.int64), per_trajectory)
    lags = rng.integers(1, last, size=len(sims))
    sources = rng.integers(1, last - lags + 1)
    return np.column_stack((sims, sources, sources + lags)).astype(np.int64)


def admissible_pairs(sim_ids: np.ndarray, n_snapshots: int) -> np.ndarray:
    """Every ``(sim, s, j)`` with ``0 <= s < j < n_snapshots``."""
    sources, targets = np.triu_indices(n_snapshots, k=1)
    per_sim = np.column_stack((sources, targets)).astype(np.int64)
    sims = np.repeat(np.asarray(sim_ids, dtype=np.int64), len(per_sim))
    return np.column_stack((sims, np.tile(per_sim, (len(sim_ids), 1))))


def training_pairs(
    sim_ids: np.ndarray,
    n_snapshots: int,
    per_trajectory: int,
    seed: int,
    epoch: int,
) -> np.ndarray:
    rng = np.random.default_rng([seed, TRAIN_PAIR_STREAM, epoch])
    return np.concatenate(
        (
            initial_source_pairs(sim_ids, n_snapshots),
            later_source_pairs(sim_ids, n_snapshots, per_trajectory, rng),
        )
    )


def validation_pairs(
    sim_ids: np.ndarray,
    n_snapshots: int,
    per_trajectory: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng([seed, VALIDATION_PAIR_STREAM])
    return np.concatenate(
        (
            initial_source_pairs(sim_ids, n_snapshots),
            later_source_pairs(sim_ids, n_snapshots, per_trajectory, rng),
        )
    )


# ------------------------------------------------------------------ encoding


@dataclass(frozen=True)
class FieldEncoder:
    """Global per-channel normalization and range-scaled conditioning.

    Source fields are never normalized per sample: absolute density and
    depletion remain visible to the model.
    """

    s_mean: float
    s_std: float
    i_mean: float
    i_std: float
    length: float
    horizon: float
    diffusivity_range: tuple[float, float]
    transmission_range: tuple[float, float]

    @classmethod
    def from_bundle(cls, bundle: DatasetBundle) -> "FieldEncoder":
        config = bundle.config
        stats = bundle.normalization
        return cls(
            s_mean=stats["S_mean"],
            s_std=stats["S_std"],
            i_mean=stats["I_mean"],
            i_std=stats["I_std"],
            length=config.domain.length,
            horizon=config.domain.horizon,
            diffusivity_range=config.sampling.diffusivity,
            transmission_range=config.sampling.transmission,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FieldEncoder":
        values = dict(payload)
        values["diffusivity_range"] = tuple(values["diffusivity_range"])
        values["transmission_range"] = tuple(values["transmission_range"])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "s_mean": self.s_mean,
            "s_std": self.s_std,
            "i_mean": self.i_mean,
            "i_std": self.i_std,
            "length": self.length,
            "horizon": self.horizon,
            "diffusivity_range": list(self.diffusivity_range),
            "transmission_range": list(self.transmission_range),
        }

    def spatial(self, susceptible: np.ndarray, infectious: np.ndarray, x_centers: np.ndarray) -> np.ndarray:
        """``[..., Nx, 3]``: normalized source S, normalized source I, x/L."""
        s = (np.asarray(susceptible, dtype=np.float64) - self.s_mean) / self.s_std
        i = (np.asarray(infectious, dtype=np.float64) - self.i_mean) / self.i_std
        x = np.broadcast_to(np.asarray(x_centers) / self.length, s.shape)
        return np.stack((s, i, x), axis=-1).astype(np.float32)

    def condition(self, diffusivity, transmission, lead_time) -> np.ndarray:
        """``[..., 3]``: range-scaled D, range-scaled beta, lead time / T."""
        d_low, d_high = self.diffusivity_range
        b_low, b_high = self.transmission_range
        return np.stack(
            (
                (np.asarray(diffusivity, dtype=np.float64) - d_low) / (d_high - d_low),
                (np.asarray(transmission, dtype=np.float64) - b_low) / (b_high - b_low),
                np.asarray(lead_time, dtype=np.float64) / self.horizon,
            ),
            axis=-1,
        ).astype(np.float32)

    def target(self, susceptible: np.ndarray, infectious: np.ndarray) -> np.ndarray:
        """``[..., Nx, 2]`` normalized target fields."""
        s = (np.asarray(susceptible, dtype=np.float64) - self.s_mean) / self.s_std
        i = (np.asarray(infectious, dtype=np.float64) - self.i_mean) / self.i_std
        return np.stack((s, i), axis=-1).astype(np.float32)

    def decode(self, normalized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Normalized ``[..., Nx, 2]`` predictions back to physical S and I."""
        values = np.asarray(normalized, dtype=np.float64)
        return (
            values[..., 0] * self.s_std + self.s_mean,
            values[..., 1] * self.i_std + self.i_mean,
        )


# ------------------------------------------------------------------ datasets


class PairDataset(Dataset):
    """Source-target samples addressed by ``(sim, s, j)`` index triples."""

    def __init__(self, bundle: DatasetBundle, pairs: np.ndarray, encoder: FieldEncoder):
        self.bundle = bundle
        self.encoder = encoder
        self.pairs = np.asarray(pairs, dtype=np.int64)
        if np.any(self.pairs[:, 1] >= self.pairs[:, 2]):
            raise ValueError("Pairs must move forward in time within one trajectory.")
        # Normalize once; samples are views into these arrays.
        self._spatial = encoder.spatial(bundle.susceptible, bundle.infectious, bundle.x_centers)
        self._target = self._spatial[..., :2]
        self._diffusivity = bundle.parameters["diffusivity"]
        self._transmission = bundle.parameters["transmission"]
        self._times = np.asarray(bundle.times, dtype=np.float64)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sim, source, target = (int(value) for value in self.pairs[index])
        condition = self.encoder.condition(
            self._diffusivity[sim],
            self._transmission[sim],
            self._times[target] - self._times[source],
        )
        return {
            "spatial": torch.from_numpy(np.ascontiguousarray(self._spatial[sim, source])),
            "cond": torch.from_numpy(condition),
            "target": torch.from_numpy(np.ascontiguousarray(self._target[sim, target])),
            "sim_id": torch.tensor(sim, dtype=torch.int64),
            "source_index": torch.tensor(source, dtype=torch.int64),
            "target_index": torch.tensor(target, dtype=torch.int64),
        }


class TrainingPairDataset(PairDataset):
    """Fixed per-epoch budget: all initial-source pairs plus fresh later-source pairs."""

    def __init__(
        self,
        bundle: DatasetBundle,
        sim_ids: np.ndarray,
        encoder: FieldEncoder,
        per_trajectory: int,
        seed: int,
    ):
        self.sim_ids = np.asarray(sim_ids, dtype=np.int64)
        self.n_snapshots = bundle.n_snapshots
        self.per_trajectory = int(per_trajectory)
        self.seed = int(seed)
        self.epoch = 0
        super().__init__(bundle, self._pairs_for(0), encoder)

    def _pairs_for(self, epoch: int) -> np.ndarray:
        return training_pairs(
            self.sim_ids,
            self.n_snapshots,
            self.per_trajectory,
            self.seed,
            epoch,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.pairs = self._pairs_for(self.epoch)


def build_datasets(
    config: FoxConfig,
    bundle: DatasetBundle,
    encoder: FieldEncoder,
) -> tuple[TrainingPairDataset, PairDataset]:
    per_trajectory = config.pairs.later_source_per_trajectory
    train = TrainingPairDataset(
        bundle, bundle.splits["train"], encoder, per_trajectory, config.seeds.pairs
    )
    val = PairDataset(
        bundle,
        validation_pairs(bundle.splits["val"], bundle.n_snapshots, per_trajectory, config.seeds.pairs),
        encoder,
    )
    return train, val


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int = 0,
    generator: torch.Generator | None = None,
) -> DataLoader:
    # persistent_workers stays off: ``set_epoch`` replaces the training pairs
    # in the parent process, and only re-created workers observe it.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
