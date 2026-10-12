"""Single-device training on admissible source-target pairs.

    python -m fno_fox.train --config conf/default.yaml \
        --data data/fox_rabies --run runs/default

Extracted from fno-tps-sizing ``src/fno_tps/training.py`` and
``src/fno_tps/runtime.py``: seeding, device resolution, per-epoch training
and validation, metrics CSV, and best/last checkpoints. Each epoch sees every
initial-source pair plus a fresh, seed-determined draw of later-source pairs.
The best checkpoint minimizes the equally weighted mean of the initial-source
and later-source validation losses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import csv
import json
import random
import time

import numpy as np
import torch

from fno_fox.config import FoxConfig, load_config
from fno_fox.data import (
    DatasetBundle,
    FieldEncoder,
    build_datasets,
    capture_git_state,
    load_dataset,
    make_loader,
)
from fno_fox.model import FNOConfig, FNO1d, build_model


CHECKPOINT_VERSION = 1
METRIC_FIELDS = [
    "epoch",
    "learning_rate",
    "train_loss",
    "train_loss_initial",
    "train_loss_later",
    "val_loss_initial",
    "val_loss_later",
    "val_loss",
    "epoch_seconds",
    "is_best",
]


def resolve_device(requested: str = "auto") -> torch.device:
    """One CUDA device when available, otherwise CPU."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_sample_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-channel normalized MSE for each sample: mean_c mean_x (y - y_hat)^2."""
    return (prediction - target).pow(2).mean(dim=1).mean(dim=1)


def _accumulate(
    totals: dict[str, float],
    counts: dict[str, int],
    losses: torch.Tensor,
    source_index: torch.Tensor,
) -> None:
    initial = source_index == 0
    for name, mask in (("initial", initial), ("later", ~initial)):
        if mask.any():
            totals[name] += float(losses[mask].sum())
            counts[name] += int(mask.sum())


def _category_means(totals: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    return {
        name: totals[name] / counts[name] if counts[name] else float("nan")
        for name in ("initial", "later")
    }


@torch.no_grad()
def validation_losses(model: FNO1d, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    totals = {"initial": 0.0, "later": 0.0}
    counts = {"initial": 0, "later": 0}
    for batch in loader:
        prediction = model(batch["spatial"].to(device), batch["cond"].to(device))
        losses = per_sample_loss(prediction, batch["target"].to(device)).cpu()
        _accumulate(totals, counts, losses, batch["source_index"])
    means = _category_means(totals, counts)
    means["combined"] = 0.5 * (means["initial"] + means["later"])
    return means


def save_checkpoint(
    path: str | Path,
    model: FNO1d,
    model_config: FNOConfig,
    encoder: FieldEncoder,
    config: FoxConfig,
    *,
    epoch: int,
    metrics: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    dataset: dict[str, Any] | None = None,
) -> None:
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_config": model_config.as_dict(),
        "model_state": model.state_dict(),
        "encoder": encoder.as_dict(),
        "config": config.as_dict(),
        "dataset": dataset or {},
        "code_provenance": capture_git_state(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, Path(path))


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[FNO1d, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(f"Unsupported checkpoint version in {path}.")
    model = build_model(FNOConfig(**payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload


def check_dataset_compatibility(config: FoxConfig, bundle: DatasetBundle) -> None:
    """Training may change model/optimizer settings, not the data definition."""
    generated = bundle.config
    for section in ("domain", "solver", "sampling", "splits"):
        if getattr(generated, section) != getattr(config, section):
            raise ValueError(
                f"Config section {section!r} differs from the one used to generate "
                f"{bundle.root}; regenerate the dataset or use its configuration."
            )


def train(
    config: FoxConfig,
    data_dir: str | Path,
    run_dir: str | Path,
    *,
    epochs: int | None = None,
    device: str | None = None,
    overwrite: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    destination = Path(run_dir)
    if (destination / "last.pt").exists() and not overwrite:
        raise FileExistsError(f"{destination} already holds a run; pass --overwrite.")
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(config.seeds.training)
    resolved = resolve_device(device or config.training.device)
    bundle = load_dataset(data_dir)
    check_dataset_compatibility(config, bundle)
    encoder = FieldEncoder.from_bundle(bundle)
    train_set, val_set = build_datasets(config, bundle, encoder)
    generator = torch.Generator().manual_seed(config.seeds.training)
    train_loader = make_loader(
        train_set,
        config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    val_loader = make_loader(
        val_set,
        config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model_config = FNOConfig.from_model_config(config.model)
    model = build_model(model_config).to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    epoch_count = config.training.epochs if epochs is None else int(epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_count)
    dataset_info = {
        "root": str(Path(data_dir).resolve()),
        "dataset_sha256": bundle.manifest["dataset_sha256"],
        "config_sha256": bundle.manifest["config_sha256"],
    }
    (destination / "config.json").write_text(
        json.dumps({"config": config.as_dict(), "epochs": epoch_count, "device": str(resolved)}, indent=2),
        encoding="utf-8",
    )
    (destination / "normalization.json").write_text(
        json.dumps(encoder.as_dict(), indent=2), encoding="utf-8"
    )

    best = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    start = time.perf_counter()
    with (destination / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for epoch in range(epoch_count):
            epoch_start = time.perf_counter()
            learning_rate = optimizer.param_groups[0]["lr"]
            train_set.set_epoch(epoch)
            model.train()
            totals = {"initial": 0.0, "later": 0.0}
            counts = {"initial": 0, "later": 0}
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch["spatial"].to(resolved), batch["cond"].to(resolved))
                losses = per_sample_loss(prediction, batch["target"].to(resolved))
                loss = losses.mean()
                loss.backward()
                optimizer.step()
                _accumulate(totals, counts, losses.detach().cpu(), batch["source_index"])
            scheduler.step()
            train_means = _category_means(totals, counts)
            validation = validation_losses(model, val_loader, resolved)
            is_best = validation["combined"] < best
            row = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "train_loss": (totals["initial"] + totals["later"])
                / max(1, counts["initial"] + counts["later"]),
                "train_loss_initial": train_means["initial"],
                "train_loss_later": train_means["later"],
                "val_loss_initial": validation["initial"],
                "val_loss_later": validation["later"],
                "val_loss": validation["combined"],
                "epoch_seconds": time.perf_counter() - epoch_start,
                "is_best": int(is_best),
            }
            writer.writerow(row)
            handle.flush()
            history.append(row)
            common = dict(
                metrics=row,
                optimizer=optimizer,
                scheduler=scheduler,
                dataset=dataset_info,
            )
            save_checkpoint(destination / "last.pt", model, model_config, encoder, config, epoch=epoch, **common)
            if is_best:
                best, best_epoch = validation["combined"], epoch
                save_checkpoint(destination / "best.pt", model, model_config, encoder, config, epoch=epoch, **common)
            if verbose:
                print(
                    f"epoch {epoch:4d}  lr {learning_rate:.2e}  train {row['train_loss']:.4e}  "
                    f"val init {validation['initial']:.4e}  later {validation['later']:.4e}"
                    f"{'  *' if is_best else ''}",
                    flush=True,
                )

    summary = {
        "device": str(resolved),
        "epochs": epoch_count,
        "train_pairs_per_epoch": len(train_set),
        "val_pairs": len(val_set),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_val_loss": best,
        "best_metrics": history[best_epoch],
        "final_metrics": history[-1],
        "wall_seconds": time.perf_counter() - start,
        "dataset": dataset_info,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = train(
        load_config(args.config),
        args.data,
        args.run,
        epochs=args.epochs,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps({k: summary[k] for k in ("best_epoch", "best_val_loss", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
