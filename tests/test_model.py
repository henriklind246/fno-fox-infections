from __future__ import annotations

import pytest
import torch

from fno_fox.data import FieldEncoder
from fno_fox.model import FNOConfig, build_model
from fno_fox.train import load_checkpoint, save_checkpoint


def _model(**overrides):
    config = FNOConfig(**{"modes": 16, "width": 32, "n_layers": 4, "cond_hidden": 64, **overrides})
    torch.manual_seed(0)
    return build_model(config), config


def _encoder() -> FieldEncoder:
    return FieldEncoder(0.98, 0.07, 0.014, 0.037, 10.0, 5.0, (0.02, 0.08), (0.6, 1.2))


def test_two_field_tensor_contract() -> None:
    model, _ = _model()
    for cells in (256, 64):
        output = model(torch.randn(5, cells, 3), torch.rand(5, 3))
        assert output.shape == (5, cells, 2)
        assert output.dtype == torch.float32


def test_padding_is_applied_then_removed() -> None:
    model, _ = _model()
    seen = []
    handle = model.spectral_layers[0].register_forward_hook(
        lambda module, inputs, output: seen.append(inputs[0].shape[-1])
    )
    output = model(torch.randn(2, 256, 3), torch.rand(2, 3))
    handle.remove()
    assert model.padding == 8
    assert seen == [256 + model.padding]
    assert output.shape[1] == 256


def test_gradients_flow_through_conditional_instance_norm() -> None:
    model, _ = _model()
    spatial = torch.randn(4, 256, 3)
    cond = torch.rand(4, 3)
    # Identity-initialized CIN: the conditioning has no effect until trained.
    with torch.no_grad():
        changed = cond.clone()
        changed[:, 2] += 0.5
        assert torch.allclose(model(spatial, cond), model(spatial, changed))
    model(spatial, cond).pow(2).mean().backward()
    assert model.cond_mlp.head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.cond_mlp.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer.step()
    with torch.no_grad():
        assert not torch.allclose(model(spatial, cond), model(spatial, changed))
    # After training starts, gradients reach the conditioning input itself.
    cond.requires_grad_(True)
    model.zero_grad()
    model(spatial, cond).pow(2).mean().backward()
    assert cond.grad.abs().sum() > 0


def test_checkpoint_round_trip_is_exact(tmp_path, tiny_config) -> None:
    model, config = _model(modes=8, width=8, n_layers=2, cond_hidden=16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.01 * torch.randn_like(parameter))
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config, _encoder(), tiny_config, epoch=3, metrics={"val_loss": 0.5})
    loaded, payload = load_checkpoint(path)
    spatial, cond = torch.randn(3, 64, 3), torch.rand(3, 3)
    model.eval()
    assert torch.equal(model(spatial, cond), loaded(spatial, cond))
    assert payload["epoch"] == 3
    assert FieldEncoder.from_dict(payload["encoder"]) == _encoder()
    assert payload["config"] == tiny_config.as_dict()


def test_dropout_disabled_by_default(default_config) -> None:
    config = FNOConfig.from_model_config(default_config.model)
    assert config.dropout == 0.0
    model = build_model(config)
    model.train()
    spatial, cond = torch.randn(2, 256, 3), torch.rand(2, 3)
    assert torch.equal(model(spatial, cond), model(spatial, cond))


def test_default_model_size(default_config) -> None:
    config = FNOConfig.from_model_config(default_config.model)
    assert (config.modes, config.width, config.n_layers, config.cond_hidden) == (16, 32, 4, 64)
    assert (config.in_channels, config.out_channels, config.cond_dim) == (3, 2, 3)
    with pytest.raises(RuntimeError):
        build_model(config)(torch.randn(1, 256, 2), torch.rand(1, 3))
