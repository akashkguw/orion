import torch

from orion.attention.base import AttentionConfig
from orion.model import loss_fn
from orion.models_factory import OrionDecoder, build_model

DENSE_CFG = AttentionConfig(backend="dense")


def _make_model(**overrides) -> OrionDecoder:
    """Small OrionDecoder for tests. Override any param via kwargs."""
    defaults = dict(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, mlp_mult=4, attention_cfg=DENSE_CFG
    )
    return OrionDecoder(**{**defaults, **overrides})


def test_orion_decoder_forward():
    model = _make_model()
    idx = torch.randint(0, 256, (2, 32))
    logits = model(idx)

    assert logits.shape == (2, 32, 256)


def test_orion_decoder_loss():
    model = _make_model()
    idx = torch.randint(0, 256, (2, 32))
    targets = torch.randint(0, 256, (2, 32))
    loss = loss_fn(model(idx), targets)

    assert loss.item() > 0
    assert loss.requires_grad


def test_orion_decoder_learns():
    torch.manual_seed(42)
    model = _make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed batch so every step trains on the same data — makes loss decrease reliable
    idx = torch.randint(0, 256, (4, 32))
    targets = torch.randint(0, 256, (4, 32))

    model.train()
    with torch.no_grad():
        initial_loss = loss_fn(model(idx), targets).item()

    for _ in range(10):
        loss = loss_fn(model(idx), targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss didn't decrease: {initial_loss:.3f} → {final_loss:.3f}"


def test_build_model_orion_dense():
    model = build_model(
        name="orion",
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        mlp_mult=4,
        device=torch.device("cpu"),
        attention_cfg=DENSE_CFG,
    )

    assert isinstance(model, OrionDecoder)
    logits = model(torch.randint(0, 256, (2, 32)))
    assert logits.shape == (2, 32, 256)
