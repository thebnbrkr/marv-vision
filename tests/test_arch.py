"""Tests against a REAL tiny Qwen3-VL, on CPU, in seconds.

`trl-internal-testing/tiny-Qwen3VLForConditionalGeneration` is an actual
`Qwen3VLForConditionalGeneration` with 3.4M parameters — same model class, same
module names, same child ordering as the 2B checkpoint, ~14 MB to download. That
makes it strictly better than a hand-written fake: it cannot drift from the real
architecture, because it IS the real architecture.

Every bug that has cost a Colab run so far is a test here.

    pip install "transformers>=4.57,<5"     # 5.x needs torch >= 2.5
    python -m pytest tests/ -q
"""
from __future__ import annotations

import pytest
import torch

from marv_vision.arch import (
    TextTowerFFN,
    VisionTowerFFN,
    describe_model,
    tower_specs,
    towers,
)

TINY = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"

# GELU's global minimum is ~-0.170 (at x ~ -0.752). Post-activation values cannot
# go below it; pre-activation values are unbounded. This is the discriminator.
GELU_FLOOR = -0.17


@pytest.fixture(scope="module")
def model():
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        pytest.skip("transformers too old for Qwen3-VL (need >= 4.57)")
    return transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        TINY, dtype=torch.float32
    ).eval()


# --- structure ----------------------------------------------------------------


def test_both_towers_resolve(model):
    t = towers(model)
    assert t["text"].path == "model.language_model.layers"
    assert t["vision"].path == "model.visual.blocks"
    assert len(t["text"]) > 0 and len(t["vision"]) > 0


def test_text_tower_is_gated(model):
    t = TextTowerFFN(model)
    assert t.gated
    mlp = t.mlp(0)
    for a in ("gate_proj", "up_proj", "down_proj"):
        assert hasattr(mlp, a)
    # gate rows are features, down columns are features
    assert t.first(0).weight.shape[0] == t.second(0).weight.shape[1]


def test_vision_tower_is_not_gated(model):
    v = VisionTowerFFN(model)
    assert not v.gated
    assert v.attrs == ("linear_fc1", "linear_fc2")
    assert not hasattr(v.mlp(0), "gate_proj")
    assert v.first(0).weight.shape[0] == v.second(0).weight.shape[1]


# --- the run-2 regression -----------------------------------------------------


def test_vision_second_is_the_matrix_not_the_activation(model):
    """Run 2 took `named_children()[-1]`, which is act_fn, and so measured
    pre-activation values and reported a confident, wrong 0.0% dead."""
    v = VisionTowerFFN(model)
    kids = [n for n, _ in v.mlp(0).named_children()]

    assert kids[-1] == "act_fn", "the trap must still exist, else this test is vacuous"
    assert v.second(0) is v.mlp(0).linear_fc2
    assert v.second(0) is not v.mlp(0).act_fn
    assert isinstance(v.second(0), torch.nn.Linear)


def test_describe_model_warns_about_the_trap(model, capsys):
    describe_model(model)
    out = capsys.readouterr().out
    assert "Never index children by position" in out


# --- the precondition the notebook asserts before reporting -------------------


def _capture(module, x):
    seen = {}
    h = module.register_forward_pre_hook(lambda m, a: seen.setdefault("v", a[0].detach()))
    try:
        return seen, x
    finally:
        h.remove()


def test_hooking_second_gives_post_activation_values(model):
    """Hooking linear_fc2 sees GELU output, which is bounded below by ~-0.17."""
    v = VisionTowerFFN(model)
    mlp = v.mlp(0)
    hid = v.first(0).weight.shape[1]
    x = torch.randn(1, 32, hid) * 5.0  # wide range, to push the activation around

    seen = []
    h = v.second(0).register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
    try:
        mlp(x)
    finally:
        h.remove()

    assert seen, "hook never fired"
    assert seen[0].min().item() >= GELU_FLOOR - 1e-3, (
        f"min {seen[0].min().item():.4f} is below GELU's floor -- these are "
        "PRE-activation values, the hook is on the wrong module"
    )


def test_hooking_act_fn_gives_pre_activation_values(model):
    """The negative control: the wrong hook must actually look wrong, otherwise
    the check above proves nothing."""
    v = VisionTowerFFN(model)
    mlp = v.mlp(0)
    hid = v.first(0).weight.shape[1]
    x = torch.randn(1, 32, hid) * 5.0

    seen = []
    h = mlp.act_fn.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
    try:
        mlp(x)
    finally:
        h.remove()

    assert seen[0].min().item() < GELU_FLOOR, (
        "pre-activations should range well below GELU's floor; if they do not, "
        "the floor check cannot tell the two hooks apart"
    )


def test_hooks_are_removed(model):
    v = VisionTowerFFN(model)
    before = len(v.second(0)._forward_pre_hooks)
    try:
        h = v.second(0).register_forward_pre_hook(lambda m, a: None)
        raise RuntimeError("boom")
    except RuntimeError:
        h.remove()
    assert len(v.second(0)._forward_pre_hooks) == before


# --- contribution -------------------------------------------------------------


def test_write_norms_shape_and_positivity(model):
    v = VisionTowerFFN(model)
    w = v.write_norms(0)
    assert w.shape == (v.features(0),)
    assert (w >= 0).all()


def test_write_norms_match_manual_column_norms(model):
    v = VisionTowerFFN(model)
    W = v.second(0).weight.detach().float()  # (hidden, intermediate)
    assert torch.allclose(v.write_norms(0), W.norm(dim=0), atol=1e-6)


# --- config math --------------------------------------------------------------


def test_tower_specs_feature_counts(model):
    s = tower_specs(model.config)
    assert s["text"].gated and not s["vision"].gated
    assert "silu" in s["text"].activation
    assert "gelu" in s["vision"].activation
    for spec in s.values():
        assert spec.num_features == spec.num_layers * spec.intermediate_size


def test_tower_specs_agree_with_live_model(model):
    s = tower_specs(model.config)
    t = towers(model)
    assert s["text"].num_layers == len(t["text"])
    assert s["vision"].num_layers == len(t["vision"])
    assert s["vision"].intermediate_size == t["vision"].features(0)
    assert s["text"].intermediate_size == t["text"].features(0)
