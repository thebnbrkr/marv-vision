"""Architecture adapters: module-name -> weight tensors, one class per FFN form.

A vision-language model has TWO towers with DIFFERENT FFN forms, and the whole
point of this file is to keep that difference in one place instead of letting it
leak into every analysis function (the rule MARV's AGENTS.md sets out).

    text tower    x -> down_proj( SiLU(gate_proj(x)) * up_proj(x) )      GATED
    vision tower  x -> fc2( GELU(fc1(x)) )                           NOT GATED

A feature is one MLP neuron either way:
    text    feature f = row f of gate_proj, column f of down_proj
    vision  feature f = row f of fc1,       column f of fc2

The consequence that matters: `gate_proj` is a pure detector bank, so "which
features respond to this direction" is a clean question in the text tower. In
the vision tower `fc1` does detection and magnitude at once, so the same query
is blurrier. That is a property of the architecture, not a bug to paper over.

EVERY ATTRIBUTE NAME BELOW IS UNVERIFIED. They are read off Qwen3-VL's config
and the usual HF naming, not off a loaded model. Run scripts/inspect_model.py
against the real checkpoint and fix them before trusting anything here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TowerSpec:
    """What one tower's FFN looks like, resolved against a live module."""

    name: str  # "text" | "vision"
    gated: bool
    activation: str  # "silu" | "gelu"
    num_layers: int
    hidden_size: int
    intermediate_size: int

    @property
    def num_features(self) -> int:
        """Total enumerable FFN neurons in this tower."""
        return self.num_layers * self.intermediate_size


# --- candidate attribute paths ------------------------------------------------
# Ordered by how likely they are on a current transformers build. inspect_model.py
# walks these and reports which one actually resolves; the rest are kept so the
# adapter survives a rename upstream instead of failing with an AttributeError
# three call-frames deep.

TEXT_LAYERS_PATHS = (
    "model.language_model.layers",
    "model.language_model.model.layers",
    "language_model.model.layers",
    "model.layers",
)

VISION_BLOCKS_PATHS = (
    "model.visual.blocks",
    "visual.blocks",
    "model.vision_tower.blocks",
)

# (gate, up, down) on a text MLP
TEXT_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj")

# (fc1, fc2) on a vision MLP -- Qwen has used both spellings across versions
VISION_MLP_ATTRS = (
    ("linear_fc1", "linear_fc2"),
    ("fc1", "fc2"),
    ("up_proj", "down_proj"),
)


def resolve(root, path: str):
    """Walk a dotted attribute path, returning None if any hop is missing."""
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def find_first(root, paths):
    """First (path, object) in `paths` that resolves on `root`."""
    for p in paths:
        obj = resolve(root, p)
        if obj is not None:
            return p, obj
    return None, None


def describe_model(model) -> dict:
    """Print what is ACTUALLY there and return the resolved paths.

    This is step 1 of the whole project. Until it runs against the real
    checkpoint, every name in this file is a guess.
    """
    out: dict = {}

    tpath, tlayers = find_first(model, TEXT_LAYERS_PATHS)
    vpath, vblocks = find_first(model, VISION_BLOCKS_PATHS)
    out["text_layers_path"] = tpath
    out["vision_blocks_path"] = vpath

    print("=" * 70)
    print(f"text layers   : {tpath}  ({len(tlayers) if tlayers is not None else '?'} layers)")
    print(f"vision blocks : {vpath}  ({len(vblocks) if vblocks is not None else '?'} blocks)")

    if tlayers is not None and len(tlayers):
        mlp = getattr(tlayers[0], "mlp", None)
        print(f"\ntext  mlp type: {type(mlp).__name__}")
        print(f"      attrs   : {[a for a in dir(mlp) if 'proj' in a or 'fc' in a]}")
        found = all(hasattr(mlp, a) for a in TEXT_MLP_ATTRS)
        out["text_mlp_attrs"] = TEXT_MLP_ATTRS if found else None
        print(f"      gated ({'/'.join(TEXT_MLP_ATTRS)}) present: {found}")

    if vblocks is not None and len(vblocks):
        mlp = getattr(vblocks[0], "mlp", None)
        print(f"\nvision mlp type: {type(mlp).__name__}")
        print(f"       attrs   : {[a for a in dir(mlp) if 'proj' in a or 'fc' in a]}")
        hit = next((p for p in VISION_MLP_ATTRS if all(hasattr(mlp, a) for a in p)), None)
        out["vision_mlp_attrs"] = hit
        print(f"       resolved: {hit}")
        if hit is None:
            print("       !! none of the candidate spellings matched -- add the")
            print("          real one to VISION_MLP_ATTRS in marv_vision/arch.py")

    print("=" * 70)
    return out


def tower_specs(config) -> dict[str, TowerSpec]:
    """Read both towers' shapes off a Qwen3VLConfig without loading weights."""
    t, v = config.text_config, config.vision_config
    return {
        "text": TowerSpec(
            name="text",
            gated=True,
            activation=getattr(t, "hidden_act", "silu"),
            num_layers=t.num_hidden_layers,
            hidden_size=t.hidden_size,
            intermediate_size=t.intermediate_size,
        ),
        "vision": TowerSpec(
            name="vision",
            gated=False,
            activation=getattr(v, "hidden_act", "gelu"),
            num_layers=v.depth,
            hidden_size=v.hidden_size,
            intermediate_size=v.intermediate_size,
        ),
    }
