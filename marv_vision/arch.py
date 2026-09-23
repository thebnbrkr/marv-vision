"""Architecture adapters: module-name -> weight tensors, one class per FFN form.

A vision-language model has TWO towers with DIFFERENT FFN forms, and the whole
point of this file is to keep that difference in one place instead of letting it
leak into every analysis function (the rule MARV's AGENTS.md sets out).

    text tower    x -> down_proj( SiLU(gate_proj(x)) * up_proj(x) )      GATED
    vision tower  x -> linear_fc2( GELU(linear_fc1(x)) )             NOT GATED

A feature is one MLP neuron either way:
    text    feature f = row f of gate_proj,  column f of down_proj
    vision  feature f = row f of linear_fc1, column f of linear_fc2

The consequence that matters: `gate_proj` is a pure detector bank, so "which
features respond to this direction" is a clean question in the text tower. In
the vision tower `linear_fc1` does detection and magnitude at once, so the same
query is blurrier. That is a property of the architecture, not a bug to paper over.

VERIFIED against Qwen/Qwen3-VL-2B-Instruct on 2026-09-23 (transformers 5.17.0):

    text layers   : model.language_model.layers      28 x Qwen3VLTextDecoderLayer
                    mlp: gate_proj / up_proj / down_proj      NO bias
    vision blocks : model.visual.blocks              24 x Qwen3VLVisionBlock
                    mlp: linear_fc1 / linear_fc2 / act_fn     HAS bias

Note `act_fn` is a *child module* of the vision MLP, so "the last named child"
is NOT the second matrix. Name the matrices explicitly.
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
# The first entry of each tuple is the VERIFIED one; the rest are kept so a
# rename upstream degrades into a clear report from describe_model() instead of
# an AttributeError three call-frames deep.

TEXT_LAYERS_PATHS = (
    "model.language_model.layers",  # verified
    "model.language_model.model.layers",
    "language_model.model.layers",
    "model.layers",
)

VISION_BLOCKS_PATHS = (
    "model.visual.blocks",  # verified
    "visual.blocks",
    "model.vision_tower.blocks",
)

TEXT_MLP_ATTRS = ("gate_proj", "up_proj", "down_proj")  # verified

VISION_MLP_ATTRS = (
    ("linear_fc1", "linear_fc2"),  # verified
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


# --- adapters -----------------------------------------------------------------


class TowerFFN:
    """Common surface: layers, and the two matrices that bracket the activation.

    `first` is the matrix whose ROWS are features (detector side).
    `second` is the matrix whose COLUMNS are features (write side) -- the one to
    hook with a forward_pre_hook to see POST-activation values, and the one whose
    column norm says how much a feature can actually move the residual.
    """

    tower: str
    gated: bool

    def __init__(self, model):
        self.model = model
        self.path, self.layers = find_first(model, self._paths)
        if self.layers is None:
            raise AttributeError(
                f"could not find the {self.tower} tower on this model; tried {self._paths}. "
                "Run marv_vision.arch.describe_model(model) and add the real path."
            )

    def __len__(self) -> int:
        return len(self.layers)

    def mlp(self, layer: int):
        return self.layers[layer].mlp

    def first(self, layer: int):
        raise NotImplementedError

    def second(self, layer: int):
        raise NotImplementedError

    def features(self, layer: int) -> int:
        return self.first(layer).weight.shape[0]

    def write_norms(self, layer: int):
        """L2 norm of each feature's output column: how much it CAN move the
        residual, independent of whether it fired. A neuron with a big activation
        and a tiny column contributes nothing."""
        return self.second(layer).weight.detach().float().norm(dim=0)


class TextTowerFFN(TowerFFN):
    """Llama-style gated FFN. MARV's existing machinery applies unchanged."""

    tower = "text"
    gated = True
    _paths = TEXT_LAYERS_PATHS

    def gate(self, layer: int):
        return self.mlp(layer).gate_proj

    def up(self, layer: int):
        return self.mlp(layer).up_proj

    def first(self, layer: int):
        return self.gate(layer)  # the detector half

    def second(self, layer: int):
        return self.mlp(layer).down_proj


class VisionTowerFFN(TowerFFN):
    """Two-matrix GELU FFN: linear_fc1 -> act_fn -> linear_fc2.

    No gate, so there is no pure-detector matrix; `first` does detection and
    magnitude at once. Both matrices carry a bias (the text tower's do not).
    """

    tower = "vision"
    gated = False
    _paths = VISION_BLOCKS_PATHS

    def __init__(self, model):
        super().__init__(model)
        mlp = self.mlp(0)
        self.attrs = next(
            (p for p in VISION_MLP_ATTRS if all(hasattr(mlp, a) for a in p)), None
        )
        if self.attrs is None:
            raise AttributeError(
                f"vision MLP has none of {VISION_MLP_ATTRS}; it has "
                f"{[n for n, _ in mlp.named_children()]}"
            )

    def first(self, layer: int):
        return getattr(self.mlp(layer), self.attrs[0])

    def second(self, layer: int):
        # NOT named_children()[-1] -- that is act_fn on this model.
        return getattr(self.mlp(layer), self.attrs[1])


def towers(model) -> dict[str, TowerFFN]:
    """Both towers, ready to use."""
    return {"text": TextTowerFFN(model), "vision": VisionTowerFFN(model)}


# --- inspection ---------------------------------------------------------------


def describe_model(model) -> dict:
    """Print what is ACTUALLY there and return the resolved paths."""
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
        print(f"      children: {[n for n, _ in mlp.named_children()]}")
        found = all(hasattr(mlp, a) for a in TEXT_MLP_ATTRS)
        out["text_mlp_attrs"] = TEXT_MLP_ATTRS if found else None
        print(f"      gated ({'/'.join(TEXT_MLP_ATTRS)}) present: {found}")

    if vblocks is not None and len(vblocks):
        mlp = getattr(vblocks[0], "mlp", None)
        kids = [n for n, _ in mlp.named_children()]
        print(f"\nvision mlp type: {type(mlp).__name__}")
        print(f"       children: {kids}")
        hit = next((p for p in VISION_MLP_ATTRS if all(hasattr(mlp, a) for a in p)), None)
        out["vision_mlp_attrs"] = hit
        print(f"       resolved: {hit}")
        if hit and kids and kids[-1] != hit[1]:
            print(
                f"       NOTE: last child is {kids[-1]!r}, NOT the second matrix "
                f"({hit[1]!r}). Never index children by position."
            )
        if hit is None:
            print("       !! no candidate spelling matched -- add the real one to")
            print("          VISION_MLP_ATTRS in marv_vision/arch.py")

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
