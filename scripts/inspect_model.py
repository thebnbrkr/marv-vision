"""Session 1: load Qwen3-VL-2B and find out what is actually in it.

Nothing else in this repo is meaningful until this runs. Every attribute name in
marv_vision/arch.py is a hypothesis read off config.json; this script checks them
against a real loaded model and tells you which ones are wrong.

T4 note: T4 (compute 7.5) has NO bfloat16. The checkpoint is bf16, so load it as
float16 (~4 GB) or float32 (~8 GB). Both fit in 16 GB. Passing dtype="auto" on a
T4 is the same trap that cost marv-hyena a round.

    python scripts/inspect_model.py                 # fp16, cuda if available
    python scripts/inspect_model.py --dtype float32 --device cpu
    python scripts/inspect_model.py --no-load       # config only, no download
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from marv_vision.arch import describe_model, tower_specs

MODEL = "Qwen/Qwen3-VL-2B-Instruct"


class _Ns(dict):
    """dict that also answers attribute access, so a raw config.json can stand in
    for a PretrainedConfig in the shape-reading path."""

    def __getattr__(self, k):
        v = self.get(k)
        return _Ns(v) if isinstance(v, dict) else v


def _raw_config(model_id: str) -> _Ns:
    """Fetch config.json without needing a transformers version that knows the
    architecture. Shapes are all we need here."""
    import json
    import urllib.request

    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return _Ns(json.load(r))
    except Exception as exc:  # offline, SSL, 404 -- fall back to vendored shapes
        from marv_vision.known_configs import KNOWN

        if model_id not in KNOWN:
            raise SystemExit(
                f"could not fetch config for {model_id} ({exc}) and it is not in "
                "marv_vision/known_configs.KNOWN"
            )
        print(f"note: network fetch failed ({type(exc).__name__}); using vendored config")
        return _Ns(KNOWN[model_id])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--no-load", action="store_true", help="config only -- no weights")
    ap.add_argument("--tree", action="store_true", help="also print the full module tree")
    args = ap.parse_args()

    import torch

    # --- shapes, from config alone: no download of weights needed -------------
    # AutoConfig needs transformers >= 4.57 to know `qwen3_vl`. Fall back to the
    # raw config.json so this half still works on an older install -- the shapes
    # are the useful part and they do not need the model class.
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(args.model)
    except (ImportError, ValueError, KeyError) as exc:
        print(f"note: AutoConfig unavailable ({type(exc).__name__}); reading config.json directly")
        cfg = _raw_config(args.model)

    specs = tower_specs(cfg)

    print(f"\n{args.model}\n")
    total = 0
    for s in specs.values():
        total += s.num_features
        print(
            f"  {s.name:<7} {s.num_layers:>3} layers  hidden {s.hidden_size:<5} "
            f"inter {s.intermediate_size:<5} {s.activation:<18} "
            f"{'gated' if s.gated else 'NOT gated':<10} {s.num_features:>8,} features"
        )
    print(f"  {'total':<7} {'':>3}{'':<45}{total:>30,} features\n")

    ds = getattr(cfg.vision_config, "deepstack_visual_indexes", None)
    if ds:
        depth = specs["vision"].num_layers
        pct = ", ".join(f"{i} ({i / depth:.0%})" for i in ds)
        print(f"  DeepStack taps vision layers: {pct} of {depth}")
    if getattr(cfg, "tie_word_embeddings", False) or getattr(
        cfg.text_config, "tie_word_embeddings", False
    ):
        print("  tie_word_embeddings=True -> lm_head IS embed; do not store two copies\n")

    if args.no_load:
        return 0

    # --- the real check: load and walk the module tree ------------------------
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dtype == "bfloat16" and device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        if major < 8:
            print("!! this GPU has no bfloat16 (compute < 8.0). Use --dtype float16.")
            return 1

    print(f"loading on {device} as {args.dtype} ...")
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=device
    )

    paths = describe_model(model)

    if args.tree:
        print(model)

    missing = [k for k, v in paths.items() if v is None]
    if missing:
        print(f"\nUNRESOLVED: {missing}")
        print("Fix the candidate lists in marv_vision/arch.py before going further.")
        return 1

    print("\nAll paths resolved. Record them in AGENTS.md and move to step 2:")
    print("  does MARV's existing LlamaStyleFFN pick up the TEXT tower unchanged?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
