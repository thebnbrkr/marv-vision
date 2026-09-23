# marv-vision

**MARV's edit-and-measure approach, pointed at vision-language models.**

[MARV](https://github.com/thebnbrkr/marv) turns a transformer's gated FFN weights
into an inspectable numpy structure, edits the live model without fine-tuning,
and **measures what the edit broke**. [marv-hyena](https://github.com/thebnbrkr/marv-hyena)
ported that to Evo 2 (StripedHyena, DNA). This repo points it at
**`Qwen/Qwen3-VL-2B-Instruct`** — the backbone of **nuVLA**, the planning
baseline for the [nuReasoning](https://arxiv.org/abs/2605.31572) driving benchmark.

The question: **which parts of a driving foundation model changed when it was
fine-tuned, and what broke when we edited them.**

> **Status: scaffold.** Nothing here has run against the real checkpoint. Every
> attribute name in `marv_vision/arch.py` is a hypothesis read off `config.json`.
> `scripts/inspect_model.py` is the bridge to the real model — run it first.

## Why this is a separate repo

marv-hyena is separate from marv because Hyena convolutions are not a gated FFN.
Vision is a bigger break still:

- the vision tower's FFN is **not gated** (`fc1 -> GELU -> fc2`)
- the vision tower has **no unembedding**, so MARV's entire `describe_*` /
  logit-lens path has no analogue there
- "position" is a **patch in a 2D grid**, not an index in a line
- there are **two towers plus a projector**, not one stack

Per MARV's own rule, that means new `ArchAdapter` subclasses — not `if` branches
inside the analysis code.

## The model

`Qwen3-VL-2B-Instruct`, from `config.json`:

| | vision tower | text tower |
|---|---|---|
| layers | 24 | 28 |
| hidden | 1024 | 2048 |
| intermediate | 4096 | 6144 |
| activation | `gelu_pytorch_tanh` | `silu` |
| FFN form | **`fc1 -> GELU -> fc2`** (not gated) | `gate/up/down_proj` (gated) |
| **FFN features** | **98,304** | **172,032** |
| unembedding | none | `lm_head`, tied to `embed` |
| | `deepstack_visual_indexes: [5, 11, 17]` | vocab 151,936 |

**270,336 features** — enumerable, which is MARV's whole design bet. At 2B this
sits inside MARV's stated envelope (135M–3B), so the exhaustive-experiment
argument survives the port intact.

**DeepStack** taps vision layers 5, 11 and 17 of 24 and feeds them into the LLM's
early hidden states — the architecture declaring its own bands. That is also
independent support for the [Perception Encoder](https://arxiv.org/abs/2504.13181)
finding that the best visual embeddings are *not* at the output of the network:
Qwen's engineers hit the same wall and hard-wired a workaround.

## What ports, and what doesn't

| MARV module | here | why |
|---|---|---|
| `diff.py` | **works as-is** | cosine on weight matrices, architecture-agnostic |
| `diagnostics.py` | **works** | health / load_bearing / find_bottlenecks / null_model are generic |
| `trace.py` | works on the **text** tower | needs a logit to decompose |
| `edit.py` | works | hooks before the second FFN matrix, either tower |
| `evaluate.py` | partly | multiple-choice VQA is first-token-scorable; coordinates and trajectories are not |
| `extract.py` | text tower only | `VindexLite` requires `lm_head` |
| `probe.py` (`describe_*`) | **no vision analogue** | no vocabulary, no unembedding |
| `arch.py` | new adapters | non-gated vision FFN |

Genuinely missing, and **not to be implied otherwise**: attention-head
decomposition (MARV is FFN-only and treats attention as one write per layer),
and trajectory attribution for a diffusion action head.

## Order of work

1. **`python scripts/inspect_model.py`** — confirm the real attribute names.
2. Confirm MARV's existing `LlamaStyleFFN` picks up the **text** tower unchanged.
   That validates half the port with zero new code.
3. Write `VisionTowerFFN`; let the vindex hold `lm_head=None`.
4. **`dead_features` on the vision tower.** First real result: what fraction of
   Qwen3-VL's vision tower never fires? Cheap, T4-sized, needs no competition
   data, and is a precondition for any feature-labeling work — marv-hyena found
   18% of Evo 2's first layer dead, and one round's headline finding had been
   built on a dead channel.

## Hardware

```bash
pip install -r requirements.txt          # needs transformers >= 4.57
python scripts/inspect_model.py --no-load   # shapes only, no download
python scripts/inspect_model.py             # fp16 on cuda
```

- **T4 has no bf16 and no fp8** (compute 7.5). Load `float16` (~4 GB) or
  `float32` (~8 GB); both fit in 16 GB. `dtype="auto"` on a T4 is a trap.
- **A100 has bf16 but no fp8** (fp8 is Ada/Hopper only).
- **Weight-space work needs no GPU.** Extraction and diff read safetensors and
  compute cosines — run them on CPU.
- **Won't fit on a T4:** 20-second multi-camera clips, and any training.
- **Don't quantize the model you're measuring.** Every self-check in this family
  of tools is a precision comparison. Quantization is a subject of study
  (`diff(f16, int4)`), never a tool for it.

## Scope

This is an **instrument, not a competition entry**. The nuReasoning challenge is
75% planning, which needs training runs this repo does not do. What it can offer
that benchmark is an answer to a question the nuReasoning paper raises and leaves
open: *reasoning supervision improves planning even with reasoning outputs
disabled at inference (NPS 64.98 → 73.09) — what moved inside the model?*

Also out of scope, same as MARV: SAE / dictionary learning. Raw neurons only, by
design — polysemanticity is the thing being measured, not an obstacle to it.
