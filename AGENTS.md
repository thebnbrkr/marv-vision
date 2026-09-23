# AGENTS.md

Guidance for AI coding agents (and humans) working in this repo.

## What marv-vision is

[MARV](../marv) turns a transformer's gated FFN weights into an inspectable
numpy structure (a **vindex**), edits the live model, and **measures what the
edit broke**. [marv-hyena](../marv-hyena) ported that to Evo 2 (StripedHyena,
DNA). marv-vision ports it to **vision-language models** — starting with
`Qwen/Qwen3-VL-2B-Instruct`, the backbone of the nuVLA driving baseline.

The target question, in one sentence: **which parts of a driving foundation
model changed when it was fine-tuned, and what broke when we edited them.**

Status: **scaffold.** Nothing here has been run against the real model yet.
Every attribute name in `arch.py` is a hypothesis until `scripts/inspect_model.py`
confirms it.

## The model: two towers, not one

`Qwen3-VL-2B-Instruct` (config.json, verified):

| | vision tower | text tower |
|---|---|---|
| layers | 24 | 28 |
| hidden | 1024 | 2048 |
| intermediate | 4096 | 6144 |
| activation | `gelu_pytorch_tanh` | `silu` |
| **FFN form** | **`fc1 -> GELU -> fc2` (NOT gated)** | `gate/up/down_proj` (gated) |
| **FFN features** | **98,304** | **172,032** |
| unembedding | **none** | `lm_head`, **tied to `embed`** |
| other | patch 16, spatial_merge 2, out_hidden 2048 | vocab 151,936 |
| | `deepstack_visual_indexes: [5, 11, 17]` | |

270,336 FFN features total — enumerable, which is the whole MARV bet.

**DeepStack** taps vision layers 5, 11 and 17 (of 24) and injects them into the
LLM's early hidden states. That is the architecture declaring its own bands, and
it disagrees with MARV's `default_layer_bands(24)` heuristic (syntax 0-9,
knowledge 10-19, output 20-23). Treat that disagreement as an open question, not
a bug in either.

## Invariants — do not break these

Inherited from MARV, still binding:

- **A "feature" is one raw MLP neuron at one layer.** Not an SAE latent.
  Features are polysemantic, and that overlap is where edit collateral damage
  comes from. In the vision tower a feature is one **`fc1` row / `fc2` column**.
- **`vindex.suppressed` never touches weights.** It filters retrieval only. To
  change a forward pass use hooks or a weight write.
- **Extraction copies tensors** so a vindex can never alias live weights.
- **Every exact decomposition carries its own check**, and a failing check kills
  the rows. Never remove a check to make something pass.
- **Direct != total.** Label which one a number is.
- **Run a health check with every edit.** A broken model fails every test at
  once; that looks like a finding and isn't.
- **Check for bottlenecks before trusting attribution.** Evo 2's block 30 made
  both direct attribution and integrated gradients useless. Nobody has checked
  whether a VLM has an equivalent. Run `find_bottlenecks` on **both towers**
  before believing any attribution number.
- **Hooks are always removed**, even on exceptions.

New, specific to vision:

- **Always say which tower.** Text-tower layer 11 and vision-tower layer 11 are
  unrelated, in the same way L3 f134 and L18 f134 are unrelated in MARV. Every
  feature id is `(tower, layer, feature)`.
- **There is no logit lens for the vision tower.** `describe_*` in MARV ends in
  `lm_head @ down[layer]` — "what tokens does this feature promote". The vision
  tower has no unembedding and no vocabulary. **Do not fake one.** If a vision
  readout is ever added it must be an explicit, named projection (e.g. through a
  contrastive text encoder), reported as such, never presented as the model's own
  output distribution.
- **Never label a feature by eyeballing.** Top-activating patches always return
  nine images — for a dead unit, for noise, for anything. marv-hyena round 3
  found 724 dead channels (18%) whose "top inputs" were meaningless orderings of
  near-identical values, and a previous round's headline finding had been built
  on one of them. Ground features against an **answer key** (3D boxes, HD map,
  segmentation) and report an enrichment number with a null distribution.
- **Run `dead_features` before any labeling work.** It is the precondition, not
  a nice-to-have.
- **Gated vs non-gated is the real difference; GELU vs SiLU is not.** Both are
  smooth pointwise nonlinearities and nothing in extraction, diff, suppression or
  ablation cares which. What matters is that the text tower has a separate
  `gate_proj` (a pure detector bank you can query alone) and the vision tower does
  not — `fc1` does detection and magnitude at once, so vision-tower retrieval is
  inherently blurrier. Say so when reporting it.
- **Do not quantize the thing you are measuring.** fp8/int4 changes the weights
  you are trying to read, and every self-check is a precision comparison.
  Quantization is a *subject* of study (`diff(f16, int4)`), never a tool for it.

## Hardware notes

- **T4 has no bf16** (compute 7.5) and no fp8. Load with `dtype=torch.float16`
  (~4 GB) or `float32` (~8 GB); both fit in 16 GB. This is the same trap that cost
  marv-hyena a round.
- **A100 has bf16 but no fp8** (fp8 is Hopper/Ada only).
- **Weight-space work needs no GPU at all.** Extraction and diff read
  safetensors and compute cosines. Run them on CPU.
- **What will not fit on a T4:** 20-second multi-camera clips (thousands of
  visual tokens), and any training.
- `transformers >= 4.57` is required for `Qwen3VLForConditionalGeneration`.

## Layout

```
marv_vision/
  arch.py        adapters: TextTowerFFN (gated) / VisionTowerFFN (fc1-fc2).
                 describe_model() prints the real module tree -- run it FIRST.
scripts/
  inspect_model.py   session 1: load the model, print what is actually there,
                     confirm every attribute name arch.py guesses at
tests/
notebooks/
```

## Order of work

1. `scripts/inspect_model.py` — confirm the real attribute names. Nothing else
   is meaningful until this passes.
2. Confirm MARV's existing `LlamaStyleFFN` picks up the **text** tower unchanged.
   That validates half the port with zero new code.
3. `VisionTowerFFN` + letting the vindex hold `lm_head=None`.
4. `dead_features` on the vision tower. **First real result**: what fraction of
   Qwen3-VL's vision tower never fires? Cheap, T4-sized, no competition data
   needed, and a precondition for everything after it.

## Not in scope (say so if asked)

- SAE / dictionary learning — raw neurons only, by design, same as MARV. Adopting
  SAEs means building a different tool; make that choice deliberately, not by
  drift.
- Competing in the nuReasoning challenge. This is an instrument, not a
  submission. Planning is 75% of that score and needs training runs this repo
  does not do.
- Trajectory attribution for a diffusion/flow-matching action head. Not yet
  possible here and should not be implied.
- Attention-head decomposition. **MARV has none** (it is FFN-only, and treats
  attention as one write per layer). This is the most important missing piece,
  because head-level routing is where the vision literature lives — but it does
  not exist yet and must not be promised as if it does.
