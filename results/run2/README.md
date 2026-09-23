# Run 2 — vision tower survey (2026-09-23)

`Qwen/Qwen3-VL-2B-Instruct` · Colab L4 · transformers 5.17.0 · torch 2.11.0+cu130 · bfloat16

[`qwen3vl_vision_tower_walkthrough.ipynb`](qwen3vl_vision_tower_walkthrough.ipynb) is the
executed notebook with every output preserved, rewritten to be readable on its own — no
prior knowledge of this project assumed, every cell explained before it runs.

## Setup

5 images (3 of the intended 8 returned HTTP errors and were skipped). Peak activation per
neuron recorded at the input to `linear_fc2` — i.e. post-nonlinearity — across every patch
of every image. Contribution = peak activation x L2 norm of that neuron's `linear_fc2`
output column.

## Validity check

```
min value at linear_fc2 input : -0.1699
GELU floor                    ~ -0.1700
POST-activation (hook correct)
```

GELU's output is bounded below at about -0.170; pre-activation values are unbounded. The
recorded minimum sits 1e-4 above the floor, so the hook was on the intended point.

## Results

**No dominant layer.** Median contribution rises ~0.5 (early) to ~6.0 (L23), about 12x
over 24 layers, steepest across L21-L23. No layer whose output dwarfs the rest by orders
of magnitude, so credit-apportioning methods should give distributed answers in this
tower. DeepStack taps (L5, L11, L17) all sit *before* the steep rise.

**~3% of neurons quiet, bimodally distributed.** 2,935 of 98,304 (2.99%) contributed under
1% of their layer median:

| layers | quiet neurons (of 4096 each) |
|---|---|
| L0 | 15 |
| L1-L5 | 198, 713, 542, 341, 245 |
| L6 | 32 |
| **L7-L13** | **0 across all seven** |
| L14-L17 | 1, 6, 14, 18 |
| L18-L21 | 107, 228, 264, 214 |
| L22, L23 | 30, 0 |

Quiet at both ends, entirely dense through the middle.

## Standing prediction

With several hundred diverse images, the quiet clusters at L1-L5 and L18-L21 should
**shrink substantially** while the dense band at L7-L13 stays dense — because at 5 images a
narrowly tuned neuron is indistinguishable from an unused one. If those clusters do **not**
shrink, the neurons are genuinely unused.

## Limits

1. **5 images.** Demonstrates the method; supports no published number.
2. **`0.0000` minimums are bfloat16 underflow, not silence.** GELU of a strongly negative
   input is ~1e-88, below bfloat16's range, so it rounds to exactly zero. Those neurons are
   suppressed, not absent.
3. **Peak |activation| merges suppression and activation.** A neuron pinned at GELU's floor
   registers the same magnitude as a mildly active one. Defensible for contribution (a
   negative write still moves the residual); wrong for "is this neuron ever on". Track
   positive and negative peaks separately next time.

## Next

1. Several hundred verified, diverse images — this turns the prediction into a result.
2. Separate positive and negative peaks.
3. Count exact zeros explicitly, labelled as underflow.
4. Only then ask what individual neurons detect, and do it against labelled data so the
   answer carries an expected baseline.
