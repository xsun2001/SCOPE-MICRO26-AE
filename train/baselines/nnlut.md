# NN-LUT Training Method Report

## Scope

This note only covers how the paper trains the nonlinear function approximator in
`papers/nnlut.pdf.md`. It does not cover hardware synthesis, accelerator
integration, or downstream task results except where they clarify the training
procedure.

## What is actually trained

NN-LUT does not train the LUT directly. It first trains a scalar
one-hidden-layer ReLU network offline, then converts the trained network into a
piecewise-linear LUT.

The trained approximator is:

$$
\mathrm{NN}(x) = \sum_{i=1}^{N-1} m_i \mathrm{ReLU}(n_i x + b_i)
$$

where:

- `x` is a scalar input to one nonlinear sub-function.
- `N-1` hidden ReLU units produce `N` linear regions after conversion.
- the learned parameters are `(m_i, n_i, b_i)`.

After training, the breakpoints are derived from `-b_i / n_i`, and each
interval is rewritten as `s_i x + t_i` for LUT deployment.

For a 16-entry LUT, this implies a 15-hidden-unit ReLU network.

## Function decomposition used for training

The paper does not train a single network for whole Transformer blocks. It
trains small scalar approximators for the elementary nonlinear functions used
inside Transformer nonlinear ops:

- GELU: train directly on the GELU function.
- Softmax: train one approximator for `exp(x)` and another for division.
- LayerNorm: train one approximator for `1 / sqrt(x)`.

This is important: the training target is the scalar mathematical function, not
the end-to-end model loss.

## Training data generation

The dataset is synthetically generated from the reference function itself:

1. Choose an input range of interest for the target scalar function.
2. Uniformly sample inputs from that range.
3. Evaluate the exact target function to get supervision pairs `(x, f(x))`.

The paper states that `100K` samples were sufficient for curve fitting.

Reported input ranges:

| Target op | Trained function | Input range |
| --- | --- | --- |
| GELU | GELU | `(-5, 5)` |
| Softmax | `exp` | `(-256, 0)` |
| Softmax | division | `(1, 1024)` |
| LayerNorm | `1 / sqrt` | `(0.1, 1024)` |

## Initialization strategy

The paper treats initialization as part of the method, not an implementation
detail. It uses different sign constraints depending on the target function:

| Target op | Trained function | `n_i` init | `b_i` init |
| --- | --- | --- | --- |
| GELU | GELU | random | random |
| Softmax | `exp` | positive random | positive random |
| Softmax | division | negative random | positive random |
| LayerNorm | `1 / sqrt` | negative random | positive random |

The paper does not give the exact sampling distribution, only the sign pattern.

Likely rationale (inference): these sign constraints bias the ReLU units toward
the monotonic direction of the target function and help produce useful
breakpoints after the `-b_i / n_i` conversion. The paper specifies the rule but
does not provide an ablation isolating why it works.

## Loss and optimizer

The training objective shown in Figure 1 is direct regression from the network
output to the target function value using `L1` loss.

The paper reports one common hyperparameter setup working across all tested
nonlinear functions:

- optimizer: Adam
- learning rate: `0.001`
- schedule: multi-step learning-rate decay
- loss: `L1`

The authors explicitly say `L1` worked slightly better than the alternatives
they tried because it penalized outliers less aggressively.

## Training workflow

The training method in the paper is:

1. Pick the scalar target function and its input domain.
2. Generate `100K` uniformly sampled training pairs from the exact function.
3. Initialize the one-hidden-layer ReLU network with the operation-specific sign
   rules above.
4. Train with Adam, `lr=1e-3`, multi-step schedule, and `L1` loss.
5. Convert the trained network parameters into LUT breakpoints and per-interval
   linear coefficients.

The authors describe this offline fitting as straightforward and one-time. They
report about two minutes of training on one NVIDIA V100 GPU.

## Special training trick for LayerNorm: input scaling

The paper identifies `1 / sqrt(x)` as the hardest case because its output grows
rapidly when `x < 1`, which makes the slope near zero too steep for a small
ReLU approximator to fit well over the whole domain.

Their fix is not a different model; it is a domain transformation:

1. Train the approximator on a more stable range `[1, K]`, where `K >> 1`.
2. At inference time, if `0 < x < 1`, scale the input by a large constant `S`
   so that `Sx` falls into `[1, K]`.
3. Query the LUT with `Sx`.
4. Multiply the LUT output by `sqrt(S)` to recover the original function value.

For `f(x) = 1 / sqrt(x)`, this works because:

$$
f(x) = \frac{1}{\sqrt{x}} = \sqrt{S} \cdot \frac{1}{\sqrt{Sx}}
$$

The paper recommends choosing `S` as a power of two, such as `2^10`, so the
scaling becomes a simple shift in hardware.

## Post-training calibration

The paper adds a second stage after offline function fitting: calibration.

Direct approximation means replacing nonlinear ops with the pretrained NN-LUT as
is. If that causes noticeable accuracy loss, each deployed NN-LUT is calibrated
against its full-precision reference while freezing all original Transformer
parameters.

Key properties of this calibration stage:

- uses a small unlabeled dataset
- updates only the NN-LUT approximator parameters
- does not fine-tune the Transformer weights
- is intended to adapt to layer-specific activation ranges

Reported calibration setup:

- data amount: one-tenth of the training dataset
- epochs: `5`
- cost: less than `5%` of normal fine-tuning time

After calibration, the updated NN is converted back to LUT parameters.

One careful reading point: the paper advertises calibration as lightweight and
effectively label-free, but the concrete procedure still uses unlabeled data. So
the real claim is "no labels and no end-to-end retraining," not literally "no
data at all."

## What seems to matter most in the method

From the paper, the main training ideas are:

- fit scalar functions directly instead of relying on approximation-aware
  end-to-end fine-tuning
- choose the training range to match the actual domain of each nonlinear
  sub-function
- constrain initialization signs by function shape
- use `L1` regression with a simple optimizer setup
- add explicit input scaling for `1 / sqrt`
- use lightweight post-training calibration to absorb layer-specific mismatch

## Gaps left unspecified by the paper

Several details are not provided, so reproducing the method exactly would still
require implementation choices:

- exact hidden width used during initial fitting in each experiment, aside from
  the statement that 16 LUT entries were enough
- batch size
- number of training epochs for the initial offline fit
- the exact multi-step schedule milestones and decay factor
- the random initialization distribution beyond the sign constraints
- whether calibration reuses the same optimizer settings as initial fitting
- the exact mechanism for collecting layerwise calibration targets inside the
  Transformer

## Practical reproduction recipe

If the goal is to reproduce the paper's training method as faithfully as the
paper allows, the recipe is:

1. Train a scalar one-hidden-layer ReLU regressor per target function.
2. Use `100K` uniform samples from the paper's operation-specific domain.
3. Use sign-constrained initialization exactly as in Table 1.
4. Optimize with Adam, `lr=1e-3`, multi-step decay, and `L1` loss.
5. For `1 / sqrt`, train on `[1, K]` and use the scaling trick for inputs below
   `1`.
6. Convert the trained model to LUT coefficients and breakpoints.
7. If model-level accuracy drops, calibrate only the NN-LUT parameters for five
   epochs on a small unlabeled set while freezing the Transformer.

## Bottom line

NN-LUT's training method is a two-stage approximation pipeline:

- stage 1: offline supervised regression of scalar nonlinear functions with a
  small one-hidden-layer ReLU model
- stage 2: optional lightweight post-training calibration in the deployed model

The paper's main contribution on the training side is not a complicated
optimization algorithm. It is the combination of operation-specific sampling
range selection, sign-aware initialization, `L1` fitting, a scaling trick for
`1 / sqrt`, and cheap unlabeled calibration after insertion into the model.
