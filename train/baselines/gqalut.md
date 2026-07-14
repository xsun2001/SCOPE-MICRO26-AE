# GQA-LUT Training Method Report

## Scope

This note only covers how the paper trains or searches the nonlinear function approximator itself, plus the model-level fine-tuning step used after replacing nonlinear operators. It does not cover the hardware results.

## What the paper actually optimizes

The paper does not train a neuron approximator of the form

$$
f(x) = \sum_i \mathrm{ReLU}(w_i x + b_i)
$$

like this repo's current `train/train.py`. Its approximator is an `N`-entry piecewise-linear (PWL) LUT:

$$
\mathrm{pwl}(x)=
\begin{cases}
k_0 x + b_0 & x < p_0 \\
k_1 x + b_1 & p_0 \le x < p_1 \\
\dots \\
k_{N-1} x + b_{N-1} & x \ge p_{N-2}
\end{cases}
$$

The key optimization variable is the breakpoint set `P = {p_i}`. Once the breakpoints are chosen, the slopes `K = {k_i}` and intercepts `B = {b_i}` are derived from them and then quantized for fixed-point storage.

The paper therefore has two different "training" stages:

1. Operator-level search of LUT parameters for each nonlinear function.
2. Downstream model fine-tuning after replacing nonlinear ops with the searched LUT approximators.

## Quantization-aware setup before operator search

The paper is built around integer-only inference, so the approximator is searched with quantization in mind rather than in pure FP32.

- Inputs are quantized as `x ~= S * q`, where `q` is the clipped integer value and `S` is the quantization scale.
- The paper uses LSQ to learn quantization scales in the model quantization stage.
- For the nonlinear-function input scale, the paper constrains `S` to a power of two:

$$
S = 2^{\lfloor \log_2 \alpha \rceil}
$$

- The rounding inside this power-of-two conversion uses STE for gradient approximation.
- This lets the runtime replace division by `S` with bit shifts.

For quantized-input operators such as GELU, EXP, and HSWISH, the paper performs PWL approximation directly on the quantized integer input `q`. The quantization-aware parameters are:

- Quantized intercept: `\tilde{b}_i = b_i / S`
- Quantized breakpoint: `\tilde{p}_i = ceil(Clip(p_i / S, Q_n, Q_p))`

The slopes are kept in their original form, while the breakpoints and intercepts are adapted to the quantized domain.

For wide-range operators such as DIV and RSQRT, the paper does not rely on the same learned-input-scale setup. Instead, it uses a manual multi-range input scaling strategy, where different power-of-two scales are assigned to different subranges of the input domain.

## Core GQA-LUT operator training method

### 1. Search space

For each target nonlinear function, the authors choose a fixed approximation interval `[R_n, R_p]`. Examples from the paper:

- GELU: `(-4, 4)`
- HSWISH: `(-4, 4)`
- EXP: `(-8, 0)`
- DIV: `(0.5, 4)`
- RSQRT: `(0.25, 4)`

For an 8-entry LUT, the breakpoint count is `N_b = 7`.

### 2. Training data

The approximator is not trained on model activations. It is trained against the target mathematical function itself.

- For each candidate breakpoint set, the paper samples the input domain uniformly from `R_n` to `R_p` with step `0.01`.
- The loss is the mean squared error between the candidate `pwl(x)` and the true nonlinear function `f(x)`.

This means the "dataset" is just a deterministic grid over the function domain. The paper reports the resulting grid sizes as:

- `0.8K` samples for GELU, HSWISH, and EXP
- `0.35K` for DIV
- `0.36K` for RSQRT

This is one of the main differences versus NN-LUT, which the paper says requires `100K` training samples.

### 3. Genetic optimization loop

GQA-LUT uses a genetic algorithm rather than SGD, Adam, or backpropagation on the approximator parameters.

Default hyperparameters from Table 1:

- Population size `N_p = 50`
- Breakpoint count `N_b = 7`
- Crossover probability `theta_c = 0.7`
- Mutation probability `theta_m = 0.2`
- Evolution rounds `T = 500`
- Fractional fixed-point bitwidth `lambda = 5`

Algorithmically, the search is:

1. Initialize a population of `N_p` breakpoint sets, each sampled randomly in `[R_n, R_p]`.
2. For each generation and each individual:
   - Construct the PWL approximation implied by that breakpoint set.
   - Evaluate MSE on the full sampled grid.
   - With probability `theta_c`, perform crossover by swapping a random segment with another individual.
   - With probability `theta_m`, perform mutation.
3. After each generation, apply tournament selection of size 3.
4. After `T` generations, keep the best breakpoint set `P*`.
5. Derive the FP32 slopes `K*` and intercepts `B*` from `P*`.
6. Quantize slopes and intercepts to fixed-point using `lambda` fractional bits.
7. Quantize breakpoints with the quantization-aware rule tied to `S`.

The fitness function is purely function-approximation MSE. No downstream task loss is used inside this genetic search.

## Rounding Mutation (RM)

The paper finds that naive post-search fixed-point conversion causes large errors when the quantization scale `S` is large. The reason is breakpoint deviation: after quantization, a breakpoint can move far enough to noticeably shift the entire approximation segment layout.

To address this, the paper replaces the usual noise-based mutation with Rounding Mutation (RM).

For each breakpoint `p` in an individual:

1. Sample a random number.
2. Choose a rounding precision `i` from a configured range `[m_a, m_b]_e`.
3. Mutate by rounding `p` to that precision:

$$
p' = \lfloor p \cdot 2^i \rceil / 2^i
$$

4. Sort the mutated breakpoints to keep them ordered.

The important idea is that the evolutionary search sees fixed-point rounding effects during training, instead of only after training. RM therefore makes the searched breakpoint sets robust to later quantization.

Function-specific RM settings from Table 1:

- `theta_r = 0.05` for GELU, HSWISH, and EXP
- `theta_r = 0` for DIV and RSQRT

The rounding-precision range `[m_a, m_b]` also depends on the function and whether the LUT has 8 or 16 entries.

## What the paper says about baseline training

The main baseline is NN-LUT.

What this paper states:

- NN-LUT is a neural-network-based method for learning LUT approximations.
- It needs much more training data: `100K` samples, versus `0.35K` to `0.8K` for GQA-LUT.
- It is harder to control breakpoint bitwidth during training.
- In this paper's comparison, the authors re-implement NN-LUT using the training procedure from the original NN-LUT paper, then directly quantize its slopes, intercepts, and breakpoints to the same precision as GQA-LUT.

What this paper does not provide:

- The NN-LUT network architecture
- Its optimizer or learning-rate schedule
- Its exact loss details beyond referencing the original NN-LUT paper

So if we want to reproduce the NN-LUT baseline faithfully, this paper alone is not enough; we would need the original NN-LUT source or paper.

RI-LUT and I-BERT are discussed as related work, but this paper does not present a comparable standalone training recipe for their approximators.

## Downstream model fine-tuning after operator search

After the operator-level approximators are obtained, the paper validates them by replacing nonlinear operators inside quantized Transformer models and then fine-tuning the full models.

The procedure is:

1. Quantize model weights and activations to INT8 using LSQ.
2. Use the dyadic integer-only pipeline as the quantized baseline.
3. Constrain the nonlinear-function input scale to power-of-two form, unlike I-BERT's operator-specific handling.
4. Replace target nonlinear ops with 8-entry PWL LUT approximators.
5. Fine-tune the end-to-end model on Cityscapes and evaluate mIoU.

This fine-tuning stage is important for validating the approximator, but it is separate from the approximator-search algorithm itself. The approximator parameters are not learned end-to-end from task loss in the GQA-LUT stage.

## Takeaways for this repo

- GQA-LUT is a search-based LUT/PWL baseline, not a ReLU-sum approximator.
- Its operator-level "training" is a genetic search over breakpoint sets on a fixed input grid, with MSE against the target function as the objective.
- Quantization-awareness is built into the search through power-of-two scaling and breakpoint/intercept quantization.
- RM is the paper's main training innovation: it injects fixed-point rounding into the mutation step so the search becomes robust to quantized deployment.
- The only gradient-based learning explicitly mentioned in this paper is the LSQ quantization stage and the later downstream model fine-tuning, not the LUT parameter search itself.

## Minimal reproduction recipe

If we want to implement a faithful GQA-LUT-style baseline in this repo, the closest interpretation of the paper is:

1. Pick a target function and approximation interval `[R_n, R_p]`.
2. Sample the function on a dense fixed grid with step `0.01`.
3. Run a genetic search over breakpoint sets.
4. Derive slopes and intercepts from the best breakpoint set.
5. Quantize the resulting parameters to fixed-point.
6. Use RM during mutation if the target function is sensitive to varying quantization scales.
7. Only after that, optionally insert the approximator into a quantized model and fine-tune the model.

This is the main conceptual gap between GQA-LUT and our current `train/train.py`: our current trainer is a direct parametric function learner, while GQA-LUT is a quantization-aware combinatorial search over piecewise-linear breakpoints.
