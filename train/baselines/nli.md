# NLI training-method report

## Bottom line

The nonlinear approximator in `NLI: Non-uniform Linear Interpolation Approximation of Nonlinear Operations for Efficient LLMs Inference` is **not trained with SGD, backpropagation, or calibration data**. The paper's approximator is a **data-free lookup-table construction procedure**:

1. choose an operator-specific FP16 domain,
2. evaluate the target function on that discrete FP16 grid,
3. run a dynamic-programming search to place macro cutpoints,
4. expand each macro interval into a hardware-friendly LUT layout,
5. deploy the resulting interpolation table directly at inference time.

So, for our purposes, NLI is better viewed as an **offline approximation-table optimization method**, not a learned neural approximator.

## What is "trained"

The paper optimizes only the **cutpoint locations** of a piecewise-linear interpolator. There are no learned hidden weights, no minibatches, and no train/validation split.

- Target function: a fixed nonlinear operator `f`, such as `SiLU`, `exp`, `rsqrt`, `sigmoid`, or `tanh`.
- Candidate domain: sorted FP16 grid points `X = {x_0, ..., x_{N-1}}` within the legal domain of `f`.
- Budget: `M` cutpoints, producing `M - 1` macro intervals.
- Objective: minimize the average relative interpolation error over the full discrete FP16 grid, including endpoint clamping penalties.

The paper repeatedly describes this as **calibration-free**. The LUT depends on the function and numeric domain, not on collected activation samples from a model.

## Optimization objective

For a segment from `x_i` to `x_k`, NLI uses the straight line through the endpoints `(x_i, f(x_i))` and `(x_k, f(x_k))`. The per-segment objective is:

```text
Err(i -> k) =
  (1 / (k - i + 1)) * sum_{j=i..k}
    |f(x_j) - P_{i,k}(x_j)| / max(|f(x_j)|, tau)
```

where:

- `P_{i,k}(x)` is the endpoint-anchored linear interpolant,
- `tau = 2^-14`,
- `tau` is chosen as the smallest positive normal FP16 value so near-zero outputs do not explode the relative error.

The DP also includes clamping penalties:

- left boundary: if the first cutpoint is at `x_k`, values left of it are approximated by the constant `f(x_k)`,
- right boundary: if the last cutpoint is at `x_k`, values right of it are clamped to `f(x_k)`.

This means the "training loss" is really a deterministic global approximation loss over the whole discrete FP16 domain.

## Actual offline procedure

### 1. Build the discrete target set

For each operator, enumerate the finite FP16 values in the chosen legal input domain and precompute:

```text
y_k = f(x_k)
```

This replaces dataset sampling entirely. The optimization target is the function itself.

### 2. Run dynamic programming over cutpoint positions

The paper defines DP tables:

- `D[L, k]`: minimum prefix error when `x_k` is the `L`-th cutpoint,
- `P[L, k]`: predecessor index that achieves that optimum.

Boundary condition:

```text
D[0, k] =
  (1 / (k + 1)) * sum_{j=0..k}
    |f(x_j) - f(x_k)| / max(|f(x_j)|, tau)
```

Transition:

```text
D[L, k] =
  min over i in {L-1, ..., k-1} of
    D[L-1, i] + Err(i -> k) + last_error(L, k)
```

The best final solution is recovered by backtracking from:

```text
argmin_k D[M - 1, k]
```

This is the core optimization step. It is global and deterministic once the domain, error metric, and cutpoint budget are fixed.

### 3. Use the hardware-constrained layout

Although a direct DP over all 259 final cutpoints is possible, the paper does not use that as the main recipe because it is too slow and hardware-unfriendly.

Their main configuration is:

- `10` macro intervals,
- `11` optimized macro cutpoints,
- first and last macro intervals are not subdivided,
- each of the middle `8` macro intervals is uniformly split into `32` bins.

This yields:

```text
2 + 8 * 32 + 1 = 259 total LUT cutpoints
```

So the only optimized part is the **11 macro endpoints**. The 248 interior micro-points are inserted uniformly after the DP search.

### 4. Precompute deployment parameters

After macro endpoints are fixed, the paper precomputes:

- LUT values at all 259 cutpoints,
- one scale factor per macro interval for address translation,
- base indices for each interval.

These are loaded into hardware or used by software kernels for linear interpolation at inference time.

## Computational characteristics

- Complexity of the straightforward search: `O(M * N^2)`.
- Reported practical setting: `M <= 11`, `N <= 63,488`.
- Reported runtime: under ten minutes on one RTX 4090 with Triton.

The paper also reports that searching all 259 non-uniform cutpoints directly is about 28x slower, with essentially no accuracy gain, so their preferred recipe keeps the DP search at the macro level only.

## What the paper does not do

The following standard training ingredients are absent from NLI:

- no gradient descent,
- no optimizer,
- no learning rate schedule,
- no batch size,
- no epoch count,
- no labeled data,
- no train/val/test split for fitting the approximator,
- no calibration pass over activation traces,
- no per-layer or per-model finetuning after replacement.

This is important if we compare NLI against our own neuron-based approximators: NLI is a **non-learned baseline**.

## Relation to activation ranges

The paper does measure activation coverage for LLM operators, but this is used to justify the approximation domain, not to train the approximator.

For example, for SiLU they report that `[-150, 150]` covers at least 99.9% of observed activations under their measurement protocol. Values outside the chosen domain are clamped at runtime. This still does not turn NLI into a calibration-based method, because the cutpoint search objective itself is defined on the target function over the FP16 grid rather than on sampled model activations.

## What the paper says about NN-LUT training

The NLI paper briefly explains the older NN-LUT baseline because it is the main contrast case:

- NN-LUT models the LUT parameters `(k, b, d)` using a `Linear -> ReLU -> Linear` network.
- The network's piecewise-linear form lets its learned weights be converted into LUT segments.
- According to the NLI authors' re-implementation, NN-LUT is sensitive to the training span:
  - training on `[-10, 10]` causes severe extrapolation error outside that range,
  - expanding to `[-150, 150]` makes optimization unstable and hurts fit quality in high-curvature regions.

However, this NLI paper does **not** provide enough NN-LUT hyperparameter detail to reproduce that baseline's optimizer settings exactly.

## Reproduction takeaway for this repo

If we add NLI as a baseline in this repository, the correct reproduction flow is:

1. define the operator and target input domain,
2. enumerate the FP16 grid in that domain,
3. compute exact function values,
4. run the DP cutpoint search for `M = 11` macro endpoints,
5. uniformly subdivide the middle 8 intervals into 32 bins each,
6. export the 259-point LUT and scale factors,
7. evaluate approximation error and downstream model accuracy.

That should be treated as an **offline table-construction baseline**, not as a neural training pipeline.
