# `end2endacc` Code Report

## Scope

`end2endacc` is the software-side evaluation harness for the PINN/PINNacle approximation idea in this repo. Its job is to:

1. load a Hugging Face causal LM,
2. replace selected math-heavy submodules with PINN-based approximations,
3. optionally quantize the approximator weights and intermediate activations,
4. run downstream accuracy evaluation.

In the current checkout, the evaluation targets are:

- LLaMA-family models: custom attention and custom MLP
- OPT-family models: custom attention only
- Evaluation backends:
  - `lm-eval` benchmark tasks
  - WikiText perplexity

This directory does not contain training for the approximators. It contains inference-time integration plus evaluation wrappers.

## 1. Code Structure

### Top-level layout

```text
end2endacc/
├── PINNacle/
│   ├── approximation_wrapper.py
│   ├── utils.py
│   ├── quantization/
│   │   ├── calibration.py
│   │   ├── clip.py
│   │   └── quantizer.py
│   └── pinn/sw/
│       ├── attn/
│       │   ├── llama_attn_pinn.py
│       │   ├── llama_attn_plf.py
│       │   ├── llama_attn_taylor.py
│       │   ├── opt_attn_pinn.py
│       │   ├── d8/, d16/, d32/
│       │   └── old/
│       └── mlp/
│           ├── llama_mlp_pinn.py
│           └── d8/, d16/, d32/
├── evaluation/
│   ├── lmeval/lmeval.py
│   └── wikitext/evaluate_hf.py
└── scripts/
    ├── evaluation_lm_eval.sh
    ├── evaluation_wikitext.sh
    └── finetune.sh
```

### What each file does

#### `PINNacle/approximation_wrapper.py`

- Main model-rewrite entry point.
- Detects model family by `model.config._name_or_path`.
- For LLaMA:
  - swaps every `self_attn` block with `LlamaAttention_PINN`
  - swaps every `mlp` block with `LlamaMLP_PINN`
- For OPT:
  - swaps every decoder `self_attn` block with `OPTAttention_PINN`
- Copies overlapping weights from the original Hugging Face module into the replacement module.

This file is the integration boundary between a standard HF model and the PINN variants.

#### `PINNacle/quantization/clip.py`

- Core fake-quantization helper.
- Implements:
  - integer quantization
  - floating-point-like quantization (`fpq`)
  - optional clipping search
  - per-tensor and group-based layout handling
- Used for approximator weight quantization and as the backend for activation quantization.

#### `PINNacle/quantization/quantizer.py`

- Defines `ActQuantizer`.
- Supports four modes:
  - `none`
  - `calibrate`
  - `static`
  - `dynamic` (declared but not implemented)
- `calibrate` collects `amax`.
- `static` uses the stored scale for fake quantization during inference.

#### `PINNacle/quantization/calibration.py`

- Runs calibration for all `ActQuantizer` modules in a rewritten model.
- Uses `mit-han-lab/pile-val-backup` as the calibration corpus.
- Sets quantizers to `calibrate`, runs forward passes, computes scales, then switches them to `static`.

#### `PINNacle/pinn/sw/attn/llama_attn_pinn.py`

- Reimplementation of LLaMA attention.
- Keeps Q/K/V/O projections from the original model.
- Replaces the exponential inside softmax with a learned scalar approximator:
  - `ScalarExpoNet_d8`
  - `ScalarExpoNet_d16`
  - `ScalarExpoNet_d32`
- Supports activation quantization around the approximated softmax path.

#### `PINNacle/pinn/sw/attn/opt_attn_pinn.py`

- Same idea as the LLaMA attention rewrite, but adapted to OPT attention layout.
- Uses the same learned scalar exponential approximators.

#### `PINNacle/pinn/sw/mlp/llama_mlp_pinn.py`

- Reimplementation of LLaMA MLP.
- Approximates the sigmoid term used inside SiLU:
  - computes `sigmoid(x)` via a learned 1D network
  - reconstructs the SiLU gate as `sigmoid(x) * x`
- Uses the original `gate_proj`, `up_proj`, and `down_proj` weights.

#### `PINNacle/pinn/sw/attn/d8|d16|d32/`

- The scalar exponential approximators used by attention.
- Each variant is a fixed-width two-layer ReLU network with constrained-positive weights:
  - raw parameters are stored in log space
  - forward pass exponentiates them before use
- Each directory also carries a pretrained checkpoint.

#### `PINNacle/pinn/sw/mlp/d8|d16|d32/`

- The sigmoid approximators used by the LLaMA MLP rewrite.
- Same basic structure as the attention approximator.
- Each directory also carries a pretrained checkpoint.

#### `PINNacle/pinn/sw/attn/llama_attn_plf.py`

- Alternative softmax-exp approximation using piecewise linear fitting rather than the learned PINN block.
- Not wired into `approximation_wrapper.py` in this checkout.

#### `PINNacle/pinn/sw/attn/llama_attn_taylor.py`

- Alternative softmax-exp approximation using Taylor expansion plus range reduction.
- Also not wired into `approximation_wrapper.py` here.

#### `evaluation/lmeval/lmeval.py`

- Loads a HF causal LM.
- Applies `approximation_wrapper`.
- Optionally calibrates activation quantizers.
- Runs `lm-eval` tasks through `lm_eval.simple_evaluate`.
- Uses four task groups:
  - `group1`: `piqa`, `hellaswag`, `winogrande`, `arc_easy`
  - `group2`: `gsm8k`
  - `group3`: `mmlu`
  - `group4`: `wikitext`

#### `evaluation/wikitext/evaluate_hf.py`

- Intended to compute perplexity on WikiText by manually calculating shifted-token cross entropy over long token chunks.
- Loads the model and tokenizer in the same way as `lmeval.py`.

#### `scripts/*.sh`

- Thin command wrappers with preset model names and quantization flags.
- They document the intended CLI usage more than they serve as reliable production wrappers in the current checkout.

## 2. How To Use The Code For Evaluation

### Recommended working directory

Run from:

```bash
cd end2endacc
```

The shell scripts use relative paths like `evaluation/lmeval/lmeval.py`, so they assume `end2endacc/` is the current working directory.

### Python dependencies implied by the code

At minimum, the evaluation scripts expect:

```bash
uv pip install torch transformers datasets tqdm accelerate lm-eval
```

You will also need access to:

- the target HF model checkpoint
- `mit-han-lab/pile-val-backup` for activation calibration
- the relevant WikiText datasets for perplexity evaluation

### Intended evaluation flow

The code is designed to run in this order:

1. load a pretrained HF model with `AutoModelForCausalLM.from_pretrained(...)`
2. set `attn_implementation="eager"`
3. call `approximation_wrapper(...)`
4. optionally call `calibrate_static_act(...)`
5. run one of:
   - `lm_eval.simple_evaluate(...)`
   - custom WikiText perplexity loop

### Intended CLI knobs

These arguments control the approximation and quantization behavior:

- `--pinn`: enable PINN module replacement
- `--pinn_dim {8,16,32}`: choose approximator hidden width / checkpoint family
- `--quant_pinn`: quantize approximator weights
- `--quant_act`: enable activation quantizers inside the approximation path
- `--calibrate_static_act`: collect activation ranges and switch quantizers to static mode
- `--w_bits`, `--w_mantissa_bit`, `--w_zero_point`, `--w_group_size`, `--w_clip`, `--w_per_tensor`: approximator weight quantization
- `--a_bits`, `--a_mantissa_bit`, `--a_group_size`, `--a_clip`, `--a_per_tensor`: activation quantization
- `--fpq`: use floating-point-like quantization instead of integer quantization

### Intended `lm-eval` command

Use the Python entry point directly. The shell wrapper currently has flag-name mistakes; the command below uses the parser names that `lmeval.py` actually accepts.

```bash
cd end2endacc

CUDA_VISIBLE_DEVICES=0 python evaluation/lmeval/lmeval.py \
  --model facebook/opt-6.7b \
  --dtype float16 \
  --task_group group1 \
  --pinn \
  --pinn_dim 8 \
  --quant_pinn \
  --w_bits 8 \
  --w_mantissa_bit 5 \
  --w_per_tensor \
  --a_bits 8 \
  --a_mantissa_bit 2 \
  --a_per_tensor
```

If you want static activation quantization:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/lmeval/lmeval.py \
  --model facebook/opt-6.7b \
  --dtype float16 \
  --task_group group1 \
  --pinn \
  --pinn_dim 8 \
  --quant_pinn \
  --quant_act \
  --calibrate_static_act \
  --w_bits 8 \
  --w_mantissa_bit 5 \
  --w_per_tensor \
  --a_bits 8 \
  --a_mantissa_bit 2 \
  --a_per_tensor
```

### Intended WikiText perplexity command

Again, use the parser names from `evaluate_hf.py` rather than the shell wrapper.

```bash
cd end2endacc

CUDA_VISIBLE_DEVICES=0 python evaluation/wikitext/evaluate_hf.py \
  --model facebook/opt-6.7b \
  --dtype float16 \
  --pinn \
  --pinn_dim 16 \
  --quant_pinn \
  --quant_act \
  --calibrate_static_act \
  --w_bits 8 \
  --w_mantissa_bit 5 \
  --w_per_tensor \
  --a_bits 8 \
  --a_mantissa_bit 2 \
  --a_per_tensor
```

### What actually blocks execution in this checkout

The current tree contains a few issues you should account for before expecting evaluation to run end-to-end:

- `PINNacle/approximation_wrapper.py` imports `PINNacle.quantization.quant_funcs`, but that file does not exist here.
- `evaluation/wikitext/evaluate_hf.py` uses `from PINNacle import approximation_wrapper`, which does not match how the callable is defined.
- `scripts/evaluation_lm_eval.sh` and `scripts/evaluation_wikitext.sh` use `--w_bit` and `--a_bit`, but the parsers expect `--w_bits` and `--a_bits`.
- `scripts/finetune.sh` points to `finetune/finetune.py`, but no such file exists in this checkout.
- `evaluation/wikitext/evaluate_hf.py` has control-flow issues in `main()`:
  - the aggregate perplexity summary block runs outside the helper function
  - it references local variables that only exist inside the helper
  - the loss accumulation is placed after the chunk loop, so only the last chunk would contribute even if the scope bug were fixed

If your goal is to run the code exactly as-is, those issues need to be fixed first.

## 3. What The Code Does

### High-level purpose

This directory evaluates whether a transformer can keep acceptable task accuracy after replacing expensive nonlinear functions with constrained neural approximators that match the PINNacle hardware story.

Concretely:

- attention softmax no longer uses the standard `torch.exp`
- LLaMA MLP no longer uses the exact SiLU gate
- the approximators themselves can be quantized
- the activations feeding and leaving those approximators can also be quantized

The output metric is end-to-end model quality, not unit-test approximation error.

### Attention path

For LLaMA and OPT attention:

1. compute Q, K, V as usual
2. compute raw attention logits
3. subtract row-wise max for stability
4. feed valid shifted logits into a scalar learned exponential approximator
5. normalize by the row sum
6. continue with attention-value matmul

The learned approximator is a tiny 1D network with positive weights enforced by exponentiating raw parameters. That is consistent with the repo's broader idea of shape-constrained approximators for monotonic nonlinear functions.

### MLP path

For LLaMA MLP:

1. compute `gate_proj(x)`
2. approximate `sigmoid(gate_proj(x))` with a learned scalar network
3. reconstruct the SiLU-style gate as `sigmoid(x) * x`
4. multiply by `up_proj(x)`
5. project back through `down_proj`

So the LLaMA rewrite is not replacing the full MLP with a learned network. It only replaces the nonlinear gating function while preserving the original projection weights.

### Quantization path

There are two distinct quantization layers in this code:

#### Approximator-weight quantization

- Controlled by `--quant_pinn`.
- Quantizes the small learned approximator weights and some biases.
- Applied on the fly inside the approximator forward pass.

#### Activation quantization

- Controlled by `--quant_act`.
- Implemented through `ActQuantizer`.
- Can run in:
  - calibration mode
  - static inference mode
- Used around the attention logits / approximated exponential path and the LLaMA MLP gating path.

### Evaluation output

The intended outputs are:

- zero-shot or few-shot accuracy on `lm-eval` task groups
- perplexity on WikiText variants

So this directory is the "software accuracy validation" side of the project. It lets you compare exact nonlinear inference against approximated and quantized nonlinear inference.

## 4. How To Extend The Code

### A. Add a new model family

To support another HF architecture:

1. create a replacement attention module under `PINNacle/pinn/sw/attn/`
2. create a replacement MLP module if the architecture has a target nonlinear there
3. mirror the original HF forward signature closely
4. load the original state dict fields that still apply
5. add a new branch in `PINNacle/approximation_wrapper.py`

The wrapper is the only place that knows how to traverse model blocks for each architecture, so every new family needs a corresponding rewrite path there.

### B. Add a new approximator size

To add a new `pinn_dim`, for example `64`:

1. add `PINNacle/pinn/sw/attn/d64/pinn_d64.py`
2. add `PINNacle/pinn/sw/mlp/d64/pinn_d64.py`
3. place trained checkpoints in those directories
4. extend the `if args.pinn_dim == ...` branches in:
   - `llama_attn_pinn.py`
   - `opt_attn_pinn.py`
   - `llama_mlp_pinn.py`
5. extend the argparse `choices=[...]` in the evaluation scripts

### C. Add a new approximation method

This tree already contains two alternative attention approximations:

- piecewise linear fitting
- Taylor approximation

The clean extension pattern is:

1. create a separate module implementing the new method
2. keep the same external attention interface
3. decide how the method is selected:
   - new CLI flag
   - new wrapper branch
   - new `exp_method` selector
4. keep the model rewrite logic separate from the approximation math itself

Right now, those alternatives exist as standalone modules but are not wired through the main evaluation entry points.

### D. Add more evaluation tasks

For `lm-eval`:

1. extend `TASK_GROUPS` in `evaluation/lmeval/lmeval.py`
2. optionally expose new CLI choices
3. tune batch size per task group to fit GPU memory

For custom perplexity or dataset evaluation:

1. add a new script under `evaluation/`
2. reuse the same load -> rewrite -> calibrate -> evaluate flow
3. keep dataset-specific preprocessing local to that script

### E. Extend quantization behavior

Current constraints:

- `ActQuantizer.update_amax()` only supports `per_tensor=True`
- dynamic activation quantization is declared but not implemented

Natural next extensions:

1. implement per-channel or grouped activation calibration
2. implement actual dynamic quantization mode
3. separate calibration corpus choice from hardcoded `pile-val`
4. move quantizer construction into reusable helpers to reduce repeated boilerplate across attention/MLP files

### F. Make the package robust

Before extending functionality heavily, it would be worth cleaning up the integration layer:

1. add proper `__init__.py` files
2. remove or restore the missing `quant_funcs` dependency
3. fix the import path in `evaluate_hf.py`
4. fix the shell-script flag names
5. fix the WikiText perplexity control flow
6. decide whether `finetune.sh` belongs here or should be removed

Without those cleanup steps, extension work will stay fragile because the current tree mixes intended design with stale wiring.

## Practical Summary

The important mental model is:

- `evaluation/*` = entry points and metrics
- `PINNacle/approximation_wrapper.py` = model surgery
- `PINNacle/pinn/sw/attn/*` = softmax-exp approximation
- `PINNacle/pinn/sw/mlp/*` = SiLU/sigmoid approximation
- `PINNacle/quantization/*` = fake quantization and calibration

If you want to understand or modify behavior, start with `evaluation/lmeval/lmeval.py`, then `PINNacle/approximation_wrapper.py`, then the replacement module for the model family you care about.
