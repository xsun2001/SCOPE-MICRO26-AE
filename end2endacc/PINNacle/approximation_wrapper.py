import torch
from tqdm import tqdm

import gc


def print_vram_usage():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(torch.cuda.memory_summary())


def _model_family(model) -> str:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if isinstance(model_type, str) and model_type:
        return model_type.lower()

    name_or_path = getattr(getattr(model, "config", None), "_name_or_path", "")
    lowered = str(name_or_path).lower()
    if "llama" in lowered:
        return "llama"
    if "opt" in lowered:
        return "opt"
    raise ValueError(f"Unsupported model family for PINN replacement: {name_or_path!r}")


def _move_replacement(module, original_module, torch_dtype):
    return module.to(torch_dtype).to(next(original_module.parameters()).device)


def _load_filtered_state_dict(new_module, original_module):
    original_state_dict = original_module.state_dict()
    new_keys = set(new_module.state_dict().keys())
    filtered_state_dict = {k: v.to("cpu") for k, v in original_state_dict.items() if k in new_keys}
    new_module.load_state_dict(filtered_state_dict, strict=False)


def apply_pinn_replacement(model, args, torch_dtype):
    model_family = _model_family(model)
    if model_family == "llama":
        if args.approx_scope in {"all", "attn"}:
            from .pinn.sw.attn.llama_attn_pinn import LlamaAttention_PINN

            for _, block in tqdm(enumerate(model.model.layers)):
                new_attn = LlamaAttention_PINN(args, model.config, block.self_attn.layer_idx).to("cpu")
                _load_filtered_state_dict(new_attn, block.self_attn)
                block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
                torch.cuda.empty_cache()
                gc.collect()
        if args.approx_scope == "all":
            from .pinn.sw.mlp.llama_mlp_pinn import LlamaMLP_PINN

            for _, block in tqdm(enumerate(model.model.layers)):
                new_mlp = LlamaMLP_PINN(model.config, args).to("cpu")
                _load_filtered_state_dict(new_mlp, block.mlp)
                block.mlp = _move_replacement(new_mlp, block.mlp, torch_dtype)
                torch.cuda.empty_cache()
                gc.collect()
        torch.cuda.empty_cache()
    elif model_family in {"qwen2", "qwen3"}:
        from .pinn.sw.attn.qwen_attn_pinn import Qwen2Attention_PINN, Qwen3Attention_PINN

        attention_cls = Qwen2Attention_PINN if model_family == "qwen2" else Qwen3Attention_PINN
        if args.approx_scope in {"all", "attn"}:
            for _, block in tqdm(enumerate(model.model.layers)):
                new_attn = attention_cls(args, model.config, block.self_attn.layer_idx).to("cpu")
                _load_filtered_state_dict(new_attn, block.self_attn)
                block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
                torch.cuda.empty_cache()
                gc.collect()
        if args.approx_scope == "all":
            from .pinn.sw.mlp.qwen_mlp_pinn import QwenMLP_PINN

            for _, block in tqdm(enumerate(model.model.layers)):
                new_mlp = QwenMLP_PINN(model.config, args).to("cpu")
                _load_filtered_state_dict(new_mlp, block.mlp)
                block.mlp = _move_replacement(new_mlp, block.mlp, torch_dtype)
                torch.cuda.empty_cache()
                gc.collect()
        torch.cuda.empty_cache()
    elif model_family == "opt":
        from .pinn.sw.attn.opt_attn_pinn import OPTAttention_PINN

        for i, block in tqdm(enumerate(model.model.decoder.layers)):
            layer_idx = getattr(block.self_attn, "layer_idx", i)
            new_attn = OPTAttention_PINN(args, model.config, layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model family for PINN replacement: {model_family}")
    return model


def apply_nnlut_replacement(model, args, torch_dtype):
    if args.approx_scope != "attn":
        raise ValueError("NNLUT replacement currently supports only `--approx_scope attn`.")

    model_family = _model_family(model)
    if model_family == "llama":
        from .nnlut.sw.attn.llama_attn_nnlut import LlamaAttention_NNLUT

        for _, block in tqdm(enumerate(model.model.layers)):
            new_attn = LlamaAttention_NNLUT(args, model.config, block.self_attn.layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    elif model_family == "opt":
        from .nnlut.sw.attn.opt_attn_nnlut import OPTAttention_NNLUT

        for i, block in tqdm(enumerate(model.model.decoder.layers)):
            layer_idx = getattr(block.self_attn, "layer_idx", i)
            new_attn = OPTAttention_NNLUT(args, model.config, layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model family for NNLUT replacement: {model_family}")
    return model


def apply_gqalut_replacement(model, args, torch_dtype):
    if args.approx_scope != "attn":
        raise ValueError("GQA-LUT replacement currently supports only `--approx_scope attn`.")

    model_family = _model_family(model)
    if model_family == "llama":
        from .gqalut.sw.attn.llama_attn_gqalut import LlamaAttention_GQALUT

        for _, block in tqdm(enumerate(model.model.layers)):
            new_attn = LlamaAttention_GQALUT(args, model.config, block.self_attn.layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    elif model_family == "opt":
        from .gqalut.sw.attn.opt_attn_gqalut import OPTAttention_GQALUT

        for i, block in tqdm(enumerate(model.model.decoder.layers)):
            layer_idx = getattr(block.self_attn, "layer_idx", i)
            new_attn = OPTAttention_GQALUT(args, model.config, layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model family for GQA-LUT replacement: {model_family}")
    return model


def apply_nli_replacement(model, args, torch_dtype):
    if args.approx_scope != "attn":
        raise ValueError("NLI replacement currently supports only `--approx_scope attn`.")

    model_family = _model_family(model)
    if model_family == "llama":
        from .nli.sw.attn.llama_attn_nli import LlamaAttention_NLI

        for _, block in tqdm(enumerate(model.model.layers)):
            new_attn = LlamaAttention_NLI(args, model.config, block.self_attn.layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    elif model_family == "opt":
        from .nli.sw.attn.opt_attn_nli import OPTAttention_NLI

        for i, block in tqdm(enumerate(model.model.decoder.layers)):
            layer_idx = getattr(block.self_attn, "layer_idx", i)
            new_attn = OPTAttention_NLI(args, model.config, layer_idx).to("cpu")
            _load_filtered_state_dict(new_attn, block.self_attn)
            block.self_attn = _move_replacement(new_attn, block.self_attn, torch_dtype)
            torch.cuda.empty_cache()
            gc.collect()
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unsupported model family for NLI replacement: {model_family}")
    return model


def apply_approximation_replacement(model, args, torch_dtype):
    backend = getattr(args, "approx_backend", "none")
    if backend == "none":
        return model
    if backend == "pinn":
        return apply_pinn_replacement(model, args, torch_dtype)
    if backend == "nnlut":
        return apply_nnlut_replacement(model, args, torch_dtype)
    if backend == "gqalut":
        return apply_gqalut_replacement(model, args, torch_dtype)
    if backend == "nli":
        return apply_nli_replacement(model, args, torch_dtype)
    raise ValueError(f"Unsupported approximation backend: {backend}")


def approximation_wrapper(model, args, torch_dtype):
    return apply_approximation_replacement(model, args, torch_dtype)
