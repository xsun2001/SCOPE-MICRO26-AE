from datasets import load_dataset
from transformers import AutoTokenizer
import gc
import torch
from tqdm import tqdm
from .quantizer import ActQuantizer
from collections import defaultdict
import numpy as np

def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


def _calibration_forward(model, input_ids):
    # For activation calibration we only need the decoder stack to run.
    # Bypassing lm_head avoids unrelated dtype constraints in mixed exact/quantized setups.
    if hasattr(model, "model"):
        return model.model(input_ids=input_ids)
    if hasattr(model, "decoder"):
        return model.decoder(input_ids=input_ids)
    return model(input_ids)


@torch.no_grad()
def calibrate_static_act(
    model,
    tokenizer,
    num_samples=512,
    seq_len=512,
):
    """
    Calibrates a model containing ActQuantizer modules.

    This function performs the following steps:
    1. Sets all ActQuantizer modules to 'calibrate' mode.
    2. Feeds calibration data through the model to collect activation statistics (amax).
    3. Calls `update_scale()` on each quantizer to compute the quantization scale.
    4. Sets all ActQuantizer modules to 'static' mode, ready for inference.

    Args:
        model (nn.Module): The model to calibrate.
        tokenizer: The tokenizer for the model.
        dataset_path (str): Path to the calibration dataset.
        num_samples (int): Number of samples to use for calibration.
        seq_len (int): Maximum sequence length.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Find all ActQuantizer modules and set to 'calibrate' mode
    quantizers = [m for m in model.modules() if isinstance(m, ActQuantizer)]
    
    if not quantizers:
        print("Warning: No ActQuantizer modules found in the model. Nothing to calibrate.")
        return

    print(f"Found {len(quantizers)} activation quantizers to calibrate.")
    for quantizer in quantizers:
        quantizer.mode = 'calibrate'
        # Ensure amax is reset before starting a new calibration
        quantizer.amax.fill_(0.0)

    # 2. Run calibration data through the model
    print("Collecting activation statistics...")
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    dataset = dataset.shuffle(seed=42)
    pbar = tqdm(range(num_samples))

    for i in pbar:
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        _calibration_forward(model, input_ids)
        
        # Optional: Display the running average of amax from the first quantizer for progress
        if quantizers:
            pbar.set_description(f"Mean amax: {quantizers[0].amax.mean().item():.2f}")

    # 3. Update scales and set to 'static' mode
    print("\nCalibration finished. Updating scales and setting to static mode.")
    calibrated = 0
    skipped = 0
    for name, module in model.named_modules():
        if isinstance(module, ActQuantizer):
            if module.amax.detach().float().item() <= 0.0:
                module.mode = 'none'
                skipped += 1
                print(f"Skipping unused quantizer '{name}' because amax stayed at zero.")
                continue
            module.update_scale(name)
            module.mode = 'static'
            calibrated += 1

    print(
        f"Model calibration complete: {calibrated} quantizers in static mode, {skipped} left in pass-through mode."
    )
