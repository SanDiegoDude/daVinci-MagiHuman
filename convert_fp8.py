#!/usr/bin/env python3
"""Convert daVinci-MagiHuman DiT weights from FP32/BF16 to static FP8 (E4M3).

Reads the sharded safetensors checkpoint, quantizes large linear-layer weight
matrices to float8_e4m3fn with per-tensor absmax scaling, and writes a new
sharded checkpoint in the same format.  Norms, biases, embeddings, and small
tensors are kept in BF16.

Usage:
    python convert_fp8.py \
        --input  /path/to/models/daVinci-MagiHuman/distill \
        --output /path/to/models/daVinci-MagiHuman/distill_fp8

The output directory will contain:
    model-00001-of-NNNNN.safetensors  …  (FP8 weights + BF16 others)
    model.safetensors.index.json          (updated index)

To use the quantized model, point the engine_config.load at the output dir
and ensure the model code supports FP8 weight dequantization.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0
KEEP_DTYPE = torch.bfloat16

MIN_ELEMENTS_FOR_FP8 = 1024 * 1024  # only quantize tensors with ≥1M elements

SKIP_PATTERNS = (
    "norm",       # RMSNorm weights
    "bands",      # RoPE frequency bands
    "bias",       # all biases
    "embedder",   # adapter embedders (small)
)


def should_quantize(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    if tensor.numel() < MIN_ELEMENTS_FOR_FP8:
        return False
    name_lower = name.lower()
    for pat in SKIP_PATTERNS:
        if pat in name_lower:
            return False
    return True


def quantize_tensor(tensor: torch.Tensor):
    """Per-tensor absmax quantization to FP8 E4M3.

    Returns (quantized_fp8, scale) where:
        original ≈ quantized_fp8.to(bf16) * scale
    """
    t = tensor.to(torch.float32)
    amax = t.abs().amax().clamp(min=1e-12)
    scale = amax / FP8_MAX
    scaled = (t / scale).clamp(-FP8_MAX, FP8_MAX)
    quantized = scaled.to(FP8_DTYPE)
    return quantized, scale.to(torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert DiT checkpoint to FP8")
    parser.add_argument("--input", required=True, help="Source checkpoint directory")
    parser.add_argument("--output", required=True, help="Destination directory for FP8 checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be quantized without writing")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)

    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        print(f"Error: {index_path} not found")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    print(f"Source: {src}")
    print(f"  {len(weight_map)} tensors across {len(shard_names)} shards")

    # Group keys by shard
    shard_to_keys = {}
    for key, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(key)

    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    new_weight_map = {}
    total_original_bytes = 0
    total_fp8_bytes = 0
    quantized_count = 0
    kept_count = 0

    for shard_idx, shard_name in enumerate(tqdm(shard_names, desc="Processing shards")):
        shard_path = src / shard_name
        tensors = load_file(str(shard_path))

        out_tensors = {}
        for key in shard_to_keys[shard_name]:
            tensor = tensors[key]
            original_bytes = tensor.numel() * tensor.element_size()
            total_original_bytes += original_bytes

            if should_quantize(key, tensor):
                q_tensor, scale = quantize_tensor(tensor)
                out_tensors[key] = q_tensor
                out_tensors[f"{key}.__fp8_scale"] = scale
                fp8_bytes = q_tensor.numel() * 1 + 4  # 1 byte per elem + 4 byte scale
                total_fp8_bytes += fp8_bytes
                quantized_count += 1

                if args.dry_run:
                    # Show quantization error
                    reconstructed = q_tensor.to(torch.float32) * scale
                    orig_f32 = tensor.to(torch.float32)
                    rel_err = (reconstructed - orig_f32).abs().mean() / orig_f32.abs().mean()
                    print(f"  QUANTIZE {key}: {list(tensor.shape)} "
                          f"({original_bytes / 1e6:.1f}MB → {fp8_bytes / 1e6:.1f}MB) "
                          f"rel_err={rel_err:.6f}")
            else:
                out_tensors[key] = tensor.to(KEEP_DTYPE)
                total_fp8_bytes += tensor.numel() * 2  # BF16 = 2 bytes
                kept_count += 1

                if args.dry_run:
                    print(f"  KEEP     {key}: {list(tensor.shape)} dtype={tensor.dtype}")

            new_weight_map[key] = shard_name
            if f"{key}.__fp8_scale" in out_tensors:
                new_weight_map[f"{key}.__fp8_scale"] = shard_name

        if not args.dry_run:
            out_path = dst / shard_name
            save_file(out_tensors, str(out_path))

        del tensors, out_tensors

    # Write new index
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map,
    }
    new_index["metadata"]["quantization"] = "fp8_e4m3fn_per_tensor"

    if not args.dry_run:
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary:")
    print(f"  Quantized: {quantized_count} tensors")
    print(f"  Kept (BF16): {kept_count} tensors")
    print(f"  Original size: {total_original_bytes / 1e9:.2f} GB")
    print(f"  FP8 size:      {total_fp8_bytes / 1e9:.2f} GB")
    print(f"  Compression:   {total_original_bytes / total_fp8_bytes:.1f}x")
    if not args.dry_run:
        print(f"\n  Output written to: {dst}")


if __name__ == "__main__":
    main()
