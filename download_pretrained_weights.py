#!/usr/bin/env python
"""Download the pretrained weights required to train ObjectClear.

This fetches two public checkpoints from the Hugging Face Hub:

  1. SDXL inpainting base  -> diffusers/stable-diffusion-xl-1.0-inpainting-0.1
  2. CLIP image encoder    -> openai/clip-vit-large-patch14

By default they are saved under ``./ckpts`` so the paths line up with the
example command at the top of ``train_objectclear.py``:

    --pretrained_model_name_or_path ./ckpts/stable-diffusion-xl-1.0-inpainting-0.1
    --image_encoder_name_or_path    ./ckpts/clip-vit-large-patch14

Usage:
    python download_pretrained_weights.py
    python download_pretrained_weights.py --output_dir /path/to/ckpts
"""
import argparse
import os

from huggingface_hub import snapshot_download


WEIGHTS = [
    {
        "repo_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        "local_name": "stable-diffusion-xl-1.0-inpainting-0.1",
        # Skip the redundant fp16 variant to save bandwidth; training defaults to fp32.
        "allow_patterns": None,
        "ignore_patterns": ["*.fp16.safetensors", "*.fp16.bin", "*.onnx", "*.onnx_data"],
    },
    {
        "repo_id": "openai/clip-vit-large-patch14",
        "local_name": "clip-vit-large-patch14",
        "allow_patterns": None,
        # Only the safetensors weights + configs are needed; drop the legacy .bin / tf / flax.
        "ignore_patterns": ["*.bin", "*.msgpack", "*.h5", "flax_model.*"],
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "-o", "--output_dir", type=str, default="./ckpts",
        help="Directory to place the downloaded checkpoints. Default: ./ckpts",
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Optional Hugging Face cache directory (defaults to the HF cache).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for w in WEIGHTS:
        local_dir = os.path.join(args.output_dir, w["local_name"])
        print(f"\n=== Downloading {w['repo_id']} -> {local_dir} ===")
        snapshot_download(
            repo_id=w["repo_id"],
            local_dir=local_dir,
            cache_dir=args.cache_dir,
            allow_patterns=w["allow_patterns"],
            ignore_patterns=w["ignore_patterns"],
        )
        print(f"[done] {w['repo_id']}")

    print("\nAll pretrained weights are ready under:", os.path.abspath(args.output_dir))
    print("Point the training script at them, e.g.:")
    print(f"    --pretrained_model_name_or_path {os.path.join(args.output_dir, 'stable-diffusion-xl-1.0-inpainting-0.1')}")
    print(f"    --image_encoder_name_or_path    {os.path.join(args.output_dir, 'clip-vit-large-patch14')}")


if __name__ == "__main__":
    main()
