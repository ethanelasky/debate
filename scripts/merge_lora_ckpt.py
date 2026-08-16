"""Merge a verl LoRA checkpoint into base HF weights for serving.

    python scripts/merge_lora_ckpt.py --ckpt <dir with model_world_size_1_rank_0.pt> \
        --base <HF snapshot dir> --out <output dir>

The checkpoint holds peft-named LoRA pairs (base_model.model.<path>.lora_{A,B}.
default.weight), possibly as single-rank DTensors. W += (alpha/r) * B @ A.
Adapter-mode vLLM serving is NOT an alternative on Qwen3.5: it silently drops
LoRA on GDN in_proj_a/b (vllm#38085).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file, save_file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base", required=True, help="HF snapshot dir with *.safetensors")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sd = torch.load(
        os.path.join(args.ckpt, "model_world_size_1_rank_0.pt"),
        map_location="cpu", weights_only=False,
    )
    sd = {k: (v.to_local() if hasattr(v, "to_local") else v) for k, v in sd.items()}
    meta = json.load(open(os.path.join(args.ckpt, "lora_train_meta.json")))
    scale = meta["lora_alpha"] / meta["r"]

    deltas = {}
    for k, v in sd.items():
        if ".lora_A.default.weight" in k:
            base_key = k[len("base_model.model."):].replace(".lora_A.default.weight", ".weight")
            b = sd[k.replace(".lora_A.", ".lora_B.")]
            deltas[base_key] = (b.float() @ v.float()) * scale

    os.makedirs(args.out, exist_ok=True)
    applied = 0
    for shard in sorted(glob.glob(os.path.join(args.base, "*.safetensors"))):
        tensors = load_file(shard)
        for name in list(tensors):
            if name in deltas:
                tensors[name] = (tensors[name].float() + deltas.pop(name)).to(tensors[name].dtype)
                applied += 1
        save_file(tensors, os.path.join(args.out, os.path.basename(shard)), metadata={"format": "pt"})
    for f in glob.glob(os.path.join(args.base, "*")):
        if not f.endswith(".safetensors") and os.path.isfile(f):
            shutil.copy(f, os.path.join(args.out, os.path.basename(f)))

    print(f"applied {applied} deltas; unmatched: {len(deltas)}")
    if deltas:
        raise SystemExit(f"UNMATCHED keys (merge incomplete): {list(deltas)[:5]}")


if __name__ == "__main__":
    main()
