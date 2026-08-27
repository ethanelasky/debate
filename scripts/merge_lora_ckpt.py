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
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def load_full_state(ckpt: str) -> dict:
    """The checkpoint's state dict with every DTensor made whole.

    A run's world size is a THROUGHPUT decision -- mathl5_qwen35_cispo_sub8
    trains on 1 GPU and its _kenton sibling on 2 -- but it changes the
    checkpoint on disk: FSDP2 saves one file per rank, each holding a DTensor
    whose local shard is a slice of the real parameter. Reading only rank 0
    silently merges HALF of every LoRA matrix, which is not an error anywhere;
    it just produces a subtly wrong model. So the shard count is discovered
    from the filenames rather than assumed, and every reassembled tensor is
    checked against the DTensor's own global shape.
    """
    files = glob.glob(os.path.join(ckpt, "model_world_size_*_rank_*.pt"))
    if not files:
        raise SystemExit(f"no model_world_size_*_rank_*.pt under {ckpt}")
    files.sort(key=lambda p: int(re.search(r"rank_(\d+)", p).group(1)))
    shards = [torch.load(f, map_location="cpu", weights_only=False) for f in files]

    out: dict = {}
    for k, v0 in shards[0].items():
        if not hasattr(v0, "placements"):  # plain tensor: not sharded
            out[k] = v0
            continue
        locals_ = [s[k].to_local() for s in shards]
        placement = v0.placements[0]
        if len(shards) == 1 or not placement.is_shard():
            out[k] = locals_[0]
        else:
            dim = placement.dim
            full = torch.cat(locals_, dim=dim)
            # FSDP pads the last shard when the dim does not divide evenly;
            # the DTensor's global shape is the authority on the real extent.
            if full.shape[dim] > v0.shape[dim]:
                full = full.narrow(dim, 0, v0.shape[dim])
            out[k] = full
        if tuple(out[k].shape) != tuple(v0.shape):
            raise SystemExit(
                f"reassembled {k} as {tuple(out[k].shape)} but the checkpoint "
                f"says it is {tuple(v0.shape)} ({len(shards)} shards, {placement})"
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--base", required=True, help="HF snapshot dir with *.safetensors")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    sd = load_full_state(args.ckpt)
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
