# D043 immutable Debate B200 runtime

This directory now closes the provider-free build inputs for the fixed runtime
root `/opt/job-scheduler/debate-runtime/v1`. It does not contact Hugging Face,
a package index, a registry, RunPod, or a GPU provider, and it does not
authorize START.

The approved runtime uses a content-addressed binary-seed provenance tier. It
does **not** claim that the portable environment came from a complete wheel
set. The older wheel-lock v1 generator path remains available for genuinely
wheel-derived runtimes; this B200 runtime uses the separate v2 seed schemas.

## Closed source identities

`dependency.lock` is canonical compact ASCII JSON and binds all of:

- base image
  `ghcr.io/ethanelasky/verl-megatron-ssh@sha256:162f12d0ba6fbceaf56294d783551b2e83aefcc6e76b9138737acd358ee0baa8`;
- the base's real `/usr/bin/python3.12`, SHA-256
  `e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7`;
- Hugging Face dataset `ethanelasky/verl-b200-env`, immutable revision
  `dd4c26f6cb38002a56d55556c6e622308394e400`;
- `verl-b200-portable.tar.zst`, exactly 3,891,912,437 bytes, SHA-256
  `65ebafdd4e02e64887cb03740c9557dfcd17e98f8b8f720700c8166179fe62e9`;
- its exact raw tar stream, exactly 10,294,538,240 bytes, SHA-256
  `171b9b1c8ccd7e5a28b8a919e16e268d5b60cb7a34186ff78e4f7a15f5fa9c59`;
- the deterministic relocation/removal policy; and
- all 279 final installed distribution names and exact METADATA versions.

The 3.9 GB seed and derived 10.3 GB tar are deliberately not stored in Git.
Their bytes are external build inputs, not ambient dependency sources.

The checked `build-spec.json` fixes Python 3.12.3, distribution Torch 2.11.0,
live `torch.__version__` 2.11.0+cu130, CUDA 13.0, Ray 2.56.1, Transformers
5.14.1, Verl 0.9.0.dev0 (whose retained HTTPS VCS origin names commit
`e9618406de5bad40041d7612554e465ec2003ec1`), vLLM 0.24.0, and compiled
extension `vllm._C`. The validated seed does not contain flash-attn, so the
spec does not invent a flash-attn import or provenance claim.

## Deterministic offline preparation

`prepare_binary_seed.py` accepts only the checked lock and one exact seed. It:

- permits only normalized regular files/directories and the four declared
  source symlinks, then removes every link from the final tree;
- binds and ignores the archive's one exact empty out-of-prefix directory,
  `workspace/uv/python`, while refusing every other out-of-prefix member;
- refuses traversal, hard links, devices, FIFOs, unknown symlinks, duplicate
  members, oversized members, editable hooks, and `file:` direct origins;
- removes the complete editable Debate installation (scripts, finder, PTH,
  bytecode, and dist-info), because the scheduler stages the reviewed Debate
  source separately;
- removes every retained `__pycache__`/`.pyc` and deterministically removes
  those rows from each distribution `RECORD`;
- admits only the three exact digest-allowlisted non-Debate PTH files;
- preserves legitimate shared `RECORD` claims in each distribution inventory
  while recording and hashing each final payload file exactly once;
- binds the seed's two intentional virtualenv bootstrap files, `_virtualenv.pth`
  and `_virtualenv.py`, as the only site-packages files not claimed by a
  distribution `RECORD`;
- replaces the portable Python link with the exact real base Python as a
  mode-0555 single-link file;
- changes `pyvenv.cfg` only from its exact source digest to the fixed
  `home = /usr/bin` bytes;
- rewrites only the exact source shebang and five declared activation files to
  the fixed runtime root, always refuses the old
  `/workspace/envs/verl-b200` prefix, and admits other `/workspace/` strings
  only in 25 exact path-and-hash allowlisted SDK documentation/example or
  native source/debug-string files; and
- normalizes every payload file and directory to mtime zero, makes each file
  exactly mode 0444 or 0555 and each directory mode 0555, and publishes a
  canonical, complete `seed-transformation.json` file/directory inventory.

The pinned outer image does not contain zstd. The preferred Docker input is
therefore the content-addressed raw tar. Produce it outside the image using a
locally reviewed decoder, then verify it before it enters the build context:

```sh
/opt/homebrew/bin/zstd -dcf verl-b200-portable.tar.zst \
  -o verl-b200-portable.tar
test "$(stat -f %z verl-b200-portable.tar)" = 10294538240
test "$(shasum -a 256 verl-b200-portable.tar | awk '{print $1}')" = \
  171b9b1c8ccd7e5a28b8a919e16e268d5b60cb7a34186ff78e4f7a15f5fa9c59
```

On Linux use `stat -c %s`; the hash is platform-independent. The preparer
rehashes the entire raw stream again, so decoder identity is not trusted.

The real transformed-tree audit found zero ELF `RPATH`/`RUNPATH` dynamic tags
containing `/workspace`; the 25 reviewed occurrences above are content-bound
in the lock. Any byte or path drift, any reappearance of the source environment
prefix, or any new `/workspace/` occurrence refuses preparation.

Inside an offline build stage based on the literal pinned image digest:

```sh
/usr/bin/python3.12 -I -S runtime_image/prepare_binary_seed.py \
  --dependency-lock runtime_image/dependency.lock \
  --binary-seed /build-inputs/verl-b200-portable.tar \
  --archive-format tar \
  --base-python /usr/bin/python3.12 \
  --staging-root /

/usr/bin/python3.12 -I -S runtime_image/generate_runtime_metadata.py \
  --dependency-lock runtime_image/dependency.lock \
  --build-spec runtime_image/build-spec.json \
  --uncompressed-seed /build-inputs/verl-b200-portable.tar \
  --staging-root /
```

`Dockerfile.binary-seed` encodes those two commands with `RUN --network=none`.
For example, if a local directory contains only the locked raw tar:

```sh
docker buildx build --pull=false --network=none \
  --build-context runtime_seed=/absolute/raw-seed-directory \
  --file runtime_image/Dockerfile.binary-seed \
  --tag debate-b200-runtime:local --load .
```

Pass the raw tar through a BuildKit bind/named build context; do not add it to
the repository or image. Build with `--network=none --pull=false`. Preparation
does not implement a compressed mode: the pinned base has no decoder, and the
raw bytes are independently content-addressed by the lock.

The Dockerfile also digest-pins its Dockerfile frontend. That frontend, the
literal base image, and the raw-tar named context must already be present in
the local builder cache for a build with no registry access.

`verify_build_inputs.sh` performs only the lock/spec/seed checks. Set exactly
one of `DEBATE_RUNTIME_BINARY_SEED` or
`DEBATE_RUNTIME_UNCOMPRESSED_SEED` before running it.

## Generated evidence and readiness

The generator publishes, without replacement:

- `dependency.lock`;
- `seed-transformation.json` (published by the preparer and bound by manifest);
- `installed-distributions.json`, including every distribution RECORD file and
  the complete transformed payload file-and-directory inventory, with exact
  mode and normalized mtime;
- `required-import-probe.json`;
- `cuda-compatibility-probe.json`; and
- `runtime-manifest.json`.

The v2 manifest binds every sibling plus the exact Python. The scheduler
bootstrap must rehash every payload file, reject any extra final-tree entry,
and execute both canonical probes under its clean UID-10001 environment before
READY. Evidence siblings are individually manifest-bound; the manifest binds
itself by hashing the canonical document without its self-digest field.

The CUDA probe requires the live runtime to report exactly two NVIDIA B200s,
SM100 `[10,0]`, CUDA 13.0, driver major at least 580, symmetric NV18 for pair
`[[0,1]]`, deterministic BF16 matmul plus synchronization on both devices, and
the compiled vLLM extension. This provider-free closure does not claim that a
real 2xB200 Debate run passed. That remains a separately authorized paid-run
certification gate.
