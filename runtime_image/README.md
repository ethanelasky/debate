# D043 immutable Debate runtime assets

This directory contains the provider-free metadata frontier for the fixed
runtime root `/opt/job-scheduler/debate-runtime/v1`. It does not build or pull
an image, resolve or install a dependency, contact RunPod, or authorize START.

The current repository cannot yet produce a truthful runtime image. Its
`uv.lock` covers the Debate application but not the GPU execution stack:
`torch`, `vllm`, `verl`, `ray`, and `flash-attn` are absent. The existing
provisioner admits `vllm==0.24.*`, lets a resolver choose transitive packages,
and patches four installed source files in place. The locally cached
flash-attn wheels are not tracked build inputs. An immutable base-image digest
and a complete artifact wheelhouse (including wheels with the approved source
patches already applied) are also absent.

For that reason, this change intentionally does **not** add a placeholder
`dependency.lock`, `build-spec.json`, or Dockerfile. A placeholder would make a
partial environment look reproducible. `verify_build_inputs.sh` fails closed
until both canonical inputs exist and every lock artifact is present and
hash-matched.

`runtime_contract.py` is the single exported machine-readable contract for
all closed key sets, schema literals, limits, fixed paths, and exact probe
programs. The generator loads that tracked module explicitly even under
isolated Python. Scheduler bootstrap tests consume the same golden source and
must reject a divergent local copy.

## Required build inputs

`dependency.lock` is canonical compact ASCII JSON with no newline:

```text
{
  "schema": "runpod-safety.debate-runtime-dependency-lock/v1",
  "runtime_root": "/opt/job-scheduler/debate-runtime/v1",
  "python_version": "<exact 3.12 version>",
  "distributions": [
    {"name":"<PEP-503 normalized>","version":"<exact>",
     "artifact":{"path":"<relative tracked wheel>","sha256":"sha256:<64hex>"}}
  ]
}
```

Every installed distribution, including installer tooling left in the runtime,
must appear exactly once. Entries are name-sorted. Each artifact is a regular,
single-link file adjacent to the lock's build context; URLs, ranges, wildcards,
editable installs, and ambient indexes are not accepted. The generator opens
each artifact as a wheel, validates its METADATA identity and every RECORD
hash/size, and compares installed package bytes with that locked source. An
in-place source patch therefore fails; an approved patch must be represented by
a new tracked, hashed wheel.

`build-spec.json` is canonical compact ASCII JSON with exact keys `schema`,
`runtime_root`, `python`, `site_packages_path`, `required_imports`, and `cuda`.
Its schema is `runpod-safety.debate-runtime-build-spec/v1`. Python fixes the
path `/opt/job-scheduler/debate-runtime/v1/python/bin/python3.12` and an exact
version. Required imports map a sorted module to its exact normalized
distribution and version. CUDA fixes the torch distribution/version, the
torch CUDA version, exact device count `1`, compute capability `[9,0]`, and a
sorted nonempty list of compiled extension modules.

## Generated image objects

After an offline, hash-enforcing installer has populated the fixed runtime,
run the generator as root in the image build. It writes, without replacement:

- `dependency.lock`
- `installed-distributions.json`
- `required-import-probe.json`
- `cuda-compatibility-probe.json`
- `runtime-manifest.json`

All JSON is canonical compact ASCII without a trailing newline. Support files
are mode `0444`; the exact Python is a real mode-`0555` file, not a symlink.
The final image step must make the complete runtime tree root-owned and
non-writable, then the scheduler bootstrap independently rehashes it.

The inventory is complete against every `.dist-info/RECORD`, rejects editable
or legacy installs, hashes every claimed regular file, records the exact
metadata directory, and binds `direct_url.json` raw bytes plus its canonical
semantic JSON when present. Its normalized distribution set and versions must
equal the dependency lock exactly.

Each probe document has exactly `schema`, `argv`, and `expected_result`.
`argv` is exactly five elements: the fixed Python, `-I`, `-c`, the generator's
fixed probe source, and canonical expected-result JSON. At START the installed
bootstrap must execute that exact argv under a sanitized environment and
require return code zero, empty stderr, and stdout byte-equal to canonical
`expected_result` with no newline. The import probe checks live import plus
distribution/version identity. The CUDA probe checks the live Python and
torch/CUDA identity, exact H100 device count/capabilities, and compiled
extension imports.
