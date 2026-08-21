"""Single exported D043 sibling-schema and fixed-probe contract.

The Debate metadata generator consumes this module directly. Scheduler tests
consume the same tracked file as their golden contract so the two repositories
cannot silently evolve independent field sets or probe programs.
"""

RUNTIME_ROOT = "/opt/job-scheduler/debate-runtime/v1"
RUNTIME_PYTHON = RUNTIME_ROOT + "/python/bin/python3.12"

MANIFEST_SCHEMA = "runpod-safety.debate-runtime-manifest/v1"
LOCK_SCHEMA = "runpod-safety.debate-runtime-dependency-lock/v1"
SPEC_SCHEMA = "runpod-safety.debate-runtime-build-spec/v1"
INVENTORY_SCHEMA = "runpod-safety.installed-distributions/v1"
IMPORT_PROBE_SCHEMA = "runpod-safety.required-import-probe/v1"
CUDA_PROBE_SCHEMA = "runpod-safety.cuda-compatibility-probe/v1"

LOCK_KEYS = frozenset({"schema", "runtime_root", "python_version", "distributions"})
LOCK_DISTRIBUTION_KEYS = frozenset({"name", "version", "artifact"})
LOCK_ARTIFACT_KEYS = frozenset({"path", "sha256"})
SPEC_KEYS = frozenset(
    {
        "schema",
        "runtime_root",
        "python",
        "site_packages_path",
        "required_imports",
        "cuda",
    }
)
PYTHON_KEYS = frozenset({"path", "version"})
REQUIRED_IMPORT_KEYS = frozenset({"module", "distribution", "version"})
CUDA_SPEC_KEYS = frozenset(
    {
        "torch_distribution",
        "torch_version",
        "torch_cuda_version",
        "device_count",
        "compute_capability",
        "compiled_extensions",
    }
)
COMPILED_EXTENSION_SPEC_KEYS = frozenset({"module"})
INVENTORY_KEYS = frozenset(
    {"schema", "runtime_root", "python_path", "distributions"}
)
INVENTORY_DISTRIBUTION_KEYS = frozenset(
    {"name", "version", "metadata_path", "direct_origin", "files"}
)
DIRECT_ORIGIN_KEYS = frozenset({"path", "sha256", "canonical_json"})
INVENTORY_FILE_KEYS = frozenset({"path", "size", "sha256"})
PROBE_KEYS = frozenset({"schema", "argv", "expected_result"})
IMPORT_RESULT_KEYS = frozenset({"python", "imports"})
CUDA_RESULT_KEYS = frozenset({"python", "torch", "gpu", "compiled_extensions"})
CUDA_TORCH_KEYS = frozenset({"version", "cuda_version"})
CUDA_GPU_KEYS = frozenset({"available", "count", "compute_capabilities"})
COMPILED_EXTENSION_RESULT_KEYS = frozenset({"module", "loaded"})

MAX_DISTRIBUTIONS = 4096
MAX_FILES = 250_000
MAX_FILE_SIZE = 16 * 1024 * 1024 * 1024
MAX_PATH_BYTES = 4096
MAX_IMPORTS = 1024
MAX_EXTENSIONS = 64
EXACT_DEVICE_COUNT = 1
EXACT_COMPUTE_CAPABILITY = [9, 0]

FIXED_IMPORT_SOURCE = """import importlib
import importlib.metadata
import json
import platform
import sys
config=json.loads(sys.argv[1])
imports=[]
for expected in config["imports"]:
    importlib.import_module(expected["module"])
    imports.append({"distribution":expected["distribution"],"module":expected["module"],"version":importlib.metadata.version(expected["distribution"])})
result={"imports":imports,"python":{"path":sys.executable,"version":platform.python_version()}}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False))"""

FIXED_CUDA_SOURCE = """import importlib
import json
import platform
import sys
import torch
config=json.loads(sys.argv[1])
extensions=[]
for expected in config["compiled_extensions"]:
    importlib.import_module(expected["module"])
    extensions.append({"loaded":True,"module":expected["module"]})
count=torch.cuda.device_count()
result={"compiled_extensions":extensions,"gpu":{"available":torch.cuda.is_available(),"compute_capabilities":[list(torch.cuda.get_device_capability(index)) for index in range(count)],"count":count},"python":{"path":sys.executable,"version":platform.python_version()},"torch":{"cuda_version":torch.version.cuda,"version":torch.__version__}}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False))"""
