"""Single exported D043 sibling-schema and fixed-probe contract.

The Debate metadata generator consumes this module directly. Scheduler tests
consume the same tracked file as their golden contract so the two repositories
cannot silently evolve independent field sets or probe programs.
"""

RUNTIME_ROOT = "/opt/job-scheduler/debate-runtime/v1"
RUNTIME_PYTHON = RUNTIME_ROOT + "/python/bin/python3.12"

MANIFEST_SCHEMA = "runpod-safety.debate-runtime-manifest/v1"
BINARY_SEED_MANIFEST_SCHEMA = "runpod-safety.debate-runtime-manifest/v2"
LOCK_SCHEMA = "runpod-safety.debate-runtime-dependency-lock/v1"
SPEC_SCHEMA = "runpod-safety.debate-runtime-build-spec/v1"
BINARY_SEED_LOCK_SCHEMA = "runpod-safety.debate-runtime-binary-seed-lock/v2"
BINARY_SEED_SPEC_SCHEMA = "runpod-safety.debate-runtime-binary-seed-build-spec/v2"
SEED_TRANSFORMATION_SCHEMA = "runpod-safety.debate-runtime-seed-transformation/v1"
INVENTORY_SCHEMA = "runpod-safety.installed-distributions/v1"
BINARY_SEED_INVENTORY_SCHEMA = "runpod-safety.installed-runtime/v2"
IMPORT_PROBE_SCHEMA = "runpod-safety.required-import-probe/v1"
CUDA_PROBE_SCHEMA = "runpod-safety.cuda-compatibility-probe/v1"

LOCK_KEYS = frozenset({"schema", "runtime_root", "python_version", "distributions"})
LOCK_DISTRIBUTION_KEYS = frozenset({"name", "version", "artifact"})
LOCK_ARTIFACT_KEYS = frozenset({"path", "sha256"})
BINARY_SEED_LOCK_KEYS = frozenset(
    {
        "schema",
        "provenance_tier",
        "runtime_root",
        "python_version",
        "base_image",
        "seed",
        "transformation",
        "distributions",
    }
)
BINARY_SEED_DISTRIBUTION_KEYS = frozenset({"name", "version"})
BINARY_SEED_BASE_IMAGE_KEYS = frozenset({"reference", "python"})
BINARY_SEED_BASE_PYTHON_KEYS = frozenset({"path", "sha256"})
BINARY_SEED_KEYS = frozenset(
    {
        "repository",
        "revision",
        "path",
        "size",
        "sha256",
        "uncompressed_path",
        "uncompressed_size",
        "uncompressed_sha256",
        "format",
        "prefix",
    }
)
BINARY_SEED_TRANSFORMATION_KEYS = frozenset(
    {
        "removed_paths",
        "removed_links",
        "source_shebang",
        "target_shebang",
        "source_pyvenv_sha256",
        "target_pyvenv",
        "remove_bytecode",
        "rewrite_prefix_paths",
        "source_prefix",
        "target_prefix",
        "allowed_pth",
        "allowed_workspace_files",
        "ignored_archive_members",
    }
)
BINARY_SEED_REMOVED_LINK_KEYS = frozenset({"path", "target"})
BINARY_SEED_PTH_KEYS = frozenset({"path", "sha256"})
BINARY_SEED_WORKSPACE_FILE_KEYS = frozenset({"path", "sha256"})
SEED_TRANSFORMATION_KEYS = frozenset(
    {
        "schema",
        "dependency_lock_sha256",
        "seed_sha256",
        "uncompressed_seed_sha256",
        "base_python_sha256",
        "files",
        "directories",
    }
)
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
        "device_name",
        "compute_capability",
        "minimum_host_driver_version",
        "nvlink_link_label",
        "nvlink_pairs",
        "compiled_extensions",
    }
)
COMPILED_EXTENSION_SPEC_KEYS = frozenset({"module"})
INVENTORY_KEYS = frozenset(
    {"schema", "runtime_root", "python_path", "distributions"}
)
BINARY_SEED_CUDA_SPEC_KEYS = CUDA_SPEC_KEYS | {"torch_distribution_version"}
BINARY_SEED_INVENTORY_KEYS = INVENTORY_KEYS | {
    "runtime_files",
    "runtime_directories",
}
INVENTORY_DISTRIBUTION_KEYS = frozenset(
    {"name", "version", "metadata_path", "direct_origin", "files"}
)
DIRECT_ORIGIN_KEYS = frozenset({"path", "sha256", "canonical_json"})
INVENTORY_FILE_KEYS = frozenset({"path", "size", "sha256"})
BINARY_SEED_RUNTIME_FILE_KEYS = frozenset(
    {"path", "mode", "mtime_ns", "size", "sha256"}
)
BINARY_SEED_RUNTIME_DIRECTORY_KEYS = frozenset({"path", "mode", "mtime_ns"})
PROBE_KEYS = frozenset({"schema", "argv", "expected_result"})
IMPORT_RESULT_KEYS = frozenset({"python", "imports"})
CUDA_RESULT_KEYS = frozenset(
    {
        "python",
        "torch",
        "gpu",
        "driver",
        "topology",
        "compute",
        "compiled_extensions",
    }
)
CUDA_TORCH_KEYS = frozenset({"version", "cuda_version"})
CUDA_GPU_KEYS = frozenset(
    {"available", "count", "device_names", "compute_capabilities"}
)
CUDA_DRIVER_KEYS = frozenset({"minimum_version", "compatible"})
CUDA_TOPOLOGY_KEYS = frozenset({"compatible", "link_label", "nvlink_pairs"})
CUDA_COMPUTE_KEYS = frozenset({"bf16_matmul"})
BF16_MATMUL_RESULT_KEYS = frozenset({"device_index", "passed"})
COMPILED_EXTENSION_RESULT_KEYS = frozenset({"module", "loaded"})

MAX_DISTRIBUTIONS = 4096
MAX_FILES = 250_000
MAX_FILE_SIZE = 16 * 1024 * 1024 * 1024
MAX_PATH_BYTES = 4096
MAX_IMPORTS = 1024
MAX_EXTENSIONS = 64
PROVENANCE_TIER_BINARY_SEED = "content-addressed-binary-seed"
B200_BASE_IMAGE_REFERENCE = (
    "ghcr.io/ethanelasky/verl-megatron-ssh@sha256:"
    "162f12d0ba6fbceaf56294d783551b2e83aefcc6e76b9138737acd358ee0baa8"
)
B200_BASE_PYTHON_PATH = "/usr/bin/python3.12"
B200_BASE_PYTHON_SHA256 = (
    "sha256:e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7"
)
B200_SEED_REPOSITORY = "ethanelasky/verl-b200-env"
B200_SEED_REVISION = "dd4c26f6cb38002a56d55556c6e622308394e400"
B200_SEED_PATH = "verl-b200-portable.tar.zst"
B200_SEED_SIZE = 3_891_912_437
B200_SEED_SHA256 = (
    "sha256:65ebafdd4e02e64887cb03740c9557dfcd17e98f8b8f720700c8166179fe62e9"
)
B200_SEED_UNCOMPRESSED_SIZE = 10_294_538_240
B200_SEED_UNCOMPRESSED_PATH = "verl-b200-portable.tar"
B200_SEED_UNCOMPRESSED_SHA256 = (
    "sha256:171b9b1c8ccd7e5a28b8a919e16e268d5b60cb7a34186ff78e4f7a15f5fa9c59"
)
B200_SEED_PREFIX = "workspace/envs/verl-b200"
B200_IGNORED_ARCHIVE_MEMBERS = ("workspace/uv/python",)
B200_ALLOWED_UNCLAIMED_SITE_FILES = (
    "python/lib/python3.12/site-packages/_virtualenv.pth",
    "python/lib/python3.12/site-packages/_virtualenv.py",
)
B200_TORCH_DISTRIBUTION_VERSION = "2.11.0"
B200_TORCH_LIVE_VERSION = "2.11.0+cu130"
RUNTIME_FILE_MODE = 0o444
RUNTIME_EXECUTABLE_MODE = 0o555
RUNTIME_DIRECTORY_MODE = 0o555
RUNTIME_MTIME_NS = 0
EXACT_DEVICE_COUNT = 2
EXACT_DEVICE_NAME = "NVIDIA B200"
EXACT_COMPUTE_CAPABILITY = [10, 0]
EXACT_TORCH_CUDA_VERSION = "13.0"
MINIMUM_HOST_DRIVER_VERSION = 580
NVIDIA_SMI = "/usr/bin/nvidia-smi"
NVIDIA_SMI_DRIVER_ARGS = (
    NVIDIA_SMI,
    "--query-gpu=driver_version",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_TOPOLOGY_ARGS = (NVIDIA_SMI, "topo", "-m")
EXACT_NVLINK_PAIRS = [[0, 1]]
EXACT_NVLINK_LINK_LABEL = "NV18"

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
import re
import subprocess
import sys
import torch
config=json.loads(sys.argv[1])
extensions=[]
for expected in config["compiled_extensions"]:
    importlib.import_module(expected["module"])
    extensions.append({"loaded":True,"module":expected["module"]})
count=torch.cuda.device_count()
bf16_matmul=[]
for expected in config["compute"]["bf16_matmul"]:
    index=expected["device_index"]
    left=torch.tensor([[1,2],[3,4]],dtype=torch.bfloat16,device=f"cuda:{index}")
    right=torch.tensor([[5,6],[7,8]],dtype=torch.bfloat16,device=f"cuda:{index}")
    product=torch.matmul(left,right)
    torch.cuda.synchronize(index)
    passed=product.dtype==torch.bfloat16 and product.device.type=="cuda" and product.device.index==index and list(product.shape)==[2,2] and product.tolist()==[[19.0,22.0],[43.0,50.0]]
    bf16_matmul.append({"device_index":index,"passed":passed})
driver=subprocess.run(["/usr/bin/nvidia-smi","--query-gpu=driver_version","--format=csv,noheader,nounits"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="ascii",env={"LC_ALL":"C"})
driver_versions=driver.stdout.splitlines()
minimum_driver=config["driver"]["minimum_version"]
driver_compatible=driver.stderr=="" and len(driver_versions)==count and all(re.fullmatch(r"[0-9]+(?:[.][0-9]+){1,3}",version) is not None and int(version.split(".",1)[0])>=minimum_driver for version in driver_versions)
topology=subprocess.run(["/usr/bin/nvidia-smi","topo","-m"],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding="ascii",env={"LC_ALL":"C"})
topology_lines=[line.split() for line in topology.stdout.splitlines() if line.split()]
topology_headers=[tokens for tokens in topology_lines if tokens[:2]==["GPU0","GPU1"] and (len(tokens)==2 or re.fullmatch(r"GPU[0-9]+",tokens[2]) is None)]
gpu_row_tokens=[tokens for tokens in topology_lines if re.fullmatch(r"GPU[0-9]+",tokens[0]) is not None and "X" in tokens[1:]]
gpu_rows={tokens[0]:tokens for tokens in gpu_row_tokens}
nvlink_pairs=config["topology"]["nvlink_pairs"]
nvlink_link_label=config["topology"]["link_label"]
topology_compatible=topology.stderr=="" and len(topology_headers)==1 and len(gpu_row_tokens)==count and set(gpu_rows)=={"GPU0","GPU1"} and all(len(row)>=count+1 for row in gpu_rows.values()) and all(gpu_rows[f"GPU{index}"][index+1]=="X" for index in range(count)) and all(gpu_rows[f"GPU{source}"][target+1]==nvlink_link_label and gpu_rows[f"GPU{target}"][source+1]==nvlink_link_label for source,target in nvlink_pairs)
result={"compiled_extensions":extensions,"compute":{"bf16_matmul":bf16_matmul},"driver":{"compatible":driver_compatible,"minimum_version":minimum_driver},"gpu":{"available":torch.cuda.is_available(),"compute_capabilities":[list(torch.cuda.get_device_capability(index)) for index in range(count)],"count":count,"device_names":[torch.cuda.get_device_name(index) for index in range(count)]},"python":{"path":sys.executable,"version":platform.python_version()},"topology":{"compatible":topology_compatible,"link_label":nvlink_link_label,"nvlink_pairs":nvlink_pairs},"torch":{"cuda_version":torch.version.cuda,"version":torch.__version__}}
sys.stdout.write(json.dumps(result,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False))"""
