from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import textwrap
import uuid
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CKPT_SYNC = ROOT / "scripts" / "ckpt_sync.sh"
POD_RUN = ROOT / "scripts" / "pod_run.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fake_commands(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "stat",
        "#!/usr/bin/env bash\n"
        "if [ \"${1:-}\" = -c ] && [ \"${2:-}\" = %Y ]; then echo 0; exit 0; fi\n"
        "exec /usr/bin/stat \"$@\"\n",
    )
    _write_executable(
        fake_bin / "flock",
        "#!/usr/bin/env bash\n"
        "if [ -n \"${FAKE_SWAP_PID_DURING_FLOCK:-}\" ]; then\n"
        "  mv \"$CKPT_SYNC_PID_FILE\" \"$FAKE_SWAP_PID_DURING_FLOCK\"\n"
        "  printf '%s\\n' 'unrelated replacement bytes' > \"$CKPT_SYNC_PID_FILE\"\n"
        "  chmod 600 \"$CKPT_SYNC_PID_FILE\"\n"
        "fi\n"
        'exit "${FAKE_FLOCK_RC:-0}"\n',
    )
    return fake_bin


def _short_write_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "short-write-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "real_write = os.write\n"
        "def short_write(fd, data):\n"
        "    caller = sys._getframe(1).f_code.co_filename\n"
        "    if caller == '<stdin>' and len(data) > 3:\n"
        "        return real_write(fd, data[:3])\n"
        "    return real_write(fd, data)\n"
        "os.write = short_write\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _wrong_euid_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "wrong-euid-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "real_euid = os.geteuid()\n"
        "os.geteuid = lambda: real_euid + 1\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _credential_swap_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "credential-swap-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "if 'def load_credential_file' in source:\n"
        "    real_open = os.open\n"
        "    swapped = False\n"
        "    target = os.environ['FAKE_SWAP_CREDENTIAL_PATH']\n"
        "    original = os.environ['FAKE_SWAP_CREDENTIAL_ORIGINAL']\n"
        "    def swapping_open(path, flags, *args, **kwargs):\n"
        "        global swapped\n"
        "        if not swapped and str(path) == target:\n"
        "            swapped = True\n"
        "            os.replace(target, original)\n"
        "            if os.environ.get('FAKE_SWAP_CREDENTIAL_KIND') == 'fifo':\n"
        "                os.mkfifo(target, mode=0o600)\n"
        "            else:\n"
        "                pathlib.Path(target).write_bytes(b'AWS_ACCESS_KEY_ID=replacement\\n')\n"
        "                pathlib.Path(target).chmod(0o600)\n"
        "        return real_open(path, flags, *args, **kwargs)\n"
        "    os.open = swapping_open\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _foreign_leaf_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "foreign-leaf-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "if 'coordination paths must be pairwise distinct' in source:\n"
        "    real_lstat = os.lstat\n"
        "    target = os.environ['FAKE_FOREIGN_ROLE_PATH']\n"
        "    def foreign_lstat(path, *args, **kwargs):\n"
        "        observed = real_lstat(path, *args, **kwargs)\n"
        "        if str(path) != target:\n"
        "            return observed\n"
        "        fields = list(observed)\n"
        "        fields[4] = os.geteuid() + 1\n"
        "        return os.stat_result(fields)\n"
        "    os.lstat = foreign_lstat\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _fsync_audit_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "fsync-audit-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import stat\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "real_open = os.open\n"
        "real_close = os.close\n"
        "real_fsync = os.fsync\n"
        "fd_paths = {}\n"
        "def audited_open(path, flags, *args, **kwargs):\n"
        "    fd = real_open(path, flags, *args, **kwargs)\n"
        "    fd_paths[fd] = str(path)\n"
        "    return fd\n"
        "def audited_close(fd):\n"
        "    try:\n"
        "        return real_close(fd)\n"
        "    finally:\n"
        "        fd_paths.pop(fd, None)\n"
        "def audited_fsync(fd):\n"
        "    mode = os.fstat(fd).st_mode\n"
        "    kind = 'DIR' if stat.S_ISDIR(mode) else 'FILE' if stat.S_ISREG(mode) else 'OTHER'\n"
        "    with open(os.environ['FAKE_FSYNC_AUDIT_LOG'], 'a', encoding='utf-8') as log:\n"
        "        log.write(kind + '\\t' + fd_paths.get(fd, '<inherited>') + '\\n')\n"
        "    return real_fsync(fd)\n"
        "os.open = audited_open\n"
        "os.close = audited_close\n"
        "os.fsync = audited_fsync\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _fail_role_publish_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "fail-role-publish-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "role = os.environ['FAKE_FAIL_ROLE_PUBLISH']\n"
        "if f'.{role}-init-v1-' in source:\n"
        "    def fail_rename(source_path, target_path):\n"
        "        raise RuntimeError('injected failure before canonical role publish')\n"
        "    os.rename = fail_rename\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _pid_prepublication_swap_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "pid-prepublication-swap-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "if 'pid canonical role changed before publish' in source:\n"
        "    real_lstat = os.lstat\n"
        "    target = os.environ['FAKE_SWAP_PID_PREPUBLISH_PATH']\n"
        "    original = os.environ['FAKE_SWAP_PID_PREPUBLISH_ORIGINAL']\n"
        "    calls = 0\n"
        "    def swapping_lstat(path, *args, **kwargs):\n"
        "        global calls\n"
        "        if str(path) == target:\n"
        "            calls += 1\n"
        "            if calls == 2:\n"
        "                os.replace(target, original)\n"
        "                pathlib.Path(target).write_bytes(b'unrelated replacement bytes\\n')\n"
        "                pathlib.Path(target).chmod(0o600)\n"
        "        return real_lstat(path, *args, **kwargs)\n"
        "    os.lstat = swapping_lstat\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _midloop_state_swap_python(tmp_path: Path) -> Path:
    wrapper = tmp_path / "midloop-state-swap-python"
    _write_executable(
        wrapper,
        f"#!{sys.executable}\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "source = sys.stdin.read()\n"
        "if 'invalid checkpoint step directory name' in source:\n"
        "    target = pathlib.Path(os.environ['FAKE_MIDLOOP_STATE_PATH'])\n"
        "    original = pathlib.Path(os.environ['FAKE_MIDLOOP_STATE_ORIGINAL'])\n"
        "    target.replace(original)\n"
        "    target.write_bytes(b'unrelated replacement state bytes\\n')\n"
        "    target.chmod(0o600)\n"
        "exec(compile(source, '<stdin>', 'exec'), {'__name__': '__main__'})\n",
    )
    return wrapper


def _fake_s3_modules(fake_modules: Path) -> None:
    botocore = fake_modules / "botocore"
    botocore.mkdir()
    (botocore / "__init__.py").write_text("", encoding="utf-8")
    (botocore / "exceptions.py").write_text(
        textwrap.dedent(
            """
            class ClientError(Exception):
                def __init__(self, code):
                    self.response = {"Error": {"Code": code}}
                    super().__init__(code)

            class ParamValidationError(Exception):
                pass
            """
        ),
        encoding="utf-8",
    )
    (botocore / "config.py").write_text(
        textwrap.dedent(
            """
            class Config:
                def __init__(self, *, retries):
                    self.retries = dict(retries)
            """
        ),
        encoding="utf-8",
    )
    (botocore / "utils.py").write_text(
        textwrap.dedent(
            """
            class S3RegionRedirector:
                def redirect_from_error(self, **kwargs):
                    return None

            class S3RegionRedirectorv2:
                def redirect_from_error(self, **kwargs):
                    return None
            """
        ),
        encoding="utf-8",
    )
    (fake_modules / "boto3.py").write_text(
        textwrap.dedent(
            """
            import base64
            import hashlib
            import io
            import json
            import os
            import pathlib
            import time
            from types import SimpleNamespace
            from botocore.exceptions import ClientError
            from botocore.utils import S3RegionRedirectorv2

            def journal_provider_call(operation):
                path = os.environ.get("FAKE_PROVIDER_CALL_LOG")
                if path:
                    with open(path, "a", encoding="utf-8") as output:
                        output.write(operation + "\\n")

            class FakeHandlerTrie:
                def __init__(self, retry_handlers):
                    self.retry_handlers = retry_handlers
                    self.event_handlers = {}

                def prefix_search(self, event_name):
                    if event_name == "needs-retry.s3":
                        return list(self.retry_handlers)
                    return list(self.event_handlers.get(event_name, []))

            class FakeEvents:
                def __init__(self):
                    redirect_shape = os.environ.get("FAKE_REDIRECT_SHAPE", "one")
                    redirects = {
                        "none": [],
                        "one": [S3RegionRedirectorv2().redirect_from_error],
                        "duplicate": [
                            S3RegionRedirectorv2().redirect_from_error,
                            S3RegionRedirectorv2().redirect_from_error,
                        ],
                    }[redirect_shape]
                    self._emitter = SimpleNamespace(
                        _handlers=FakeHandlerTrie(
                            [lambda **kwargs: None, *redirects]
                        )
                    )

                def unregister(self, event_name, *, handler):
                    if event_name == "needs-retry.s3":
                        self._emitter._handlers.retry_handlers.remove(handler)
                    else:
                        self._emitter._handlers.event_handlers[event_name].remove(handler)

                def register_first(self, event_name, handler):
                    self._emitter._handlers.event_handlers.setdefault(
                        event_name, []
                    ).insert(0, handler)

            class FakeS3:
                def __init__(self, client_kwargs):
                    self.list_calls = 0
                    self.client_kwargs = client_kwargs
                    self.meta = SimpleNamespace(events=FakeEvents())
                    expected_credentials = json.loads(
                        os.environ.get("FAKE_EXPECTED_CREDENTIAL_DIGESTS", "{}")
                    )
                    for name, expected_digest in expected_credentials.items():
                        observed = os.environ.get(name)
                        if observed is None or hashlib.sha256(
                            observed.encode("utf-8")
                        ).hexdigest() != expected_digest:
                            raise RuntimeError(f"missing or incorrect credential: {name}")
                    self.credential_names = sorted(expected_credentials)
                    store = os.environ.get("FAKE_S3_STORE")
                    self.store = pathlib.Path(store) if store else None
                    if self.store is not None:
                        self.store.mkdir(parents=True, exist_ok=True)
                    raw = json.loads(os.environ.get("FAKE_S3_EXISTING", "{}"))
                    self.objects = {}
                    for key, body in raw.items():
                        encoded = body.encode("utf-8")
                        self._create(
                            key,
                            encoded,
                            len(encoded),
                            {"sha256": hashlib.sha256(encoded).hexdigest()},
                            exclusive=False,
                        )

                def _path(self, key):
                    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
                    return self.store / f"{digest}.json"

                def _assert_transport_budget(self, operation_name):
                    if os.environ.get("FAKE_REQUIRE_TRANSPORT_BUDGET", "1") == "0":
                        return
                    event_name = f"before-send.s3.{operation_name}"
                    handlers = self.meta.events._emitter._handlers.prefix_search(
                        event_name
                    )
                    guards = [
                        handler
                        for handler in handlers
                        if getattr(handler, "__name__", None)
                        == "transport_budget_guard"
                    ]
                    if len(guards) != 1:
                        raise RuntimeError(
                            f"missing exact transport budget for {operation_name}"
                        )
                    guards[0](request=None)

                def _record(self, key, body, content_length, metadata):
                    return {
                        "Key": key,
                        "BodyBase64": base64.b64encode(body).decode("ascii"),
                        "ContentLength": content_length,
                        "Metadata": dict(metadata),
                    }

                def _create(self, key, body, content_length, metadata, *, exclusive=True):
                    record = self._record(key, body, content_length, metadata)
                    if self.store is None:
                        if exclusive and key in self.objects:
                            raise ClientError("PreconditionFailed")
                        self.objects[key] = record
                        return
                    target = self._path(key)
                    flags = os.O_WRONLY | os.O_CREAT
                    flags |= os.O_EXCL if exclusive else os.O_TRUNC
                    try:
                        fd = os.open(target, flags, 0o600)
                    except FileExistsError:
                        raise ClientError("PreconditionFailed") from None
                    payload = json.dumps(record, sort_keys=True).encode("utf-8")
                    try:
                        os.write(fd, payload)
                        os.fsync(fd)
                    finally:
                        os.close(fd)

                def _read_shared_record(self, path):
                    # O_EXCL publishes the claim before its tiny JSON body is
                    # fully written. Wait only for that bounded in-progress
                    # fake-client write; production S3 returns whole objects.
                    deadline = time.monotonic() + 2
                    while True:
                        try:
                            return json.loads(path.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, OSError):
                            if time.monotonic() >= deadline:
                                raise
                            time.sleep(0.001)

                def _all(self):
                    if self.store is None:
                        return dict(self.objects)
                    records = {}
                    for path in self.store.glob("*.json"):
                        record = self._read_shared_record(path)
                        records[record["Key"]] = record
                    return records

                def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
                    self._assert_transport_budget("ListObjectsV2")
                    journal_provider_call("LIST")
                    self.list_calls += 1
                    snapshot = self._all()
                    keys = sorted(key for key in snapshot if key.startswith(Prefix))
                    pagination = os.environ.get("FAKE_S3_PAGINATION")
                    pagination_start = int(
                        os.environ.get("FAKE_S3_PAGINATION_START_CALL", "1")
                    )
                    if pagination and self.list_calls >= pagination_start:
                        page_key = (
                            f"{Prefix}duplicate-page-key"
                            if pagination == "duplicate-key"
                            else f"{Prefix}synthetic-page-{self.list_calls}"
                        )
                        response = {
                            "Contents": []
                            if pagination == "empty-page"
                            else [{"Key": page_key, "Size": 1}],
                            "IsTruncated": True,
                        }
                        if pagination == "missing-token":
                            return response
                        response["NextContinuationToken"] = (
                            ""
                            if pagination == "empty-token"
                            else f"token-{self.list_calls}"
                            if pagination == "novel-overflow"
                            else "stuck-token"
                        )
                        return response
                    swap_target = os.environ.get("FAKE_SWAP_STATE_AFTER_FIRST_LIST")
                    if swap_target and self.list_calls == 1:
                        state_path = pathlib.Path(os.environ["CKPT_SYNC_STATE"])
                        original = pathlib.Path(swap_target)
                        content = state_path.read_bytes()
                        state_path.replace(original)
                        state_path.write_bytes(content)
                        state_path.chmod(0o600)
                    barrier = os.environ.get("FAKE_S3_INITIAL_LIST_BARRIER")
                    if barrier and self.list_calls == 1:
                        barrier_path = pathlib.Path(barrier)
                        barrier_path.mkdir(parents=True, exist_ok=True)
                        client_id = os.environ["FAKE_S3_CLIENT_ID"]
                        (barrier_path / client_id).write_text("ready", encoding="utf-8")
                        deadline = time.monotonic() + 5
                        while len(list(barrier_path.iterdir())) < 2:
                            if time.monotonic() >= deadline:
                                raise RuntimeError("fake initial-list barrier timed out")
                            time.sleep(0.005)
                    return {
                        "Contents": [
                            {"Key": key, "Size": snapshot[key]["ContentLength"]}
                            for key in keys
                        ],
                        "IsTruncated": False,
                        "KeyCount": len(keys),
                    }

                def head_object(self, *, Bucket, Key):
                    self._assert_transport_budget("HeadObject")
                    journal_provider_call("HEAD")
                    race_key = os.environ.get("FAKE_S3_HEAD_RACE_KEY")
                    objects = self._all()
                    if race_key == Key and Key not in objects:
                        body = os.environ.get("FAKE_S3_HEAD_RACE_BODY", "raced").encode("utf-8")
                        self._create(
                            Key,
                            body,
                            len(body),
                            {"sha256": hashlib.sha256(body).hexdigest()},
                        )
                        objects = self._all()
                    if Key in objects:
                        item = objects[Key]
                        return {
                            "ContentLength": item["ContentLength"],
                            "Metadata": dict(item["Metadata"]),
                        }
                    raise ClientError("NoSuchKey")

                def get_object(self, *, Bucket, Key):
                    self._assert_transport_budget("GetObject")
                    journal_provider_call("GET")
                    objects = self._all()
                    if Key not in objects:
                        raise ClientError("NoSuchKey")
                    item = objects[Key]
                    body = base64.b64decode(item["BodyBase64"])
                    metadata = dict(item["Metadata"])
                    size = item["ContentLength"]
                    if os.environ.get("FAKE_S3_CORRUPT_GET_KEY") == Key:
                        mode = os.environ.get("FAKE_S3_CORRUPT_GET_MODE", "body")
                        if mode == "body":
                            body += b"corrupt"
                        elif mode == "metadata":
                            metadata["sha256"] = "0" * 64
                        elif mode == "size":
                            size += 1
                        else:
                            raise RuntimeError("unknown fake corruption mode")
                    return {
                        "Body": io.BytesIO(body),
                        "ContentLength": size,
                        "Metadata": metadata,
                    }

                def put_object(self, **kwargs):
                    self._assert_transport_budget("PutObject")
                    journal_provider_call("PUT")
                    conflict_key = os.environ.get("FAKE_S3_PUT_CONFLICT_KEY")
                    if conflict_key == kwargs["Key"]:
                        raise ClientError("PreconditionFailed")
                    supplied_body = kwargs["Body"]
                    body = (
                        supplied_body.read()
                        if hasattr(supplied_body, "read")
                        else bytes(supplied_body)
                    )
                    self._create(
                        kwargs["Key"],
                        body,
                        kwargs["ContentLength"],
                        kwargs["Metadata"],
                    )
                    foreign_key = os.environ.get("FAKE_S3_FOREIGN_AFTER_PUT_KEY")
                    if foreign_key and not kwargs["Key"].endswith(
                        ".ckpt-sync-reservation-v1.json"
                    ):
                        foreign_body = b"foreign"
                        self._create(
                            foreign_key,
                            foreign_body,
                            len(foreign_body),
                            {
                                "sha256": hashlib.sha256(foreign_body).hexdigest()
                            },
                        )
                    record = {
                        "Bucket": kwargs["Bucket"],
                        "Key": kwargs["Key"],
                        "Endpoint": self.client_kwargs.get("endpoint_url"),
                        "Region": self.client_kwargs.get("region_name"),
                        "RetryConfig": self.client_kwargs["config"].retries,
                        "RegionRedirectHandlers": sum(
                            getattr(getattr(handler, "__func__", None), "__name__", None)
                            == "redirect_from_error"
                            for handler in self.meta.events._emitter._handlers.retry_handlers
                        ),
                        "CredentialNames": self.credential_names,
                        "IfNoneMatch": kwargs.get("IfNoneMatch"),
                        "ContentLength": kwargs["ContentLength"],
                        "Metadata": kwargs["Metadata"],
                        "BodyBase64": base64.b64encode(body).decode("ascii"),
                    }
                    with open(os.environ["FAKE_UPLOAD_LOG"], "a", encoding="utf-8") as out:
                        out.write(json.dumps(record) + "\\n")

            def client(service, **kwargs):
                assert service == "s3"
                journal_provider_call("CLIENT")
                return FakeS3(kwargs)
            """
        ),
        encoding="utf-8",
    )


def _sync_env(
    tmp_path: Path,
    checkpoint_dir: Path,
    namespace: str,
    fake_modules: Path,
    *,
    run_name: str | None = None,
) -> tuple[dict[str, str], Path]:
    fake_bin = _fake_commands(tmp_path)
    upload_log = tmp_path / "uploads.jsonl"
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("AWS_"):
            env.pop(name)
    env.update(
        {
            "CKPT_DIR": str(checkpoint_dir),
            "RUN_NAME": checkpoint_dir.parent.name if run_name is None else run_name,
            "DEBATE_LAUNCH_NAMESPACE": namespace,
            "PYBIN": sys.executable,
            "PYTHONPATH": str(fake_modules),
            "PATH": f"{fake_bin}{os.pathsep}{env['PATH']}",
            "QUIESCENT_SECS": "0",
            "INTERVAL": "0",
            "CKPT_SYNC_ONCE": "1",
            "CKPT_SYNC_STATE": str(tmp_path / "state"),
            "CKPT_SYNC_PID_FILE": str(tmp_path / "pid"),
            "CKPT_SYNC_LOCK_FILE": str(tmp_path / "lock"),
            "S3_ENV_FILE": str(tmp_path / "no-s3-env"),
            "CKPT_DESTINATION_JSON": json.dumps(
                {
                    "kind": "bucket",
                    "endpoint": "https://objects.example.test",
                    "region": "test-region-1",
                    "bucket": "test-bucket",
                    "prefix": "checkpoints",
                }
            ),
            "FAKE_UPLOAD_LOG": str(upload_log),
            "FAKE_PROVIDER_CALL_LOG": str(tmp_path / "provider-calls.log"),
        }
    )
    return env, upload_log


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _provider_journal(env: dict[str, str]) -> bytes:
    path = Path(env["FAKE_PROVIDER_CALL_LOG"])
    return path.read_bytes() if path.exists() else b""


def _state_identity_header(env: dict[str, str], checkpoint_dir: Path) -> str:
    destination = json.loads(env["CKPT_DESTINATION_JSON"])
    identity = {
        "safe_run": checkpoint_dir.parent.name,
        "namespace": env["DEBATE_LAUNCH_NAMESPACE"],
        "checkpoint_dir": str(checkpoint_dir),
        "destination": destination,
    }
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"identity-v1\t{hashlib.sha256(canonical).hexdigest()}\n"


def _attempt_identity_digest(env: dict[str, str], checkpoint_dir: Path) -> str:
    return _state_identity_header(env, checkpoint_dir).split("\t", 1)[1].strip()


def _lock_role(env: dict[str, str], checkpoint_dir: Path) -> str:
    return f"lock-v1\t{_attempt_identity_digest(env, checkpoint_dir)}\n"


def _pid_role(env: dict[str, str], checkpoint_dir: Path, pid: int = 999999) -> str:
    return f"pid-v1\t{_attempt_identity_digest(env, checkpoint_dir)}\t{pid}\n"


def _state_completion_record(checkpoint: Path) -> str:
    digest = hashlib.sha256(str(checkpoint).encode("utf-8")).hexdigest()
    return f"complete-v1\t{checkpoint.name}\t{digest}\n"


def _assert_state_has_only_identity(
    env: dict[str, str], checkpoint_dir: Path
) -> None:
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == (
        _state_identity_header(env, checkpoint_dir)
    )


def test_sync_refuses_missing_destination_even_with_ambient_bucket_credentials(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env.pop("CKPT_DESTINATION_JSON")
    env["AWS_ACCESS_KEY_ID"] = "ambient-credential-must-not-select-a-destination"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode != 0
    assert "set CKPT_DESTINATION_JSON from run submission config" in result.stderr
    assert "ambient-credential-must-not-select-a-destination" not in (
        result.stdout + result.stderr
    )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "destination",
    [
        "not-json",
        "[]",
        '{"kind":"bucket","kind":"local","directory":"/"}',
        json.dumps({"kind": "unknown"}),
        json.dumps(
            {
                "kind": "bucket",
                "endpoint": "https://objects.invalid",
                "region": "region-1",
                "bucket": "bucket",
                "prefix": "checkpoints",
                "access_key": "destination-must-not-carry-secrets",
            }
        ),
        json.dumps(
            {
                "kind": "bucket",
                "endpoint": "https://objects.invalid",
                "region": "region-1",
                "bucket": "bucket",
            }
        ),
    ],
)
def test_sync_refuses_malformed_duplicate_unknown_or_incomplete_destination(
    tmp_path: Path, destination: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = destination

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "invalid CKPT_DESTINATION_JSON" in result.stderr
    assert "destination-must-not-carry-secrets" not in (result.stdout + result.stderr)
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://objects.example.test",
        " https://objects.example.test",
        "https://objects.example.test ",
        "https:\\objects.example.test",
        "https:objects.example.test",
        "https:///missing-host",
        "https://objects.example.test/",
        "https://objects.example.test//",
        "https://OBJECTS.example.test",
        "https://objects.example.test:443",
        "https://bad host.example",
        "https://bad_host.example",
        "https://percent%20escape.example",
        "https://-leading.example",
        "https://trailing-.example",
        "https://empty..label",
        "https://unicode-é.example",
        "https://trailing.example.",
        "https://127.000.0.1",
        "https://[0:0:0:0:0:0:0:1]",
    ],
)
def test_bucket_destination_requires_an_exact_canonical_https_origin(
    tmp_path: Path, endpoint: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    destination = json.loads(env["CKPT_DESTINATION_JSON"])
    destination["endpoint"] = endpoint
    env["CKPT_DESTINATION_JSON"] = json.dumps(destination)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "invalid CKPT_DESTINATION_JSON" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://storage",
        "https://objects.example.test",
        "https://192.0.2.10:9443",
        "https://[2001:db8::1]:9443",
    ],
)
def test_bucket_destination_accepts_canonical_dns_ipv4_and_ipv6_origins(
    tmp_path: Path, endpoint: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    destination = json.loads(env["CKPT_DESTINATION_JSON"])
    destination["endpoint"] = endpoint
    env["CKPT_DESTINATION_JSON"] = json.dumps(destination)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert {record["Endpoint"] for record in _read_jsonl(upload_log)} == {endpoint}


@pytest.mark.parametrize("destination_at_checkpoint", [False, True])
def test_local_destination_noops_only_when_checkpoint_is_physically_contained(
    tmp_path: Path, destination_at_checkpoint: bool
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    launch = tmp_path / "durable" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    destination = launch if destination_at_checkpoint else tmp_path / "durable"
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(destination)}
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert "configured durable local destination; nothing to do" in result.stdout
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)
    assert Path(env["CKPT_SYNC_LOCK_FILE"]).read_text(encoding="utf-8") == _lock_role(
        env, launch
    )
    pid_fields = Path(env["CKPT_SYNC_PID_FILE"]).read_text(encoding="utf-8").split("\t")
    assert pid_fields[:2] == ["pid-v1", _attempt_identity_digest(env, launch)]
    assert _provider_journal(env) == b""


def test_local_destination_refuses_checkpoint_outside_and_prefix_sibling(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    destination = tmp_path / "durable"
    destination.mkdir()
    launch = tmp_path / "durable-other" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(destination)}
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "outside configured local destination" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("unsafe_kind", ["symlink", "hardlink", "fifo", "socket"])
def test_local_destination_refuses_unsafe_existing_tree_before_success(
    tmp_path: Path, unsafe_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    durable = tmp_path / "durable"
    launch = durable / "checkpoints" / "run" / "attempt-a"
    final = launch / "final"
    final.mkdir(parents=True)
    outside = tmp_path / "outside-artifact"
    outside.write_bytes(b"must-remain-unchanged")
    unsafe = final / "unsafe"
    if unsafe_kind == "symlink":
        unsafe.symlink_to(outside)
    elif unsafe_kind == "hardlink":
        os.link(outside, unsafe)
    elif unsafe_kind == "fifo":
        os.mkfifo(unsafe)
    else:
        previous_cwd = Path.cwd()
        os.chdir(final)
        try:
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix_socket.bind(unsafe.name)
            unix_socket.close()
        finally:
            os.chdir(previous_cwd)
    before = outside.read_bytes()
    before_stat = outside.stat()
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(durable)}
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "unsafe existing local checkpoint tree" in result.stderr
    assert "nothing to do" not in result.stdout
    assert outside.read_bytes() == before
    after_stat = outside.stat()
    assert (
        after_stat.st_ino,
        after_stat.st_mode,
        after_stat.st_size,
        after_stat.st_mtime_ns,
    ) == (
        before_stat.st_ino,
        before_stat.st_mode,
        before_stat.st_size,
        before_stat.st_mtime_ns,
    )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""
    assert not Path(env["CKPT_SYNC_STATE"]).exists()
    assert not Path(env["CKPT_SYNC_PID_FILE"]).exists()
    assert not Path(env["CKPT_SYNC_LOCK_FILE"]).exists()


@pytest.mark.parametrize("path_form", ["symlink", "noncanonical"])
def test_local_destination_requires_an_exact_canonical_non_symlink_directory(
    tmp_path: Path, path_form: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    durable = tmp_path / "durable"
    launch = durable / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    if path_form == "symlink":
        configured = tmp_path / "durable-link"
        configured.symlink_to(durable, target_is_directory=True)
    else:
        configured = durable / ".." / "durable"
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(configured)}
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "invalid CKPT_DESTINATION_JSON" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("wrong_part", ["namespace", "run"])
def test_sync_rejects_a_checkpoint_dir_inconsistent_with_exact_identity(
    tmp_path: Path, wrong_part: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"final")
    namespace = "attempt-b" if wrong_part == "namespace" else "attempt-a"
    run_name = "other-run" if wrong_part == "run" else "run"
    env, upload_log = _sync_env(
        tmp_path, launch, namespace, fake_modules, run_name=run_name
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "CKPT_DIR must be exactly" in result.stderr
    assert not upload_log.exists()


def test_same_namespace_lock_contention_refuses_a_second_synchronizer(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"final")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_FLOCK_RC"] = "1"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 5
    assert "another synchronizer already owns namespace attempt-a" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "coord_scenario",
    [
        "lock_is_adapter",
        "lock_symlink_adapter",
        "lock_symlink_state",
        "state_symlink_adapter",
        "pid_is_adapter",
        "pid_symlink_adapter",
        "state_pid_alias",
        "state_lock_alias",
        "pid_lock_alias",
        "lock_wrong_mode",
        "pid_wrong_mode",
        "lock_parent_symlink",
        "lock_inside_checkpoint",
        "pid_inside_checkpoint",
        "state_inside_checkpoint",
    ],
)
def test_unsafe_coordination_aliases_refuse_without_mutating_targets(
    tmp_path: Path, coord_scenario: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    adapter = launch / "final" / "adapter.bin"
    adapter.write_bytes(b"original-adapter-bytes")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    created_aliases: list[Path] = []
    extra_targets: list[Path] = []
    if coord_scenario == "lock_is_adapter":
        env["CKPT_SYNC_LOCK_FILE"] = str(adapter)
    elif coord_scenario == "lock_symlink_adapter":
        alias = tmp_path / "lock-alias"
        alias.symlink_to(adapter)
        created_aliases.append(alias)
        env["CKPT_SYNC_LOCK_FILE"] = str(alias)
    elif coord_scenario == "lock_symlink_state":
        target = tmp_path / "state-target"
        target.write_bytes(b"original-state-bytes")
        target.chmod(0o600)
        alias = tmp_path / "lock-alias"
        alias.symlink_to(target)
        extra_targets.append(target)
        created_aliases.append(alias)
        env["CKPT_SYNC_LOCK_FILE"] = str(alias)
    elif coord_scenario == "state_symlink_adapter":
        alias = tmp_path / "state-alias"
        alias.symlink_to(adapter)
        created_aliases.append(alias)
        env["CKPT_SYNC_STATE"] = str(alias)
    elif coord_scenario == "pid_is_adapter":
        env["CKPT_SYNC_PID_FILE"] = str(adapter)
    elif coord_scenario == "pid_symlink_adapter":
        alias = tmp_path / "pid-alias"
        alias.symlink_to(adapter)
        created_aliases.append(alias)
        env["CKPT_SYNC_PID_FILE"] = str(alias)
    elif coord_scenario == "state_pid_alias":
        env["CKPT_SYNC_PID_FILE"] = env["CKPT_SYNC_STATE"]
    elif coord_scenario == "state_lock_alias":
        env["CKPT_SYNC_LOCK_FILE"] = env["CKPT_SYNC_STATE"]
    elif coord_scenario == "pid_lock_alias":
        env["CKPT_SYNC_LOCK_FILE"] = env["CKPT_SYNC_PID_FILE"]
    elif coord_scenario == "lock_wrong_mode":
        target = tmp_path / "wrong-mode-lock"
        target.write_bytes(b"original-lock")
        target.chmod(0o644)
        extra_targets.append(target)
        env["CKPT_SYNC_LOCK_FILE"] = str(target)
    elif coord_scenario == "pid_wrong_mode":
        target = tmp_path / "wrong-mode-pid"
        target.write_bytes(b"original-pid")
        target.chmod(0o644)
        extra_targets.append(target)
        env["CKPT_SYNC_PID_FILE"] = str(target)
    elif coord_scenario == "lock_parent_symlink":
        real_parent = tmp_path / "real-coord-parent"
        real_parent.mkdir()
        parent_alias = tmp_path / "coord-parent-alias"
        parent_alias.symlink_to(real_parent, target_is_directory=True)
        created_aliases.append(parent_alias)
        env["CKPT_SYNC_LOCK_FILE"] = str(parent_alias / "lock")
    elif coord_scenario == "lock_inside_checkpoint":
        env["CKPT_SYNC_LOCK_FILE"] = str(launch / "coord.lock")
    elif coord_scenario == "pid_inside_checkpoint":
        env["CKPT_SYNC_PID_FILE"] = str(launch / "coord.pid")
    else:
        env["CKPT_SYNC_STATE"] = str(launch / "coord.state")
    before_adapter = adapter.read_bytes()
    before_adapter_stat = adapter.stat()
    before_alias_stats = {path: path.lstat() for path in created_aliases}
    before_target_data = {path: path.read_bytes() for path in extra_targets}
    before_target_stats = {path: path.stat() for path in extra_targets}

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "unsafe coordination path configuration" in result.stderr
    assert adapter.read_bytes() == before_adapter
    after_adapter_stat = adapter.stat()
    assert (
        after_adapter_stat.st_ino,
        after_adapter_stat.st_mode,
        after_adapter_stat.st_size,
        after_adapter_stat.st_mtime_ns,
    ) == (
        before_adapter_stat.st_ino,
        before_adapter_stat.st_mode,
        before_adapter_stat.st_size,
        before_adapter_stat.st_mtime_ns,
    )
    for path, before in before_alias_stats.items():
        after = path.lstat()
        assert (after.st_ino, after.st_mode, after.st_size) == (
            before.st_ino,
            before.st_mode,
            before.st_size,
        )
    for path, before in before_target_stats.items():
        after = path.stat()
        assert path.read_bytes() == before_target_data[path]
        assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_existing_owner_only_lock_is_not_truncated_and_stale_pid_is_safely_updated(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, _ = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    lock_path = Path(env["CKPT_SYNC_LOCK_FILE"])
    pid_path = Path(env["CKPT_SYNC_PID_FILE"])
    valid_lock = _lock_role(env, launch)
    stale_pid = _pid_role(env, launch)
    lock_path.write_text(valid_lock, encoding="utf-8")
    pid_path.write_text(stale_pid, encoding="utf-8")
    lock_path.chmod(0o600)
    pid_path.chmod(0o600)
    lock_inode = lock_path.stat().st_ino

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert lock_path.read_text(encoding="utf-8") == valid_lock
    assert lock_path.stat().st_ino == lock_inode
    pid_fields = pid_path.read_text(encoding="utf-8").strip().split("\t")
    assert pid_fields[:2] == ["pid-v1", _attempt_identity_digest(env, launch)]
    assert pid_fields[2].isdigit()
    assert pid_path.read_text(encoding="utf-8") != stale_pid
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(pid_path.stat().st_mode) == 0o600
    first_journal = _provider_journal(env)

    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert second.returncode == 0, second.stderr
    assert lock_path.read_text(encoding="utf-8") == valid_lock
    assert _provider_journal(env) == first_journal


@pytest.mark.parametrize("parent_mode", [0o755, 0o770])
def test_coordination_parent_must_be_private_and_is_never_mutated(
    tmp_path: Path, parent_mode: int
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    coord_parent = tmp_path / "untrusted-coordination-parent"
    coord_parent.mkdir(mode=parent_mode)
    coord_parent.chmod(parent_mode)
    env["CKPT_SYNC_STATE"] = str(coord_parent / "state")
    env["CKPT_SYNC_PID_FILE"] = str(coord_parent / "pid")
    env["CKPT_SYNC_LOCK_FILE"] = str(coord_parent / "lock")
    before = coord_parent.stat()

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    after = coord_parent.stat()
    assert result.returncode == 2
    assert "parent must be owned by euid with mode 0700" in result.stderr
    assert list(coord_parent.iterdir()) == []
    assert (after.st_ino, after.st_mode, after.st_mtime_ns) == (
        before.st_ino,
        before.st_mode,
        before.st_mtime_ns,
    )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_coordination_parent_and_leaves_must_match_effective_uid(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["PYBIN"] = str(_wrong_euid_python(tmp_path))

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "parent must be owned by euid with mode 0700" in result.stderr
    assert not Path(env["CKPT_SYNC_STATE"]).exists()
    assert not Path(env["CKPT_SYNC_PID_FILE"]).exists()
    assert not Path(env["CKPT_SYNC_LOCK_FILE"]).exists()
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("role", ["state", "pid", "lock"])
@pytest.mark.parametrize("content_kind", ["empty", "arbitrary", "wrong_role", "mismatch"])
def test_existing_coordination_leaf_requires_exact_role_and_attempt_identity(
    tmp_path: Path, role: str, content_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    role_paths = {
        "state": Path(env["CKPT_SYNC_STATE"]),
        "pid": Path(env["CKPT_SYNC_PID_FILE"]),
        "lock": Path(env["CKPT_SYNC_LOCK_FILE"]),
    }
    digest = _attempt_identity_digest(env, launch)
    contents = {
        "empty": b"",
        "arbitrary": b"unrelated owner-owned bytes\n",
        "wrong_role": {
            "state": _lock_role(env, launch).encode(),
            "pid": _state_identity_header(env, launch).encode(),
            "lock": _pid_role(env, launch).encode(),
        }[role],
        "mismatch": {
            "state": f"identity-v1\t{'0' * 64}\n".encode(),
            "pid": f"pid-v1\t{'0' * 64}\t999999\n".encode(),
            "lock": f"lock-v1\t{'0' * 64}\n".encode(),
        }[role],
    }
    target = role_paths[role]
    target.write_bytes(contents[content_kind])
    target.chmod(0o600)
    before = target.stat()
    before_bytes = target.read_bytes()

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    after = target.stat()
    assert digest not in result.stdout
    assert result.returncode == 2
    assert "unsafe coordination path configuration" in result.stderr
    assert target.read_bytes() == before_bytes
    assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("foreign_role", ["state", "pid", "lock"])
def test_each_coordination_leaf_owner_check_is_independently_enforced(
    tmp_path: Path, foreign_role: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    paths = {
        "state": Path(env["CKPT_SYNC_STATE"]),
        "pid": Path(env["CKPT_SYNC_PID_FILE"]),
        "lock": Path(env["CKPT_SYNC_LOCK_FILE"]),
    }
    contents = {
        "state": _state_identity_header(env, launch),
        "pid": _pid_role(env, launch),
        "lock": _lock_role(env, launch),
    }
    for role, path in paths.items():
        path.write_text(contents[role], encoding="utf-8")
        path.chmod(0o600)
    before = {
        role: (path.read_bytes(), path.stat()) for role, path in paths.items()
    }
    env["PYBIN"] = str(_foreign_leaf_python(tmp_path))
    env["FAKE_FOREIGN_ROLE_PATH"] = str(paths[foreign_role])

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "existing " + foreign_role + " leaf must be regular" in result.stderr
    for role, path in paths.items():
        before_bytes, before_stat = before[role]
        after = path.stat()
        assert path.read_bytes() == before_bytes
        assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
            before_stat.st_ino,
            before_stat.st_mode,
            before_stat.st_size,
            before_stat.st_mtime_ns,
        )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_new_coordination_roles_fsync_each_file_and_its_parent_directory(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    launch = tmp_path / "durable" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(tmp_path / "durable")}
    )
    audit_log = tmp_path / "fsync-audit.log"
    env["PYBIN"] = str(_fsync_audit_python(tmp_path))
    env["FAKE_FSYNC_AUDIT_LOG"] = str(audit_log)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    records = audit_log.read_text(encoding="utf-8").splitlines()
    parent_record = f"DIR\t{tmp_path}"
    lock_prefix = f"FILE\t{tmp_path}/.lock.lock-init-v1-"
    state_prefix = f"FILE\t{tmp_path}/.state.state-init-v1-"
    pid_prefix = f"FILE\t{tmp_path}/.pid.pid-init-v1-"
    lock_index = next(i for i, line in enumerate(records) if line.startswith(lock_prefix))
    state_index = next(i for i, line in enumerate(records) if line.startswith(state_prefix))
    pid_index = next(i for i, line in enumerate(records) if line.startswith(pid_prefix))
    assert records[lock_index + 1] == parent_record
    assert records[state_index + 1] == parent_record
    assert records[pid_index + 1] == parent_record
    assert lock_index < state_index < pid_index
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("failed_role", ["lock", "state", "pid"])
def test_failed_first_role_publish_leaves_only_temp_evidence_and_reruns_cleanly(
    tmp_path: Path, failed_role: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    launch = tmp_path / "durable" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(tmp_path / "durable")}
    )
    env["PYBIN"] = str(_fail_role_publish_python(tmp_path))
    env["FAKE_FAIL_ROLE_PUBLISH"] = failed_role
    canonical = {
        "lock": Path(env["CKPT_SYNC_LOCK_FILE"]),
        "state": Path(env["CKPT_SYNC_STATE"]),
        "pid": Path(env["CKPT_SYNC_PID_FILE"]),
    }[failed_role]

    failed = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert failed.returncode == 2
    assert "injected failure before canonical role publish" in failed.stderr
    assert not canonical.exists()
    leftovers = list(tmp_path.glob(f".{canonical.name}.{failed_role}-init-v1-*"))
    assert len(leftovers) == 1
    assert leftovers[0].stat().st_size > 0
    leftover_bytes = leftovers[0].read_bytes()
    env["PYBIN"] = sys.executable
    env.pop("FAKE_FAIL_ROLE_PUBLISH")

    rerun = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert rerun.returncode == 0, rerun.stderr
    assert canonical.exists()
    assert leftovers[0].read_bytes() == leftover_bytes
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_concurrent_first_start_publishes_only_complete_coordination_roles(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    launch = tmp_path / "durable" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(tmp_path / "durable")}
    )
    processes = [
        subprocess.Popen(
            ["bash", str(CKPT_SYNC)],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert [process.returncode for process in processes] == [0, 0], results
    assert Path(env["CKPT_SYNC_LOCK_FILE"]).read_text(encoding="utf-8") == _lock_role(
        env, launch
    )
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == (
        _state_identity_header(env, launch)
    )
    pid_line = Path(env["CKPT_SYNC_PID_FILE"]).read_text(encoding="utf-8")
    assert pid_line.startswith(f"pid-v1\t{_attempt_identity_digest(env, launch)}\t")
    assert not list(tmp_path.glob(".*-init-v1-*"))
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_pid_path_swap_after_preflight_never_overwrites_unrelated_replacement(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    pid_path = Path(env["CKPT_SYNC_PID_FILE"])
    valid_pid = _pid_role(env, launch)
    pid_path.write_text(valid_pid, encoding="utf-8")
    pid_path.chmod(0o600)
    renamed_original = tmp_path / "renamed-original-pid"
    env["FAKE_SWAP_PID_DURING_FLOCK"] = str(renamed_original)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "cannot securely write pid file" in result.stderr
    assert renamed_original.read_text(encoding="utf-8") == valid_pid
    assert pid_path.read_bytes() == b"unrelated replacement bytes\n"
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_pid_swap_immediately_before_publish_preserves_unrelated_replacement(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    pid_path = Path(env["CKPT_SYNC_PID_FILE"])
    valid_pid = _pid_role(env, launch)
    pid_path.write_text(valid_pid, encoding="utf-8")
    pid_path.chmod(0o600)
    renamed_original = tmp_path / "renamed-prepublish-pid"
    env.update(
        {
            "PYBIN": str(_pid_prepublication_swap_python(tmp_path)),
            "FAKE_SWAP_PID_PREPUBLISH_PATH": str(pid_path),
            "FAKE_SWAP_PID_PREPUBLISH_ORIGINAL": str(renamed_original),
        }
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "pid canonical role changed before publish" in result.stderr
    assert renamed_original.read_text(encoding="utf-8") == valid_pid
    assert pid_path.read_bytes() == b"unrelated replacement bytes\n"
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "entry_kind", ["file_symlink", "directory_symlink", "hardlink", "fifo", "socket"]
)
def test_sync_rejects_nested_nonprivate_or_nonregular_entries_before_provider(
    tmp_path: Path, entry_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    nested = launch / "final" / "nested"
    nested.mkdir(parents=True)
    outside_file = tmp_path / "outside-secret"
    outside_file.write_bytes(b"must-not-escape")
    entry = nested / "unsafe"
    if entry_kind == "file_symlink":
        entry.symlink_to(outside_file)
    elif entry_kind == "directory_symlink":
        outside_dir = tmp_path / "outside-directory"
        outside_dir.mkdir()
        (outside_dir / "secret").write_bytes(b"must-not-escape")
        entry.symlink_to(outside_dir, target_is_directory=True)
    elif entry_kind == "hardlink":
        os.link(outside_file, entry)
    elif entry_kind == "fifo":
        os.mkfifo(entry)
    else:
        # Darwin's AF_UNIX path cap is shorter than pytest's temp path. Bind a
        # relative name while cwd is the actual nested checkpoint directory.
        previous_cwd = Path.cwd()
        os.chdir(nested)
        try:
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix_socket.bind(entry.name)
            unix_socket.close()
        finally:
            os.chdir(previous_cwd)
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    expected = {
        "file_symlink": "refusing symlink",
        "directory_symlink": "refusing symlink",
        "hardlink": "refusing hard-linked checkpoint entry",
        "fifo": "refusing non-regular checkpoint entry",
        "socket": "refusing non-regular checkpoint entry",
    }[entry_kind]
    assert expected in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "entry_kind",
    ["regular", "fifo", "socket", "file_symlink", "directory_symlink", "broken_symlink"],
)
def test_sync_refuses_every_unsafe_top_level_checkpoint_match(
    tmp_path: Path, entry_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    launch.mkdir(parents=True)
    entry = launch / "final"
    outside_file = tmp_path / "outside-file"
    outside_file.write_bytes(b"outside")
    if entry_kind == "regular":
        entry.write_bytes(b"not-a-directory")
    elif entry_kind == "fifo":
        os.mkfifo(entry)
    elif entry_kind == "socket":
        previous_cwd = Path.cwd()
        os.chdir(launch)
        try:
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            unix_socket.bind(entry.name)
            unix_socket.close()
        finally:
            os.chdir(previous_cwd)
    elif entry_kind == "file_symlink":
        entry.symlink_to(outside_file)
    elif entry_kind == "directory_symlink":
        outside_dir = tmp_path / "outside-directory"
        outside_dir.mkdir()
        entry.symlink_to(outside_dir, target_is_directory=True)
    else:
        entry.symlink_to(tmp_path / "missing-target")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    if "symlink" in entry_kind:
        assert "refusing symlinked checkpoint entry" in result.stderr
    else:
        assert "refusing non-directory checkpoint entry" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    "run_name",
    [
        "unsafe scientific/run name",
        "../escape",
        "unicode-é-science",
        "x" * 200,
        "---",
        "has\tcontrol",
    ],
)
def test_checkpoint_dir_run_component_matches_shared_safe_renderer(
    tmp_path: Path, run_name: str
) -> None:
    from infra.launch_namespace import safe_path_component

    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    safe_run = safe_path_component(run_name, fallback="run")
    launch = tmp_path / "checkpoints" / safe_run / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"final")
    env, upload_log = _sync_env(
        tmp_path, launch, "attempt-a", fake_modules, run_name=run_name
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    records = _read_jsonl(upload_log)
    expected_prefix = f"checkpoints/{safe_run}/attempt-a/final/"
    assert [record["Key"] for record in records] == [
        expected_prefix + ".ckpt-sync-reservation-v1.json",
        expected_prefix + "adapter.bin",
    ]
    claim = json.loads(base64.b64decode(records[0]["BodyBase64"]))
    assert claim["run"] == safe_run
    assert run_name not in upload_log.read_text(encoding="utf-8")


def test_s3_sync_uses_conditional_namespaced_keys_for_exact_launch(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    sibling = launch.parent / "attempt-b"
    (launch / "step-00025").mkdir(parents=True)
    (sibling / "step-00050").mkdir(parents=True)
    (launch / "step-00025" / "rank-0.bin").write_bytes(b"selected")
    (sibling / "step-00050" / "rank-0.bin").write_bytes(b"not-selected")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {
            "kind": "bucket",
            "endpoint": "https://storage.vendor.invalid:9443",
            "region": "arbitrary-region-9",
            "bucket": "configured-bucket",
            "prefix": "team/adapter-checkpoints",
        }
    )
    env["AWS_MAX_ATTEMPTS"] = "99"
    env["AWS_RETRY_MODE"] = "adaptive"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    uploads = _read_jsonl(upload_log)
    prefix = f"team/adapter-checkpoints/{env['RUN_NAME']}/attempt-a/step-00025/"
    assert [call["Key"] for call in uploads] == [
        prefix + ".ckpt-sync-reservation-v1.json",
        prefix + "rank-0.bin",
    ]
    assert all(call["IfNoneMatch"] == "*" for call in uploads)
    assert {call["Endpoint"] for call in uploads} == {
        "https://storage.vendor.invalid:9443"
    }
    assert {call["Region"] for call in uploads} == {"arbitrary-region-9"}
    assert {call["Bucket"] for call in uploads} == {"configured-bucket"}
    assert {json.dumps(call["RetryConfig"], sort_keys=True) for call in uploads} == {
        json.dumps({"mode": "standard", "total_max_attempts": 1}, sort_keys=True)
    }
    assert {call["RegionRedirectHandlers"] for call in uploads} == {0}
    claim_bytes = base64.b64decode(uploads[0]["BodyBase64"])
    claim = json.loads(claim_bytes)
    assert claim["run"] == env["RUN_NAME"]
    assert claim["namespace"] == "attempt-a"
    assert claim["step"] == "step-00025"
    assert len(claim["claim_nonce"]) == 32
    claim_digest = claim.pop("claim_digest_sha256")
    claim_core = json.dumps(
        claim, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert claim_digest == hashlib.sha256(claim_core).hexdigest()
    manifest = [
        {
            "path": "rank-0.bin",
            "size": len(b"selected"),
            "sha256": hashlib.sha256(b"selected").hexdigest(),
        }
    ]
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert claim["checkpoint_manifest_sha256"] == hashlib.sha256(
        manifest_bytes
    ).hexdigest()
    assert uploads[1]["Metadata"]["claim-digest-sha256"] == claim_digest
    assert "step-00050" not in upload_log.read_text(encoding="utf-8")
    expected_state = _state_identity_header(env, launch) + _state_completion_record(
        launch / "step-00025"
    )
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == expected_state

    first_log = upload_log.read_bytes()
    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert second.returncode == 0, second.stderr
    assert upload_log.read_bytes() == first_log
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == expected_state


def test_coordination_and_state_writes_complete_under_short_os_write(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["PYBIN"] = str(_short_write_python(tmp_path))

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert len(_read_jsonl(upload_log)) == 2
    assert Path(env["CKPT_SYNC_LOCK_FILE"]).read_text(encoding="utf-8") == _lock_role(
        env, launch
    )
    pid_fields = Path(env["CKPT_SYNC_PID_FILE"]).read_text(encoding="utf-8").split("\t")
    assert pid_fields[:2] == ["pid-v1", _attempt_identity_digest(env, launch)]
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == (
        _state_identity_header(env, launch)
        + _state_completion_record(launch / "final")
    )


@pytest.mark.parametrize("checkpoint_content", ["empty", "reserved"])
def test_bucket_refuses_empty_or_reserved_checkpoint_before_provider_mutation(
    tmp_path: Path, checkpoint_content: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    if checkpoint_content == "reserved":
        (launch / "final" / ".ckpt-sync-reservation-v1.json").write_text(
            "caller content", encoding="utf-8"
        )
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    expected = "empty checkpoint directory" if checkpoint_content == "empty" else "reserved relative path"
    assert expected in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)
    assert _provider_journal(env) == b""


def test_credential_env_file_cannot_override_validated_identity_or_controls(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    access_canary = "loaded-access-canary"
    credential_secret = "must-never-appear-in-output"
    command_side_effect = tmp_path / "must-not-exist"
    env_file = tmp_path / "s3.env"
    env_file.write_text(
        "\n".join(
            [
                f"export AWS_ACCESS_KEY_ID='{access_canary}'",
                f"AWS_SECRET_ACCESS_KEY={credential_secret}",
                "RUN_NAME=redirected-run",
                "DEBATE_LAUNCH_NAMESPACE=redirected-namespace",
                "EXPECTED_RUN_COMPONENT=redirected-component",
                "DESTINATION_PREFIX=redirected-prefix",
                "DESTINATION_BUCKET=redirected-bucket",
                "PYBIN=/definitely/not/python",
                "CKPT_DIR=/redirected/checkpoints",
                "QUIESCENT_SECS=999999",
                "CKPT_SYNC_ONCE=0",
                "SYNC_ONCE=0",
                f"UNTRUSTED_CONTROL=$(touch {command_side_effect})",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    env_file.chmod(0o600)
    env["S3_ENV_FILE"] = str(env_file)
    env["FAKE_EXPECTED_CREDENTIAL_DIGESTS"] = json.dumps(
        {
            "AWS_ACCESS_KEY_ID": hashlib.sha256(
                access_canary.encode("utf-8")
            ).hexdigest(),
            "AWS_SECRET_ACCESS_KEY": hashlib.sha256(
                credential_secret.encode("utf-8")
            ).hexdigest(),
        }
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    records = _read_jsonl(upload_log)
    assert [record["Key"] for record in records] == [
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json",
        "checkpoints/run/attempt-a/final/adapter.bin",
    ]
    assert {record["Bucket"] for record in records} == {"test-bucket"}
    assert {tuple(record["CredentialNames"]) for record in records} == {
        ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    }
    combined_output = result.stdout + result.stderr
    assert credential_secret not in combined_output
    assert access_canary not in combined_output
    assert not command_side_effect.exists()
    serialized_uploads = upload_log.read_text(encoding="utf-8")
    assert credential_secret not in serialized_uploads
    assert access_canary not in serialized_uploads
    assert "redirected-" not in serialized_uploads


@pytest.mark.parametrize(
    "credential_path_kind",
    ["symlink", "directory", "hardlink", "wrong_mode", "fifo"],
)
def test_bucket_refuses_symlink_or_nonregular_credential_file(
    tmp_path: Path, credential_path_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    credential_path = tmp_path / "credentials"
    if credential_path_kind == "symlink":
        target = tmp_path / "actual-credentials"
        target.write_text("AWS_ACCESS_KEY_ID=canary\n", encoding="utf-8")
        credential_path.symlink_to(target)
    elif credential_path_kind == "directory":
        credential_path.mkdir()
    elif credential_path_kind == "hardlink":
        target = tmp_path / "actual-credentials"
        target.write_text("AWS_ACCESS_KEY_ID=canary\n", encoding="utf-8")
        target.chmod(0o600)
        credential_path.hardlink_to(target)
    elif credential_path_kind == "wrong_mode":
        credential_path.write_text("AWS_ACCESS_KEY_ID=canary\n", encoding="utf-8")
        credential_path.chmod(0o644)
    else:
        os.mkfifo(credential_path, mode=0o600)
    env["S3_ENV_FILE"] = str(credential_path)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "euid-owned regular single-link mode 0600" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    ("pagination_mode", "expected_lists", "expected_error"),
    [
        ("missing-token", 1, "unsafe nonprogressing truncated"),
        ("empty-token", 1, "unsafe nonprogressing truncated"),
        ("empty-page", 1, "unsafe nonprogressing truncated"),
        ("repeated-token", 2, "unsafe nonprogressing truncated"),
        ("duplicate-key", 2, "unsafe nonprogressing S3 prefix-list page"),
        ("novel-overflow", 3, "unsafe overfull S3 prefix-list response"),
    ],
)
def test_bucket_refuses_nonprogressing_pagination_with_bounded_lists(
    tmp_path: Path,
    pagination_mode: str,
    expected_lists: int,
    expected_error: str,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_S3_PAGINATION"] = pagination_mode
    env["FAKE_S3_PAGINATION_START_CALL"] = "2"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert expected_error in result.stderr
    journal = _provider_journal(env)
    assert journal.count(b"LIST\n") == expected_lists + 1
    assert journal.startswith(b"CLIENT\nLIST\n")
    assert len(_read_jsonl(upload_log)) == 2
    _assert_state_has_only_identity(env, launch)


@pytest.mark.parametrize("replacement_kind", ["regular", "fifo"])
def test_credential_path_swap_between_lstat_and_open_refuses_both_files(
    tmp_path: Path, replacement_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    credential_path = tmp_path / "credentials"
    original_bytes = b"AWS_ACCESS_KEY_ID=original\n"
    credential_path.write_bytes(original_bytes)
    credential_path.chmod(0o600)
    renamed_original = tmp_path / "renamed-original-credentials"
    env.update(
        {
            "S3_ENV_FILE": str(credential_path),
            "PYBIN": str(_credential_swap_python(tmp_path)),
            "FAKE_SWAP_CREDENTIAL_PATH": str(credential_path),
            "FAKE_SWAP_CREDENTIAL_ORIGINAL": str(renamed_original),
            "FAKE_SWAP_CREDENTIAL_KIND": replacement_kind,
        }
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "euid-owned regular single-link mode 0600" in result.stderr
    assert renamed_original.read_bytes() == original_bytes
    if replacement_kind == "fifo":
        assert stat.S_ISFIFO(credential_path.stat().st_mode)
    else:
        assert credential_path.read_bytes() == b"AWS_ACCESS_KEY_ID=replacement\n"
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


def test_bucket_sync_refuses_a_non_adapter_sized_file_before_reservation(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    oversized = launch / "final" / "not-an-adapter.bin"
    with oversized.open("wb") as output:
        output.truncate(5 * 1024**3 + 1)
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "exceeds the conditional single-PUT limit" in result.stderr
    assert "multipart" in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)


@pytest.mark.parametrize(("shape", "observed"), [("none", 0), ("duplicate", 2)])
def test_bucket_fails_closed_on_unknown_botocore_redirector_shape(
    tmp_path: Path, shape: str, observed: int
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_REDIRECT_SHAPE"] = shape

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert f"observed {observed}" in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)


@pytest.mark.parametrize("corruption", ["body", "metadata", "size"])
def test_bucket_remote_byte_verification_failure_is_terminal(
    tmp_path: Path, corruption: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_S3_CORRUPT_GET_KEY"] = (
        "checkpoints/run/attempt-a/final/adapter.bin"
    )
    env["FAKE_S3_CORRUPT_GET_MODE"] = corruption
    env.pop("CKPT_SYNC_ONCE")

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "refusing unverified S3 object after PUT" in result.stderr
    assert [record["Key"] for record in _read_jsonl(upload_log)] == [
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json",
        "checkpoints/run/attempt-a/final/adapter.bin",
    ]
    _assert_state_has_only_identity(env, launch)


def test_bucket_reservation_body_corruption_refuses_before_data_put(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    reservation_key = (
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json"
    )
    env["FAKE_S3_CORRUPT_GET_KEY"] = reservation_key
    env["FAKE_S3_CORRUPT_GET_MODE"] = "body"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing unverified S3 object after PUT" in result.stderr
    assert [record["Key"] for record in _read_jsonl(upload_log)] == [
        reservation_key
    ]
    _assert_state_has_only_identity(env, launch)


def test_state_path_swap_after_first_list_never_receives_completion(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    renamed_original = tmp_path / "renamed-original-state"
    env["FAKE_SWAP_STATE_AFTER_FIRST_LIST"] = str(renamed_original)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "durable state append FAILED" in result.stderr
    header = _state_identity_header(env, launch)
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == header
    assert renamed_original.read_text(encoding="utf-8") == header
    assert len(_read_jsonl(upload_log)) == 2
    assert b"CLIENT\nLIST\nPUT\nGET\nHEAD\nPUT\nGET\nLIST\n" == _provider_journal(env)


def test_midloop_state_inode_tamper_is_validation_exit_two_before_provider(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    state_path = Path(env["CKPT_SYNC_STATE"])
    renamed_original = tmp_path / "renamed-midloop-state"
    env.update(
        {
            "PYBIN": str(_midloop_state_swap_python(tmp_path)),
            "FAKE_MIDLOOP_STATE_PATH": str(state_path),
            "FAKE_MIDLOOP_STATE_ORIGINAL": str(renamed_original),
        }
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "cannot validate completion state" in result.stderr
    assert "state inode changed" in result.stderr
    assert renamed_original.read_text(encoding="utf-8") == _state_identity_header(
        env, launch
    )
    assert state_path.read_bytes() == b"unrelated replacement state bytes\n"
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("state_kind", ["directory", "symlink", "wrong_mode"])
def test_unsafe_state_path_refuses_before_any_provider_mutation(
    tmp_path: Path, state_kind: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    unsafe_state = tmp_path / "unsafe-state"
    if state_kind == "directory":
        unsafe_state.mkdir()
    elif state_kind == "symlink":
        target = tmp_path / "state-target"
        target.write_text("", encoding="utf-8")
        unsafe_state.symlink_to(target)
    else:
        unsafe_state.write_text("", encoding="utf-8")
        unsafe_state.chmod(0o400)
    env["CKPT_SYNC_STATE"] = str(unsafe_state)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "unsafe coordination path configuration" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("endpoint", "https://other.example.test"),
        ("region", "other-region-2"),
        ("bucket", "other-bucket"),
        ("prefix", "other/checkpoints"),
    ],
)
def test_state_identity_refuses_switching_bucket_destination_before_provider(
    tmp_path: Path, field: str, replacement: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    first = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert first.returncode == 0, first.stderr
    first_log = upload_log.read_bytes()
    first_provider_journal = _provider_journal(env)
    changed = json.loads(env["CKPT_DESTINATION_JSON"])
    changed[field] = replacement
    env["CKPT_DESTINATION_JSON"] = json.dumps(changed)

    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert second.returncode == 2
    assert "unsafe coordination path configuration" in second.stderr
    assert upload_log.read_bytes() == first_log
    assert _provider_journal(env) == first_provider_journal


def test_state_identity_refuses_bucket_to_local_switch_before_noop(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "durable" / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    first = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert first.returncode == 0, first.stderr
    first_log = upload_log.read_bytes()
    first_provider_journal = _provider_journal(env)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(tmp_path / "durable")}
    )

    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert second.returncode == 2
    assert "unsafe coordination path configuration" in second.stderr
    assert "nothing to do" not in second.stdout
    assert upload_log.read_bytes() == first_log
    assert _provider_journal(env) == first_provider_journal


def test_state_identity_refuses_switching_between_two_containing_local_roots(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    outer = tmp_path / "durable"
    inner = outer / "nested-root"
    launch = inner / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(outer)}
    )
    first = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert first.returncode == 0, first.stderr
    state_path = Path(env["CKPT_SYNC_STATE"])
    first_state = state_path.read_bytes()
    first_stat = state_path.stat()
    env["CKPT_DESTINATION_JSON"] = json.dumps(
        {"kind": "local", "directory": str(inner)}
    )

    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    after = state_path.stat()
    assert second.returncode == 2
    assert "unsafe coordination path configuration" in second.stderr
    assert "nothing to do" not in second.stdout
    assert state_path.read_bytes() == first_state
    assert (after.st_ino, after.st_mode, after.st_size, after.st_mtime_ns) == (
        first_stat.st_ino,
        first_stat.st_mode,
        first_stat.st_size,
        first_stat.st_mtime_ns,
    )
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("changed_identity", ["run", "directory"])
def test_state_identity_refuses_different_run_or_checkpoint_dir(
    tmp_path: Path, changed_identity: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    first = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )
    assert first.returncode == 0, first.stderr
    first_log = upload_log.read_bytes()
    first_provider_journal = _provider_journal(env)
    if changed_identity == "run":
        other_launch = tmp_path / "checkpoints" / "other-run" / "attempt-a"
        other_run_name = "other-run"
    else:
        other_launch = tmp_path / "other-checkpoints" / "run" / "attempt-a"
        other_run_name = "run"
    (other_launch / "final").mkdir(parents=True)
    (other_launch / "final" / "adapter.bin").write_bytes(b"other")
    env["CKPT_DIR"] = str(other_launch)
    env["RUN_NAME"] = other_run_name

    second = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert second.returncode == 2
    assert "unsafe coordination path configuration" in second.stderr
    assert upload_log.read_bytes() == first_log
    assert _provider_journal(env) == first_provider_journal


@pytest.mark.parametrize(
    "existing_state",
    [
        "attempt-a\t/legacy/checkpoint\n",
        "identity-v1\tnot-a-digest\n",
        "identity-v1\t" + "0" * 64 + "\ncomplete-v1\tmalformed\n",
    ],
)
def test_headerless_malformed_or_mismatched_state_is_never_adopted(
    tmp_path: Path, existing_state: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    existing_path = Path(env["CKPT_SYNC_STATE"])
    existing_path.write_text(existing_state, encoding="utf-8")
    existing_path.chmod(0o600)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "unsafe coordination path configuration" in result.stderr
    assert not upload_log.exists()
    assert _provider_journal(env) == b""
    assert Path(env["CKPT_SYNC_STATE"]).read_text(encoding="utf-8") == existing_state


@pytest.mark.parametrize("state_error", ["malformed", "duplicate"])
def test_correct_identity_header_with_invalid_records_is_preserved_and_refused(
    tmp_path: Path, state_error: str
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    header = _state_identity_header(env, launch)
    valid_record = _state_completion_record(launch / "final")
    existing = (
        header + "complete-v1\tmalformed\n"
        if state_error == "malformed"
        else header + valid_record + valid_record
    )
    state_path = Path(env["CKPT_SYNC_STATE"])
    state_path.write_text(existing, encoding="utf-8")
    state_path.chmod(0o600)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 2
    assert "unsafe coordination path configuration" in result.stderr
    assert state_path.read_text(encoding="utf-8") == existing
    assert not upload_log.exists()
    assert _provider_journal(env) == b""


@pytest.mark.parametrize("existing_bytes", ["new", "different"])
def test_s3_sync_refuses_every_existing_destination_even_if_bytes_match(
    tmp_path: Path,
    existing_bytes: str,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "rank-0.bin").write_bytes(b"new")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    key = "checkpoints/run/attempt-a/final/rank-0.bin"
    env["FAKE_S3_EXISTING"] = json.dumps({key: existing_bytes})

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing occupied S3 step prefix" in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)


@pytest.mark.parametrize("with_partial", [False, True])
def test_bucket_refuses_seeded_crash_state_without_writes_or_adoption(
    tmp_path: Path, with_partial: bool
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "adapter.bin").write_bytes(b"adapter")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    prefix = "checkpoints/run/attempt-a/final/"
    existing = {prefix + ".ckpt-sync-reservation-v1.json": "prior marker"}
    if with_partial:
        existing[prefix + "adapter.bin"] = "prior partial"
    env["FAKE_S3_EXISTING"] = json.dumps(existing)

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing occupied S3 step prefix" in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)


def test_fake_s3_head_preserves_size_and_sha256_for_existing_content(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(fake_modules)
    env["FAKE_S3_EXISTING"] = json.dumps({"prefix/object": "exact-bytes"})
    env["FAKE_REQUIRE_TRANSPORT_BUDGET"] = "0"
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import boto3, json; "
            "print(json.dumps(boto3.client('s3').head_object("
            "Bucket='bucket', Key='prefix/object'), sort_keys=True))",
        ],
        env=env,
        text=True,
        capture_output=True,
    )

    assert probe.returncode == 0, probe.stderr
    response = json.loads(probe.stdout)
    assert response["ContentLength"] == len(b"exact-bytes")
    assert response["Metadata"]["sha256"] == hashlib.sha256(
        b"exact-bytes"
    ).hexdigest()


def test_s3_sync_refuses_a_foreign_object_anywhere_under_step_prefix(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "rank-0.bin").write_bytes(b"new")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    foreign = "checkpoints/run/attempt-a/final/not-in-local-checkpoint.bin"
    env["FAKE_S3_EXISTING"] = json.dumps({foreign: "foreign"})

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing occupied S3 step prefix" in result.stderr
    assert not upload_log.exists()
    _assert_state_has_only_identity(env, launch)


def test_s3_sync_refuses_object_appearing_between_prefix_list_and_head(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "rank-0.bin").write_bytes(b"new")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_S3_HEAD_RACE_KEY"] = (
        "checkpoints/run/attempt-a/final/rank-0.bin"
    )
    env["FAKE_S3_HEAD_RACE_BODY"] = "new"

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing occupied S3 destination" in result.stderr
    uploads = _read_jsonl(upload_log)
    assert [record["Key"] for record in uploads] == [
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json"
    ]
    _assert_state_has_only_identity(env, launch)


def test_s3_sync_refuses_and_does_not_adopt_a_conditional_put_race(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "rank-0.bin").write_bytes(b"new")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_S3_PUT_CONFLICT_KEY"] = (
        "checkpoints/run/attempt-a/final/rank-0.bin"
    )
    # Exercise normal daemon mode, not the test-only one-scan exit: an S3
    # reservation/data failure is terminal and must not loop into adoption.
    env.pop("CKPT_SYNC_ONCE")

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=5,
    )

    assert result.returncode == 1
    assert "refusing concurrently occupied S3 destination" in result.stderr
    uploads = _read_jsonl(upload_log)
    assert [record["Key"] for record in uploads] == [
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json"
    ]
    _assert_state_has_only_identity(env, launch)


def test_atomic_step_marker_allows_only_one_of_two_empty_prefix_writers(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    remote_store = tmp_path / "remote-s3"
    barrier = tmp_path / "initial-list-barrier"
    processes = []
    writer_envs = {}
    payloads = {"writer-a": b"from-a", "writer-b": b"from-b"}
    for writer, payload in payloads.items():
        writer_root = tmp_path / writer
        writer_root.mkdir()
        writer_root.chmod(0o700)
        launch = writer_root / "checkpoints" / "run" / "attempt-a"
        (launch / "final").mkdir(parents=True)
        (launch / "final" / f"{writer}.bin").write_bytes(payload)
        env, _ = _sync_env(writer_root, launch, "attempt-a", fake_modules)
        env.update(
            {
                "FAKE_S3_STORE": str(remote_store),
                "FAKE_S3_INITIAL_LIST_BARRIER": str(barrier),
                "FAKE_S3_CLIENT_ID": writer,
            }
        )
        writer_envs[writer] = env

    for writer in payloads:
        processes.append(
            (
                writer,
                subprocess.Popen(
                    ["bash", str(CKPT_SYNC)],
                    cwd=ROOT,
                    env=writer_envs[writer],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ),
            )
        )
    results = {}
    for writer, process in processes:
        stdout, stderr = process.communicate(timeout=10)
        results[writer] = (process.returncode, stdout, stderr)

    assert sorted(result[0] for result in results.values()) == [0, 1]
    loser = next(writer for writer, result in results.items() if result[0] == 1)
    assert "refusing concurrently reserved S3 step prefix" in results[loser][2]

    remote_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in remote_store.glob("*.json")
    ]
    prefix = "checkpoints/run/attempt-a/final/"
    reservation_key = prefix + ".ckpt-sync-reservation-v1.json"
    assert len(remote_records) == 2
    assert sum(record["Key"] == reservation_key for record in remote_records) == 1
    data_records = [record for record in remote_records if record["Key"] != reservation_key]
    assert len(data_records) == 1
    data_record = data_records[0]
    winner = data_record["Key"].removeprefix(prefix).removesuffix(".bin")
    assert winner in payloads
    assert base64.b64decode(data_record["BodyBase64"]) == payloads[winner]

    reservation = next(
        record for record in remote_records if record["Key"] == reservation_key
    )
    claim = json.loads(base64.b64decode(reservation["BodyBase64"]))
    assert (claim["run"], claim["namespace"], claim["step"]) == (
        "run",
        "attempt-a",
        "final",
    )
    expected_manifest = [
        {
            "path": f"{winner}.bin",
            "size": len(payloads[winner]),
            "sha256": hashlib.sha256(payloads[winner]).hexdigest(),
        }
    ]
    expected_manifest_bytes = json.dumps(
        expected_manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert claim["checkpoint_manifest_sha256"] == hashlib.sha256(
        expected_manifest_bytes
    ).hexdigest()
    assert data_record["Metadata"]["claim-digest-sha256"] == claim[
        "claim_digest_sha256"
    ]
    winner_launch = Path(writer_envs[winner]["CKPT_DIR"])
    assert Path(writer_envs[winner]["CKPT_SYNC_STATE"]).read_text(
        encoding="utf-8"
    ) == _state_identity_header(
        writer_envs[winner], winner_launch
    ) + _state_completion_record(winner_launch / "final")
    _assert_state_has_only_identity(
        writer_envs[loser], Path(writer_envs[loser]["CKPT_DIR"])
    )


def test_s3_sync_detects_a_foreign_prefix_race_after_conditional_put(
    tmp_path: Path,
) -> None:
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    _fake_s3_modules(fake_modules)
    launch = tmp_path / "checkpoints" / "run" / "attempt-a"
    (launch / "final").mkdir(parents=True)
    (launch / "final" / "rank-0.bin").write_bytes(b"new")
    env, upload_log = _sync_env(tmp_path, launch, "attempt-a", fake_modules)
    env["FAKE_S3_FOREIGN_AFTER_PUT_KEY"] = (
        "checkpoints/run/attempt-a/final/foreign-race.bin"
    )

    result = subprocess.run(
        ["bash", str(CKPT_SYNC)], cwd=ROOT, env=env, text=True, capture_output=True
    )

    assert result.returncode == 1
    assert "refusing raced or incomplete S3 step prefix" in result.stderr
    uploads = _read_jsonl(upload_log)
    assert [record["Key"] for record in uploads] == [
        "checkpoints/run/attempt-a/final/.ckpt-sync-reservation-v1.json",
        "checkpoints/run/attempt-a/final/rank-0.bin"
    ]
    _assert_state_has_only_identity(env, launch)


def test_real_util_linux_flock_excludes_a_second_process(tmp_path: Path) -> None:
    flock = shutil.which("flock")
    if flock is None:
        pytest.skip("util-linux flock is not installed on this platform")
    version = subprocess.run(
        [flock, "--version"], text=True, capture_output=True, check=False
    )
    if "util-linux" not in (version.stdout + version.stderr):
        pytest.skip("installed flock is not the util-linux implementation used on pods")
    lock = tmp_path / "real-flock.lock"
    holder = subprocess.Popen(
        [
            "bash",
            "-c",
            'exec 9>"$1"; "$2" -n 9 || exit 9; printf "ready\\n"; IFS= read -r _',
            "holder",
            str(lock),
            flock,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    try:
        assert holder.stdout.readline() == "ready\n"
        contender = subprocess.run(
            [
                "bash",
                "-c",
                'exec 9>"$1"; "$2" -n 9',
                "contender",
                str(lock),
                flock,
            ],
            text=True,
            capture_output=True,
        )
        assert contender.returncode != 0
    finally:
        if holder.stdin is not None:
            holder.stdin.write("release\n")
            holder.stdin.flush()
        holder.wait(timeout=5)


def test_installed_botocore_put_object_supports_if_none_match_read_only() -> None:
    session_module = pytest.importorskip(
        "botocore.session", reason="botocore is not installed in this local environment"
    )
    service = session_module.get_session().get_service_model("s3")
    put_object = service.operation_model("PutObject")
    assert "IfNoneMatch" in put_object.input_shape.members


def test_installed_botocore_client_uses_exact_single_attempt_policy_despite_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boto3 = pytest.importorskip(
        "boto3", reason="boto3 is not installed in this local environment"
    )
    config_module = pytest.importorskip(
        "botocore.config", reason="botocore is not installed in this local environment"
    )
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "99")
    monkeypatch.setenv("AWS_RETRY_MODE", "adaptive")
    config = config_module.Config(
        retries={"total_max_attempts": 1, "mode": "standard"}
    )
    client = boto3.client(
        "s3",
        endpoint_url="https://objects.example.test",
        region_name="test-region-1",
        aws_access_key_id="read-only-test",
        aws_secret_access_key="read-only-test",
        config=config,
    )
    assert client.meta.config.retries == {
        "total_max_attempts": 1,
        "mode": "standard",
    }


@pytest.mark.parametrize(
    ("status", "error_code"),
    [(301, "PermanentRedirect"), (412, "PreconditionFailed"), (200, None)],
)
def test_real_botocore_conditional_put_has_exactly_one_send_without_network(
    status: int, error_code: str | None
) -> None:
    boto3 = pytest.importorskip(
        "boto3", reason="boto3 is not installed in this local environment"
    )
    awsrequest = pytest.importorskip(
        "botocore.awsrequest",
        reason="botocore is not installed in this local environment",
    )
    config_module = pytest.importorskip("botocore.config")
    exceptions_module = pytest.importorskip("botocore.exceptions")

    # Execute the production helper itself, extracted from the embedded Python
    # driver, so this independent transport probe cannot drift to a test-only
    # reimplementation of the handler selection logic.
    script = CKPT_SYNC.read_text(encoding="utf-8")
    helper_start = script.index("    def disable_s3_region_redirect_retries")
    helper_end = script.index(
        "\n\n    disable_s3_region_redirect_retries(s3)", helper_start
    )
    helper_namespace: dict[str, object] = {}
    exec(textwrap.dedent(script[helper_start:helper_end]), helper_namespace)
    disable_redirect = helper_namespace["disable_s3_region_redirect_retries"]

    client = boto3.client(
        "s3",
        endpoint_url="https://objects.example.test",
        region_name="test-region-1",
        aws_access_key_id="read-only-test",
        aws_secret_access_key="read-only-test",
        config=config_module.Config(
            retries={"total_max_attempts": 1, "mode": "standard"}
        ),
    )
    emitter = client.meta.events._emitter
    signing_before = list(emitter._handlers.prefix_search("before-sign.s3"))
    endpoint_before = list(
        emitter._handlers.prefix_search("before-endpoint-resolution.s3")
    )
    utils_module = pytest.importorskip("botocore.utils")
    redirect_types = tuple(
        redirect_type
        for name in ("S3RegionRedirector", "S3RegionRedirectorv2")
        if (redirect_type := getattr(utils_module, name, None)) is not None
    )
    redirect_handlers = [
        handler
        for handler in emitter._handlers.prefix_search("needs-retry.s3")
        if isinstance(getattr(handler, "__self__", None), redirect_types)
        and getattr(handler, "__name__", None) == "redirect_from_error"
    ]
    assert len(redirect_handlers) == 1
    operation_redirect = redirect_handlers[0]
    disable_redirect(client)  # type: ignore[operator]
    assert list(emitter._handlers.prefix_search("before-sign.s3")) == signing_before
    assert list(
        emitter._handlers.prefix_search("before-endpoint-resolution.s3")
    ) == endpoint_before
    assert any(
        getattr(handler, "__name__", None) == "needs_retry"
        for handler in emitter._handlers.prefix_search("needs-retry.s3")
    )
    if status == 301:
        # Simulate a future/version-specific operation-level redirect hook that
        # appears after the production global redirect cleanup. The absolute
        # before-send budget must still abort its retry before transport.
        client.meta.events.register(
            "needs-retry.s3.PutObject", operation_redirect
        )

    wrapper_start = script.index("    def call_s3_once")
    wrapper_end = script.index("\n\n    bucket =", wrapper_start)
    wrapper_namespace: dict[str, object] = {"s3": client}
    exec(textwrap.dedent(script[wrapper_start:wrapper_end]), wrapper_namespace)
    call_once = wrapper_namespace["call_s3_once"]

    class RawResponse:
        def __init__(self, body: bytes):
            self.body = body
            self.offset = 0

        def read(self, amt=None):
            if amt is None:
                chunk = self.body[self.offset :]
                self.offset = len(self.body)
                return chunk
            chunk = self.body[self.offset : self.offset + amt]
            self.offset += len(chunk)
            return chunk

        def stream(self, amt=None, decode_content=False):
            del amt, decode_content
            yield self.body

    sends = 0

    def before_send(request, **kwargs):
        del kwargs
        nonlocal sends
        sends += 1
        conditional_header = request.headers.get("If-None-Match")
        if isinstance(conditional_header, bytes):
            conditional_header = conditional_header.decode("ascii")
        assert conditional_header == "*"
        if status == 200:
            body = b""
            headers = {"etag": '"test-etag"'}
        elif status == 301:
            body = (
                b"<Error><Code>PermanentRedirect</Code><Message>wrong region</Message>"
                b"<Bucket>bucket</Bucket></Error>"
            )
            headers = {
                "content-type": "application/xml",
                "x-amz-bucket-region": "redirect-target",
            }
        else:
            body = (
                b"<Error><Code>PreconditionFailed</Code>"
                b"<Message>occupied</Message></Error>"
            )
            headers = {"content-type": "application/xml"}
        headers["content-length"] = str(len(body))
        return awsrequest.AWSResponse(
            request.url, status, headers, RawResponse(body)
        )

    client.meta.events.register("before-send.s3.PutObject", before_send)
    put_arguments = {
        "Bucket": "bucket",
        "Key": "prefix/object",
        "Body": b"payload",
        "ContentLength": 7,
        "Metadata": {"sha256": hashlib.sha256(b"payload").hexdigest()},
        "IfNoneMatch": "*",
    }
    if error_code is None:
        response = call_once(  # type: ignore[operator]
            "PutObject", client.put_object, **put_arguments
        )
        assert response["ETag"] == '"test-etag"'
    elif status == 301:
        with pytest.raises(RuntimeError, match="second transport attempt"):
            call_once(  # type: ignore[operator]
                "PutObject", client.put_object, **put_arguments
            )
    else:
        with pytest.raises(exceptions_module.ClientError) as exc_info:
            call_once(  # type: ignore[operator]
                "PutObject", client.put_object, **put_arguments
            )
        assert exc_info.value.response["Error"]["Code"] == error_code
    assert sends == 1
    assert not any(
        getattr(handler, "__name__", None) == "transport_budget_guard"
        for handler in emitter._handlers.prefix_search("before-send.s3.PutObject")
    )


@pytest.mark.parametrize("operation_name", ["ListObjectsV2", "HeadObject", "GetObject"])
@pytest.mark.parametrize("status", [301, 200])
def test_real_botocore_read_operations_have_one_transport_send_without_network(
    operation_name: str, status: int
) -> None:
    boto3 = pytest.importorskip(
        "boto3", reason="boto3 is not installed in this local environment"
    )
    awsrequest = pytest.importorskip("botocore.awsrequest")
    config_module = pytest.importorskip("botocore.config")
    utils_module = pytest.importorskip("botocore.utils")

    script = CKPT_SYNC.read_text(encoding="utf-8")
    helper_start = script.index("    def disable_s3_region_redirect_retries")
    helper_end = script.index(
        "\n\n    disable_s3_region_redirect_retries(s3)", helper_start
    )
    helper_namespace: dict[str, object] = {}
    exec(textwrap.dedent(script[helper_start:helper_end]), helper_namespace)

    client = boto3.client(
        "s3",
        endpoint_url="https://objects.example.test",
        region_name="test-region-1",
        aws_access_key_id="read-only-test",
        aws_secret_access_key="read-only-test",
        config=config_module.Config(
            retries={"total_max_attempts": 1, "mode": "standard"}
        ),
    )
    emitter = client.meta.events._emitter
    signing_before = list(emitter._handlers.prefix_search("before-sign.s3"))
    endpoint_before = list(
        emitter._handlers.prefix_search("before-endpoint-resolution.s3")
    )
    redirect_types = tuple(
        redirect_type
        for name in ("S3RegionRedirector", "S3RegionRedirectorv2")
        if (redirect_type := getattr(utils_module, name, None)) is not None
    )
    redirect_handlers = [
        handler
        for handler in emitter._handlers.prefix_search("needs-retry.s3")
        if isinstance(getattr(handler, "__self__", None), redirect_types)
        and getattr(handler, "__name__", None) == "redirect_from_error"
    ]
    assert len(redirect_handlers) == 1
    operation_redirect = redirect_handlers[0]
    helper_namespace["disable_s3_region_redirect_retries"](client)  # type: ignore[index,operator]
    assert list(emitter._handlers.prefix_search("before-sign.s3")) == signing_before
    assert list(
        emitter._handlers.prefix_search("before-endpoint-resolution.s3")
    ) == endpoint_before
    if status == 301:
        client.meta.events.register(
            f"needs-retry.s3.{operation_name}", operation_redirect
        )

    wrapper_start = script.index("    def call_s3_once")
    wrapper_end = script.index("\n\n    bucket =", wrapper_start)
    wrapper_namespace: dict[str, object] = {"s3": client}
    exec(textwrap.dedent(script[wrapper_start:wrapper_end]), wrapper_namespace)
    call_once = wrapper_namespace["call_s3_once"]

    class RawResponse:
        def __init__(self, body: bytes):
            self.body = body
            self.offset = 0

        def read(self, amt=None):
            if amt is None:
                chunk = self.body[self.offset :]
                self.offset = len(self.body)
                return chunk
            chunk = self.body[self.offset : self.offset + amt]
            self.offset += len(chunk)
            return chunk

        def stream(self, amt=None, decode_content=False):
            del amt, decode_content
            yield self.body

    sends = 0

    def before_send(request, **kwargs):
        del kwargs
        nonlocal sends
        sends += 1
        if status == 301:
            body = (
                b"<Error><Code>PermanentRedirect</Code><Message>wrong region</Message>"
                b"<Bucket>bucket</Bucket></Error>"
            )
            headers = {
                "content-type": "application/xml",
                "x-amz-bucket-region": "redirect-target",
            }
        elif operation_name == "ListObjectsV2":
            body = (
                b'<?xml version="1.0" encoding="UTF-8"?>'
                b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                b"<Name>bucket</Name><Prefix>prefix/</Prefix>"
                b"<KeyCount>0</KeyCount><MaxKeys>1000</MaxKeys>"
                b"<IsTruncated>false</IsTruncated></ListBucketResult>"
            )
            headers = {"content-type": "application/xml"}
        elif operation_name == "GetObject":
            body = b"payload"
            headers = {
                "content-type": "application/octet-stream",
                "x-amz-meta-sha256": hashlib.sha256(body).hexdigest(),
            }
        else:
            body = b""
            headers = {"etag": '"test-etag"'}
        headers["content-length"] = str(len(body))
        return awsrequest.AWSResponse(
            request.url, status, headers, RawResponse(body)
        )

    client.meta.events.register(f"before-send.s3.{operation_name}", before_send)
    method = {
        "ListObjectsV2": client.list_objects_v2,
        "HeadObject": client.head_object,
        "GetObject": client.get_object,
    }[operation_name]
    arguments = {"Bucket": "bucket", "Prefix": "prefix/"} if operation_name == "ListObjectsV2" else {"Bucket": "bucket", "Key": "prefix/object"}
    if status == 301:
        with pytest.raises(RuntimeError, match="second transport attempt"):
            call_once(operation_name, method, **arguments)  # type: ignore[operator]
    else:
        response = call_once(operation_name, method, **arguments)  # type: ignore[operator]
        if operation_name == "ListObjectsV2":
            assert response["IsTruncated"] is False
        elif operation_name == "HeadObject":
            assert response["ContentLength"] == 0
        else:
            assert response["Body"].read() == b"payload"
    assert sends == 1
    assert not any(
        getattr(handler, "__name__", None) == "transport_budget_guard"
        for handler in emitter._handlers.prefix_search(
            f"before-send.s3.{operation_name}"
        )
    )


def _run_pod_namespace_prefix(
    tmp_path: Path, supplied: str | None, *, fake_python: bool = False
) -> subprocess.CompletedProcess[str]:
    text = POD_RUN.read_text(encoding="utf-8")
    start = text.index('MODE="${1:?usage: pod_run.sh')
    export_line = "export DEBATE_LAUNCH_NAMESPACE"
    end = text.index("\n", text.index(export_line, start)) + 1
    prefix = text[start:end]
    env = os.environ.copy()
    if supplied is None:
        env.pop("DEBATE_LAUNCH_NAMESPACE", None)
    else:
        env["DEBATE_LAUNCH_NAMESPACE"] = supplied
    if fake_python:
        fake_bin = tmp_path / "pod-bin"
        fake_bin.mkdir()
        count = tmp_path / "uuid-count"
        _write_executable(
            fake_bin / "python3",
            "#!/usr/bin/env bash\n"
            f"printf x >> {count!s}\n"
            "printf '%s\\n' 123e4567-e89b-42d3-a456-426614174000\n",
        )
        env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    command = "set -euo pipefail\n" + prefix + "printf '%s\\n' \"$DEBATE_LAUNCH_NAMESPACE\"\n"
    return subprocess.run(
        ["bash", "-c", command, "pod_run.sh", "rlvr", "experiment"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_pod_run_preserves_a_supplied_namespace_without_generating(
    tmp_path: Path,
) -> None:
    result = _run_pod_namespace_prefix(tmp_path, "scheduler.attempt-001")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "scheduler.attempt-001"


def test_pod_run_generates_exactly_one_canonical_uuid4(tmp_path: Path) -> None:
    result = _run_pod_namespace_prefix(tmp_path, None, fake_python=True)
    assert result.returncode == 0, result.stderr
    generated = result.stdout.strip()
    parsed = uuid.UUID(generated)
    assert parsed.version == 4
    assert str(parsed) == generated
    assert (tmp_path / "uuid-count").read_text(encoding="utf-8") == "x"


@pytest.mark.parametrize(
    "namespace",
    ["", "-leading", "has/slash", "has space", "é", "a" * 129],
)
def test_pod_run_rejects_invalid_supplied_namespaces(
    tmp_path: Path, namespace: str
) -> None:
    result = _run_pod_namespace_prefix(tmp_path, namespace)
    assert result.returncode == 2
    assert "must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}" in result.stderr


def test_shell_scripts_parse() -> None:
    for script in (CKPT_SYNC, POD_RUN):
        subprocess.run(["bash", "-n", script], check=True)
