from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from codecontests_executor.protocol import encode_envelope, sign_payload, static_identity


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_lima_deploy_shell_and_unit_policy_are_static() -> None:
    for relative in (
        "scripts/deploy_codecontests_executor_lima.sh",
        "scripts/deploy_codecontests_executor_lima_guest.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", str(REPO_ROOT / relative)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr

    deploy_script = (
        REPO_ROOT / "scripts/deploy_codecontests_executor_lima.sh"
    ).read_text(encoding="utf-8")
    assert 'readonly DEFAULT_HOST_PORT="18081"' in deploy_script
    assert 'DEFAULT_ARTIFACT_ROOT="${REPO_ROOT}/../ai-debate-run-artifacts/' in deploy_script
    assert "/Users/" not in deploy_script

    unit = (
        REPO_ROOT
        / "deploy/codecontests-executor/codecontests-executor.lima.service.in"
    ).read_text(encoding="utf-8")
    assert "/opt/palaestra" not in unit
    assert "ExecStart=/usr/bin/python3.12 -B -m codecontests_executor.service" in unit
    assert "--bind 127.0.0.1 --port 8080" in unit
    assert "CPUQuota=400%" in unit
    assert "CPUAffinity=0 1 2 3" in unit
    assert "MemoryMax=25769803776" in unit
    assert "MemorySwapMax=0" in unit
    assert "NoNewPrivileges=no" in unit
    assert "MemoryDenyWriteExecute=no" in unit
    assert "Delegate=cpu memory pids" in unit
    assert (
        "BindReadOnlyPaths=@ROOTFS_MANIFEST_PATH@:"
        "/var/lib/codecontests-executor/rootfs.manifest.json"
    ) in unit

    hostile_unit = (
        REPO_ROOT
        / "deploy/codecontests-executor/"
        "codecontests-executor-hostile-probe.lima.service.in"
    ).read_text(encoding="utf-8")
    assert "Type=oneshot" in hostile_unit
    assert "probe_codecontests_executor_live.py" in hostile_unit
    assert "CPUQuota=400%" in hostile_unit
    assert "CPUAffinity=0 1 2 3" in hostile_unit
    assert "MemoryMax=25769803776" in hostile_unit
    assert "ReadWritePaths=@PROBE_OUTPUT_DIR@" in hostile_unit

    mount = (
        REPO_ROOT
        / "deploy/codecontests-executor/codecontests-rootfs.lima.mount.in"
    ).read_text(encoding="utf-8")
    assert "What=/var/lib/codecontests-executor/rootfs" in mount
    assert "Where=/var/lib/codecontests-executor/rootfs" in mount
    assert "Options=bind,ro,nosuid,nodev" in mount


def test_capture_identity_fetches_and_verifies_same_signed_payload_twice(
    tmp_path: Path,
) -> None:
    hmac_key = b"h" * 32
    bearer = b"b" * 32
    identity = static_identity(
        service_id="capture-test",
        launcher_sha256="1" * 64,
        cgroup_gate_sha256="2" * 64,
        rootfs_manifest_sha256="3" * 64,
        rootfs_manifest_file_sha256="4" * 64,
        server_bundle_sha256="5" * 64,
    )
    response = encode_envelope(sign_payload(identity, hmac_key))
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal request_count
            assert self.path == "/v1/identity"
            assert self.headers["Authorization"] == f"Bearer {bearer.decode()}"
            request_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        bearer_path = tmp_path / "bearer"
        hmac_path = tmp_path / "hmac-key"
        identity_path = tmp_path / "identity.json"
        envelope_path = tmp_path / "identity.envelope.json"
        bearer_path.write_bytes(bearer + b"\n")
        hmac_path.write_bytes(hmac_key + b"\n")
        os.chmod(tmp_path, 0o700)
        os.chmod(bearer_path, 0o600)
        os.chmod(hmac_path, 0o600)
        completed = subprocess.run(
            [
                "python3",
                "-B",
                str(REPO_ROOT / "scripts/capture_codecontests_executor_identity.py"),
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--bearer-file",
                str(bearer_path),
                "--hmac-key-file",
                str(hmac_path),
                "--identity-output",
                str(identity_path),
                "--envelope-output",
                str(envelope_path),
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert completed.returncode == 0, completed.stderr
    assert request_count == 2
    assert json.loads(identity_path.read_text(encoding="utf-8")) == identity
    assert json.loads(envelope_path.read_text(encoding="utf-8"))["payload"] == identity
    assert stat.S_IMODE(identity_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(envelope_path.stat().st_mode) == 0o600
    summary = json.loads(completed.stdout)
    assert summary["service_id"] == "capture-test"
    assert summary["server_bundle_sha256"] == "5" * 64
