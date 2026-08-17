#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

host_port="${PISTON_HOST_PORT:-2000}"
base_url="http://127.0.0.1:${host_port}"
runtime_version="3.12.0"
runtime_archive="data/piston/packages/python/${runtime_version}/pkg.tar.gz"
runtime_archive_sha256="abc40b3231fc7e713799da2cd79844545c72b3904a4d2ffcc28c4d133ed21d0b"

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "required command is missing: $1" >&2
        exit 1
    fi
}

check_host() {
    require_command docker
    require_command curl
    require_command jq
    require_command sha256sum
    require_command timeout

    if [[ "$(uname -s)" != "Linux" ]]; then
        echo "Piston must run on a dedicated Linux host" >&2
        exit 1
    fi
    case "$(uname -m)" in
        x86_64 | amd64) ;;
        *)
            echo "the pinned Piston image requires an amd64 host" >&2
            exit 1
            ;;
    esac
    if [[ ! -f /sys/fs/cgroup/cgroup.controllers ]]; then
        echo "cgroup v2 is not mounted at /sys/fs/cgroup" >&2
        exit 1
    fi
    if awk '$0 ~ / - cgroup / { found = 1 } END { exit(found ? 0 : 1) }' \
        /proc/self/mountinfo; then
        echo "a cgroup v1 mount is active; Piston requires unified cgroup v2" >&2
        exit 1
    fi

    docker info >/dev/null
    docker compose version >/dev/null
}

wait_for_root() {
    local attempt
    for attempt in $(seq 1 60); do
        if curl --fail --silent --show-error --connect-timeout 2 \
            --max-time 3 "$base_url/" \
            | jq --exit-status \
                '.message | type == "string" and test("^Piston v[^[:space:]]+$")' \
                >/dev/null 2>&1; then
            return
        fi
        sleep 1
    done
    echo "Piston did not answer GET / within 60 seconds" >&2
    exit 1
}

runtime_ready() {
    curl --fail --silent --show-error --connect-timeout 2 --max-time 5 \
        "$base_url/api/v2/runtimes" \
        | jq --exit-status --arg version "$runtime_version" \
            'any(.[]; .language == "python" and .version == $version)' \
            >/dev/null
}

verify_runtime_archive() {
    if [[ ! -f "$runtime_archive" ]]; then
        echo "installed runtime archive is missing: $runtime_archive" >&2
        return 1
    fi
    if ! printf '%s  %s\n' "$runtime_archive_sha256" "$runtime_archive" \
        | sha256sum --check --status; then
        echo "python-${runtime_version} runtime archive checksum mismatch" >&2
        return 1
    fi
}

verify_execution() {
    local container_id restart_count_before restart_count_after response
    container_id="$(docker compose ps --quiet piston)"
    if [[ -z "$container_id" ]]; then
        echo "Piston container is not running" >&2
        return 1
    fi
    restart_count_before="$(docker inspect --format '{{.RestartCount}}' "$container_id")"
    response="$(
        jq --null-input --compact-output --arg version "$runtime_version" '{
            language: "python",
            version: $version,
            files: [{
                name: "health.py",
                content: "import sys\ndata = sys.stdin.buffer.read()\nprint(len(data), data[-1:].hex())",
                encoding: "utf8"
            }],
            stdin: ("a" * (600 * 1024)),
            run_timeout: 2000,
            run_cpu_time: 2000
        }' \
            | curl --fail --silent --show-error --connect-timeout 2 \
                --max-time 5 -H 'Content-Type: application/json' \
                --data-binary @- "$base_url/api/v2/execute"
    )"
    if ! jq --exit-status --arg version "$runtime_version" '
        .language == "python"
        and .version == $version
        and .run.status == null
        and .run.code == 0
        and .run.signal == null
        and .run.stdout == "614400 61\n"
    ' <<<"$response" >/dev/null; then
        echo "Piston execution probe failed, truncated stdin, or mutated unterminated stdin" >&2
        return 1
    fi

    response="$(
        jq --null-input --compact-output --arg version "$runtime_version" '{
            language: "python",
            version: $version,
            files: [{name: "health.py", content: "pass\n", encoding: "utf8"}],
            stdin: ("a" * (600 * 1024)),
            run_timeout: 2000,
            run_cpu_time: 2000
        }' \
            | curl --fail --silent --show-error --connect-timeout 2 \
                --max-time 5 -H 'Content-Type: application/json' \
                --data-binary @- "$base_url/api/v2/execute"
    )"
    if ! jq --exit-status --arg version "$runtime_version" '
        .language == "python"
        and .version == $version
        and .run.status == null
        and .run.code == 0
        and .run.signal == null
        and .run.stdout == ""
    ' <<<"$response" >/dev/null; then
        echo "Piston large-stdin early-exit probe failed" >&2
        return 1
    fi
    sleep 1
    restart_count_after="$(docker inspect --format '{{.RestartCount}}' "$container_id")"
    if [[ "$restart_count_after" != "$restart_count_before" ]]; then
        echo "Piston API restarted during execution probes" >&2
        return 1
    fi
    wait_for_root
}

install_runtime() {
    if runtime_ready; then
        verify_runtime_archive
        echo "python-${runtime_version} is already installed"
        return
    fi

    echo "installing exact Piston runtime python-${runtime_version}"
    curl --fail --silent --show-error --connect-timeout 5 --max-time 1200 \
        -H 'Content-Type: application/json' \
        -X POST \
        --data "{\"language\":\"python\",\"version\":\"${runtime_version}\"}" \
        "$base_url/api/v2/packages" >/dev/null

    if ! runtime_ready; then
        echo "package install returned but python-${runtime_version} is not ready" >&2
        exit 1
    fi
    verify_runtime_archive
}

health() {
    wait_for_root
    if ! runtime_ready; then
        echo "GET /api/v2/runtimes lacks exact python-${runtime_version}" >&2
        exit 1
    fi
    verify_runtime_archive
    verify_execution
    echo "Piston is healthy at ${base_url} with python-${runtime_version}; execution and exact stdin passed"
}

stop_after_failed_bootstrap() {
    local status="$?"
    local running_ids
    if [[ "$status" -ne 0 ]]; then
        echo "Piston bootstrap failed; stopping the service" >&2
        if ! timeout --kill-after=5s 30s docker compose stop piston; then
            echo "ERROR: failed to stop Piston after bootstrap failure" >&2
        fi
        if ! running_ids="$(
            timeout --kill-after=2s 10s \
                docker compose ps --status running --quiet piston 2>/dev/null
        )"; then
            echo "ERROR: unable to verify whether Piston stopped" >&2
        elif [[ -n "$running_ids" ]]; then
            echo "ERROR: Piston is still running; operator cleanup is required" >&2
        else
            echo "Piston is stopped after bootstrap failure" >&2
        fi
    fi
    exit "$status"
}

case "${1:-bootstrap}" in
    bootstrap)
        trap stop_after_failed_bootstrap EXIT
        check_host
        mkdir -p data/piston/packages
        timeout --kill-after=30s 20m \
            docker compose up --detach --build piston
        wait_for_root
        install_runtime
        health
        trap - EXIT
        ;;
    check-host)
        check_host
        echo "host satisfies Piston's amd64 and cgroup v2 prerequisites"
        ;;
    health)
        require_command docker
        require_command curl
        require_command jq
        require_command sha256sum
        health
        ;;
    *)
        echo "usage: $0 [bootstrap|check-host|health]" >&2
        exit 2
        ;;
esac
