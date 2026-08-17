FROM --platform=linux/amd64 ghcr.io/engineer-man/piston@sha256:2f66b7456189c4d713aa986d98eccd0b6ee16d26c7ec5f21b30e942756fd127a

# Stock body-parser rejects CodeContests cases above roughly 100 KiB. Keep
# this as a fail-closed, one-line upstream patch so a changed base image cannot
# silently build with an unreviewed source edit.
RUN set -eu; \
    target=/piston_api/src/index.js; \
    expected='    app.use(body_parser.json());'; \
    replacement="    app.use(body_parser.json({ limit: 3 * 1024 * 1024 }));"; \
    count="$(grep -Fxc "$expected" "$target" || true)"; \
    if [ "$count" -ne 1 ]; then \
        echo "expected exactly one stock body_parser.json() line, found $count" >&2; \
        exit 1; \
    fi; \
    line="$(grep -nFx "$expected" "$target" | cut -d: -f1)"; \
    sed -i "${line}c\\${replacement}" "$target"; \
    grep -Fqx "$replacement" "$target"

# CodeContests treats each test input as exact bytes. Stock Piston silently
# appends a newline (including to empty input), changing 17.5% of RLVR cases.
# Remove only that exact upstream block and fail closed if the pinned image no
# longer has the reviewed source shape.
RUN set -eu; \
    target=/piston_api/src/job.js; \
    marker="        // Add a trailing newline if it doesn't exist"; \
    count="$(grep -Fxc "$marker" "$target" || true)"; \
    if [ "$count" -ne 1 ]; then \
        echo "expected exactly one stock stdin-newline block, found $count" >&2; \
        exit 1; \
    fi; \
    line="$(grep -nFx "$marker" "$target" | cut -d: -f1)"; \
    actual="$(sed -n "${line},$((line + 3))p" "$target")"; \
    expected="$(printf '%s\n' \
        "$marker" \
        "        if (this.stdin.slice(-1) !== '\\n') {" \
        "            this.stdin += '\\n';" \
        "        }")"; \
    if [ "$actual" != "$expected" ]; then \
        echo "stock stdin-newline block did not match reviewed bytes" >&2; \
        exit 1; \
    fi; \
    sed -i "${line},$((line + 3))d" "$target"; \
    ! grep -Fq "$marker" "$target"

# Stock Piston writes, ends, and immediately destroys the child stdin stream.
# The destroy truncates buffered large inputs. Merely deleting it is also
# unsafe: a candidate that exits without reading a large stdin makes Node emit
# an unhandled EPIPE and restart the API. Replace the exact reviewed block with
# one buffered end(data) plus an error listener. A closed candidate pipe is a
# normal child-process outcome; the candidate's exit/output still determine
# its verdict.
RUN set -eu; \
    target=/piston_api/src/job.js; \
    first='        if (event_bus === null) {'; \
    count="$(grep -Fxc "$first" "$target" || true)"; \
    if [ "$count" -ne 1 ]; then \
        echo "expected exactly one stock stdin write block, found $count" >&2; \
        exit 1; \
    fi; \
    line="$(grep -nFx "$first" "$target" | cut -d: -f1)"; \
    actual="$(sed -n "${line},$((line + 3))p" "$target")"; \
    expected="$(printf '%s\n' \
        "$first" \
        '            proc.stdin.write(this.stdin);' \
        '            proc.stdin.end();' \
        '            proc.stdin.destroy();')"; \
    if [ "$actual" != "$expected" ]; then \
        echo "stock stdin write/end/destroy block did not match reviewed bytes" >&2; \
        exit 1; \
    fi; \
    { \
        head -n "$((line - 1))" "$target"; \
        printf '%s\n' \
            '        let stdin_completion = Promise.resolve(null);' \
            '        if (event_bus === null) {' \
            '            stdin_completion = new Promise(resolve => {' \
            "                proc.stdin.once('error', resolve);" \
            "                proc.stdin.once('finish', () => resolve(null));" \
            '            });' \
            '            proc.stdin.end(this.stdin);'; \
        tail -n "+$((line + 4))" "$target"; \
    } > "${target}.new"; \
    mv "${target}.new" "$target"; \
    grep -Fqx '            proc.stdin.end(this.stdin);' "$target"; \
    ! grep -Fq '            proc.stdin.destroy();' "$target"; \
    marker='        const data = await new Promise((res, rej) => {'; \
    count="$(grep -Fxc "$marker" "$target" || true)"; \
    if [ "$count" -ne 1 ]; then \
        echo "expected exactly one child-exit promise, found $count" >&2; \
        exit 1; \
    fi; \
    line="$(grep -nFx "$marker" "$target" | cut -d: -f1)"; \
    if [ "$(sed -n "$((line + 12))p" "$target")" != '        });' ]; then \
        echo "child-exit promise did not match reviewed shape" >&2; \
        exit 1; \
    fi; \
    { \
        head -n "$((line + 12))" "$target"; \
        printf '%s\n' \
            '' \
            '        const stdin_error = await stdin_completion;' \
            '        if (' \
            '            stdin_error !== null &&' \
            "            !['EPIPE', 'ERR_STREAM_DESTROYED'].includes(stdin_error.code)" \
            '        ) {' \
            "            throw new Error('Unexpected child stdin error: ' + stdin_error.message);" \
            '        }'; \
        tail -n "+$((line + 13))" "$target"; \
    } > "${target}.new"; \
    mv "${target}.new" "$target"; \
    grep -Fqx '        const stdin_error = await stdin_completion;' "$target"

COPY healthcheck.js /usr/local/libexec/piston-healthcheck.js
