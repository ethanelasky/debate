# CodeContests executor deployment

This canary executor runs only in the dedicated credential-free Ubuntu VM.
The HTTP listener is exactly `127.0.0.1:8080`; Lima forwards it to host port
`18081` by default. Non-loopback clients must use HTTPS, and the client rejects
redirects.

## Immutable inputs

- pinned official release archive
  `/opt/gvisor/20260721.0/gvisor.tar.bz2` (131,637,510 bytes, SHA-512
  `27a6d5103c36ef11c7e8c6158b7039ac43623d77147227d4e6d083835d0cd20fe3b100680ffeca9e6691ee7ba31de19b613c10ef860771e861e7971b24bc2947`);
- independently pinned extracted `/opt/gvisor/20260721.0/runsc`
  (96,196,105 bytes, SHA-512
  `23465f6a5d7c1da2c31ac25af95e0db783e2776f0fb2afb3a3c421b8928c51d7d4d3a680c555ff821d12504d51af985a226855f56d22dc258d3058b537995734`),
  whose exact version output is release `20260721.0`, OCI spec `1.2.1`;
- root-owned `/var/lib/codecontests-executor/rootfs`, mounted at that same
  canonical path as a dedicated read-only bind mount;
- root-owned `/opt/palaestra/codecontests_executor` containing only the
  inventory-listed `.py` sources, with no cache, symlink, or extra entry;
- root-owned rootfs manifest, server source inventory, client provenance,
  environment, bearer, and HMAC files.

The pinned gofer rejects a proc-FD OCI root (`self-bind-mounting ... invalid
argument`), so OCI uses the canonical ordinary rootfs path. The service gives
it no mount helper/CAP_SYS_ADMIN workaround. Instead it freezes the rootfs fd
for inode evidence and checks the exact mount ID/root/source/options plus
dev/inode/mode/ownership before and after every request. Startup measures the
entire tree twice (content, metadata, xattrs, links, and exact path set). The
supervisor additionally snapshots every path's topology, inode, size, link,
mode, ownership, device, mtime, ctime, and symlink target through the retained
root descriptor, then revalidates that full metadata tree before and after
each request. This avoids re-hashing the whole rootfs on every case; the held
dedicated read-only mount prevents request writes, while any backing-tree
content or topology mutation changes the revalidated inode metadata. The
hostile deployment suite still repeats the full content measurement before
sign-off and shutdown.

The v2 rootfs must expose `sympy==1.14.0` and `mpmath==1.3.0` to
`/usr/bin/python3 -I`. Build those packages only from a root-owned offline
wheelhouse and a reviewed `--require-hashes` lock containing both exact pins;
do not give the build VM credentials or permit an online `pip install`. After
staging, normalize the credential-free tree to root ownership without
following symlinks or crossing mounted filesystems, and remove every setuid or
setgid bit:

```sh
sudo find /var/lib/codecontests-executor/rootfs -xdev -exec chown -h 0:0 -- {} +
sudo find /var/lib/codecontests-executor/rootfs -xdev -perm /6000 -exec chmod a-s -- {} +
```

Then construct a candidate normalized artifact. The builder chroots into
the tree, verifies Python and both package versions, verifies that isolated
mode can import them, and prints the new digest:

```sh
sudo scripts/build_codecontests_rootfs_artifact.sh --candidate \
  /var/lib/codecontests-executor/rootfs \
  /var/lib/codecontests-executor/rootfs.candidate.tar
```

The normalized artifact was built twice byte-for-byte identically and is
pinned as SHA-256
`83e694da5d1e0b94700da2a195d760527ce609ea631f7302ec930666bae136d0`
in both `ROOTFS_SHA256` and `EXPECTED_SHA256`. Reconstruct the exact pinned
artifact and manifest; both commands refuse overwrite unless `--force` is
explicit:

```sh
sudo scripts/build_codecontests_rootfs_artifact.sh \
  /var/lib/codecontests-executor/rootfs \
  /var/lib/codecontests-executor/rootfs.tar
sudo /usr/bin/python3.12 -B scripts/build_codecontests_rootfs_manifest.py \
  --rootfs /var/lib/codecontests-executor/rootfs \
  --rootfs-artifact /var/lib/codecontests-executor/rootfs.tar \
  --output /var/lib/codecontests-executor/rootfs.manifest.json
```

Stage only the server sources and create the root-owned per-file pre-import
inventory:

```sh
sudo /usr/bin/python3.12 -I -B \
  scripts/measure_codecontests_executor_bundle.py \
  --package-dir /opt/palaestra/codecontests_executor \
  --output /etc/codecontests-executor/server-bundle.inventory.json
```

Put the printed bundle digest and manifest `file_sha256` in the environment
file. `ExecStartPre` is a standalone non-package verifier: it validates every
source against the inventory and rejects extras/pyc before Python imports the
service or reads credentials. The service repeats the bundle check internally.

Build and install the trusted-driver provenance JSON with
`measure_codecontests_client_provenance.py`. Requests bind the exact
`client.py`, `protocol.py`, and verifier digests.

## Runtime isolation

Each stdin case gets a fresh rootful `runsc run` OCI bundle. A nonce-holding
UID-0 monitor has only KILL, SETUID, SETGID, SETPCAP, and in-sandbox
SYS_PTRACE. KILL lets that monitor terminate the candidate after its credential
transition; it is never retained by the candidate. The monitor starts a fresh
nonce-free interpreter and opens the candidate gate only after that process
proves UID/GID 65534, no supplementary groups, all five capability sets empty,
and `PR_GET_NO_NEW_PRIVS=1`. The workload has network none, root readonly,
`/dev` a read-only 64 KiB tmpfs, and `/tmp` a separate read-only 64 KiB tmpfs.
There is no candidate-writable filesystem.

The candidate interpreter uses `python3 -I -B -c`: isolated safe-path mode
removes the working directory and candidate-controlled environment from import
resolution while retaining standard `site` initialization and its commonly
generated `exit()`/`quit()` conveniences. Startup attests both conveniences
and the absence of `""`/`/tmp` from `sys.path`; the live acceptance matrix
executes the same probe with the pinned SymPy/mpmath imports.

The semantic task maximum is exactly two: candidate main plus one concurrent
same-process pthread. Pinned gVisor does not charge the already-existing fresh
interpreter when it changes real UID, so raw `RLIMIT_NPROC=1` supplies the one
thread slot and a second concurrent thread receives EAGAIN. A candidate that
catches that denial and exits successfully remains an executed result (with
signed denial evidence); an uncaught denial is `PROCESS_LIMIT`. Seccomp returns
EPERM for `fork`, `vfork`, process-clone layouts, namespace clone flags, and
all non-pinned legacy-clone layouts. It traces only the exact complete
CPython/glibc pthread legacy-clone layout; the ptrace monitor then requires its
matching clone event and exact task inventory. Candidate `clone3` receives
ENOSYS before ptrace or candidate-memory inspection, forcing pinned glibc onto
the fully BPF-validated legacy pthread path. User/mount namespace transitions,
mount API families, chroot/pivot-root, and process-group/session escape syscalls
remain denied across exec. Other guest semantic limits are independent:
RLIMIT_AS is exactly 4 GiB, stdout and stderr are each capped at 2 MiB, and
RLIMIT_FSIZE is exactly 2 MiB.

The systemd delegated root stays process-free. Each request creates an outer
cgroup with `controller/` (the gated runsc CLI) and `candidate/` (OCI runtime)
leaves. A pinned helper blocks before exec; the parent configures the outer
cgroup, moves and verifies the blocked PID, then releases it. At the private
READY boundary the entire request subtree is frozen while every runtime PID,
PID starttime, ancestor, process group, session, controller membership, and cap
is re-read twice; that exact descendant/helper inventory must equal the
controller and candidate cgroups. The blocked pre-exec gate PID/starttime is
captured before release, and an immutable union retains every later observed
PID/starttime through cleanup even after individual helpers exit. Same-token
ancestry, process-group, or session drift fails closed. The subtree is then
unfrozen before candidate release. The same equality is required at the later
frozen teardown boundary, and cleanup proves that no recorded,
container-identified, session-associated, or descendant helper remains. Outer caps are
pids 128, one CPU, and semantic AS +384 MiB gVisor
headroom. READY is a second stdin gate: only after cgroup/controller/event
attestation and the CPU baseline does the host send `G` followed by byte-exact
candidate stdin. Aggregate CPU/memory/pids event evidence is signed but can
only produce `UNKNOWN/AMBIGUOUS_RESOURCE_OR_CONTROLLER_FAILURE`; only private
guest monitor/kernel evidence can identify a candidate resource failure.
The nonce-bound terminal status also signs whether the tracer's immutable
teardown ledger targeted the main candidate. After the runsc, controller,
cgroup, and guest-resource gates are all clean, every otherwise unexplained
main `SIGKILL` remains `UNKNOWN`; a tracer-owned or otherwise evidenced
resource `SIGKILL` retains its specific limit verdict. The executor never
turns an origin-ambiguous `SIGKILL` into a candidate runtime failure.
Before any guest forced teardown, the ptrace owner sends one process-group
`SIGSTOP` pulse and drains `wait4(__WALL | WNOHANG)` into a physical held-task
set. The only legal new task is the exact admitted pthread clone; it remains
provisional until the matching clone event names the same initial-stop TID.
Procfs alone never admits one, and denied creation calls must return without a
new task.
The tracer claims and sends its single classified `SIGKILL` only after every
procfs-known task is held and every creation transition is attributed.
Every subtree is recursively killed while still frozen when applicable, then
proved empty and removed on every path.
The hostile acceptance artifact is linked only provisionally: its retained
host-only marker is rehashed and its full stable metadata is remeasured at the
terminal publication boundary, and a failed terminal check removes the newly
published artifact inode.

Guest RLIMIT_CPU remains a candidate-specific fail-safe, but is intentionally
unreachable on valid requests: exact 1x wall precedes
`RLIMIT_CPU=ceil(raw)+1` while `cpu.max=100000 100000` limits the candidate to
one CPU. The outer counter is a conservative infrastructure failsafe at
`2 * effective_guest_cpu_seconds + 1s`, not a science limit. The one-CPU quota
also prevents the optional worker thread from shortening the exact wall budget
by consuming CPU in parallel.

The unit pins `MemoryMax=25769803776` (24 GiB), `MemorySwapMax=0`, and
`TasksMax=768`. The VM has 32 GiB, while each request has a separate cgroup
ceiling of 4 GiB candidate RSS plus 384 MiB of runsc headroom. RLIMIT_AS is a
virtual-address-space limit, not an RSS reservation: four active sandboxes do
not preallocate 16 GiB. The 24 GiB aggregate ceiling nevertheless covers four
simultaneous per-request ceilings (17.5 GiB total) and leaves 6.5 GiB for the
service, shared page cache, and measurement overhead without exposing all VM
memory to the service.
Startup refuses swap, sysctl, CPU-affinity, cgroup-cap, runtime-version,
identity, rootfs, source, or teardown drift.

## Identity freeze

Install/start the unit and fetch `/v1/identity` through the live loopback
service. A trusted driver-side utility must verify its HMAC envelope, save only
the payload to the root-owned frozen identity file, then immediately re-fetch
and require exact equality. Do not use `--print-identity` as the authoritative
production capture: running outside the unit measures a different cgroup.
That option is only an offline prestart diagnostic.
