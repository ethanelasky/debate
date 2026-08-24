"""The pod-side deadman: what stops a pod when its lifecycle owner dies.

jobd and scheduler_debate_supervisor.py both set POD_IDLE_STOP=0, because they
own the pod's lifecycle themselves. Both of them run on a laptop. If that
laptop crashes, sleeps or loses the network, nothing on the pod stops it
billing -- the one failure the owner cannot protect against, and the most
expensive one available here.

These tests pin the invariants of the backstop that covers it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
POD_RUN = REPO_ROOT / "scripts" / "pod_run.sh"
IDLE_STOP = REPO_ROOT / "scripts" / "pod_idle_stop.sh"


def test_pod_run_is_syntactically_valid():
    subprocess.run(["bash", "-n", str(POD_RUN)], check=True)


def test_turning_the_primary_watchdog_off_still_arms_a_deadman():
    body = POD_RUN.read_text()
    assert 'pod_idle_stop.sh" --deadman' in body, (
        "POD_IDLE_STOP=0 must arm a deadman, not leave the pod unwatched"
    )
    assert "POD_DEADMAN_MINUTES" in body


def test_a_deadman_is_not_mistaken_for_a_primary_watchdog():
    """Otherwise a later run asking for 30-minute cover silently gets 3 hours."""
    body = POD_RUN.read_text()
    exclusion = re.search(r"case \"\$args\" in ([^\n]*?) continue", body)
    assert exclusion is not None, "live_watchdog_pids lost its exclusion case"
    assert "--deadman" in exclusion.group(1)
    assert "--once" in exclusion.group(1), "the pre-existing --once exclusion"


def test_the_deadman_defers_to_the_reaper_rather_than_racing_it():
    """It is a backstop: its timeout must dwarf the primary watchdog's.

    jobd retires an idle pod after 10 minutes. A deadman anywhere near that
    would start stopping pods the owner still wants, which turns a safety net
    into an outage.
    """
    body = POD_RUN.read_text()
    deadman = re.search(r'POD_DEADMAN_MINUTES:-(\d+)', body)
    assert deadman is not None
    primary = re.search(r'IDLE_MINUTES:-(\d+)', IDLE_STOP.read_text())
    assert primary is not None
    assert int(deadman.group(1)) >= 4 * int(primary.group(1)), (
        f"deadman {deadman.group(1)}m is not comfortably longer than the "
        f"primary watchdog's {primary.group(1)}m"
    )


def test_the_deadman_can_be_given_up_deliberately():
    """An interactive pod you mean to leave idle is a legitimate case."""
    body = POD_RUN.read_text()
    assert '"$POD_DEADMAN_MINUTES" = 0' in body
    assert "DISABLED" in body


def test_the_watchdog_ignores_the_deadman_flag():
    """--deadman is only a ps marker; pod_idle_stop.sh must not choke on it."""
    body = IDLE_STOP.read_text()
    handled = re.findall(r'\[ "\$\{1:-\}" = "(--[a-z]+)" \]', body)
    assert handled == ["--once"], (
        "pod_idle_stop.sh parses argv beyond --once; --deadman may no longer "
        "be inert"
    )
