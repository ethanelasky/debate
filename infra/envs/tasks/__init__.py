"""Task registry: dataset.type -> TaskFamily. Adding a task = one module
implementing TaskFamily + one entry here."""

from infra.envs.tasks.base import TaskFamily
from infra.envs.tasks.codecontests import CodeContestsFamily
from infra.envs.tasks.math import MathFamily
from infra.envs.tasks.monitoringbench import MonitoringBenchFamily

TASK_FAMILIES: dict[str, type[TaskFamily]] = {
    "codecontests": CodeContestsFamily,
    "math": MathFamily,
    "monitoringbench": MonitoringBenchFamily,
}


def get_family(name) -> TaskFamily:
    """Fresh instance per call: source() may set per-config state on the
    family (e.g. the codecontests grading timeout), which must not leak
    between builds in one process."""
    if name not in TASK_FAMILIES:
        raise ValueError(f"dataset.type is {name!r}; declare one of {sorted(TASK_FAMILIES)} explicitly")
    return TASK_FAMILIES[name]()


__all__ = ["TaskFamily", "TASK_FAMILIES", "get_family"]
