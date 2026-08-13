from pathlib import PurePath

from infra.run_debate import _docent_launch_id, _docent_run_dir


def test_docent_run_dirs_namespace_concurrent_runs():
    assert _docent_run_dir("debate-lr1e-5", launch_id="pid-100") != _docent_run_dir(
        "debate-lr2e-5", launch_id="pid-100"
    )
    assert (
        _docent_run_dir("debate-lr1e-5", launch_id="pid-100")
        == "docent/debate-lr1e-5/pid-100"
    )


def test_docent_run_dirs_namespace_same_run_across_launches():
    first = _docent_run_dir("debate", launch_id=_docent_launch_id(100))
    second = _docent_run_dir("debate", launch_id=_docent_launch_id(101))

    assert first != second


def test_docent_run_dir_is_stable_within_one_launch():
    launch_id = _docent_launch_id(100)

    assert _docent_run_dir("debate", launch_id=launch_id) == _docent_run_dir(
        "debate", launch_id=launch_id
    )


def test_docent_run_dir_sanitizes_separators_without_collisions_or_escape():
    slash = PurePath(_docent_run_dir("../../arm/a", launch_id="pid-100"))
    backslash = PurePath(_docent_run_dir(r"..\..\arm\a", launch_id="pid-100"))

    assert slash != backslash
    for path in (slash, backslash):
        assert path.parts[0] == "docent"
        assert len(path.parts) == 3
        assert all(part not in {"", ".", ".."} for part in path.parts[1:])


def test_docent_run_dir_sanitizes_injected_launch_id():
    path = PurePath(_docent_run_dir("debate", launch_id="../../other"))

    assert path.parts[0:2] == ("docent", "debate")
    assert len(path.parts) == 3
    assert path.parts[2] not in {"", ".", ".."}
