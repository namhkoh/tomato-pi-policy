"""Native worker admission is read-only and rejects unrelated Kit processes."""
import pytest
from .native_generated_pair import check_process_admission


def test_only_worker_or_exact_parent_child_pair_allowed():
    assert check_process_admission([dict(ProcessId=12, ParentProcessId=3)], 12)["allowed_kit_pids"] == [12]
    result = check_process_admission([dict(ProcessId=12, ParentProcessId=9), dict(ProcessId=9, ParentProcessId=3)], 12, 9)
    assert result["allowed_kit_pids"] == [9, 12]


@pytest.mark.parametrize("rows,worker,parent", [
    ([], 12, None),
    ([dict(ProcessId=12, ParentProcessId=3), dict(ProcessId=14, ParentProcessId=3)], 12, None),
    ([dict(ProcessId=12, ParentProcessId=3), dict(ProcessId=9, ParentProcessId=3)], 12, 9),
    ([dict(ProcessId=12, ParentProcessId=9)], 12, 9),
    ([dict(ProcessId=12, ParentProcessId=9), dict(ProcessId=9, ParentProcessId=3)], 12, 12),
    ([dict(ProcessId=12, ParentProcessId=9)], True, None),
    ([dict(ProcessId=12, ParentProcessId=9)], 12, True),
])
def test_bad_or_unrelated_processes_block(rows, worker, parent):
    with pytest.raises(ValueError):
        check_process_admission(rows, worker, parent)
