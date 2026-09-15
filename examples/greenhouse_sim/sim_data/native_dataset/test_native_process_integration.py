"""CPU-only guard wiring tests; never launch, stop or modify native workers."""
import ast
from pathlib import Path

from ..native_capture_v4 import campaign
from ..native_capture_v5 import campaign_queue
from . import native_process_guard


def test_campaign_delegates_without_discarding_blockers(monkeypatch):
    blockers = [{'ProcessId': 123, 'reason': 'synthetic_unknown_native'}]
    monkeypatch.setattr(native_process_guard, 'native_processes', lambda: blockers)
    assert campaign.native_processes() is blockers


def test_guard_failure_propagates(monkeypatch):
    import pytest
    def fail():
        raise ValueError('Synthetic unavailable process inventory')
    monkeypatch.setattr(native_process_guard, 'native_processes', fail)
    with pytest.raises(ValueError, match='unavailable'):
        campaign.native_processes()


def test_both_coordinators_pin_guard_implementation():
    for module in (campaign, campaign_queue):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and isinstance(node.func.value, ast.Name)
                 and node.func.value.id == 'native_process_guard'
                 and node.func.attr == 'implementation_bindings']
        assert calls, module.__name__


def test_guard_bindings_are_actual_source_bytes():
    from hashlib import sha256
    bindings = native_process_guard.implementation_bindings()
    assert bindings
    assert str(Path(native_process_guard.__file__).resolve()) in bindings
    for path, expected in bindings.items():
        assert sha256(Path(path).read_bytes()).hexdigest() == expected
