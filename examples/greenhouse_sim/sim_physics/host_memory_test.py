import builtins
import json
import sys

import pytest

from sim_physics import host_memory as memory


def healthy():
    return dict(platform='win32', supported=True, read_succeeded=True,
                committed_bytes=32 * memory.GIB, commit_limit_bytes=128 * memory.GIB,
                physical_total_bytes=64 * memory.GIB, physical_available_bytes=20 * memory.GIB,
                kernel_paged_bytes=memory.GIB, kernel_nonpaged_bytes=memory.GIB)


def test_reserve_pass_does_not_claim_runtime_capacity():
    result = memory.assess_memory(healthy())
    assert result['allowed'] and result['checked']
    assert result['guarantees_runtime_capacity'] is False


@pytest.mark.parametrize('field,value,reason', [
    ('committed_bytes', 120 * memory.GIB, 'commit'),
    ('committed_bytes', 129 * memory.GIB, 'commit'),
    ('physical_available_bytes', 3 * memory.GIB, 'physical'),
])
def test_low_reserve_rejects(field, value, reason):
    snapshot = healthy(); snapshot[field] = value
    result = memory.assess_memory(snapshot)
    assert not result['allowed'] and result['checked']
    assert any(reason in text for text in result['reasons'])


@pytest.mark.parametrize('field,value', [
    ('committed_bytes', None), ('commit_limit_bytes', 0),
    ('physical_available_bytes', -1), ('physical_available_bytes', 65 * memory.GIB),
    ('kernel_paged_bytes', float('nan')), ('kernel_nonpaged_bytes', True),
    ('read_succeeded', False),
])
def test_failed_or_invalid_native_read_cannot_authorize(field, value):
    snapshot = healthy(); snapshot[field] = value
    result = memory.assess_memory(snapshot)
    assert not result['allowed'] and not result['checked']


def test_other_platform_explicitly_has_no_check():
    result = memory.assess_memory(dict(platform='linux', supported=False))
    assert result['allowed'] and not result['checked']
    assert result['state'] == 'unsupported_platform_not_checked'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows read-only native API')
def test_installed_windows_counters_are_valid_bytes():
    snapshot = memory.memory_snapshot()
    assert snapshot['read_succeeded']
    assert snapshot['commit_limit_bytes'] > memory.GIB
    assert 0 <= snapshot['physical_available_bytes'] <= snapshot['physical_total_bytes']
    assert memory.assess_memory(snapshot)['checked']


def test_blocked_benchmark_writes_receipt_without_importing_isaac(tmp_path, monkeypatch):
    from sim_physics.benchmark import main
    snapshot = healthy(); snapshot['committed_bytes'] = 120 * memory.GIB
    monkeypatch.setattr(memory, 'memory_snapshot', lambda: snapshot)
    original_import = builtins.__import__
    def checked_import(name, *args, **kwargs):
        if name == 'isaacsim':
            pytest.fail('Low-memory startup must not import or create SimulationApp')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    output = tmp_path / 'blocked'
    assert main(['--output', str(output), '--scene', 'package', '--full-robot-probe',
                 '--sparse-contacts', '--solver', 'PGS', '--spring-mode', 'implicit_effort',
                 '--force-newton', '0', '--physics-hz', '240', '--seconds', '20']) == 2
    report = json.loads((output / 'report.json').read_text())
    assert report['state'] == 'blocked_host_memory'
    assert report['simulation_started'] is False and report['training_eligible'] is False
    assert set(p.name for p in output.iterdir()) == {'report.json'}
