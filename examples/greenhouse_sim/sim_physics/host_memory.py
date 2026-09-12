"""Read-only Windows launch reserve, not a simulator performance certificate.

GetPerformanceInfo returns page counts, NOT byte counts. See Microsoft's
PERFORMANCE_INFORMATION documentation. No process, pagefile or driver edits.
"""
import ctypes
import sys
from datetime import datetime, timezone

GIB = 1024**3
MINIMUM_COMMIT_HEADROOM = 16 * GIB
MINIMUM_AVAILABLE_PHYSICAL = 4 * GIB


class PerformanceInformation(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal',
            'PhysicalAvailable', 'SystemCache', 'KernelTotal', 'KernelPaged',
            'KernelNonpaged', 'PageSize')
    ] + [(name, ctypes.c_uint32) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]


def memory_snapshot():
    result = dict(platform=sys.platform, sampled_utc=datetime.now(timezone.utc).isoformat(),
                  supported=sys.platform == 'win32', source='GetPerformanceInfo')
    if not result['supported']:
        return result
    try:
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        function = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
        function.argtypes = [ctypes.POINTER(PerformanceInformation), ctypes.c_uint32]
        function.restype = ctypes.c_int
        if not function(ctypes.byref(info), info.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        if not info.PageSize:
            raise ValueError('Missing native page size')
        for key, field in (
            ('committed_bytes', 'CommitTotal'), ('commit_limit_bytes', 'CommitLimit'),
            ('physical_total_bytes', 'PhysicalTotal'), ('physical_available_bytes', 'PhysicalAvailable'),
            ('kernel_paged_bytes', 'KernelPaged'), ('kernel_nonpaged_bytes', 'KernelNonpaged')):
            result[key] = int(getattr(info, field)) * int(info.PageSize)
        result['read_succeeded'] = True
    except Exception as exc:
        result.update(read_succeeded=False, error=type(exc).__name__ + ': ' + str(exc))
    return result


def assess_memory(snapshot):
    """Conservative engineering reserve; a pass does not guarantee enough RAM.

    Native90 exhausted commit during full-context setup with <12 GB available
    commit before launch.16 GiB is a startup floor, NOT a measured peak bound.
    Other OSes retain their existing behavior, explicitly without this check.
    """
    result = dict(snapshot=dict(snapshot), policy='full_greenhouse_windows_launch_reserve_v1',
                  minimum_commit_headroom_bytes=MINIMUM_COMMIT_HEADROOM,
                  minimum_available_physical_bytes=MINIMUM_AVAILABLE_PHYSICAL,
                  allowed=False, checked=False, reasons=[], guarantees_runtime_capacity=False)
    if snapshot.get('supported') is False and snapshot.get('platform') != 'win32':
        result.update(allowed=True, state='unsupported_platform_not_checked')
        return result
    keys = ('committed_bytes', 'commit_limit_bytes', 'physical_total_bytes',
            'physical_available_bytes', 'kernel_paged_bytes', 'kernel_nonpaged_bytes')
    if (snapshot.get('read_succeeded') is not True
            or any(type(snapshot.get(k)) is not int or snapshot[k] < 0 for k in keys)
            or snapshot['commit_limit_bytes'] <= 0 or snapshot['physical_total_bytes'] <= 0
            or snapshot['physical_available_bytes'] > snapshot['physical_total_bytes']):
        result.update(state='memory_read_unavailable', reasons=['Valid native memory counters unavailable'])
        return result
    result['checked'] = True
    headroom = snapshot['commit_limit_bytes'] - snapshot['committed_bytes']
    result['commit_headroom_bytes'] = headroom
    if headroom < MINIMUM_COMMIT_HEADROOM:
        result['reasons'].append('Insufficient system commit headroom for full greenhouse startup')
    if snapshot['physical_available_bytes'] < MINIMUM_AVAILABLE_PHYSICAL:
        result['reasons'].append('Insufficient available physical RAM reserve')
    result.update(allowed=not result['reasons'],
                  state='reserve_available_not_runtime_guarantee' if not result['reasons'] else 'insufficient_launch_reserve')
    return result


def preflight():
    return assess_memory(memory_snapshot())


if __name__ == '__main__':
    import json
    report = preflight()
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report['allowed'] else 2)
