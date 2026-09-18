"""Separate an unsafe current state from insufficient capacity for a new Kit."""
from .dataset_review import require


def decide(evidence):
    memory=evidence['memory'];gpu=evidence['gpu'];inventory=evidence['process_inventory']
    require(type(evidence['projected_additional_vram_mib']) is int
        and evidence['projected_additional_vram_mib']>=10240
        and evidence['required_remaining_vram_mib']==4096,'Unchanged conservative additional-worker reserve required')
    actual_safe=bool(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30
        and evidence['disk_free_bytes']>=60*2**30 and gpu['free_mib']>=4096
        and not inventory['blockers'] and inventory['no_blockers_observed'] is True)
    projected_safe=gpu['free_mib']-evidence['projected_additional_vram_mib']>=4096
    if not actual_safe:return 'unsafe_current_state'
    if not projected_safe:
        require(evidence['allowed'] is False,'Reservation decision inconsistent with measured GPU memory')
        return 'defer_additional_worker'
    require(evidence['allowed'] is True,'Fresh resource evidence did not admit a supported new worker')
    return 'admit_additional_worker'
