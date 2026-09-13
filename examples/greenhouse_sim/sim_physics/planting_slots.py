"""Permute original preview slots without removing plants or changing spacing."""


def validate(slot):
    if type(slot) is not int or slot not in (0,12,23):
        raise ValueError('Explicit original centre or end-of-row slot (0,12,23) required')
    return slot


def target_y(slot):
    return (validate(slot)-12)*.5


def backdrop_source_slot(slot,target_slot,*,target_side):
    validate(target_slot)
    if type(slot) is not int or not 0<=slot<24 or type(target_side) is not bool:
        raise ValueError('Original preview slot and explicit target-side flag required')
    if not target_side:return slot
    if slot==target_slot:return None  # Detailed target occupies this slot.
    return target_slot if slot==12 else slot  # Move that SAME backdrop to 12.
