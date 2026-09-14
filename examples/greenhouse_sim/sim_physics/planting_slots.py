"""Permute original preview slots without removing plants or changing spacing."""


def validate(slot):
    if type(slot) is not int or slot not in (0,12,23):
        raise ValueError('Explicit original centre or end-of-row slot (0,12,23) required')
    return slot


def target_y(slot):
    return (validate(slot)-12)*.5


def validate_side(side):
    if type(side) is not int or side not in (-1,1):
        raise ValueError('Explicit original negative/positive X planting side (-1,1) required')
    return side


def backdrop_source_address(side,slot,target_slot,*,target_planting_side=1,selected_gutter=False):
    """Swap the detailed target with exactly one original backdrop address.

    The preview target starts at (+X,12). Its displaced backdrop moves there,
    including the ORIGINAL side's asset offset. The detailed neighbor (+X,13)
    is never moved. Other gutters, spacing, rotations and plant count stay put.
    """
    validate_side(side);validate_side(target_planting_side);validate(target_slot)
    if type(slot) is not int or not 0<=slot<24 or type(selected_gutter) is not bool:
        raise ValueError('Original preview address and explicit selected-gutter flag required')
    if not selected_gutter:return side,slot
    if (side,slot)==(target_planting_side,target_slot):return None
    if (side,slot)==(1,12):return target_planting_side,target_slot
    return side,slot


def backdrop_source_slot(slot,target_slot,*,target_side):
    validate(target_slot)
    if type(slot) is not int or not 0<=slot<24 or type(target_side) is not bool:
        raise ValueError('Original preview slot and explicit target-side flag required')
    if not target_side:return slot
    if slot==target_slot:return None  # Detailed target occupies this slot.
    return target_slot if slot==12 else slot  # Move that SAME backdrop to 12.
