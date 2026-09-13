"""Current measured pad footprint on a connected shaft; not grasp evidence.

Caller supplies validated collider geometry/frames and live adjacent joint IDs.
Only identity eligibility changes. Exact side/inner-face witnesses, signed
loads, tensor reconciliation, dwell and slip still have to pass independently.
"""
import numpy as np


def eligible_span(chain, pads, world, links, selected_index, cut_index):
    selected=chain[selected_index]
    axis=world[selected.collider][:3,2];origin=world[selected.collider][:3,3]
    bounds=[]
    # Use only existing authored contact offsets, not an adjustable margin.
    shaft_offset=max(s.contact_offset_m for s in chain[cut_index:])
    for pad in pads:
        frame=world[pad.collider]
        centre=float((frame[:3,3]-origin)@axis)
        half=float(np.abs(axis@frame[:3,:3])@pad.half_extents_m)
        margin=shaft_offset+pad.contact_offset_m
        bounds.append((centre-half-margin,centre+half+margin))
    interval=(min(v[0] for v in bounds),max(v[1] for v in bounds))
    def intersects(i):
        shaft=chain[i];frame=world[shaft.collider]
        centre=float((frame[:3,3]-origin)@axis)
        # Enclose cylinders too; exact native material-side witnesses remain
        # mandatory per row. A broad identity candidate never supplies force.
        half=abs(float(axis@frame[:3,2]))*shaft.half_height_m+shaft.radius_m
        return centre+half>=interval[0] and centre-half<=interval[1]
    accepted=set()
    if intersects(selected_index):
        accepted.add(selected_index)
        for direction in (-1,1):
            previous=selected_index;i=previous+direction
            while cut_index<=i<len(chain):
                if (frozenset((chain[previous].body,chain[i].body)) not in links
                        or not intersects(i)):break
                accepted.add(i);previous=i;i+=direction
    return {chain[i].collider for i in accepted},dict(
        model='current_connected_physical_pad_span_v1',
        eligible_indices=sorted(accepted),pad_axial_interval_m=list(interval),
        axis_world=axis.tolist(),origin_world_m=origin.tolist(),
        uses_authored_contact_offsets=True,grasp_verified=False,
        contact_filters_changed=False)
