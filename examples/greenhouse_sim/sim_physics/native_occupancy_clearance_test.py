import numpy as np
import pytest
from sim_physics.native_occupancy_clearance import NativeOccupancyClearance


def fixture(tool_centre,*,min_cell=.001,max_nodes=4096):
    # Two tiny solids whose LARGE enclosing boxes overlap. Calls stand in for
    # native overlap; this is algorithm validation, not PhysX evidence.
    centres={'/leaf':np.array([-.009,0.,0.]),'/tool':np.asarray(tool_centre,float)}
    calls=[]
    def query(path,c,a,h):
        calls.append(path)
        local=a.T@(centres[path]-c)
        return bool(np.all(np.abs(local)<=h+.0003))
    n=NativeOccupancyClearance(query,{'/leaf':[(np.zeros(3),np.eye(3),np.full(3,.01))]},
        guard=lambda:None,positive_control=lambda p:p in centres,min_cell_m=min_cell,max_nodes=max_nodes)
    return n,calls


def test_overlapping_root_boxes_can_be_resolved_without_disabling_solids():
    n,calls=fixture([.009,0,0])
    assert n.clear('/tool',np.eye(4),np.eye(4),'/leaf',.001)
    assert n.nodes>1 and '/leaf' in calls and '/tool' in calls
    before=calls.count('/leaf')
    assert n.clear('/tool',np.eye(4),np.eye(4),'/leaf',.001)
    assert calls.count('/leaf')==before  # Only frozen obstacle occupancy reused.


@pytest.mark.parametrize('tool',[[-.009,0,0],[-.0085,0,0],[-.0075,0,0]])
def test_overlap_or_insufficient_margin_never_clears(tool):
    n,_=fixture(tool)
    assert not n.clear('/tool',np.eye(4),np.eye(4),'/leaf',.001)


def test_countertransform_checks_proposed_not_parked_tool_location():
    n,_=fixture([.009,0,0]);actual=np.eye(4);proposed=np.eye(4)
    # Translating the future tool left18 mm places it on the actual leaf.
    proposed[0,3]=-.018
    assert not n.clear('/tool',actual,proposed,'/leaf',.001)


@pytest.mark.parametrize('fault',['budget','none','actor','epoch','margin'])
def test_unknown_results_and_budget_failures_do_not_become_clearance(fault):
    n,_=fixture([.009,0,0],max_nodes=1 if fault=='budget' else 4096)
    if fault=='none':n.overlap=lambda *args:None
    if fault=='actor':n.positive_control=lambda *args:False
    if fault=='epoch':n.guard=lambda:(_ for _ in ()).throw(RuntimeError('stale'))
    with pytest.raises((RuntimeError,ValueError)):
        n.clear('/tool',np.eye(4),np.eye(4),'/leaf',.0001 if fault=='margin' else .001)


def test_unknown_obstacle_not_cleared_and_input_bounds_are_copied():
    n,_=fixture([.009,0,0])
    assert not n.clear('/tool',np.eye(4),np.eye(4),'/missing',.001)
    assert n.calls==0
