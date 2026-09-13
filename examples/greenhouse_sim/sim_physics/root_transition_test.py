from types import SimpleNamespace

import numpy as np
import pytest

from .root_transition import snapshot, continuity, verify_free_mass, _FIELDS
from .root_transition import jacobian_velocity_audit
from .root_transition import FixedRootTransition


class View:
    def __init__(self,fixed=True):
        self.count=1
        self.shared_metatype=SimpleNamespace(fixed_base=fixed,dof_names=['x','y','z'])
        self.link_paths=[['root','tip']]
        self.values={key:np.ones((1,3)) for key in _FIELDS}
        self.values['poses']=np.zeros((1,2,7));self.values['poses'][:,:,6]=1
        self.values['velocities']=np.zeros((1,2,6))
        self.matrix=np.eye(9)[None]

    def __getattr__(self,name):
        for key,method in _FIELDS.items():
            if method==name:return lambda: self.values[key]
        raise AttributeError(name)

    def get_generalized_mass_matrices(self):return self.matrix


def test_no_step_continuity_changes_only_root_metadata():
    old=View();new=View(False)
    assert not any(continuity(snapshot(old),snapshot(new),release=True).values())
    assert not any(continuity(snapshot(old),snapshot(old),release=False).values())
    assert verify_free_mass(new,3)==[1,9,9]


def test_snapshot_owns_its_data():
    v=View();s=snapshot(v);v.values['q'][0,0]=123
    assert s['q'][0,0]==1


def test_even_initial_native_import_failure_is_latched(monkeypatch):
    import sys
    t=object.__new__(FixedRootTransition)
    t.rig=SimpleNamespace(cut=False);t.attempted=False;t.error=None
    monkeypatch.setitem(sys.modules,'omni',None)
    with pytest.raises(ImportError):t.release()
    assert t.attempted and t.error is not None and not t.rig.cut
    with pytest.raises(RuntimeError,match='cannot be retried'):t.release()


def test_jacobian_audit_preserves_and_reports_preexisting_velocity_residual():
    v=View();v.get_jacobians=lambda:np.zeros((1,1,6,3))
    state=snapshot(v);state['velocities'][0,1,4]=.00014
    j,r=jacobian_velocity_audit(v,state)
    assert j.shape==(1,6,3)
    assert r['max_angular_velocity_error_rad_s']==.00014
    assert not r['native_velocity_replaced']
    assert state['velocities'][0,1,4]==.00014


@pytest.mark.parametrize('key',list(_FIELDS))
def test_mutated_state_or_commands_fail(key):
    a=snapshot(View());b=snapshot(View(False));b[key].flat[0]+=.0001
    with pytest.raises(RuntimeError,match='changed across no-step'):
        continuity(a,b,release=True)


@pytest.mark.parametrize('key',list(_FIELDS))
def test_invalid_native_value_fails(key):
    v=View();v.values[key].flat[0]=np.nan
    with pytest.raises(RuntimeError,match='Nonfinite'):
        snapshot(v)


@pytest.mark.parametrize('mutation',['names','links','topology','shape','nan'])
def test_inventory_and_layout_cannot_silently_change(mutation):
    a=snapshot(View());b=snapshot(View(False))
    if mutation=='names':b['names']=('z','y','x')
    elif mutation=='links':b['links']=('tip','root')
    elif mutation=='topology':b['fixed_base']=True
    elif mutation=='shape':b['q']=np.zeros((1,9))
    else:b['q'][0,0]=np.nan
    with pytest.raises(RuntimeError):continuity(a,b,release=True)


@pytest.mark.parametrize('kind',['fixed','missing_root','nan','negative','asymmetric'])
def test_free_mass_required(kind):
    v=View(False)
    if kind=='fixed':v.shared_metatype.fixed_base=True
    elif kind=='missing_root':v.matrix=np.eye(3)[None]
    elif kind=='nan':v.matrix[0,0,0]=np.nan
    elif kind=='negative':v.matrix[0,0,0]=-1
    else:v.matrix[0,0,1]=.1
    with pytest.raises((RuntimeError,np.linalg.LinAlgError)):verify_free_mass(v,3)
