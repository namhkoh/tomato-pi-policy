import numpy as np
import pytest
from types import SimpleNamespace as S
from .seam_escape import bind,gaps,monotone,execution_guard


def test_escaped_face_stays_clear_and_uncleared_face_must_separate():
    a={'a':.0002,'b':.003}
    assert monotone(a,{'a':.0003,'b':.002})
    assert not monotone(a,{'a':.0001,'b':.004})
    assert not monotone(a,{'a':.0012,'b':.0008})
    assert not monotone({'a':-.0001,'b':.003},{'a':-.0002,'b':.003})


@pytest.mark.parametrize('load',[.01,.03,.5,-.001,True,None,float('nan')])
def test_exit_is_stricter_than_original_absolute_force_guard(load):
    with pytest.raises(RuntimeError):execution_guard(dict(cut=True,native_guards_passed=True,
        knife=dict(tool_contact_upper_bound_n=load)))


def test_signed_capsule_gap_increases_when_box_leaves_cut_faces():
    shape=('knife','wrist','ee_right','box',(np.zeros(3),np.eye(3),np.ones(3)*.001))
    obstacles=[(p,'capsule',(np.array([x,0.,0.]),np.array([x,0.,.01]),.001),None,None)
        for p,x in [('a',.0015),('b',.0016)]]
    binding=bind(S(shapes=[shape],blade_path='knife',seam_paths={'a','b'},obstacles=obstacles))
    f=np.eye(4);before=gaps(binding,{'ee_right':f})
    f[0,3]-=.002;after=gaps(binding,{'ee_right':f})
    assert monotone(before,after) and all(x>.001 for x in after.values())
    assert not monotone(after,before)


def test_missing_face_or_nonfinite_bound_never_authorizes_escape():
    with pytest.raises(ValueError):monotone({'a':0},{'a':1})
    with pytest.raises(ValueError):monotone({'a':0,'b':0},{'a':np.nan,'b':1})
