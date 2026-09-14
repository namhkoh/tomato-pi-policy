import numpy as np
import pytest
from .native_resolution_smoke import known_surface
from .capture_sensor import HIRES_RESOLUTION


def test_memory_block_persists_failure_without_starting_isaac(tmp_path,monkeypatch):
    import json
    from sim_physics import host_memory
    from .native_resolution_smoke import main
    monkeypatch.setattr(host_memory,'preflight',lambda:dict(allowed=False,checked=True))
    output=tmp_path/'native_smoke'
    with pytest.raises(ValueError,match='memory reserve'):
        main(['--output',str(output)])
    assert json.loads((output/'request.json').read_text())['host_memory_preflight']['allowed'] is False
    failure=json.loads((output/'failure.json').read_text())
    assert failure['training_approved'] is False
    assert not (output/'result.json').exists()


def buffers():
    z=np.full((816,1696),2.,np.float32)
    valid=np.ones(z.shape,bool);ids=np.full(z.shape,7,np.uint32)
    return z,valid,ids,{7:"/World/CalibrationCube"}


def test_known_native_surface_probes_metric_Z_and_identity():
    z,v,ids,m=buffers()
    checks=known_surface(z,v,ids,m,HIRES_RESOLUTION,"/World/CalibrationCube",2.)
    assert [x["pixel_xy"] for x in checks]==[[848,408],[940,408]]
    assert all(x["native_z_m"]==2. for x in checks)


@pytest.mark.parametrize("fault",["invalid","wrong_z","off_axis_range","identity","shape","nonmetric"])
def test_known_surface_rejects_mismatched_depth_or_identity(fault):
    z,v,ids,m=buffers();expected=2.
    if fault=="invalid":v[408,848]=False
    if fault=="wrong_z":z[408,848]=1.8
    if fault=="off_axis_range":z[408,940]=2.02
    if fault=="identity":m[7]="/World/WrongObject"
    if fault=="shape":z=z[:408]
    if fault=="nonmetric":expected=float("nan")
    with pytest.raises(ValueError):
        known_surface(z,v,ids,m,HIRES_RESOLUTION,"/World/CalibrationCube",expected)
