"""Curved recipe dispatch retains native-camera and original-target cap contracts."""
from pathlib import Path
import pytest
from sim_data.generated_capture_test import simple_plan
from sim_data.generated_capture import check_plan
from sim_data.dataset_review import write_json
from sim_data.depth_preview import sha256


def curved_plan(tmp_path):
    p=simple_plan(tmp_path)
    directory=tmp_path/'variant';directory.mkdir()
    version='curved_relocated_petiole_static.v1'
    write_json(directory/'qualification.json',dict(version=version,code_sha256={'procedural_petiole_usd.py':'fixture'}))
    root=Path(__file__).parent
    files=[root/name for name in ('generated_capture.py','plant_variant_catalogue.py',
                                 'procedural_petiole_catalogue.py','procedural_petiole_usd.py')]
    p.update(variant_directory=str(directory),generator_version=version,independent_target_novelty_approved=False,
             generator_code_bindings={str(f.resolve()):sha256(f) for f in files})
    return p


def test_curved_native_plan_preserves_source_cap(tmp_path):
    p=curved_plan(tmp_path);check_plan(p)
    assert p['sample_count_limit']==2 and p['conservative_view_cap_group']=='seed7/SubStem_42'


@pytest.mark.parametrize('fault',['discriminator','missing_binding','wrong_hash','novelty','cap'])
def test_curved_plan_rejects_binding_or_admission_bypass(tmp_path,fault):
    p=curved_plan(tmp_path)
    if fault=='discriminator':p.pop('generator_version')
    elif fault=='missing_binding':p['generator_code_bindings'].pop(next(iter(p['generator_code_bindings'])))
    elif fault=='wrong_hash':p['generator_code_bindings'][next(iter(p['generator_code_bindings']))]='0'*64
    elif fault=='novelty':p['independent_target_novelty_approved']=True
    else:p['conservative_view_cap_group']='new_geometry'
    with pytest.raises(ValueError):check_plan(p)
