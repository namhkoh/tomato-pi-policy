import pytest
from pxr import Usd, UsdGeom, Gf
from sim_physics.branch_contact_fixture import select_components
from sim_physics.benchmark import parser, main


def fixture(monkeypatch):
    components = {
        'stem': dict(type='main_stem', parent=None),
        'stem2': dict(type='main_stem', parent='stem'),
        'petiole': dict(type='sub_stem', parent='stem2'),
        'leaf1': dict(type='leaf', parent='petiole'),
        'leaf2': dict(type='leaf', parent='petiole'),
        'other': dict(type='sub_stem', parent='stem'),
        'fruit': dict(type='fruit', parent='other'),
    }
    paths = dict(stem='/World/Plant/stem', stem2='/World/Plant/stem/stem2',
        petiole='/World/Plant/stem/stem2/petiole', leaf1='/World/Plant/stem/stem2/petiole/leaf1',
        leaf2='/World/Plant/stem/stem2/petiole/leaf2', other='/World/Plant/stem/other',
        fruit='/World/Plant/stem/other/fruit')
    stage = Usd.Stage.CreateInMemory()
    for i, path in enumerate(paths.values()):
        UsdGeom.Xform.Define(stage, path).AddTranslateOp().Set(Gf.Vec3d(i*.01, 0, 0))
        mesh=UsdGeom.Mesh.Define(stage,path+'/Mesh');mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    audit=dict(components=components, manifest_sha256='source-hash',
        targets=[dict(component_id='petiole',status='candidate',protected_descendant_ids=[],
                      expected_detached_component_ids=['petiole','leaf1','leaf2'])])
    monkeypatch.setattr('sim_data.audit.audit_manifest',lambda _:audit)
    return stage,dict(manifest_path='source',plant_root='/World/Plant',component_paths=paths),audit


def test_only_session_components_change_all_target_leaves_and_main_stem_survive(monkeypatch):
    stage,record,audit=fixture(monkeypatch)
    source=stage.GetRootLayer().ExportToString();edit=stage.GetEditTarget()
    kept=['stem','stem2','petiole','leaf1','leaf2'];cache=UsdGeom.XformCache()
    transforms={k:cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record['component_paths'][k])) for k in kept}
    report=select_components(stage,record,'petiole')
    assert stage.GetRootLayer().ExportToString()==source and stage.GetEditTarget()==edit
    assert report['retained_component_ids']==sorted(kept)
    assert report['inactive_session_roots']==[record['component_paths']['other']]
    assert not report['training_eligible'] and not report['full_greenhouse_qualified']
    assert not report['intact_source_plant_present'] and report['all_target_leaves_retained']
    cache.Clear()
    for k in kept:
        prim=stage.GetPrimAtPath(record['component_paths'][k]);assert prim.IsActive()
        assert cache.GetLocalToWorldTransform(prim)==transforms[k]
        assert len(UsdGeom.Mesh.Get(stage,prim.GetPath().AppendChild('Mesh')).GetPointsAttr().Get())==3
    assert not stage.GetPrimAtPath(record['component_paths']['other']).IsActive()


@pytest.mark.parametrize('bad',['missing_target','protected','excluded','parent','missing_leaf','foreign_leaf',
    'missing_path','duplicate_path','outside','ancestor','already_inactive','robot','rig'])
def test_invalid_fixture_fails_before_any_session_mutation(monkeypatch,bad):
    stage,record,audit=fixture(monkeypatch);paths=record['component_paths'];target=audit['targets'][0]
    if bad=='missing_target':target['component_id']='unknown'
    if bad=='protected':target['protected_descendant_ids']=['fruit']
    if bad=='excluded':target['status']='excluded'
    if bad=='parent':audit['components']['petiole']['parent']='other'
    if bad=='missing_leaf':target['expected_detached_component_ids'].remove('leaf1')
    if bad=='foreign_leaf':audit['components']['leaf1']['type']='fruit'
    if bad=='missing_path':paths.pop('other')
    if bad=='duplicate_path':paths['other']=paths['stem']
    if bad=='outside':paths['other']='/World/Other';UsdGeom.Xform.Define(stage,paths['other'])
    if bad=='ancestor':paths['other']='/World/Plant';paths['fruit']='/World/Plant/stem/other'
    if bad=='already_inactive':stage.GetPrimAtPath(paths['other']).SetActive(False)
    if bad=='robot':UsdGeom.Xform.Define(stage,'/World/RBY1')
    if bad=='rig':UsdGeom.Xform.Define(stage,'/World/InteractionPhysics')
    before=stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):select_components(stage,record,'petiole')
    assert stage.GetSessionLayer().ExportToString()==before


def test_branch_fixture_is_explicit_and_cannot_enable_production_or_default_run(tmp_path):
    assert not parser().parse_args(['--output','unused']).branch_contact_fixture
    with pytest.raises(ValueError,match='isolated cut contact'):
        main(['--output',str(tmp_path/'unused'),'--branch-contact-fixture'])
    assert not (tmp_path/'unused').exists()


def test_inactive_source_robot_namespace_is_not_a_constructed_dynamic_robot(monkeypatch):
    stage,record,audit=fixture(monkeypatch)
    for path in ('/World/RBY1','/World/InteractionPhysics'):
        UsdGeom.Xform.Define(stage,path).GetPrim().SetActive(False)
    source=stage.GetRootLayer().ExportToString()
    assert select_components(stage,record,'petiole')['all_target_leaves_retained']
    assert stage.GetRootLayer().ExportToString()==source


def test_original_package_undefined_robot_override_is_not_a_constructed_rig(monkeypatch):
    stage,record,audit=fixture(monkeypatch)
    stub=stage.OverridePrim('/World/RBY1')
    assert stub.IsActive() and not stub.IsDefined()
    assert select_components(stage,record,'petiole')['all_target_leaves_retained']
    assert not stub.IsDefined()
