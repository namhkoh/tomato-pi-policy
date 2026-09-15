"""CPU-only V3 retention, same-callback storage and native-loop parity tests."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

from ..depth_preview import sha256
from ..native_budget import REFERENCE, TRIAL
from . import compact_query_v2 as v2
from . import compact_query_v3 as worker
from . import query_audit_v3 as audit
from . import capture_storage
from .bundle import SampleReader
from .test_compact_query_v2 import function_source, assert_scoped_ast_parity
from .test_compact_views import main_fixture as legacy_main_fixture

V2 = Path(v2.__file__)
V3 = Path(worker.__file__)
CORE_EDITS = [
    [
        "def _collect(app,output,plan,base,*,profile_render=False,instance_backend='legacy',render_budget=REFERENCE):",
        "def _collect(app,output,plan,base,*,plan_sha256,profile_render=False,instance_backend='legacy',\n             render_budget=REFERENCE,persistence=ALL_FRAMES,qualification_witness=False):"
    ],
    [
        "    import omni.replicator.core as rep",
        "    audit3._hash(plan_sha256)\n    audit3.validate_plan_mode(plan, persistence, qualification_witness)\n    import omni.replicator.core as rep"
    ],
    [
        "    captured_count=0",
        "    validated_callbacks=0"
    ],
    [
        "            render_settings=budget_evidence(render_budget,captured_count)",
        "            render_settings=budget_evidence(render_budget,validated_callbacks)"
    ],
    [
        "            previous=freshness",
        "            previous=freshness\n            validated_callbacks+=1"
    ],
    [
        "            stored=write_compact_native_sample(folder,rgb,depth,valid,metadata,instances,\n                mapping,catalogue,components,organs,mask,label,trace)",
        "            saved=persist_callback(output,name,rgb,depth,valid,metadata,instances,\n                mapping,catalogue,components,organs,mask,label,trace,plan_sha256=plan_sha256,\n                persistence=persistence,qualification_witness=qualification_witness)"
    ],
    [
        "            decision.update(state='native_captured_pending_review',eligible_annotation=label['eligible'],\n                automatic_annotation_eligible=bool(trace is not None and trace['passed']),\n                label_reason=label['reason'],label_sha256=stored['label_sha256'],\n                sample_sha256=stored['sample_sha256'],query_trace=stored['query_trace'],\n                render_budget=render_settings,\n                elapsed_seconds=time.perf_counter()-view_started)",
        "            decision.update(state=CALLBACK_STATE,**saved,callback_ordinal=validated_callbacks,\n                source_checks_passed=True,render_budget=render_settings,\n                elapsed_seconds=time.perf_counter()-view_started)"
    ],
    [
        "            captured_count+=1\n",
        ""
    ],
    [
        "        return dict(state=CAPTURE_STATE,\n            records=records,captured_frames=sum(r['state']=='native_captured_pending_review' for r in records),\n            eligible_annotation_candidates=sum(r.get('eligible_annotation',False) for r in records),\n            automatically_clear_annotation_candidates=sum(r.get('automatic_annotation_eligible',False) for r in records),",
        "        counts=audit3.callback_counts(records)\n        require(counts['validated_callbacks']==validated_callbacks,'Callback accounting differs from native loop')\n        return dict(state=CAPTURE_STATE,records=records,counts=counts,\n            counter_scope='worker_reported_target_callbacks_and_separate_warmup_inclusive_render_requests',"
    ]
]
MAIN_EDITS = [
    [
        "    p.add_argument('--annotation-policy-sha256',required=True)",
        "    p.add_argument('--annotation-policy-sha256',required=True)\n    p.add_argument('--persistence-policy-sha256',required=True)\n    p.add_argument('--persistence',choices=audit3.MODES,default=ALL_FRAMES)\n    p.add_argument('--qualification-witness-save-all',action='store_true')"
    ],
    [
        "    a=p.parse_args(argv)",
        "    a=p.parse_args(argv)\n    require(a.persistence_policy_sha256==audit3.PERSISTENCE_POLICY_SHA256,'Caller persistence policy pin mismatch')"
    ],
    [
        "    require('target_cases' in plan, 'Initial query-V2 capture requires a batch plan')",
        "    audit3.validate_plan_mode(plan,a.persistence,a.qualification_witness_save_all)"
    ],
    [
        "        **annotation_fields(),annotation_policy=annotation_policy(),",
        "        **annotation_fields(),annotation_policy=annotation_policy(),\n        **audit3.configuration(a.persistence,a.qualification_witness_save_all),"
    ],
    [
        "                       instance_backend=a.instance_backend,render_budget=a.render_budget)",
        "                       instance_backend=a.instance_backend,render_budget=a.render_budget,plan_sha256=a.plan_sha256,\n                       persistence=a.persistence,qualification_witness=a.qualification_witness_save_all)"
    ]
]


def test_exact_frozen_native_loop_parity():
    assert sha256(V2) == audit._FROZEN["compact_query_v2.py"]
    assert sha256(Path(audit.v2.__file__)) == audit._FROZEN["query_audit_v2.py"]
    assert_scoped_ast_parity(function_source(V2,"_collect"), function_source(V3,"_collect"), reviewed_edits=CORE_EDITS)


def test_exact_main_parity_only_explicit_persistence_handoff():
    assert_scoped_ast_parity(function_source(V2,"main"), function_source(V3,"main"), reviewed_edits=MAIN_EDITS)


def test_collect_wrapper_only_adds_explicit_configuration():
    assert_scoped_ast_parity(function_source(V2,"collect"), function_source(V3,"collect"),
                            reviewed_edits=[
    [
        "def collect(app, output, plan, base, *, storage_qualification,\n            storage_qualification_sha256, profile_render=False,\n            instance_backend='legacy', render_budget=REFERENCE):",
        "def collect(app, output, plan, base, *, plan_sha256, storage_qualification,\n            storage_qualification_sha256, profile_render=False,\n            instance_backend='legacy', render_budget=REFERENCE,\n            persistence=ALL_FRAMES, qualification_witness=False):"
    ],
    [
        "    require('target_cases' in plan, 'Initial query-V2 capture requires a batch plan')\n    result = _collect",
        "    audit3._hash(plan_sha256)\n    audit3.validate_plan_mode(plan, persistence, qualification_witness)\n    result = _collect"
    ],
    [
        "                      instance_backend=instance_backend, render_budget=render_budget)",
        "                      instance_backend=instance_backend, render_budget=render_budget,\n                      plan_sha256=plan_sha256, persistence=persistence, qualification_witness=qualification_witness)"
    ],
    [
        "                  worker_module=WORKER_MODULE, worker_implementation_bindings=bindings)",
        "                  worker_module=WORKER_MODULE, worker_implementation_bindings=bindings,\n                  **audit3.configuration(persistence, qualification_witness))"
    ]
])


@pytest.mark.parametrize("before,after", [
    ("previous=freshness", "previous=None"),
    ("validated_callbacks+=1", "validated_callbacks+=0"),
    ("budget_evidence(render_budget,validated_callbacks)", "budget_evidence(render_budget,0)"),
    ("range(6)", "range(0)"),
    ("source_hashes(stage)==stage_hashes", "True"),
    ("verify_bindings(plan['source_bindings'])", "None"),
    ("set_reference_snapshot(", "unmounted_view("),
    ("validate_native_static(", "unvalidated_payload("),
    ("decode_native_instances(", "invented_ids("),
    ("annotate_for_storage(", "loose_annotation("),
])
def test_parity_catches_persistence_regressions(before, after):
    candidate = function_source(V3,"_collect")
    assert before in candidate
    with pytest.raises(AssertionError, match="Unlisted"):
        assert_scoped_ast_parity(function_source(V2,"_collect"), candidate.replace(before,after,1),
                                reviewed_edits=CORE_EDITS)


def test_no_frozen_global_rebinding_boundary_import_or_native_import_at_load():
    for module in (worker, audit):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "exec(" not in source and "setattr(" not in source and "sys.modules" not in source
        tree = ast.parse(source)
        imports = [n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(n and "boundary_bindings" in n for n in imports)
    assert worker.implementation_bindings() == audit.implementation_bindings()
    assert len(worker.implementation_bindings()) == 25


def test_live_array_contract_table_equals_frozen_writer():
    def table(path, function):
        node = ast.parse(function_source(path,function))
        return next(n.iter for n in ast.walk(node) if isinstance(n,ast.For)
                    and isinstance(n.target,ast.Tuple)
                    and [x.id for x in n.target.elts] == ["name","array","dtype","expected_shape"])
    assert ast.dump(table(Path(audit.__file__),"validate_live_buffers")) == ast.dump(
        table(Path(capture_storage.__file__),"write_compact_native_sample"))


def invoke_native_loop_stub(tmp_path, dispositions, *, profile=TRIAL, fail_source=False, witness=False):
    """Run the actual whole _collect AST with ONLY imports replaced by CPU fakes.

    These are NOT native frames. AST parity above guards the production calls.
    The separate real-array tests exercise persist_callback and exact storage.
    """
    events, previous_tokens, budgets = [], [], []
    class NoImports(ast.NodeTransformer):
        def visit_Import(self,node): return None
        def visit_ImportFrom(self,node): return None
    tree = NoImports().visit(ast.parse(function_source(V3,"_collect")))
    writer = SimpleNamespace(request_index=0, sequence=0,
        attach=lambda x: events.append("attach"), detach=lambda: events.append("detach"))
    product = SimpleNamespace(destroy=lambda: events.append("destroy"))
    def make_product(*args):
        events.append("product")
        return product
    def step(rep, current, *, subframes):
        assert current is writer and subframes == 8
        writer.request_index += 1
        writer.sequence += 1
        return dict(camera_params={},pilot_render_frame={})
    def validate(payload,cal,before,after,sequence,*,previous):
        previous_tokens.append(deepcopy(previous))
        return None,None,None,None,dict(callback_sequence=sequence,fixture_only=True)
    source_calls = []
    def source_hashes(stage):
        source_calls.append("hash")
        return {"fixture":len(source_calls)} if fail_source and len(source_calls)>1 else {"fixture":1}
    class Cache:
        def __init__(self,*args,**kwargs): pass
        def __call__(self): return {"passed":True}
        def diagnostics(self): return {"fixture_only":True}
        def close(self): events.append("cache_close")
    class Monitor:
        def __init__(self,*args): pass
        def begin(self): return {}
        def token(self): return {}
        def close(self): events.append("monitor_close")
    row = dict(variant_id="fixture",component_id="Petiole",target_id="fixture/Petiole",
               cut_region_proposal={"nominal":{"petiole_radius_m":.002}})
    base_plan = dict(generated_row=row,source_row={"target_id":"fixture/Petiole"},
                     expected_robot_snapshot={"joint_degrees":{}},split_group="fixture",
                     conservative_view_cap_group="fixture/Petiole")
    world = dict(nominal_world_m=[0,0,1],interval_world_m=[[0,0,1],[0,0,1]])
    target_plan = dict(expected_nominal_world_m=[0,0,1])
    specs = [dict(candidate_id=f"frame_{i}",desired_pixel_xy=[10,10]) for i in range(len(dispositions))]
    plan = dict(target_cases=[dict(views=specs)],source_bindings={"fixture":"source"},
                implementation_bindings={"fixture":"code"})
    optical = {k:None for k in ("camera_path","resolution","intrinsics","focal_length_mm","apertures_mm",
                                "aperture_offsets_mm","depth_convention","crop_resize")}
    stage = SimpleNamespace(GetUsedLayers=lambda: [])
    context = dict(stage=stage,generated={"report":{}},original_variant={},records=[],variants=[],
        reports=[],counts={"components":1},robot={"root":"/Robot"},settings={"/rtx/rendermode":"fixture"},
        old_manifest={"renderer":"fixture","lighting":{}})
    def annotate(meta,*args,**kwargs):
        index = int(meta["sample_id"].split("_")[-1])
        d = dispositions[index]
        return {"eligible":d!="exclude","reason":d}, None if d=="exclude" else {"passed":d=="strict"}
    def persist(output,name,*args,**kwargs):
        i = int(name.split("_")[-1]); d=dispositions[i]
        events.append("persist_"+d)
        primary = {"fixture":True} if d=="strict" else None
        return dict(annotation={"disposition":d},persistence={"role":"strict_candidate" if d=="strict" else
            "unpersisted_rejection"},primary=primary,witness={"fixture":True} if witness else None,
            pixels_persisted=primary is not None or witness,independent_pixel_replay_possible=primary is not None or witness)
    def budget(profile,count):
        value = worker.budget_evidence(profile,count)
        budgets.append(value["requested_subframes"])
        return value
    namespace = dict(worker.__dict__, require=worker.require, budget_evidence=budget,
        omni=SimpleNamespace(timeline=SimpleNamespace(get_timeline_interface=lambda:SimpleNamespace(pause=lambda:None))),
        rep=SimpleNamespace(create=SimpleNamespace(render_product=make_product)), prepare_native_scene=lambda *a:context,
        substitute_plant=lambda *a:dict(report={},records=[],variants=[{"variant_id":"fixture"}]),
        component_catalogue=lambda *a:[{"variant_id":"fixture","component_id":"Petiole"}],
        mounted_camera_to_head=lambda *a:np.eye(4), calibration=lambda *a:optical,
        calibration_for_native_resolution=lambda c,r:c, source_hashes=source_hashes, make_writer=lambda *a,**kw:writer,
        StaticGeometryScreenCache=Cache, StaticSceneMonitor=Monitor, HEAD_CAMERA="/Robot/Head",
        capture_jobs=lambda *a:((base_plan,target_plan,s) for s in specs), target_world_geometry=lambda *a:world,
        set_reference_snapshot=lambda *a:{"fixture":True}, step_payload=step, validate_native_static=validate,
        assert_same_camera=lambda *a:None, decode_native_instances=lambda *a:(None,None),
        component_masks=lambda *a:(None,None,None), interval_visibility=lambda *a:({},np.array([True])),
        view_quality=lambda *a:{}, sensor_profile=lambda *a:{}, depth_evidence=lambda *a:{},
        project=lambda points,cal:[{"projection_status":"in_frame","pixel_xy":[20,20]} for p in points],
        annotate_for_storage=annotate,persist_callback=persist,
        verify_bindings=lambda b:events.append("source_check" if b==plan["source_bindings"] else "code_check"),
        write_json=lambda path,value:events.append("decision"))
    exec(compile(ast.fix_missing_locations(tree),"<CPU-native-loop-stubs>","exec"),namespace)
    error = None; result = None
    try:
        result = namespace["_collect"](None,tmp_path,plan,base_plan,plan_sha256="a"*64,
            render_budget=profile,persistence=audit.STRICT_ONLY,qualification_witness=witness)
    except ValueError as exc:
        error = exc
    return result,error,events,previous_tokens,budgets,source_calls,writer


@pytest.mark.parametrize("dispositions", [
    ["exclude","hold","strict"], ["hold","hold","strict"], ["exclude","hold","exclude"],
])
@pytest.mark.parametrize("witness", [False,True])
def test_first_reject_and_consecutive_holds_advance_actual_callback_budget(tmp_path, dispositions, witness):
    result,error,events,previous,budgets,hashes,writer = invoke_native_loop_stub(tmp_path,dispositions,witness=witness)
    assert error is None
    assert budgets == [56,56,8,8]  # initial profile validation, then three views
    assert writer.request_index == 9
    assert [p["callback_sequence"] if p else None for p in previous] == [None,7,8]
    assert result["counts"]["validated_callbacks"] == 3
    assert result["counts"]["persisted_samples"] == dispositions.count("strict")
    assert result["counts"]["unpersisted_attempts"] == 3-dispositions.count("strict")
    assert result["counts"]["witness_samples"] == (3 if witness else 0)
    assert len(hashes) == 4 and events.count("source_check") == 4
    assert events.count("product") == events.count("attach") == events.count("destroy") == 1


def test_reference_budget_never_silently_changes(tmp_path):
    result,error,events,previous,budgets,hashes,writer = invoke_native_loop_stub(
        tmp_path,["exclude","hold","strict"],profile=REFERENCE)
    assert error is None and budgets == [56,56,56,56] and writer.request_index == 21


@pytest.mark.parametrize("disposition", ["hold","strict"])
def test_source_integrity_error_never_becomes_a_successful_skip(tmp_path, disposition):
    result,error,events,previous,budgets,hashes,writer = invoke_native_loop_stub(
        tmp_path,[disposition,"strict"],fail_source=True)
    assert result is None and error is not None and "source geometry changed" in str(error)
    assert "decision" not in events and events[-3:] == ["cache_close","detach","destroy"]


@pytest.fixture
def cli_fixture(legacy_main_fixture, monkeypatch):
    output,events,proof,modules=legacy_main_fixture
    plan_path=Path(sys.argv[sys.argv.index("--batch-plan")+1])
    plan=json.loads(plan_path.read_text())
    plan["target_cases"]=[{"views":[{"candidate_id":"fixture"}]}]
    plan_path.write_text(json.dumps(plan),encoding="utf-8")
    def collect(*args,**kwargs):
        events.append("collect")
        assert kwargs["plan_sha256"]==sha256(plan_path)
        assert kwargs["persistence"]==audit.ALL_FRAMES and kwargs["qualification_witness"] is False
        return dict(state=audit.CAPTURE_STATE,training_approved=False,source_cap_reset=False)
    monkeypatch.setattr(worker,"_collect",collect)
    argv=sys.argv[1:]+["--plan-sha256",sha256(plan_path),"--annotation-policy-sha256",audit.v2.POLICY_SHA256,
                       "--persistence-policy-sha256",audit.PERSISTENCE_POLICY_SHA256]
    return output,events,proof,modules,argv


def test_cli_default_all_frames_and_exact_new_bindings(cli_fixture):
    output,events,proof,modules,argv=cli_fixture
    worker.main(argv)
    request=json.loads((output/"request.json").read_text())
    result=json.loads((output/"result.json").read_text())
    assert request["persistence_mode"]==result["persistence_mode"]==audit.ALL_FRAMES
    assert request["qualification_witness_save_all"] is False
    assert result["native_persistence_qualified"] is False
    assert result["request_sha256"]==sha256(output/"request.json")
    assert result["worker_implementation_bindings"]==worker.implementation_bindings()
    assert events[-1]=="close"


@pytest.mark.parametrize("flag", ["--plan-sha256","--annotation-policy-sha256","--persistence-policy-sha256"])
def test_wrong_pins_before_native_app(cli_fixture, flag):
    output,events,proof,modules,argv=cli_fixture
    argv[argv.index(flag)+1]="0"*64
    with pytest.raises(ValueError):worker.main(argv)
    assert not output.exists() and "app" not in events


def test_cli_integrity_error_preserves_failure_and_closes(cli_fixture, monkeypatch):
    output,events,proof,modules,argv=cli_fixture
    def fail(*a,**kw):raise ValueError("fixture integrity")
    monkeypatch.setattr(worker,"_collect",fail)
    with pytest.raises(ValueError,match="fixture integrity"):worker.main(argv)
    assert (output/"failure.json").exists() and not (output/"result.json").exists()
    assert events[-1]=="close"
