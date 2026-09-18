"""Explicit two-frame diagnostic for source-frame controlled geometry only."""
from copy import deepcopy
from pathlib import Path
import ast
import importlib.util
from .dataset_review import require, read_json, verify_bindings, safe_file
from .depth_preview import sha256
from . import native848_original_reference_v2 as references

SCHEMA = 'greenhouse.native848_source_frame_controlled_pair_plan.v1'
SAMPLE_SCHEMA = 'greenhouse.native848_source_frame_controlled_pair_sample.v1'
RESULT_STATE = 'native848_source_frame_controlled_pair_captured_pending_audit'
FLAGS = dict(training_approved=False, independent_target_novelty_approved=False,
    source_cap_reset=False, physical_motion_commanded=False, paired_848_1696_proof=False,
    label_contract_qualified=False, global_geometry_novelty_qualified=False)
COMMON = ('source_capture', 'source_sample', 'source_collection_plan', 'source_row',
    'original_variant', 'expected_scene_counts', 'expected_robot_snapshot', 'source_family',
    'split', 'split_group', 'conservative_view_cap_group', 'expected_calibration',
    'expected_original_world', 'scene_variants', 'scene_code_bindings')


def implementation_bindings():
    root = Path(__file__).resolve().parent
    roots = (root.parent, root.parents[1])
    pending = [root/name for name in ('native848_controlled_pair_plan_v1.py',
        'native848_controlled_pair_worker_v1.py', 'native848_controlled_pair_audit_v1.py',
        'native848_controlled_scene_v1.py')]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = sha256(path)
        base = next(p for p in roots if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.'*node.level+(node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name]+[name+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in roots:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def generated_bindings(directory):
    directory = Path(directory).resolve()
    receipt = directory/'qualification.json'
    value = read_json(receipt)
    pins = {str(receipt): sha256(receipt), **value['source_bindings']}
    for relative, digest in value['output_hashes'].items():
        path = safe_file(directory, relative)
        require(sha256(path) == digest, 'Changed controlled generated asset')
        require(str(path) not in pins or pins[str(path)] == digest, 'Conflicting controlled source pin')
        pins[str(path)] = digest
    verify_bindings(pins)
    return pins


def prepare(bank_path, entry_id, variant_directory):
    from .procedural_petiole_controlled_catalogue_v2 import load_for_inspection
    anchor = references.prepare(bank_path, entry_id)
    directory = Path(variant_directory).resolve()
    generated = load_for_inspection(directory, anchor['source_collection_plan'])
    rows = [r for r in generated['rows'] if r['component_id'] == anchor['source_row']['component_id']]
    require(len(rows) == 1, 'One matching original target required')
    plan = {k: deepcopy(anchor[k]) for k in COMMON}
    source_pins = {**anchor['source_bindings'], **generated_bindings(directory)}
    verify_bindings(source_pins)
    plan.update(schema_version=SCHEMA, state='controlled_native848_pair_planned_not_executed',
        resolution=[848,408], reference_anchor=anchor, variant_directory=str(directory),
        generated_row=deepcopy(rows[0]), controlled_qualification_sha256=sha256(directory/'qualification.json'),
        explicit_control=deepcopy(generated['explicit_control']),
        protected_attribute_proof=deepcopy(generated['protected_attribute_proof']),
        modes=['original_control','generated_variant'], sample_count_limit=2,
        render_subframes=56, instance_backend='legacy', source_bindings=source_pins,
        implementation_bindings=implementation_bindings(), **FLAGS)
    check(plan, full=True)
    return plan


def check(plan, *, full=False):
    from .procedural_petiole_controlled_catalogue_v2 import load_for_inspection
    require(plan['schema_version'] == SCHEMA and plan['state'] == 'controlled_native848_pair_planned_not_executed'
        and plan['resolution'] == [848,408] and all(plan[k] is v for k,v in FLAGS.items())
        and plan['modes'] == ['original_control','generated_variant'] and plan['sample_count_limit'] == 2
        and plan['render_subframes'] == 56 and plan['instance_backend'] == 'legacy', 'Changed controlled pair scope')
    require(plan['implementation_bindings'] == implementation_bindings(), 'Changed controlled implementation')
    verify_bindings(plan['implementation_bindings'])
    verify_bindings(plan['source_bindings'])
    anchor = plan['reference_anchor']
    references.check(anchor, full=full)
    require(all(plan[k] == anchor[k] for k in COMMON), 'Changed original reference or scene')
    require(plan['source_family'] == 'seed41_full' and plan['source_row']['component_id'] == 'SubStem_40'
        and plan['split'] == 'train' and plan['conservative_view_cap_group'] == 'seed41_full/SubStem_40',
        'Bounded source-frame control target required')
    directory = Path(plan['variant_directory'])
    require(sha256(directory/'qualification.json') == plan['controlled_qualification_sha256'],
        'Changed controlled qualification')
    generated = load_for_inspection(directory, plan['source_collection_plan'])
    require(generated['source_family'] == generated['split_group'] == plan['source_family']
        and generated['training_eligible'] is False and generated['independent_target_novelty_approved'] is False,
        'Changed controlled lineage or authority')
    control = generated['explicit_control']
    require(control == plan['explicit_control']
        and control['schema'] == 'greenhouse.source_frame_explicit_displacement_control.v1'
        and control['amplitude_m'] in (0.002, 0.025), 'Predeclared controlled amplitude required')
    require(generated['protected_attribute_proof'] == plan['protected_attribute_proof'],
        'Changed surrounding attribute proof')
    rows = [r for r in generated['rows'] if r['component_id'] == 'SubStem_40']
    require(rows == [plan['generated_row']] and len(generated['rows']) == 1
        and rows[0]['target_id'] == generated['variant_id']+'/SubStem_40'
        and rows[0]['source_plant_id'] == rows[0]['split_group'] == plan['source_family']
        and rows[0]['conservative_view_cap_group'] == plan['conservative_view_cap_group'],
        'Changed controlled row or source pool')
    expected_pins = {**anchor['source_bindings'], **generated_bindings(directory)}
    require(plan['source_bindings'] == expected_pins, 'Incomplete controlled source bindings')
    return generated if full else None
