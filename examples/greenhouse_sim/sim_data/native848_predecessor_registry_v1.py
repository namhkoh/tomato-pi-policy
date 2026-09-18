"""Finite pinned closure dispatch only; this registry never authorizes capture.

Foreign predecessors use exactly five immutable receipt pins. Existing serial
and coordinator validators retain their native waits, full identities, assets,
source verification and current-absence checks. The caller must still perform
its reviewed capture contract, native lock and fresh process/resource admission.
"""
from pathlib import Path
import importlib
from . import native848_bulk_sibling_gate_v5 as gate
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256

ROOT=Path('D:/research/tomato-pi-policy')
H=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'
M=ROOT/'examples/greenhouse_sim/sim_data'
SCHEMA='greenhouse.native848_predecessor_registry.v1'
KEYS=frozenset(('metadata','result','terminal','outer_launch','outer_exit'))
REGISTRY={
    'greenhouse.native848_fully_labeled_owner.v5':(
        'run_native848_fully_labeled_serial_v5',H/'run_native848_fully_labeled_serial_v5.py',
        '5ea36f87e8ee4a0a1a5a914b9743230d23c95f8943a7010fa13f018d48668bc5','serial'),
    'greenhouse.native848_fully_labeled_two_owner.v1':(
        'sim_data.native848_fully_labeled_two_closure_v1',M/'native848_fully_labeled_two_closure_v1.py',
        '381153e90f3bb952ff67fc709752786a7800f6800445e9ce4a79954e77906021','two'),
    'greenhouse.native848_reset6_production_owner.v1':(
        'run_native848_reset6_production_serial_v1',H/'run_native848_reset6_production_serial_v1.py',
        '128adbf868745a3f800dc6ff868704fa362761af14e98b425abf3c4b0151fd74','serial'),
}
FIXED_BINDINGS={
    str(H/'run_native848_fully_labeled_two_v1.py'):'181622b9c174940ef33b278daf3552f1626bc581e58dbca6a753df31844524d5',
    str(M/'native848_fully_labeled_sibling_worker_v1.py'):'1605add9268f769eebe852946b57b083a3056b5a53fc29927e433dd377bc2f91',
    str(M/'native848_fully_labeled_two_admission_v1.py'):'49bf35c87d600104f0d899bc7f883a9dc390d6ff3954cff8f435d71c9792ae1d',
    str(M/'native848_bulk_sibling_gate_v5.py'):'3766afb02212dfa0bb432bb61c6dc9b3e45d7fbd259d778be6e7bc2b9a7150d4',
    str(M/'native848_reset6_production_worker_v1.py'):'5e773e7c296903083f34b7357ca015e47dc92101fcad59a0409b1f27826a2dcf',
    str(M/'native848_reset6_production_plan_v1.py'):'75df5eb3eeeff110040d7ae9052d7e0b6211d83e19adb17980709b7119e6da23',
}


def source_bindings():
    pins={str(Path(__file__).resolve()):sha256(__file__),**FIXED_BINDINGS}
    pins.update({str(path):digest for _,path,digest,_ in REGISTRY.values()})
    verify_bindings(pins)
    return pins


def review_pin():
    return dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__))


def validate_review(review,expected_sha256):
    require(sha256(__file__)==expected_sha256 and review.get('predecessor_registry')==review_pin(),
        'Explicit reviewed exact predecessor registry source pin required')
    return source_bindings()


def pin(spec):
    from . import native848_data_roots_v1 as roots
    require(isinstance(spec,dict) and set(spec)=={'path','sha256'},'Exact immutable receipt pin required')
    path=roots.resolve_evidence(spec['path'])
    require(path.is_file() and sha256(path)==spec['sha256'],'Changed predecessor receipt')
    return path


def _load(schema):
    require(schema in REGISTRY,'Unregistered predecessor schema; no fallback or dynamic plugin')
    name,path,digest,kind=REGISTRY[schema]
    require(sha256(path)==digest,'Frozen predecessor validator changed before import')
    module=importlib.import_module(name)
    require(Path(module.__file__).resolve()==path.resolve() and sha256(module.__file__)==digest,
        'Loaded predecessor validator differs from exact registry source')
    return module,kind


def authenticate(value,*,rows=None):
    """Validate one completed registered predecessor, including actual outer0."""
    require(isinstance(value,dict) and set(value)==KEYS,
        'Registry requires exact metadata/result/terminal/outer_launch/outer_exit pins')
    bindings=source_bindings()
    paths={key:pin(spec) for key,spec in value.items()}
    bindings.update({str(paths[key]):spec['sha256'] for key,spec in value.items()})
    result=read_json(paths['result']);meta=read_json(paths['metadata'])
    gate.identity(meta['exact_identity'])  # require all six original CIM fields
    validator,kind=_load(result.get('schema'))
    if kind=='serial':
        proof=validator.check_predecessor({key:value[key] for key in ('metadata','result','terminal')},rows=rows)
        outer=validator.check_outer_predecessor(dict(predecessor_outer_launch=value['outer_launch'],
            predecessor_outer_exit=value['outer_exit']),proof)
        proof['source_bindings'].update(outer)
    else:
        entries=result.get('results',[])
        require(len(entries)==2 and {entry.get('slot') for entry in entries}=={0,1},
            'Both exact completed finite-two slots required before dispatch')
        entry=next(entry for entry in entries if entry['slot']==0)
        checked=validator.authenticate_slot(Path(entry['result']['path']).resolve().parent,
            entry['result']['sha256'],value,rows=rows)
        require(checked['both_actual_waits_verified'] is True and checked['actual_outer_wait0_verified'] is True
            and checked['coordinator_lock_released'] is True,'Whole finite-two closure incomplete')
        proof=dict(metadata=meta,source_bindings=checked['source_bindings'],
            completion_method='actual_both_slots_and_coordinator_outer_wait0',synthetic_root_native_exit=False,
            additional_full_native_process_inventory_required_before_launch=True)
    require(proof['metadata']==meta and proof.get('synthetic_root_native_exit') is False
        and proof.get('parent_failed') is not True,'Exact successful predecessor identity required')
    for path,digest in bindings.items():
        require(path not in proof['source_bindings'] or proof['source_bindings'][path]==digest,
            'Conflicting registered predecessor source')
        proof['source_bindings'][path]=digest
    proof.update(predecessor_registry=review_pin(),registered_predecessor_schema=result['schema'],
        actual_outer_wait0_verified=True,outer_launch=value['outer_launch'],outer_exit=value['outer_exit'],
        capture_authorized_by_registry=False)
    return proof
