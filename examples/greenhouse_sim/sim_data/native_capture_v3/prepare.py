"""Build new generated capture jobs from exact, freshly qualified bank anchors.

CPU asset/plan construction only. Original plans, source geometry and reviews
remain immutable. A proposed robot snapshot is not an executable robot motion.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import json

from . import reference_bank as rb
from . import reference_schedule as schedule_api
from ..dataset_review import read_json, write_json, require, verify_bindings
from ..depth_preview import sha256


def choose_references(bank, job, attempt, seed, max_targets=3):
    """Same-group references only; anchor retains its exact sensor-proof pose."""
    require(type(seed) is int and 0 <= seed < 2**32, 'uint32 seed required')
    require(type(max_targets) is int and 1 <= max_targets <= 12, 'One to twelve targets')
    for key in ('source_family', 'source_collection_plan', 'source_collection_plan_sha256',
                'compatibility_group'):
        require(attempt[key] == job[key], 'Anchor/group mismatch: '+key)
    require(job['split'] == attempt['split'] == 'train', 'TRAIN-only construction')
    entries = {entry['id']: entry for entry in bank['entries']}
    require(attempt['id'] in entries, 'Anchor missing from checked reference bank')
    original = entries[attempt['id']]
    require(all(attempt.get(k) == v for k, v in original.items()), 'Anchor original evidence changed')
    require(attempt['target_id'] in job['reviewed_target_ids'], 'Unreviewed anchor target')
    candidates = []
    for target in job['reviewed_target_ids']:
        if target == attempt['target_id']:
            continue
        references = rb.select_views(bank, job['compatibility_group'], target, limit=6)
        require(references, 'Missing compatible target reference')
        # Rotate among genuinely different historical poses over generated seeds.
        # This does not grant independent image/context credit.
        reference = references[seed % len(references)]
        require(reference['source_collection_plan'] == job['source_collection_plan']
                and reference['source_collection_plan_sha256'] == job['source_collection_plan_sha256']
                and reference['source_family'] == job['source_family'] and reference['split'] == 'train',
                'Target reference escaped original plan/family')
        candidates.append(reference)
    candidates.sort(key=lambda e: (rb.rank_key(e), e['target_id'], e['id']))
    return deepcopy([original, *candidates[:max_targets-1]])


def validate_destination(output, bank, schedule):
    """Never place generated copies inside the original package or receipts."""
    output = Path(output).resolve()
    protected = {Path(bank['checkpoint']).resolve(), Path(schedule['qualification_output_root']).resolve()}
    protected.update(Path(e['source_capture']).resolve() for e in bank['entries'])
    protected.update(Path(p).resolve().parent for p in bank['source_bindings'])
    plans = {e['source_collection_plan']:e['source_collection_plan_sha256'] for e in bank['entries']}
    for path, expected in plans.items():
        require(sha256(path) == expected, 'Original collection plan changed')
        protected.add(Path(path).resolve().parent)
        protected.add(Path(read_json(path)['package']).resolve())
    require(not output.exists() and output != Path(output.anchor)
            and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            'New generated job disjoint from source, plan and qualification roots required')
    return output


def verify_fresh_anchor(attempt):
    """Validate native observations BEFORE any geometry generation or writes."""
    from ..generated_capture import verify_sensor_prerequisite
    from ..capture_sensor import calibration_for_native_resolution, HIRES_RESOLUTION
    proof = Path(attempt['qualification_output'])
    require((proof/'result.json').is_file() and not (proof/'failure.json').exists(),
            'Fresh exact-anchor native camera qualification required first')
    return verify_sensor_prerequisite(dict(prerequisite_directory=str(proof),
        source_capture=attempt['source_capture'], source_sample=attempt['source_sample'],
        source_row=attempt['source_row'], expected_calibration=calibration_for_native_resolution(
            attempt['calibration'], HIRES_RESOLUTION)))


def verify_prepared(output, schedule_path, job, attempt, seed, *, max_targets=3):
    """Launcher handoff: a plan without its completed preparation is insufficient."""
    output, schedule_path = Path(output).resolve(), Path(schedule_path).resolve()
    result = read_json(output/'prepared.json')
    request = read_json(output/'prepare_request.json')
    require(result.get('state') == 'generated_reference_bank_job_prepared_pending_native_capture',
            'Preparation did not complete')
    require(all(result.get(k) == v for k,v in request.items()), 'Preparation request/result changed')
    expected = dict(schedule=str(schedule_path),schedule_sha256=sha256(schedule_path),
        job_id=job['job_id'],attempt_id=attempt['attempt_id'],seed=seed,max_targets=max_targets,
        source_collection_plan=job['source_collection_plan'],
        source_collection_plan_sha256=job['source_collection_plan_sha256'],
        compatibility_group=job['compatibility_group'],source_family=job['source_family'])
    require(all(result.get(k)==v for k,v in expected.items()), 'Preparation differs from scheduled job')
    require(result.get('training_approved') is False and result.get('source_cap_reset') is False
            and result.get('native_capture_started') is False and result.get('physical_execution_approved') is False,
            'Preparation cannot grant approval')
    require(result.get('reference_ids') and result['reference_ids'][0]==attempt['id'],
            'Exact qualified anchor missing')
    require(result.get('implementation_bindings')=={str(Path(__file__).resolve()):sha256(__file__)},
            'Preparation implementation changed')
    require(result.get('plan_path')==str(output/'plan.json')
            and result.get('plan_sha256')==sha256(output/'plan.json'), 'Prepared plan missing or changed')
    proof=verify_fresh_anchor(attempt)['bindings']
    require(result.get('sensor_prerequisite_bindings')==proof, 'Prepared proof belongs to another anchor')
    verify_bindings(proof)
    schedule=read_json(schedule_path)
    require(sha256(schedule['source_reference_bank'])==schedule['source_reference_bank_sha256'], 'Changed bank')
    bank=read_json(schedule['source_reference_bank']);entries={e['id']:e for e in bank['entries']}
    plan=read_json(output/'plan.json')
    from ..native_multitarget_plan import check as check_plan
    check_plan(plan,replay_geometry=False)
    cases=plan['target_cases'];ids=result['reference_ids']
    require(1<=len(cases)==len(ids)<=max_targets and len(set(ids))==len(ids), 'Prepared target bound changed')
    require(plan['prerequisite_bindings']==proof and plan['source_family']==job['source_family']
            and plan['split']=='train' and plan['resolution']==[1696,816], 'Plan/proof/family mismatch')
    require(plan['anchor_pair_plan']==cases[0]['base_pair_plan'] and plan['views_per_target']==6
            and plan['maximum_native_frames']==result['maximum_native_frames']==6*len(cases),
            'Prepared anchor or view bound changed')
    source_targets=[]
    for identity,case in zip(ids,cases,strict=True):
        require(identity in entries, 'Prepared reference absent from pinned bank')
        reference=entries[identity]
        require(reference['compatibility_group']==job['compatibility_group'], 'Mixed reference groups')
        base_path=Path(case['base_pair_plan']).resolve()
        require(base_path.parent==output and sha256(base_path)==case['base_pair_plan_sha256'],
                'Prepared base missing, changed or outside job')
        base=read_json(base_path)
        for key in ('source_capture','source_sample','source_collection_plan','source_family','source_row'):
            require(base[key]==reference[key], 'Prepared base does not match reviewed reference: '+key)
        require(base['source_collection_plan']==job['source_collection_plan']
                and sha256(base['source_collection_plan'])==job['source_collection_plan_sha256']
                and Path(base['prerequisite_directory']).resolve()==Path(attempt['qualification_output']).resolve(),
                'Prepared base plan/proof changed')
        require(base['generated_row']['target_id']==case['target_id']
                and base['generated_row']['component_id']==reference['source_row']['component_id']
                and case['conservative_view_cap_group']==reference['target_id'] and len(case['views'])==6,
                'Prepared target identity or source cap changed')
        source_targets.append(reference['target_id'])
    require(result['targets']==source_targets, 'Prepared target inventory changed')
    return result


def prepare(schedule_path, job_id, attempt_id, seed, output, *, max_targets=3):
    schedule_path = Path(schedule_path).resolve()
    schedule = read_json(schedule_path)
    schedule_api.check(schedule)
    job = next(j for j in schedule['jobs'] if j['job_id'] == job_id)
    attempt = next(a for a in [job['anchor'], *job['fallback_anchors']] if a['attempt_id'] == attempt_id)
    bank = read_json(schedule['source_reference_bank'])
    require(sha256(schedule['source_reference_bank']) == schedule['source_reference_bank_sha256'],
            'Source bank changed')
    references = choose_references(bank, job, attempt, seed, max_targets)
    output = validate_destination(output, bank, schedule)
    sensor_proof = verify_fresh_anchor(attempt)
    proof = Path(attempt['qualification_output'])

    # Lazy imports: no USD/Kit import occurs when inspecting CLI or pure selection.
    from ..generated_capture import prepare_plan, check_plan, verify_sensor_prerequisite
    from ..plant_variants import load_training_sources, training_envelope
    from ..procedural_petiole_v2 import generate, plan_change
    from ..native_multitarget_plan import build, check

    frozen, sources = load_training_sources(job['source_collection_plan'])
    envelope = training_envelope(frozen, sources)
    source = sources[job['source_family']]
    selected, exclusions = [], []
    for reference in references:
        component = reference['source_row']['component_id']
        try:
            plan_change(source, component, envelope, seed+len(selected)*104729)
        except ValueError as exc:
            if reference['id'] == attempt['id']:
                raise ValueError('Exact qualified anchor cannot generate this seed: '+str(exc)) from exc
            exclusions.append(dict(target_id=reference['target_id'], reason=str(exc)))
            continue
        selected.append(reference)
    require(selected and selected[0]['id'] == attempt['id'], 'Exact anchor must remain first')
    output.mkdir(parents=True)
    implementation = {str(Path(__file__).resolve()):sha256(__file__)}
    request = dict(schedule=str(schedule_path), schedule_sha256=sha256(schedule_path),
        job_id=job_id, attempt_id=attempt_id, seed=seed, max_targets=max_targets,
        reference_ids=[e['id'] for e in selected], source_collection_plan=job['source_collection_plan'],
        source_collection_plan_sha256=job['source_collection_plan_sha256'],
        compatibility_group=job['compatibility_group'], source_family=job['source_family'],
        training_approved=False, implementation_bindings=implementation,
        sensor_prerequisite_bindings=sensor_proof['bindings'])
    write_json(output/'prepare_request.json', request)
    variant = output/(job['source_family']+'_cr_'+str(seed))
    generate(job['source_collection_plan'], job['source_family'],
             [e['source_row']['component_id'] for e in selected], seed, variant)
    bases = []
    for reference in selected:
        base = prepare_plan(reference['source_capture'], reference['source_sample'], variant, proof)
        check_plan(base)
        if reference['id'] == attempt['id']:
            verify_sensor_prerequisite(base)
        path = output/(reference['source_row']['component_id']+'_base.json')
        write_json(path, base)
        bases.append(path)
    plan = build(bases[0], bases, 6)
    check(plan)
    write_json(output/'plan.json', plan)
    verify_bindings(implementation)
    verify_bindings(schedule['implementation_bindings'])
    verify_bindings(sensor_proof['bindings'])
    result = dict(state='generated_reference_bank_job_prepared_pending_native_capture',
        **request, plan_path=str(output/'plan.json'), plan_sha256=sha256(output/'plan.json'),
        targets=[e['target_id'] for e in selected], excluded_proposals=exclusions,
        maximum_native_frames=plan['maximum_native_frames'], native_capture_started=False,
        source_cap_reset=False, physical_execution_approved=False, new_biological_families=0)
    write_json(output/'prepared.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-targets', type=int, default=3)
    args = parser.parse_args()
    result = prepare(args.schedule, args.job_id, args.attempt_id, args.seed, args.output,
                     max_targets=args.max_targets)
    print('REFERENCE_JOB_PREPARED', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
