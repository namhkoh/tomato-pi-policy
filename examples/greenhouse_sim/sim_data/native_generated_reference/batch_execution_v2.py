"""Separate bounded native-batch contract; hash-only preflight before Kit.

The reviewed CPU plan stays immutable. This contract binds its producer, owner,
native replay and inventory reader, plus the previously qualified lossless
storage. Qualification of this producer still requires actual native execution.
"""
import ast
import importlib.util
from pathlib import Path
import sys

from ..native_original_capture import contracts as oc
from .execution_v1 import require_running_app, runtime_bindings

SCHEMA = 'greenhouse.generated_original_batch_execution.v2'
RESULT = 'greenhouse.generated_original_batch_result.v2'
SAMPLE = 'greenhouse.generated_original_batch_sample.v2'
AUDIT = 'greenhouse.generated_original_batch_audit.v2'
RECEIPT = 'greenhouse.generated_original_batch_owned_exit.v2'
WORKER = 'sim_data.native_generated_reference.batch_worker_v2'
FLAGS = dict(training_approved=False, source_cap_reset=False,
    independent_geometry_qualification=False, global_complete=False,
    depth_recomputed=False, physical_execution_approved=False)


def _closure():
    root = Path(__file__).resolve().parent
    search = (root.parent.parent, root.parents[2])
    pending = [root/n for n in ('batch_prepare_v2.py', 'batch_scene_v2.py',
        'batch_pose_v2.py', 'batch_execution_v2.py', 'batch_worker_v2.py',
        'batch_owner_v2.py', 'batch_audit_v2.py', 'batch_inventory_v2.py')]
    pending.append(root.parent/'native_dataset/original_reference_batch_owner_v2.py')
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes(); found[str(path)] = oc.digest(raw)
        base = next(p for p in search if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.'*node.level + (node.module or '')
                name = importlib.util.resolve_name(name, package) if node.level else name
                names = [name] + [name+'.'+a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in search:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


_LOADED = _closure()


def implementation_bindings():
    oc.bind_all(_LOADED)
    return dict(_LOADED)


def jobs(plan):
    return [(case, spec) for case in plan['target_cases'] for spec in case['views']]


def structural_plan(plan):
    oc.require(plan.get('schema') == 'greenhouse.generated_original_native_batch_plan.v2'
        and plan.get('state') == 'cpu_prepared_batch_pending_bound_native_producer'
        and plan.get('profile') == 'reviewed_original_current_scene_mounted_views_reference56.v2',
        'Wrong frozen CPU batch seam')
    oc.require(plan['resolution'] == oc.RESOLUTION and plan['camera_path'] == oc.HEAD_CAMERA
        and type(plan['render_subframes_per_view']) is int and plan['render_subframes_per_view'] == 56
        and plan['native_instance_backend'] == 'legacy' and plan['split'] == 'train'
        and plan['source_family'] == plan['split_group'] and oc.FROZEN_SPLITS.get(plan['source_family']) == 'train',
        'Changed native batch policy')
    for key in ('training_approved', 'source_cap_reset', 'native_launched', 'independent_geometry_qualification',
        'generated_native_qualification', 'physical_motion_commanded', 'paired_848_1696_proof',
        'hidden_cut_coordinates_executable', 'native_launch_supported'):
        oc.require(plan[key] is False, 'CPU batch acquired authority: '+key)
    cases = plan['target_cases']; count = plan['views_per_target']
    oc.require(isinstance(cases, list) and 1 <= len(cases) <= 12 and type(count) is int and 1 <= count <= 6,
        'One-to-twelve targets and one-to-six views required')
    names = []
    for case in cases:
        row = case['source_row']; generated = case['generated_row']
        oc.require(row['source_plant_id'] == plan['source_family']
            and case['conservative_view_cap_group'] == row['target_id'] == generated['conservative_view_cap_group']
            and generated['source_plant_id'] == generated['split_group'] == plan['source_family']
            and case['target_id'] == generated['target_id'] and len(case['views']) == count,
            'Batch source lineage or bound differs')
        names.extend(s['candidate_id'] for s in case['views'])
    oc.require(len({c['target_id'] for c in cases}) == len(cases)
        and len(names) == len(set(names)) == plan['maximum_native_frames']
        and type(plan['maximum_native_frames']) is int and 1 <= len(names) <= 72
        and all(isinstance(n, str) and n and n not in ('.', '..')
            and not any(c in n for c in '/\\:') for n in names), 'Repeated, unsafe or unbounded native cases')
    oc.require(plan['scene_authority']['source_row'] == plan['source_row'] == cases[0]['source_row']
        and plan['scene_authority']['policy'] == oc.SCENE_POLICY, 'Changed current scene authority')
    oc.bind_all(plan['source_bindings']); oc.bind_all(plan['implementation_bindings'])
    return plan


def checked_storage(request):
    """Full proof replay; CPU owner or initialized Kit only."""
    from ..native_dataset import compact_qualification as compact
    declared = request['storage_qualification']
    actual = compact.check_qualification(declared['path'], expected_sha256=declared['sha256'])
    oc.require(actual == declared and request['storage_backend'] == compact.STORAGE_BACKEND,
        'Storage qualification differs from the bound request')
    return actual


def make_request(plan_path, *, plan_sha256, isaac_python, native_deps, output,
                 storage_qualification, storage_qualification_sha256):
    oc.require(not Path(sys.executable).resolve().is_relative_to(Path(isaac_python).resolve().parent)
        and not any(n.split('.')[0] in ('isaacsim', 'omni', 'carb') for n in sys.modules),
        'Use a separate CPU interpreter for preparation')
    from .batch_prepare_v2 import check_plan
    from ..native_dataset import compact_qualification as compact
    plan_path = oc.pin(plan_path, plan_sha256)
    plan = structural_plan(oc.read_json(plan_path)); check_plan(plan)
    proof = compact.check_qualification(storage_qualification, expected_sha256=storage_qualification_sha256)
    code = implementation_bindings(); runtime = runtime_bindings(isaac_python)
    deps = Path(native_deps).resolve(strict=True)
    oc.require(deps.is_dir(), 'Native dependency directory required')
    bindings = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'], code, runtime,
        proof['bindings'], {str(plan_path): plan_sha256})
    output = oc.new_destination(output, [plan_path.parent, plan['scene_authority']['package'],
        plan['variant_directory'], Path(proof['path']).parent, *bindings])
    return dict(schema=SCHEMA, plan_path=str(plan_path), plan_sha256=plan_sha256,
        output=str(output), isaac_python=str(Path(isaac_python).resolve()), native_deps=str(deps),
        storage_backend=compact.STORAGE_BACKEND, storage_qualification=proof,
        maximum_frames=plan['maximum_native_frames'], render_subframes_per_view=56,
        worker_module=WORKER, implementation_bindings=code, source_bindings=bindings,
        runtime_bindings=runtime, dependency_environment_fully_attested=False,
        automatic_retries=False, **FLAGS)


def preflight_request(path, sha256):
    """Stdlib/hashes only: never load USD before the native SimulationApp."""
    request_path = oc.pin(path, sha256); request = oc.read_json(request_path)
    for value in (str(request_path), request['plan_path'], request['output'], request['isaac_python'], request['native_deps']):
        oc.require(isinstance(value, str) and not any(c in value for c in '&|<>^%!"\r\n'), 'Unsafe Windows bootstrap path')
    oc.require(request['schema'] == SCHEMA and request['worker_module'] == WORKER
        and type(request['maximum_frames']) is int and 1 <= request['maximum_frames'] <= 72
        and type(request['render_subframes_per_view']) is int and request['render_subframes_per_view'] == 56
        and request['automatic_retries'] is False and request['dependency_environment_fully_attested'] is False
        and all(request[k] is v for k, v in FLAGS.items()), 'Changed batch executor request')
    oc.require(request['implementation_bindings'] == implementation_bindings(), 'Executor code differs from request')
    oc.bind_all(request['source_bindings']); oc.bind_all(request['runtime_bindings'])
    oc.require(request['runtime_bindings'] == runtime_bindings(request['isaac_python']), 'Different native runtime')
    proof = request['storage_qualification']
    oc.pin(proof['path'], proof['sha256']); oc.bind_all(proof['bindings'])
    oc.require(proof['schema'] == 'greenhouse.compact_storage_qualification_binding.v1'
        and proof['storage_backend'] == request['storage_backend'] == 'greenhouse.compact_native_sample.v1'
        and proof['training_approved'] is False and type(proof['training_diversity_increment']) is int
        and proof['training_diversity_increment'] == 0, 'Changed storage scope')
    plan = structural_plan(oc.read_json(oc.pin(request['plan_path'], request['plan_sha256'])))
    oc.require(request['maximum_frames'] == plan['maximum_native_frames'], 'Request changes planned frame count')
    expected = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'],
        request['implementation_bindings'], request['runtime_bindings'], proof['bindings'],
        {request['plan_path']: request['plan_sha256']})
    oc.require(request['source_bindings'] == expected, 'Incomplete/expanded executor source closure')
    output = Path(request['output']).resolve()
    oc.require(str(output) == request['output'] and output != Path(output.anchor), 'Canonical non-root output required')
    for source in [Path(plan['scene_authority']['package']), Path(plan['variant_directory']),
        Path(request['plan_path']).parent, Path(proof['path']).parent, request_path, *map(Path, expected)]:
        oc.require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output overlaps pinned inputs')
    return request, plan


def worker_command(request_path, request_sha256, request):
    return [request['isaac_python'], '-m', WORKER, '--request', str(Path(request_path).resolve()),
        '--request-sha256', request_sha256]
