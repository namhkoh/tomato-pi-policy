"""Separate two-frame executor contract; stdlib/hash-only before SimulationApp.

Does not alter the frozen CPU plan. Literal local import closure includes every
new producer/context/audit/inventory helper, but no queue or test-file glob.
"""
import ast
import importlib.util
from pathlib import Path
import sys

from ..native_original_capture import contracts as oc

SCHEMA = 'greenhouse.generated_original_reference_execution.v1'
RESULT = 'greenhouse.generated_original_reference_result.v1'
SAMPLE = 'greenhouse.generated_original_reference_sample.v1'
AUDIT = 'greenhouse.generated_original_reference_audit.v1'
RECEIPT = 'greenhouse.generated_original_reference_owned_exit.v1'
WORKER = 'sim_data.native_generated_reference.worker_v1'
MODES = ['original_control', 'generated_variant']
FLAGS = dict(training_approved=False, source_cap_reset=False,
    independent_geometry_qualification=False, global_complete=False,
    depth_recomputed=False, physical_execution_approved=False)


def _closure():
    root = Path(__file__).resolve().parent
    search = (root.parent.parent, root.parents[2])
    pending = [root/n for n in ('__init__.py', 'prepare.py', 'scene.py',
        'execution_v1.py', 'worker_v1.py', 'owner_v1.py', 'audit_v1.py', 'inventory_v1.py')]
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
                name = '.' * node.level + (node.module or '')
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


def require_running_app(app):
    # Source verified: isaacsim.simulation_app SimulationApp.is_running() returns
    # the underlying Kit lifetime state. Never import Isaac merely to check it.
    cls = getattr(sys.modules.get('isaacsim'), 'SimulationApp', None)
    oc.require(isinstance(cls, type) and isinstance(app, cls), 'Initialized SimulationApp instance required')
    oc.require(app.is_running() is True, 'SimulationApp must be running before native imports')


def structural_plan(plan):
    oc.require(plan.get('schema') == 'greenhouse.generated_from_original_reference_plan.v1'
        and plan.get('state') == 'cpu_prepared_scene_only_no_native_producer_bound'
        and plan.get('profile') == 'one_target_exact_pose_original_generated_56.v1', 'Wrong frozen CPU seam')
    oc.require(plan['execution'] == dict(native_launch_supported=False, producer_module=None,
        producer_bindings=None, owned_launcher_receipt=None, postexit_audit_adapter=None), 'CPU plan acquired execution authority')
    oc.require(plan['modes'] == MODES and type(plan['sample_count_limit']) is int and plan['sample_count_limit'] == 2
        and plan['resolution'] == oc.RESOLUTION and plan['camera_path'] == oc.HEAD_CAMERA
        and type(plan['render_subframes_per_view']) is int and plan['render_subframes_per_view'] == 56
        and plan['native_instance_backend'] == 'legacy' and plan['pose_changes_between_modes'] is False
        and plan['split'] == 'train' and oc.FROZEN_SPLITS.get(plan['source_family']) == 'train', 'Changed bounded native policy')
    for key in ('training_approved','source_cap_reset','native_launched','generated_native_qualification',
                'independent_target_novelty_approved','physical_motion_commanded','paired_848_1696_proof'):
        oc.require(plan[key] is False, 'Unexpected CPU authority: '+key)
    oc.require(plan['scene_authority']['source_row'] == plan['source_row']
        and plan['scene_authority']['policy'] == oc.SCENE_POLICY
        and plan['source_row']['source_plant_id'] == plan['source_family'] == plan['split_group']
        and plan['source_row']['target_id'] == plan['conservative_view_cap_group']
        == plan['generated_row']['conservative_view_cap_group'], 'CPU source ancestry changed')
    oc.bind_all(plan['source_bindings']); oc.bind_all(plan['implementation_bindings'])
    return plan


def runtime_bindings(isaac_python):
    python = Path(isaac_python).resolve(strict=True)
    oc.require(python.name == 'python.bat', 'Explicit Windows Isaac python.bat required')
    root = python.parent/'exts/isaacsim.simulation_app'
    paths = [python, root/'isaacsim/simulation_app/simulation_app.py', root/'config/python_api.md']
    return {str(p): oc.sha256(p) for p in paths}


def make_request(plan_path, *, plan_sha256, isaac_python, native_deps, output):
    # CPU-only entrypoint: full replay is allowed here, never worker preflight.
    oc.require(not Path(sys.executable).resolve().is_relative_to(Path(isaac_python).resolve().parent)
        and not any(n.split('.')[0] in ('isaacsim','omni','carb') for n in sys.modules), 'Use a separate CPU interpreter for preparation')
    from .prepare import check_plan
    plan_path = oc.pin(plan_path, plan_sha256)
    plan = structural_plan(oc.read_json(plan_path)); check_plan(plan)
    code = implementation_bindings()
    deps = Path(native_deps).resolve(strict=True)
    oc.require(deps.is_dir(), 'Native dependency directory required')
    runtime = runtime_bindings(isaac_python)
    bindings = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'], code, runtime,
                                {str(plan_path): plan_sha256})
    output = oc.new_destination(output, [plan_path.parent, plan['scene_authority']['package'],
        plan['variant_directory'], *bindings])
    return dict(schema=SCHEMA, plan_path=str(plan_path), plan_sha256=plan_sha256,
        output=str(output), isaac_python=str(Path(isaac_python).resolve()), native_deps=str(deps),
        storage='raw_native_v1', modes=list(MODES), maximum_frames=2, render_subframes_per_view=56,
        worker_module=WORKER, implementation_bindings=code, source_bindings=bindings,
        runtime_bindings=runtime, dependency_environment_fully_attested=False,
        automatic_retries=False, **FLAGS)


def preflight_request(path, sha256):
    """Hash/strict structural checks ONLY; safe before SimulationApp imports USD."""
    request_path = oc.pin(path, sha256); request = oc.read_json(request_path)
    for value in (str(request_path),request['plan_path'],request['output'],request['isaac_python'],request['native_deps']):
        oc.require(isinstance(value,str) and not any(c in value for c in '&|<>^%!"\r\n'), 'Unsafe Windows bootstrap path')
    oc.require(request['schema'] == SCHEMA and request['worker_module'] == WORKER
        and request['storage'] == 'raw_native_v1' and request['modes'] == MODES
        and type(request['maximum_frames']) is int and request['maximum_frames'] == 2
        and type(request['render_subframes_per_view']) is int and request['render_subframes_per_view'] == 56
        and request['automatic_retries'] is False
        and all(request[k] is v for k,v in FLAGS.items()), 'Changed executor request')
    oc.require(request['implementation_bindings'] == implementation_bindings(), 'Executor code differs from request')
    oc.bind_all(request['source_bindings']); oc.bind_all(request['runtime_bindings'])
    oc.require(request['runtime_bindings'] == runtime_bindings(request['isaac_python']), 'Different native runtime')
    plan = structural_plan(oc.read_json(oc.pin(request['plan_path'], request['plan_sha256'])))
    expected = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'],
        request['implementation_bindings'], request['runtime_bindings'], {request['plan_path']:request['plan_sha256']})
    oc.require(request['source_bindings'] == expected, 'Incomplete/expanded executor source closure')
    output = Path(request['output']).resolve()
    oc.require(str(output) == request['output'] and output != Path(output.anchor), 'Canonical non-root output required')
    for source in [Path(plan['scene_authority']['package']), Path(plan['variant_directory']),
                   Path(request['plan_path']).parent, request_path, *map(Path, expected)]:
        oc.require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Output overlaps pinned inputs')
    return request, plan


def worker_command(request_path, request_sha256, request):
    return [request['isaac_python'], '-m', WORKER, '--request', str(Path(request_path).resolve()),
            '--request-sha256', request_sha256]
