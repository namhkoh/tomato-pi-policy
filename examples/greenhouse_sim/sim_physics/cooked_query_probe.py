"""Compare captured convex unions with native collider-labelled scene rays.

Diagnostic only. Finite rays establish sampled geometry/transform agreement,
not whole-shape equivalence, path clearance, motion safety or a cut. This does
not use overlap_mesh (which cooks a separate query convex hull).
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import threading
import time

import numpy as np

from .cooked_geometry_probe import fingerprint, write_new_report


def ray_interval_distance(planes, origin, direction, maximum):
    """Ray entry into intersection of outward unit-normal halfspaces <= 0."""
    planes = np.asarray(planes, float)
    origin, direction = np.asarray(origin, float), np.asarray(direction, float)
    if (planes.ndim != 2 or planes.shape[1:] != (4,) or len(planes) < 4
            or origin.shape != (3,) or direction.shape != (3,)
            or not np.isfinite(planes).all() or not np.isfinite([origin, direction]).all()
            or not np.isfinite(maximum) or maximum <= 0
            or not np.isclose(np.linalg.norm(direction), 1., atol=1e-10, rtol=0)
            or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1., atol=1e-8, rtol=0)):
        raise ValueError("Finite unit-normal planes and a bounded unit ray required")
    distance = planes[:, :3] @ origin + planes[:, 3]
    slopes = planes[:, :3] @ direction
    parallel = abs(slopes) < 1e-14
    if np.any(parallel & (distance > 1e-10)):
        return None
    entry, leave = 0., float(maximum)
    for value, slope in zip(distance[~parallel], slopes[~parallel]):
        bound = -value / slope
        if slope < 0:
            entry = max(entry, float(bound))
        else:
            leave = min(leave, float(bound))
    return entry if entry <= leave + 1e-10 else None


def world_planes(planes_local, local_to_world, meters_per_unit):
    """Preserve native polygon halfspaces, including noncoplanar face fitting.

    Re-hulling returned vertices is NOT equivalent: native polygon planes can
    extend beyond that vertex hull. Transform covectors with inverse transpose.
    """
    from .cooked_geometry_probe import world_vertices
    world_vertices([[0,0,0]], local_to_world, meters_per_unit)  # affine/unit validation
    matrix=np.array(local_to_world,float,copy=True)
    matrix[:3]*=meters_per_unit
    planes=np.asarray(planes_local,float)
    if (planes.ndim!=2 or planes.shape[1:]!=(4,) or len(planes)<4
            or not np.isfinite(planes).all()):
        raise ValueError('Finite native polygon planes required')
    result=planes@np.linalg.inv(matrix)
    norm=np.linalg.norm(result[:,:3],axis=1)
    if np.any(norm<=1e-15):raise ValueError('Degenerate native plane')
    return result/norm[:,None]


def ray_fixture(parts, *, planes=None):
    """Deterministic axis rays through bounds AND each convex-piece centroid.

    Native callers MUST pass transformed raw polygon planes. The default hull
    of vertices is for synthetic helper tests only. Both directions and
    outside-bounds misses exercise axis/scale errors and false occupancy.
    """
    from scipy.spatial import ConvexHull
    values = [np.asarray(p, float) for p in parts]
    if not values or any(v.ndim != 2 or v.shape[1:] != (3,) or len(v) < 4
                         or not np.isfinite(v).all() for v in values):
        raise ValueError("Nonempty finite native convex pieces required")
    planes = [ConvexHull(v).equations for v in values] if planes is None else planes
    if len(planes)!=len(values):raise ValueError('One native plane set per convex piece required')
    all_points = np.concatenate(values)
    low, high = all_points.min(0), all_points.max(0)
    if np.any(high <= low):
        raise ValueError("Degenerate geometry bounds")
    padding = max(.01, float(np.linalg.norm(high-low)) * .1)
    rays = []
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        coordinates = [low[other] + np.asarray(uv) * (high-low)[other]
                       for uv in itertools.product(np.linspace(-.1, 1.1, 9), repeat=2)]
        coordinates.extend(v.mean(0)[other] for v in values)
        for sign in (-1, 1):
            for uv in coordinates:
                origin = np.zeros(3)
                origin[other] = uv
                origin[axis] = low[axis]-padding if sign == 1 else high[axis]+padding
                direction = np.eye(3)[axis] * sign
                maximum = float(high[axis]-low[axis]+2*padding)
                hits = [ray_interval_distance(p, origin, direction, maximum) for p in planes]
                hits = [h for h in hits if h is not None]
                rays.append(dict(origin_m=origin.tolist(), direction=direction.tolist(),
                                 maximum_m=maximum, expected_m=min(hits) if hits else None))
    return rays


def compare_rays(rays, query, collider_path, *, tolerance_m=2e-5):
    """Query returns nearest native distance for this EXACT collider, or None."""
    if not np.isfinite(tolerance_m) or not 0 < tolerance_m <= 2e-5:
        raise ValueError("Explicit finite <=20 micrometre diagnostic tolerance required")
    rows = []
    for ray in rays:
        actual = query(collider_path, ray)
        expected = ray['expected_m']
        if actual is not None and (not np.isfinite(actual) or not 0 <= actual <= ray['maximum_m']):
            raise ValueError("Invalid native distance")
        error = None if actual is None or expected is None else abs(float(actual)-expected)
        passed = ((actual is None and expected is None)
                  or (error is not None and error <= tolerance_m))
        rows.append(dict(**ray, native_m=actual, absolute_error_m=error, passed=bool(passed)))
    positive = sum(r['expected_m'] is not None for r in rows)
    negative = len(rows)-positive
    return dict(passed=bool(rows) and positive > 0 and negative > 0 and all(r['passed'] for r in rows),
                ray_count=len(rows), positive_cases=positive, negative_cases=negative,
                mismatches=sum(not r['passed'] for r in rows), tolerance_m=tolerance_m,
                maximum_distance_error_m=max((r['absolute_error_m'] for r in rows
                                             if r['absolute_error_m'] is not None), default=None),
                rows=rows)


def compare_current_stage(capture, output):
    """Caller already parsed actors in a stopped diagnostic stage; never step/play."""
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_scene_query_interface
    from .cooked_geometry_probe import _source_snapshot
    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    if stage is None or not timeline.is_stopped():
        raise ValueError("Expected caller-owned stopped stage")
    if capture.get('status') != 'captured_advisory':
        raise ValueError("Successful source-bound capture required")
    payload = dict(capture)
    digest = payload.pop('payload_sha256')
    if fingerprint(payload) != digest:
        raise ValueError("Captured payload changed")
    paths = list(capture['colliders'])
    before = _source_snapshot(stage, paths)
    if before['source_sha256'] != capture['source']['source_sha256']:
        raise ValueError("Current geometry/transforms differ from captured stage")
    # Actor parsing belongs BEFORE source capture: parsing can author generated
    # session-layer bookkeeping. Do not silently accept that mutation mid-query.
    query = get_physx_scene_query_interface()

    def native(path, ray):
        if not timeline.is_stopped() or omni.usd.get_context().get_stage() != stage:
            raise RuntimeError("Stage/timeline changed during native query")
        hits = []
        def collect(hit):
            if hit.collision == path:
                hits.append(float(hit.distance))
            return True  # Other scene objects must not hide the target collider.
        query.raycast_all(tuple(ray['origin_m']), tuple(ray['direction']), ray['maximum_m'], collect)
        return min(hits) if hits else None

    results = {}
    for path, row in capture['colliders'].items():
        source=capture['source']
        planes=[world_planes([face['plane_local_raw'] for face in part['polygons']],
                             source['meshes'][path]['local_to_world_column_matrix'],
                             source['meters_per_unit']) for part in row['convexes']]
        results[path] = compare_rays(ray_fixture([p['vertices_world_m'] for p in row['convexes']],planes=planes), native, path)
    after = _source_snapshot(stage, paths)
    unchanged = after['source_sha256'] == before['source_sha256']
    result = dict(schema='native_cooked_query_comparison_v1',
                  status='sampled_query_agreement' if unchanged and all(r['passed'] for r in results.values()) else 'blocked',
                  source_unchanged=unchanged, capture_payload_sha256=digest, colliders=results,
                  source_after=after,
                  native_actors_loaded=True, physics_steps_requested=0, timeline_play_requested=False,
                  collision_shapes_modified=False, all_shapes_equivalence_proved=False,
                  expected_solid='intersection_of_raw_native_polygon_halfspaces_not_vertex_hull',
                  eligible_to_replace_screen=False, physical_cut_verified=False,
                  limitations=['Finite ray samples, not exhaustive shape/overlap/clearance validation.',
                               'Stopped-stage transforms, not moving Fabric-body synchronization.',
                               'Native solid queries do not bake in contact/rest offsets.',
                               'No robot motion, blade loading, grasp or training episode.'])
    result['payload_sha256'] = fingerprint(result)
    write_new_report(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    project = Path(__file__).resolve().parents[3]
    output = args.output.resolve()
    root = (project/'data/sim_physics').resolve()
    if output == root or not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
        raise ValueError("NEW output beneath data/sim_physics with existing parent required")
    output.mkdir()
    finished = threading.Event()
    def watchdog():
        if not finished.wait(180.):
            os._exit(124)  # Only this owned process; outer native timeout still recommended.
    threading.Thread(target=watchdog, daemon=True).start()
    started = time.monotonic()
    app = None
    result = None
    code = 2
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, multi_gpu=False, sync_loads=False,
                                 create_new_stage=True, disable_viewport_updates=True, fast_shutdown=True))
        import omni.timeline
        import omni.usd
        from omni.kit.async_engine import run_coroutine
        from pxr import Gf, UsdGeom, UsdPhysics
        from sim_data.audit import DEFAULT_PACK, audit_manifest
        from sim_data.geometry import assemble_plant
        from .plant import build
        from .bimanual import BimanualRobot
        from .cooked_geometry_probe import capture_current_stage, select_targets
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.Xform.Define(stage, '/World')
        UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
        manifest = DEFAULT_PACK/'plants/components/seed101_full/manifest.json'
        paths = assemble_plant(stage, '/World/Plant', audit_manifest(manifest))
        stage.SetEditTarget(stage.GetSessionLayer())
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005, 0, .9))
        rig = build(stage, dict(manifest_path=str(manifest), component_paths=paths,
                               plant_root='/World/Plant'), 'SubStem_41')
        robot = BimanualRobot(stage, rig, sparse_contacts=True, torso_degrees=[0.]*6,
                             ground_height=lambda x,y:.101, arc=.05, approach_tilt=10,
                             grasp_roll=180, station_pose=[.530398411918,.553263301559,-163.12111184],
                             grasp_depth=.125, approach_distance=.02,
                             finger_gravity=True, compliant_fingers=True)
        targets = select_targets([str(p.GetPath()) for p in stage.Traverse()
                                  if p.HasAPI(UsdPhysics.CollisionAPI)])
        targets.append(robot.root+'/ee_right/attachments/RightWristCamera/AdapterCollision')
        # Bind the already-parsed scene. This loads native objects but does not
        # start the timeline, step, move the robot, or touch source asset files.
        from omni.physx import get_physx_interface
        print('COOKED_QUERY native actor load before source capture', flush=True)
        get_physx_interface().force_load_physics_from_usd()
        print('COOKED_QUERY capture '+str(len(targets)), flush=True)
        task = run_coroutine(capture_current_stage(output/'convexes.json', collider_paths=targets))
        while not task.done():
            if not omni.timeline.get_timeline_interface().is_stopped():
                raise RuntimeError('Unexpected timeline play')
            app.update()
        capture = task.result()
        print('COOKED_QUERY native actor load and rays', flush=True)
        result = compare_current_stage(capture, output/'queries.json')
        code = 0 if result['status'] == 'sampled_query_agreement' else 2
        print('COOKED_QUERY '+json.dumps({p:{k:v for k,v in r.items() if k!='rows'}
                                        for p,r in result['colliders'].items()}), flush=True)
    except Exception as exc:
        import traceback
        write_new_report(output/'failure.json', dict(error=str(exc), traceback=traceback.format_exc()))
    finally:
        write_new_report(output/'run.json', dict(exit_code=code, wall_s=time.monotonic()-started,
                                                status=None if result is None else result['status'],
                                                physics_steps_requested=0, full_robot_motion=False))
        try:
            if app is not None:
                app.close(wait_for_replicator=False, skip_cleanup=True, exit_code=code)
        finally:
            finished.set()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
