"""Pinned actual-mesh comparison of a donor and exactly two whole-plant variants.

CPU orthographic painter only: diagnostic colors and approximate occlusion,
without textures, native RGB-D, robot calibration, or training admission.
"""
import argparse
from pathlib import Path

import numpy as np

from .audit import audit_manifest, safe_asset
from .cut_regions import _oriented_chain, _sample
from .dataset_review import read_json, require, write_json
from .depth_preview import sha256
from .geometry import assemble_plant
from .plant_variant_preview import camera_basis, triangles_by_component, draw_mesh

SCHEMA = 'greenhouse.whole_plant_morphology_mesh_preview.v1'
QUALIFICATION_SCHEMA = 'greenhouse.donor_whole_plant_morphology_qualification.v1'


def remember(bindings, path, expected):
    path = Path(path).resolve()
    require(isinstance(expected, str) and len(expected) == 64
            and all(c in '0123456789abcdef' for c in expected), 'Explicit SHA256 required')
    require(path.is_file() and sha256(path) == expected, 'Changed or missing pinned file: '+str(path))
    require(str(path) not in bindings or bindings[str(path)] == expected, 'Conflicting input pins')
    bindings[str(path)] = expected


def geometry_9mm(report, target):
    component = report['components'][target]
    require(component['type'] == 'sub_stem' and component['deleafed'] is False
            and report['components'][component['parent']]['type'] == 'main_stem',
            'Intact direct-main-stem petiole required')
    chain, arc, reversed_chain, endpoint_error = _oriented_chain(component, 1e-6)
    require(arc[-1] >= .030, 'At least 30 mm of centerline required')
    origin = component['translation_plant_m']
    nominal = _sample(chain, arc, .009, origin)
    support = [_sample(chain, arc, float(s), origin)['point_plant_m']
               for s in np.linspace(.009, .019, 11)]
    return dict(attachment_plant_m=component['attachment_plant_m'],
                nominal_arc_m=.009, nominal_point_plant_m=nominal['point_plant_m'],
                visibility_support_arc_m=[.009, .019], visibility_support_points_plant_m=support,
                support_is_permissible_cut_interval=False,
                centerline_length_m=float(arc[-1]), source_chain_reversed=bool(reversed_chain),
                attachment_endpoint_error_m=float(endpoint_error),
                capsule_radius_used_for_visual_width_or_eligibility=False,
                surface_or_native_visibility_verified=False)


def check_receipt(receipt, family, manifest, manifest_sha):
    require(receipt.get('schema') == QUALIFICATION_SCHEMA, 'Wrong generated qualification schema')
    require(receipt.get('source_family') == family and receipt.get('split_group') == family
            and receipt.get('original_source_family_cap_group') == family,
            'Original donor lineage must be retained')
    require(receipt.get('source_split') in ('train', 'validation', 'test'), 'Missing original split')
    require(receipt.get('variant_id') == Path(manifest).parent.name, 'Variant directory mismatch')
    require(receipt.get('output_hashes', {}).get('manifest.json') == manifest_sha,
            'Generated manifest is not bound by qualification')
    require(receipt.get('full_component_graph_preserved') is True
            and receipt.get('source_deleaf_state_preserved') is True, 'Incomplete plant receipt')
    require(receipt.get('source_assets_unchanged') is True, 'Source preservation receipt missing')
    require(receipt.get('training_approved') is False
            and receipt.get('native_capture_validated') is False
            and receipt.get('accepted_training_increment') == 0
            and receipt.get('new_independent_source_family') is False,
            'Diagnostic-only generated morphology required')


def load_inputs(source_manifest, source_sha, manifests, manifest_shas, qualification_shas):
    require(len(manifests) == len(manifest_shas) == len(qualification_shas) == 2,
            'Exactly two generated manifests and explicit pins required')
    source_manifest = Path(source_manifest).resolve()
    manifests = [Path(p).resolve() for p in manifests]
    require(len(set([source_manifest, *manifests])) == 3, 'Three distinct manifests required')
    bindings = {}
    remember(bindings, source_manifest, source_sha)
    source = audit_manifest(source_manifest)
    require(source['status'] != 'blocked', 'Blocked source manifest')
    reports, receipts = [source], []
    family = source_manifest.parent.name
    source_graph = {k: (v['parent'], v['type'], v.get('deleafed'))
                    for k, v in source['components'].items()}
    for manifest, manifest_sha, qualification_sha in zip(manifests, manifest_shas, qualification_shas):
        remember(bindings, manifest, manifest_sha)
        qp = manifest.parent/'qualification.json'
        remember(bindings, qp, qualification_sha)
        receipt = read_json(qp)
        check_receipt(receipt, family, manifest, manifest_sha)
        require(receipt['source_bindings'].get(str(source_manifest)) == source_sha,
                'Qualification does not bind this exact donor')
        for key in ('source_bindings', 'texture_source_bindings', 'implementation_hashes'):
            require(isinstance(receipt.get(key), dict) and receipt[key], 'Missing '+key)
            for path, expected in receipt[key].items():
                if str(Path(path).resolve()) not in bindings:
                    remember(bindings, path, expected)
                else:
                    require(bindings[str(Path(path).resolve())] == expected, 'Conflicting shared input pin')
        for relative, expected in receipt['output_hashes'].items():
            remember(bindings, safe_asset(manifest.parent, relative), expected)
        report = audit_manifest(manifest)
        require(report['status'] != 'blocked', 'Blocked generated manifest')
        graph = {k: (v['parent'], v['type'], v.get('deleafed'))
                 for k, v in report['components'].items()}
        require(graph == source_graph, 'Full source organ graph/deleaf state differs')
        for component in report['components'].values():
            require(receipt['output_hashes'].get(component['file']) == component['asset_sha256'],
                    'Generated component absent from qualification')
        for component in source['components'].values():
            path = safe_asset(source_manifest.parent, component['file'])
            require(receipt['source_bindings'].get(str(path)) == component['asset_sha256'],
                    'Donor component absent from qualification')
        reports.append(report)
        receipts.append(receipt)
    require(receipts[0]['source_split'] == receipts[1]['source_split'], 'Generated split mismatch')
    return reports, receipts, bindings


def shared_limits(meshes, basis):
    points = np.concatenate([t.reshape(-1, 3) for group in meshes
                             for t in group.values() if len(t)]) @ basis
    low, high = points[:, :2].min(0), points[:, :2].max(0)
    pad = max(high-low)*.04
    return (low[0]-pad, high[0]+pad, low[1]-pad, high[1]+pad)


def preview(source_manifest, source_manifest_sha256, generated_manifests,
            generated_manifest_sha256, qualification_sha256, output, target='SubStem_42'):
    from pxr import Usd, UsdGeom
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    reports, receipts, bindings = load_inputs(source_manifest, source_manifest_sha256,
        generated_manifests, generated_manifest_sha256, qualification_sha256)
    output = Path(output).resolve()
    require(not output.exists() and all(not output.is_relative_to(Path(r['manifest_path']).parent)
            for r in reports), 'Create-only preview output outside all plant assets required')
    geometry = [geometry_9mm(report, target) for report in reports]
    meshes, stages = [], []
    for report in reports:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        UsdGeom.SetStageUpAxis(stage, 'Z')
        paths = assemble_plant(stage, '/Plant', report)
        meshes.append(triangles_by_component(stage, paths))
        stages.append(stage)
    require(all(set(m) == set(reports[0]['components']) for m in meshes), 'Missing component ownership')
    output.mkdir(parents=True, exist_ok=False)
    titles = ['Original donor', *[r['variant_id'] for r in receipts]]
    bases = [camera_basis(-50, 9), camera_basis(40, 9)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 13))
    for row, basis in enumerate(bases):
        limits = shared_limits(meshes, basis)
        for col, (mesh, report) in enumerate(zip(meshes, reports)):
            draw_mesh(axes[row, col], mesh, report['components'], set(), basis, limits)
            axes[row, col].set_title(titles[col], fontsize=10)
    fig.suptitle('Actual whole-plant USD meshes — two shared orthographic views', fontsize=16)
    fig.text(.5, .016, 'DIAGNOSTIC ONLY: flat organ colors, approximate triangle occlusion; '
             'no textures or native RGB-D. Generated plants retain the original donor family.',
             ha='center', fontsize=9)
    fig.subplots_adjust(top=.945, bottom=.04, left=.025, right=.975, wspace=.04, hspace=.09)
    whole = output/'whole_plant_comparison.png'
    fig.savefig(whole, dpi=140, facecolor='white'); plt.close(fig)
    basis = camera_basis(-50, 9)
    centers = [np.asarray(g['attachment_plant_m']) @ basis for g in geometry]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (mesh, report, g, center) in enumerate(zip(meshes, reports, geometry, centers)):
        limits = (center[0]-.04, center[0]+.04, center[1]-.03, center[1]+.03)
        draw_mesh(axes[i], mesh, report['components'], set(), basis, limits)
        support = np.asarray(g['visibility_support_points_plant_m']) @ basis
        cut = np.asarray(g['nominal_point_plant_m']) @ basis
        axes[i].plot(support[:, 0], support[:, 1], color='#ce078e', lw=2)
        axes[i].scatter(*center[:2], color='#ffb000', s=40, edgecolors='black', linewidths=.6)
        axes[i].scatter(*cut[:2], marker='+', color='white', s=100, linewidths=2)
        x, y = limits[0]+.005, limits[2]+.005
        axes[i].plot([x, x+.01], [y, y], color='black', lw=2)
        axes[i].text(x, y+.002, '10 mm', fontsize=9)
        axes[i].set_title(titles[i], fontsize=10)
    fig.suptitle(target+': freshly recomputed metric 9 mm point — same orientation / scale', fontsize=14)
    fig.text(.5, .045, 'Orange: attachment. White +: 9 mm centerline point. Magenta: 9–19 mm visibility '
             'support, NOT a permissible cut interval.', ha='center', fontsize=9)
    fig.text(.5, .012, 'Markers overlay all geometry and do NOT establish surface ownership, visibility, '
             'first-leaf clarity, physical radius or cutting eligibility.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.89, bottom=.105, left=.02, right=.98, wspace=.06)
    close = output/'junction_9mm_comparison.png'
    fig.savefig(close, dpi=160, facecolor='white'); plt.close(fig)
    # Recheck the complete distinct input union after assembly/rendering.
    for path, expected in bindings.items():
        require(sha256(path) == expected, 'Input changed during preview: '+path)
    result = dict(schema=SCHEMA, state='actual_mesh_preview_not_native_or_training',
        source_family=receipts[0]['source_family'], source_split=receipts[0]['source_split'],
        generated_variant_ids=[r['variant_id'] for r in receipts], selected_target=target,
        component_counts=[len(r['components']) for r in reports],
        triangle_counts=[sum(len(t) for t in mesh.values()) for mesh in meshes],
        recomputed_centerline_geometry=geometry, source_bindings=bindings,
        preview_files={p.name: sha256(p) for p in (whole, close)},
        implementation_hashes={str(Path(__file__).with_name(name).resolve()):
                              sha256(Path(__file__).with_name(name)) for name in
                              ('plant_morphology_preview_v1.py', 'plant_variant_preview.py',
                               'geometry.py', 'audit.py', 'cut_regions.py')},
        camera_type='shared_orthographic_not_robot_camera',
        rendering='actual_authored_triangle_painter_approximate_occlusion_flat_organ_colors',
        texture_rendering=False, native_depth_created=False, actual_visual_review=False,
        eligibility_verified=False, accepted_training_increment=0, training_approved=False)
    write_json(output/'preview.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--generated-manifests', nargs=2, type=Path, required=True)
    p.add_argument('--generated-manifest-sha256', nargs=2, required=True)
    p.add_argument('--qualification-sha256', nargs=2, required=True)
    p.add_argument('--target', default='SubStem_42')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = preview(**vars(a))
    print('MORPHOLOGY_MESH_PREVIEW', a.output, result['triangle_counts'], flush=True)


if __name__ == '__main__':
    main()
