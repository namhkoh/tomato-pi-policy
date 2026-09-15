"""Read-only CPU mesh preview of an original donor and generated static plant.

Actual authored triangles, shared orthographic views and diagnostic organ colors.
No Isaac capture, textures, depth output, training approval or source edits.
"""
import argparse
from pathlib import Path

import numpy as np

from .audit import audit_manifest
from .dataset_review import read_json, write_json, require
from .depth_preview import sha256
from .plant_variant_catalogue import load_for_inspection, assemble_for_inspection


def camera_basis(azimuth_degrees, elevation_degrees):
    a, e = np.radians([azimuth_degrees, elevation_degrees])
    toward_viewer = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.])
    up = np.cross(toward_viewer, right)
    return np.column_stack([right, up, toward_viewer])


def triangles_by_component(stage, paths):
    from pxr import UsdGeom
    owners = {path: key for key, path in paths.items()}
    cache = UsdGeom.XformCache()
    result = {key: [] for key in paths}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        owner = prim
        while owner and str(owner.GetPath()) not in owners:
            owner = owner.GetParent()
        if not owner:
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        matrix = np.asarray(cache.GetLocalToWorldTransform(prim))
        world = points @ matrix[:3, :3] + matrix[3, :3]
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        require(sum(counts) == len(indices), "Invalid mesh topology")
        start, triangles = 0, []
        for count in counts:
            face = indices[start:start+count]
            require(count >= 3, "Invalid polygon")
            triangles.extend([[face[0], face[i], face[i+1]] for i in range(1, count-1)])
            start += count
        if triangles:
            result[owners[str(owner.GetPath())]].append(world[np.asarray(triangles)])
    return {key: np.concatenate(parts) if parts else np.empty((0, 3, 3))
            for key, parts in result.items()}


def draw_mesh(ax, meshes, components, changed_members, basis, limits):
    from matplotlib.collections import PolyCollection
    faces, colors = [], []
    light = np.array([.3, -.5, 1.])
    light /= np.linalg.norm(light)
    palette = dict(leaf=(.23, .47, .20), sub_stem=(.28, .43, .17),
                   main_stem=(.31, .40, .15), fruit=(.78, .25, .10),
                   flower=(.9, .65, .08))
    for key, triangles in meshes.items():
        if not len(triangles):
            continue
        projected = triangles @ basis
        # Clip only whole faces outside the plot; never simplify or invent mesh.
        selected = ((projected[:, :, 0].max(axis=1) >= limits[0])
                    & (projected[:, :, 0].min(axis=1) <= limits[1])
                    & (projected[:, :, 1].max(axis=1) >= limits[2])
                    & (projected[:, :, 1].min(axis=1) <= limits[3]))
        projected, triangles = projected[selected], triangles[selected]
        if not len(triangles):
            continue
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-15)
        brightness = .5 + .5*np.abs(normals @ light)
        color = (.10, .55, .77) if key in changed_members else palette.get(components[key]["type"], (.4, .48, .2))
        faces.append(projected)
        colors.append(brightness[:, None]*np.asarray(color))
    faces, colors = np.concatenate(faces), np.concatenate(colors)
    order = np.argsort(faces[:, :, 2].mean(axis=1), kind="stable")
    ax.add_collection(PolyCollection(faces[order, :, :2], facecolors=colors[order],
                                    edgecolors="none", antialiaseds=False))
    ax.set(xlim=limits[:2], ylim=limits[2:], aspect="equal")
    ax.set_facecolor("#f1f4ed")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d4dcd0")


def preview(variant_directory, source_plan, output, target):
    from pxr import Usd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .geometry import assemble_plant
    from .cut_regions import propose_cut_region, load_rule

    catalogue = load_for_inspection(variant_directory, source_plan)
    receipt = read_json(Path(variant_directory)/"qualification.json")
    source = audit_manifest(receipt["source_manifest_path"])
    row = next((r for r in catalogue["rows"] if r["component_id"] == target), None)
    require(row is not None, "Target absent or withheld; cannot preview it as a qualified candidate")
    output = Path(output).resolve()
    require(not output.exists() and all(not output.is_relative_to(Path(p).resolve())
            for p in (variant_directory, Path(receipt["source_manifest_path"]).parent,
                      read_json(source_plan)["package"])), "New diagnostic output outside source assets required")
    stage = Usd.Stage.CreateInMemory()
    old_paths = assemble_plant(stage, "/Original", source)
    old_meshes = triangles_by_component(stage, old_paths)
    # Distinct stages prevent accidental ownership of the other plant's meshes.
    new_stage = Usd.Stage.CreateInMemory()
    assembled = assemble_for_inspection(new_stage, "/Generated", catalogue)
    new_meshes = triangles_by_component(new_stage, assembled["record"]["component_paths"])
    changed = {m for change in receipt["changes"] for m in change["members"]}
    selected = next(c for c in receipt["changes"] if c["component_id"] == target)
    old_component = source["components"][target]
    old_proposal = propose_cut_region(old_component, source["components"][old_component["parent"]], load_rule())
    proposals = [old_proposal, row["cut_region_proposal"]]
    anchor = np.asarray(row["attachment_plant_m"])
    full_basis = camera_basis(-50, 9)
    both_points = np.concatenate([v.reshape(-1, 3) for meshes in (old_meshes, new_meshes)
                                  for v in meshes.values() if len(v)])
    full_projected = both_points @ full_basis
    low, high = full_projected[:, :2].min(axis=0), full_projected[:, :2].max(axis=0)
    pad = max(high-low)*.04
    full_limits = (low[0]-pad, high[0]+pad, low[1]-pad, high[1]+pad)
    tangent = np.asarray(old_proposal["nominal"]["tangent_plant"])
    close_basis = camera_basis(np.degrees(np.arctan2(tangent[1], tangent[0]))-90, 5)
    center = anchor @ close_basis
    close_limits = (center[0]-.023, center[0]+.09, center[1]-.045, center[1]+.045)

    fig, axes = plt.subplots(2, 2, figsize=(13, 12), gridspec_kw={"height_ratios": [2.0, 1.]})
    fig.patch.set_facecolor("white")
    for col, (meshes, report, proposal, title) in enumerate(zip(
            (old_meshes, new_meshes), (source, catalogue["report"]), proposals,
            ("Original donor", "Generated variant"), strict=True)):
        draw_mesh(axes[0, col], meshes, report["components"], changed, full_basis, full_limits)
        axes[0, col].set_title(title, fontsize=17, fontweight="bold", pad=10)
        location = anchor @ full_basis
        axes[0, col].scatter(*location[:2], s=95, facecolors="none", edgecolors="#fc9c00", linewidths=2)
        axes[0, col].annotate(target, location[:2], xytext=(12, 0), textcoords="offset points",
                              fontsize=10, color="#222", bbox=dict(fc="white", ec="none", alpha=.85))
        draw_mesh(axes[1, col], meshes, report["components"], changed, close_basis, close_limits)
        interval = np.asarray([s["point_plant_m"] for s in proposal["accepted_centerline_interval"]["samples"]]) @ close_basis
        point = np.asarray(proposal["nominal"]["point_plant_m"]) @ close_basis
        axes[1, col].plot(interval[:, 0], interval[:, 1], color="#ed139c", lw=3, zorder=3)
        axes[1, col].scatter(*center[:2], color="#fca000", s=42, edgecolors="black", linewidths=.6, zorder=4)
        axes[1, col].scatter(*point[:2], marker="+", color="white", s=110, linewidths=2.5, zorder=5)
        axes[1, col].set_title(target + " junction - same view / scale", fontsize=12)
        x0, y0 = close_limits[0]+.008, close_limits[2]+.008
        axes[1, col].plot([x0, x0+.01], [y0, y0], color="#222", lw=3)
        axes[1, col].text(x0, y0+.003, "10 mm", fontsize=9)
    legends = [Line2D([], [], color="#337838", lw=6, label="Unchanged organs"),
               Line2D([], [], color="#198cc4", lw=6, label="Selected subtrees (both panels)"),
               Line2D([], [], color="#fca000", marker="o", lw=0, label="Attachment"),
               Line2D([], [], color="#ed139c", lw=3, label="10-20 mm cut interval")]
    fig.suptitle("Source-derived tomato plant generator", fontsize=21, fontweight="bold", y=.98)
    fig.legend(handles=legends, loc="upper center", bbox_to_anchor=(.5, .953), ncol=2, frameon=False)
    fig.text(.5, .049, f'{catalogue["source_family"]} | {len(source["components"])} components | '
             f'{len(receipt["changes"])} transformed petiole/leaf subtrees', ha="center", fontsize=11)
    fig.text(.5, .027, f'{target}: scale {selected["scale"]:.2f}, azimuth {selected["azimuth_deg"]:+.0f} deg, '
             f'tilt {selected["tilt_deg"]:+.0f} deg. White + = recomputed nominal 10 mm cut point.',
             ha="center", fontsize=10)
    fig.text(.5, .009, "ACTUAL USD MESH PREVIEW - diagnostic colors, no textures. "
             "Not an Isaac RGB/depth capture or physical/training approval.", ha="center", fontsize=9, color="#555")
    fig.subplots_adjust(top=.875, bottom=.08, left=.04, right=.96, wspace=.07, hspace=.13)
    output.mkdir(parents=True)
    image_path = output/"plant_comparison.png"
    fig.savefig(image_path, dpi=155, facecolor="white")
    plt.close(fig)
    result = dict(state="cpu_mesh_preview_not_native_or_training", source_family=catalogue["source_family"],
        generated_variant=catalogue["variant_id"], component_count=len(source["components"]),
        transformed_subtree_count=len(receipt["changes"]), selected_target=target, selected_change=selected,
        source_triangles=sum(len(v) for v in old_meshes.values()),
        generated_triangles=sum(len(v) for v in new_meshes.values()),
        qualification_sha256=sha256(Path(variant_directory)/"qualification.json"),
        preview_sha256=sha256(image_path), implementation_sha256=sha256(__file__),
        camera_type="shared_orthographic_diagnostic_not_robot_camera",
        rendering="CPU depth-sorted authored triangle polygons; approximate occlusion, flat organ colors",
        texture_rendering=False, native_depth_created=False, training_approved=False, physics_validated=False)
    write_json(output/"preview.json", result)
    print("GENERATOR_MESH_PREVIEW", image_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="SubStem_42")
    args = parser.parse_args()
    preview(args.variant, args.source_plan, args.output, args.target)


if __name__ == "__main__":
    main()
