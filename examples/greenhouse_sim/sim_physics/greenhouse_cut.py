"""Explicit intact-greenhouse integration of the guarded diagnostic controller.

No isolated success is promoted to greenhouse qualification. The same native
startup, scene, force, grasp and release checks run in the restored scene.
Nearby context plants have static collision, not whole-greenhouse deformation.
"""


def validate(args):
    if not getattr(args,'greenhouse_cut_trial',False):return False
    if not (args.scene=='package' and args.full_robot_probe and args.bimanual_cut
            and args.fixed_root_cut_trial and args.cut_action_trial
            and args.isolated_cut_contact_trial and args.seam_contact_yield
            and args.knife_edge_mode=='source_crossbar_edge_v1'
            and args.cut_model=='loaded_downward_lower_rim_seam_v1'
            and args.native_startup_clearance and args.native_static_clearance
            and args.native_capsule_sphere_cover and args.native_drives_after_cut
            and args.local_wire_physics and args.context_gutters==3
            and args.physics_window_half_m==2 and args.batch_gutter_visuals
            and args.stream_trajectory and not args.isolate_station
            and not args.branch_contact_fixture and not args.robot_interactive
            and not args.bimanual_hold_control and not args.measured_withdrawal):
        raise ValueError('Greenhouse cut requires explicit intact three-gutter preview layout and complete guarded process-zone profile')
    return True


def intact_plant(stage,record):
    """Read-only inventory before replacing just the selected target's physics."""
    paths=record['component_paths']
    if not paths or any(not stage.GetPrimAtPath(p) or not stage.GetPrimAtPath(p).IsActive()
                        for p in paths.values()):
        raise RuntimeError('Full greenhouse trial cannot omit source plant components')
    for path in ('/World/Gutters','/World/GutterWires'):
        if not stage.GetPrimAtPath(path) or not stage.GetPrimAtPath(path).IsActive():
            raise RuntimeError('Original greenhouse infrastructure missing')
    return dict(scope='intact_source_plant_in_original_greenhouse',
        intact_component_count=len(paths),all_original_components_active=True,
        source_geometry_deleted=False,greenhouse_motion_qualified=False,
        context_compliance='static_contacts_near_robot_selected_petiole_dynamic',
        training_eligible=False)
