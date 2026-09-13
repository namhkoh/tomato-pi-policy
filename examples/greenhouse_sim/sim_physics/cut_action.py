"""Limited grasp/cut action milestone; never hides full-task failure gates."""


def assess(*,cut_only,grasp_verified,planned,cut_event,cut_time,records,fault):
    last=records[-1] if records else {}
    evidence=(cut_event or {}).get('evidence',{})
    mode='right_only' if cut_only else 'bimanual'
    released=(bool(cut_event) and cut_event.get('event')=='blade_contact_joint_release'
        and evidence.get('cut_strategy','bimanual')==mode and cut_time is not None
        and last.get('cut') is True)
    gates=dict(no_execution_fault=fault is None,checked_right_plan=planned is True,
        blade_contact_release=bool(released),
        required_pre_cut_support=(not grasp_verified and evidence.get('stable_left_grasp') is False
            and evidence.get('grasp_slip_m') is None and evidence.get('cut_only_ready') is True)
            if cut_only else (grasp_verified is True and evidence.get('stable_left_grasp') is True),
        post_cut_observation=bool(released and last.get('t',0)>=cut_time+2),
        native_guards=bool(records) and all(r.get('native_guards_passed') is True for r in records))
    return dict(model='ground_truth_cut_action_milestone_v1',strategy=mode,
        passed=all(gates.values()),gates=gates,
        verified_withdrawal_required=False,retention_required=False,deposit_required=False,
        scope='approach_and_measured_cut_plus_two_seconds_observation_not_complete_robot_task',
        landing_safety_verified=False,tissue_fracture_calibrated=False,training_eligible=False)
