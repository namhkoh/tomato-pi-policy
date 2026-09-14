"""Explicit retention experiment policy; never a replacement for native guards.

A static six-axis load balance in a fixed patch is a conservative prerequisite,
not a physical proof that a flexible retained branch cannot rotate or droop.
The default remains static-screened. An opt-in native trial measures the actual
outcome with the unchanged force, slip, bilateral-grasp and cutting guards.
"""
import math


def evaluate(preflight,*,native_trial=False):
    if type(native_trial) is not bool:raise ValueError('Explicit retention policy required')
    if (preflight.get('model')!='native_patch_retention_preflight_v1'
            or preflight.get('native_sample_binding_checked') is not True
            or type(preflight.get('prerequisite_passed')) is not bool):
        raise ValueError('Bound current native retention preflight required')
    static=preflight['prerequisite_passed']
    result=dict(model='explicit_retention_assessment_policy_v1',
        mode='native_outcome_experiment' if native_trial else 'static_prerequisite',
        static_prerequisite_passed=static,static_failure_is_diagnostic_only=False,
        continue_to_independent_cut_planning=static,retention_verified=False,
        cut_authorized=False,contact_limits_changed=False,slip_limit_changed=False,
        per_step_native_guards_required=True,production_qualified=False)
    if not native_trial or static:return result
    capacity=preflight.get('capacity') or {}
    value=capacity.get('minimum_worst_finger_utilization')
    # This comparison may question only a VALID solved over-budget static
    # approximation. Missing, unbalanced or invalid contact evidence is still
    # a refusal; the caller's actual bilateral/dwell/slip binding is unchanged.
    if (preflight.get('rejection')!='static_retention_not_established_within_contact_budgets'
            or capacity.get('model')!='fixed_patch_static_wrench_audit_v1'
            or capacity.get('balance_found') is not True or capacity.get('solver_status')!=0
            or capacity.get('within_contact_budgets') is not False
            or capacity.get('contact_budgets_n')!=[.5,.5]
            or isinstance(value,bool) or not isinstance(value,(float,int))
            or not math.isfinite(value) or value<=1):
        raise ValueError('Native experiment requires a valid solved over-budget static patch')
    result.update(static_failure_is_diagnostic_only=True,
        continue_to_independent_cut_planning=True,
        warning='Static retention not established; dynamic retention is unknown until physically tested')
    return result
