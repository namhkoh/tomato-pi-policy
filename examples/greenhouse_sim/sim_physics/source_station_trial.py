"""Explicit existing-source trial selection, not a target execution certificate."""
import re


def configure(options,source):
    if (not isinstance(source,str)
            or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*/SubStem_[0-9]+',source) is None):
        raise ValueError('Existing plant/SubStem_N source identity required')
    plant,target=source.split('/')
    # These poses belong to seed101/SubStem_41; never carry them to a new
    # anatomy while pretending they are a general robot placement policy.
    result=list(options)
    for flag,count in (('--station-pose',3),('--left-ik-seed-degrees',7),('--right-ready-degrees',7)):
        while flag in result:
            index=result.index(flag)
            if index+count>=len(result):raise ValueError('Incomplete initial-pose recipe')
            del result[index:index+count+1]
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES
    ready=[SDK_READY_POSE_DEGREES[f'right_arm_{i}'] for i in range(7)]
    return result+['--plant',plant,'--target',target,'--right-ready-degrees',*map(str,ready)]
