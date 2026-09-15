"""Explicit native render-budget experiment; never an implicit quality change.

The high-resolution warm56_then8 profile is a TRIAL until matched native
observations and visual/task review qualify it. Initial warmup is never skipped.
"""
REFERENCE='reference56'
TRIAL='warm56_then8_trial'


def requested_subframes(profile,captured_frames):
    if profile not in (REFERENCE,TRIAL):
        raise ValueError('Unknown explicit native render budget')
    if type(captured_frames) is not int or captured_frames<0:
        raise ValueError('Nonnegative captured-frame count required')
    return 56 if profile==REFERENCE or captured_frames==0 else 8


def evidence(profile,captured_frames):
    budget=requested_subframes(profile,captured_frames)
    return dict(profile=profile,requested_subframes=budget,
        native_step_calls=7 if budget==56 else 1,subframes_per_step=8,
        high_resolution_short_profile_qualified=False,
        experimental_short_profile=profile==TRIAL,
        same_scene_meshes_optics_lighting_and_renderer=True,
        photometric_equivalence_claimed=False,training_release_approved=False)

