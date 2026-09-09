"""Bounded native-buffer convergence for a frozen scene after a camera snapshot.

Short renders are accepted only after repeated geometry/identity agreement and
RGB convergence. Native callback/camera/content guards remain the caller's duty.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .capture_contract import validate_payload
from .capture_visibility import decode_instances

# Calibrated against seven SAME-POSE pairs using the original long warmup:
# global MAE 1.085-1.652 / p95 5-7, ROI MAE .481-2.104 / p95 2-8.
# A sub-noise-floor threshold never converges, even on the long reference.
RGB_LIMITS=dict(mean=2.,p95=9.,roi_mean=2.5,roi_p95=10.)
# Longer-gap samples have less correlated Monte Carlo noise than adjacent ones.
# Keep the critical target-region limits unchanged; allow <=1.18% mean full-image
# difference. These are declared image-quality limits, NOT pixel identity.
LONG_RGB_LIMITS=dict(mean=3.,p95=14.,roi_mean=2.5,roi_p95=10.)


def same_visible_identities(ids_a,mapping_a,ids_b,mapping_b):
    """Renderer-local IDs may renumber; compare the prim identity at every pixel.

    Unobserved entries in a native mapping table are not image geometry.
    """
    if ids_a.shape!=ids_b.shape: return False
    pairs=np.unique((ids_a.astype(np.uint64)<<32)|ids_b.astype(np.uint64))
    for pair in pairs:
        a,b=int(pair>>np.uint64(32)),int(pair&np.uint64(0xffffffff))
        if mapping_a.get(a,('unmapped_sentinel',a))!=mapping_b.get(b,('unmapped_sentinel',b)):
            return False
    return True


def convergence(previous,current,calibration,*,roi_pixel=None,limits=None):
    limits=RGB_LIMITS if limits is None else limits
    a,za,va,_=validate_payload(previous,calibration)
    b,zb,vb,_=validate_payload(current,calibration)
    ia,ma=decode_instances(previous); ib,mb=decode_instances(current)
    validity_equal=bool(np.array_equal(va,vb))
    depth_equal=bool(validity_equal and np.allclose(za[va],zb[vb],atol=.0002,rtol=0))
    identity_equal=same_visible_identities(ia,ma,ib,mb)
    geometry=depth_equal and identity_equal
    delta=np.abs(a.astype(np.float32)-b.astype(np.float32))
    mean=float(delta.mean()); p95=float(np.percentile(delta,95))
    roi_mean=mean; roi_p95=p95
    if roi_pixel is not None:
        x,y=np.floor(roi_pixel).astype(int)
        crop=delta[max(0,y-32):min(408,y+33),max(0,x-32):min(848,x+33)]
        if crop.size: roi_mean=float(crop.mean()); roi_p95=float(np.percentile(crop,95))
    return dict(passed=geometry and mean<=limits['mean'] and p95<=limits['p95']
                and roi_mean<=limits['roi_mean'] and roi_p95<=limits['roi_p95'],
                geometry_and_identity_equal=geometry,depth_validity_equal=validity_equal,
                native_depth_equal=depth_equal,visible_prim_identity_equal=identity_equal,
                renderer_local_ID_buffers_equal=bool(np.array_equal(ia,ib)),
                identity_comparison='resolved_prim_path_at_every_pixel_not_unused_mapping_entries',
                rgb_mean_absolute_delta=mean,rgb_p95_absolute_delta=p95,
                target_roi_mean_absolute_delta=roi_mean,target_roi_p95_absolute_delta=roi_p95,
                native_Z_tolerance_m=.0002,rgb_limits_0_255=limits)


def reference_payload(rep,writer,calibration):
    """Established six warmups plus one final eight-subframe native capture.

    This is the production default until a shorter profile is qualified.
    A fixed render budget is not a claim of pixel-identical convergence.
    """
    from .capture_pilot import step_payload
    for _ in range(7):
        current=step_payload(rep,writer,subframes=8)
        validate_payload(current,calibration)
    return current,dict(method='established_fixed_56_subframe_reference.v1',
                        render_steps=7,rt_subframes_per_step=8,short_profile_qualified=False)


def settled_payload(rep,writer,calibration,*,roi_pixel=None,max_steps=16,verify_long_reference=False):
    from .capture_pilot import step_payload
    previous=None; consecutive=0; history=[]; recent=deque(maxlen=5)
    for i in range(max_steps):
        current=step_payload(rep,writer,subframes=4)
        # Validate the actual rendered camera on EVERY candidate, not only at the end.
        validate_payload(current,calibration)
        if previous is not None:
            evidence=convergence(previous,current,calibration,roi_pixel=roi_pixel)
            if len(recent)>=2:
                evidence['eight_subframe_drift']=convergence(recent[-2],current,calibration,roi_pixel=roi_pixel,limits=LONG_RGB_LIMITS)
                evidence['passed']=evidence['passed'] and evidence['eight_subframe_drift']['passed']
            else:
                evidence['passed']=False
            history.append(evidence)
            consecutive=consecutive+1 if evidence['passed'] else 0
            if i>=3 and consecutive>=2:
                result=dict(method='repeated_native_geometry_identity_and_RGB_convergence.v1',
                    render_steps=i+1,rt_subframes_per_step=4,consecutive_passes=consecutive,comparisons=history)
                if verify_long_reference:
                    reference=current
                    for _ in range(7): reference=step_payload(rep,writer,subframes=8)
                    comparison=convergence(current,reference,calibration,roi_pixel=roi_pixel,limits=LONG_RGB_LIMITS)
                    if not comparison['passed']:
                        raise ValueError('Short render differs from long reference: '+repr(comparison))
                    result['long_reference_comparison']=comparison
                    result['additional_reference_steps']=7
                    result['additional_reference_subframes_per_step']=8
                    result['_short_rgb']=np.asarray(current['rgb'])[:,:,:3].copy()
                    # In comparison mode save the final fresh reference callback;
                    # never attach a later callback counter to an earlier payload.
                    current=reference
                return current,result
        previous=current
        recent.append(current)
    raise ValueError('Native buffers did not settle within bounded render budget: '+repr(history[-2:]))


def consolidated_payload(rep,writer,calibration,*,roi_pixel=None,compare=False,subframes=56):
    """Same requested 56 subframes in one native step; opt-in qualification only."""
    from .capture_pilot import step_payload
    import time
    if type(subframes) is not int or subframes not in (8,16,32,56):
        raise ValueError('Unsupported bounded native subframe count')
    started=time.perf_counter()
    current=step_payload(rep,writer,subframes=subframes)
    validate_payload(current,calibration)
    result=dict(method=f'single_native_{subframes}_subframe_request.v1',render_steps=1,
                rt_subframes_per_step=subframes,consolidated_capture_seconds=time.perf_counter()-started)
    if compare:
        reference,reference_settings=reference_payload(rep,writer,calibration)
        evidence=convergence(current,reference,calibration,roi_pixel=roi_pixel,limits=LONG_RGB_LIMITS)
        if not evidence['passed']:
            raise ValueError('Consolidated render differs from established reference: '+repr(evidence))
        result.update(long_reference_comparison=evidence,reference_settings=reference_settings,
                      _short_rgb=np.asarray(current['rgb'])[:,:,:3].copy())
        current=reference
    return current,result


def noise_probe_payload(rep,writer,calibration,*,roi_pixel=None):
    """Characterize repeated-reference noise; never approve the experimental RGB.

    The saved observation is the LAST established-budget reference. Candidate
    and first-reference RGB are review-only evidence, including failed metrics.
    This distinguishes real renderer variation from a shortcut's quality loss.
    """
    candidate,settings=consolidated_payload(rep,writer,calibration,roi_pixel=roi_pixel)
    reference,_=reference_payload(rep,writer,calibration)
    repeated,_=reference_payload(rep,writer,calibration)
    comparisons={name:convergence(a,b,calibration,roi_pixel=roi_pixel,limits=LONG_RGB_LIMITS)
                 for name,a,b in [('candidate_reference',candidate,reference),
                                  ('reference_repeat',reference,repeated),
                                  ('candidate_repeat',candidate,repeated)]}
    if not all(c['geometry_and_identity_equal'] for c in comparisons.values()):
        raise ValueError('Geometry/identity changed during fixed-scene noise probe: '+repr(comparisons))
    return repeated,dict(method='repeated_reference_noise_characterization.v1',
        experimental_candidate_approved=False,returned_observation='last_established_56_subframe_reference',
        candidate_settings=settings,comparisons=comparisons,
        _short_rgb=np.asarray(candidate['rgb'])[:,:,:3].copy(),
        _reference_rgb=np.asarray(reference['rgb'])[:,:,:3].copy())
