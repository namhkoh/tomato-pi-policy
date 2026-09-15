"""Resolution-aware render-budget experiment; final observation stays reference.

No production budget relaxation. Native optical-Z, visible prim identities and
full/target-region RGB are compared. Camera/static guards remain caller's duty.
"""
import time
import numpy as np
from .native_sensor_payload import validate_native_payload, decode_native_instances
from .static_render import same_visible_identities, LONG_RGB_LIMITS


def identity_differences(a, ma, b, mb, roi):
    pairs, inverse, counts = np.unique((a.astype(np.uint64)<<32)|b.astype(np.uint64),
                                      return_inverse=True, return_counts=True)
    different = np.zeros(len(pairs),bool)
    examples=[]
    for i,pair in enumerate(pairs):
        left,right=int(pair>>np.uint64(32)),int(pair&np.uint64(0xffffffff))
        pa,pb=ma.get(left,('unmapped',left)),mb.get(right,('unmapped',right))
        if pa!=pb:
            different[i]=True
            examples.append(dict(previous=pa,current=pb,pixels=int(counts[i])))
    mask=different[inverse].reshape(a.shape)
    return dict(changed_pixels=int(mask.sum()),changed_fraction=float(mask.mean()),
        target_roi_changed_pixels=int(mask[roi].sum()),
        largest_changed_prim_pairs=sorted(examples,key=lambda r:-r['pixels'])[:8])


def comparison(previous, current, calibration, *, roi_pixel):
    a, za, va, _ = validate_native_payload(previous, calibration)
    b, zb, vb, _ = validate_native_payload(current, calibration)
    ia, ma = decode_native_instances(previous, calibration['resolution'])
    ib, mb = decode_native_instances(current, calibration['resolution'])
    width, height = calibration['resolution']
    xy = np.asarray(roi_pixel, float)
    if xy.shape != (2,) or not np.isfinite(xy).all() or not (0 <= xy[0] < width and 0 <= xy[1] < height):
        raise ValueError('In-frame target ROI required')
    validity = bool(np.array_equal(va, vb))
    depth = bool(validity and np.allclose(za[va], zb[vb], atol=.0002, rtol=0))
    identities = same_visible_identities(ia, ma, ib, mb)
    delta = np.abs(a.astype(np.float32) - b.astype(np.float32))
    x, y = np.floor(xy).astype(int)
    radius = round(32 * width / 848)
    roi=(slice(max(0,y-radius),min(height,y+radius+1)),
         slice(max(0,x-radius),min(width,x+radius+1)))
    crop = delta[roi]
    metrics = dict(mean=float(delta.mean()), p95=float(np.percentile(delta,95)),
                   roi_mean=float(crop.mean()), roi_p95=float(np.percentile(crop,95)))
    rgb = all(metrics[k] <= LONG_RGB_LIMITS[k] for k in metrics)
    return dict(passed=bool(depth and identities and rgb),
        geometry_and_identity_equal=bool(depth and identities),
        depth_validity_equal=validity, native_depth_equal=depth,
        visible_prim_identity_equal=identities, rgb=metrics,
        rgb_limits_0_255=dict(LONG_RGB_LIMITS), native_Z_tolerance_m=.0002,
        target_roi_radius_px=radius, resolution=[width,height],
        identity_differences=identity_differences(ia,ma,ib,mb,roi),
        thresholds_status='predeclared_legacy_tolerances_not_native_recalibrated')


def noise_probe(rep, writer, calibration, *, roi_pixel):
    """Compare one56 request with7x8 and a second7x8; return only LAST reference."""
    from .capture_pilot import step_payload
    times = {}; requests = {}
    def render(name, steps):
        started = time.perf_counter()
        request_before=getattr(writer,'request_index',None)
        sequence_before=getattr(writer,'sequence',None)
        for subframes in steps:
            payload = step_payload(rep, writer, subframes=subframes)
            validate_native_payload(payload, calibration)
        times[name] = time.perf_counter() - started
        request_after=getattr(writer,'request_index',None)
        requests[name]=dict(helper_calls=len(steps),requested_subframes_per_call=steps[0],
            native_orchestrator_requests=None if request_before is None else request_after-request_before,
            callback_sequence_before=sequence_before,callback_sequence_after=getattr(writer,'sequence',None))
        print('NATIVE_RENDER_PROBE_BLOCK',name,times[name],requests[name],flush=True)
        return payload
    candidate = render('consolidated56_seconds', [56])
    reference = render('reference7x8_seconds', [8]*7)
    repeated = render('repeat7x8_seconds', [8]*7)
    comparisons = {name: comparison(a,b,calibration,roi_pixel=roi_pixel)
        for name,a,b in [('candidate_reference',candidate,reference),
                         ('reference_repeat',reference,repeated),
                         ('candidate_repeat',candidate,repeated)]}
    # Failed experiments must retain measurements and images too. This function
    # never qualifies a faster production profile, even when every metric passes.
    return repeated, dict(method='native_resolution_repeated_reference_probe.v1',
        experimental_candidate_approved=False, training_diversity_increment=0,
        returned_observation='last_established_56_subframe_reference',
        requested_subframes_total=168, requested_callbacks=15,
        actual_request_evidence=requests,
        all_comparisons_passed=all(c['passed'] for c in comparisons.values()),
        final_observation_reference_budget=56, timings=times, comparisons=comparisons,
        _candidate_rgb=np.asarray(candidate['rgb'])[:,:,:3].copy(),
        _first_reference_rgb=np.asarray(reference['rgb'])[:,:,:3].copy())
