"""Complete saved Q4 proof checks and predeclared visual-sampling heuristics.

No annotation changes, capture, visual-review claim, or dataset admission.
"""
import math

SCHEMA = 'greenhouse.native848_q4_group_margin.v1'
POLICY = dict(schema=SCHEMA, hard_centerline_clearance_px=4.0,
    hard_background_depth_gap_m=0.20, flag_centerline_clearance_below_px=4.25,
    flag_background_depth_gap_below_m=0.25,
    every_exception_probe_and_violating_pixel_required=True,
    endpoint_exceptions_prohibited=True,
    weakest_clearance_and_depth_representatives_required=True,
    sampling_threshold_status='predeclared_uncalibrated_engineering_review_heuristics',
    per_image_quality_thresholds_unchanged=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    require(not isinstance(value, bool), 'Boolean margin is invalid')
    x = float(value)
    require(math.isfinite(x), 'Missing or nonfinite Q4 margin')
    return x


def hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def summarize(proof):
    """Fail closed; summarize every saved exception probe, never a truncated list."""
    require(proof['schema'] == 'greenhouse.native848_conditional_background_separation.v1'
        and all(proof[k] is True for k in ('passed', 'protected_foreground_passed',
            'conditional_background_passed', 'Q1passed', 'background_exception_applied',
            'all_route_probes_recorded', 'all_violating_pixels_at_each_probe_checked'))
        and proof['q3_passed'] is False, 'Q4 protected proof incomplete')
    probes = proof['probes']
    require(isinstance(probes, list) and probes and type(proof['probe_count']) is int
        and len(probes) == proof['probe_count'], 'Incomplete route probe population')
    require(all(hash_value(proof[k]) for k in ('route_arcs_sha256', 'route_projected_pixels_sha256')),
        'Missing complete route hashes')
    endpoint = proof['endpoint']
    require(endpoint['background_exception_applied'] is False and endpoint['background_pixels'] is None
        and endpoint['protected_foreground_passed'] is True
        and endpoint['conditional_background_passed'] is True
        and endpoint['q3_probe']['passed'] is True and endpoint['q3_probe']['reasons'] == [],
        'Endpoint exception is prohibited')
    distances = []; gaps = []; pixels = 0
    for p in probes:
        require(p['protected_foreground_passed'] is True and p['conditional_background_passed'] is True,
            'A route probe failed protected checks')
        original = p['q3_probe']
        require(original['unknown_surface_pixels'] == 0, 'Unknown route pixels remain')
        if p['background_exception_applied'] is True:
            require(original['passed'] is False
                and original['reasons'] == ['visible_foreign_plant_too_close_to_route'],
                'Exception includes a protected foreground or other failure')
            b = p['background_pixels']
            require(b['passed'] is True and b['all_violating_pixels_checked'] is True
                and type(b['violating_pixel_count']) is int and b['violating_pixel_count'] > 0
                and hash_value(b['pixel_distance_and_gap_sha256']), 'Incomplete violating-pixel evidence')
            d = number(b['minimum_centerline_distance_px']); gap = number(b['minimum_depth_gap_m'])
            require(d >= POLICY['hard_centerline_clearance_px']
                and gap >= POLICY['hard_background_depth_gap_m'], 'Existing Q4 acceptance gate failed')
            distances.append(d); gaps.append(gap); pixels += b['violating_pixel_count']
        else:
            require(p['background_exception_applied'] is False and p['background_pixels'] is None
                and original['passed'] is True and original['reasons'] == [], 'Unaccounted route exception')
    require(distances and type(proof['exception_probe_count']) is int
        and len(distances) == proof['exception_probe_count'], 'Exception probe count differs')
    distance = min(distances); gap = min(gaps); flags = []
    if distance < POLICY['flag_centerline_clearance_below_px']:
        flags.append('Q4_near_background_centerline_boundary')
    if gap < POLICY['flag_background_depth_gap_below_m']:
        flags.append('Q4_near_background_depth_boundary')
    return dict(schema=SCHEMA, minimum_centerline_distance_px=distance, minimum_depth_gap_m=gap,
        exception_probe_count=len(distances), checked_route_probe_count=len(probes),
        violating_pixel_observations=pixels, all_exception_probes_checked=True,
        flags=flags, flagged=bool(flags), policy=POLICY)
