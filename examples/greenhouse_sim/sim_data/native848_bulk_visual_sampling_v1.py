"""Pure selection for the presealed native848 sampled visual-QA protocol.

No files, renderer, image review, admission, or quality claims are produced here.
`rows` are one chunk's automated-pass rows. `all_capture_rows` is that sealed
chunk's actual capture inventory, including automated exclusions. A reset segment
is a run/scene reinitialization, not each frame's accumulation reset.
"""
import hashlib
import json
import math

PROTOCOL_SHA256 = '35c817385ab5c2fa37a0b58a70703a9f85f0190bfd6d67d265373a6114aaa4d4'
SALT = '0eff5f4b4eb2240b71009224e26d7524503167c55d6568f134a5ce3d304fbca6'
SCHEMA = 'greenhouse.native848_bulk_visual_sample_request.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'Missing/invalid finite metric: ' + name)
    return float(value)


def identity(row):
    value = row['observation_id']
    require(isinstance(value, str) and value and '\n' not in value, 'Unique nonempty observation ID required')
    return value


def order(row):
    value = row['capture_index']
    require(type(value) is int and value >= 0, 'Nonnegative integer capture_index required')
    return value, identity(row)


def segment(row):
    keys = ('source_family', 'profile', 'reset_segment_id')
    require(all(isinstance(row[k], str) and row[k] for k in keys), 'Explicit family/profile/reset segment required')
    return tuple(row[k] for k in keys)


def angles(row):
    head = row['head_joint_degrees']
    require(set(head) == {'head_0', 'head_1'}, 'Exactly the two actual head joints required')
    return tuple(finite(head[k], k) for k in ('head_0', 'head_1'))


def metrics(row):
    clarity, query = row['label_metrics'], row['query_metrics']
    values = (finite(clarity['width_proxy_px'], 'label.clarity.width_proxy_px'),
              finite(clarity['interval_length_px'], 'label.clarity.interval_length_px'),
              finite(query['local_contrast_8bit'], 'label.query_usability.local_contrast_8bit'),
              finite(query['interior_radius_px'], 'label.query_usability.interior_radius_px'),
              finite(query['frame_margin_px'], 'label.query_usability.frame_margin_px'))
    require(values[0] > 0 and values[1] > 0 and all(v >= 0 for v in values[2:]), 'Invalid sampling metric range')
    return values


def sample_bulk_chunk(rows, plan_hash, chunk_id, startup, all_capture_rows):
    """Return deterministic observation IDs/reasons, never visual decisions.

All row fields are JSON data. Eligibility is `automated_pass is True` and its
unaccepted state must be `automated_candidate_pending_batch_QA`. The caller binds
the returned request to the sealed chunk inventory and actual image/label hashes.
Missing metric fields fail closed; rows are never omitted to make QA pass.
"""
    require(isinstance(plan_hash, str) and len(plan_hash) == 64
            and all(c in '0123456789abcdef' for c in plan_hash), 'Canonical plan SHA256 required')
    require(isinstance(chunk_id, str) and chunk_id and '\n' not in chunk_id, 'Explicit chunk ID required')
    require(type(startup) is bool, 'Explicit startup flag required')
    require(isinstance(rows, (list, tuple)) and isinstance(all_capture_rows, (list, tuple)), 'Row sequences required')
    require(len(rows) <= 256, 'Chunk exceeds256 eligible frames')
    eligible = sorted(rows, key=order)
    captures = sorted(all_capture_rows, key=order)
    for inventory in (eligible, captures):
        require(len({identity(r) for r in inventory}) == len(inventory), 'Duplicate observation ID')
        require(len({r['capture_index'] for r in inventory}) == len(inventory), 'Duplicate capture_index')
    by_capture = {identity(r): r for r in captures}
    eligible_ids = {identity(r) for r in eligible}
    require(eligible_ids <= set(by_capture), 'Eligible frame missing from sealed capture inventory')
    groups = {segment(r) for r in captures}
    require(len(groups) <= 1, 'Split family/profile/reset groups into separate chunks')
    actual_angles = {identity(r): angles(r) for r in captures}
    metric_map = {}
    for row in eligible:
        name = identity(row); observed = by_capture[name]
        require(row['automated_pass'] is True and row['decision'] == 'automated_candidate_pending_batch_QA',
                'Only pending automated-pass candidates may enter eligible sample population')
        require(order(row) == order(observed) and segment(row) == segment(observed)
                and angles(row) == actual_angles[name], 'Eligible metadata differs from capture inventory')
        metric_map[name] = metrics(row)
    selected = {}
    def add(row, reason):
        name = identity(row)
        selected.setdefault(name, [])
        if reason not in selected[name]:
            selected[name].append(reason)
    if len(eligible) <= 8:
        for row in eligible:
            add(row, 'all_eligible_small_chunk')
    else:
        add(eligible[0], 'first_eligible_capture')
        add(eligible[-1], 'last_eligible_capture')
        def random_key(row):
            raw = '\n'.join((SALT, plan_hash, chunk_id, identity(row))).encode('utf-8')
            return hashlib.sha256(raw).hexdigest(), identity(row)
        remaining = [r for r in eligible if identity(r) not in selected]
        for row in sorted(remaining, key=random_key)[:2]:
            add(row, 'presealed_salted_SHA256_random')
        edge_metrics = [('minimum_cut_width', lambda v: v[0]),
                        ('minimum_cut_interval_length', lambda v: v[1]),
                        ('minimum_query_local_contrast', lambda v: v[2]),
                        ('minimum_query_interior_radius_then_frame_margin', lambda v: (v[3], v[4]))]
        for reason, value in edge_metrics:
            remaining = [r for r in eligible if identity(r) not in selected]
            chosen = min(remaining, key=lambda r: (value(metric_map[identity(r)]), *order(r)))
            add(chosen, reason)
    regular_ids = set(selected)
    transition = None
    if startup and eligible:
        for row in eligible[:3]:
            add(row, 'startup_first_three_eligible')
        if captures:
            transitions = [(0.0, captures[0], None)]
            for before, after in zip(captures, captures[1:]):
                distance = math.dist(actual_angles[identity(before)], actual_angles[identity(after)])
                transitions.append((distance, after, before))
            magnitude, after, before = min(transitions, key=lambda entry: (-entry[0], *order(entry[1])))
            add(after, 'startup_largest_actual_head_transition')
            if before is not None:
                add(before, 'startup_largest_head_transition_predecessor')
            transition = dict(magnitude_degrees=magnitude, after=identity(after),
                before=identity(before) if before is not None else None,
                head_joint_names=['head_0', 'head_1'], after_degrees=list(actual_angles[identity(after)]),
                before_degrees=list(actual_angles[identity(before)]) if before is not None else None)
    limit = 13 if startup else 8
    require(len(selected) <= limit, 'Predeclared visual sample bound exceeded')
    result_rows = [dict(observation_id=identity(r), capture_index=r['capture_index'],
        selection_reasons=selected[identity(r)], eligible_automated_candidate=identity(r) in eligible_ids,
        automated_exclusion_remains_excluded=identity(r) not in eligible_ids,
        individual_visual_review=False) for r in captures if identity(r) in selected]
    encoded = json.dumps(dict(rows=rows, all_capture_rows=all_capture_rows), sort_keys=True,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return dict(schema=SCHEMA, protocol_sha256=PROTOCOL_SHA256, plan_sha256=plan_hash,
        chunk_id=chunk_id, startup=startup, source_family_profile_reset=list(next(iter(groups))) if groups else None,
        selection_input_sha256=hashlib.sha256(encoded).hexdigest(),
        eligible_population_count=len(eligible), actual_capture_population_count=len(captures),
        regular_sample_count=len(regular_ids), total_requested_visual_reviews=len(result_rows),
        sample_ids=[r['observation_id'] for r in result_rows], samples=result_rows,
        largest_head_transition=transition, images_viewed=False,
        chunk_accepted=False, training_approved=False)
