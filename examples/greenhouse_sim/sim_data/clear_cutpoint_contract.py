"""Visible-only experiment; stricter sampling, not a rewrite of task-v3 labels."""
import hashlib
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from .dataset_review import require

PROFILE = 'clear_cutpoint_v1'
SCHEMA = 'greenhouse.clear_cutpoint_release.v1'
POLICY = dict(minimum_interval_px=12., minimum_width_proxy_px=8.,
              minimum_local_median_luma=40., maximum_local_dark_fraction=.10,
              minimum_proximal_visible_fraction=.95, views_per_target=12,
              crop_source_size=384, crop_output_size=768, selection_seed=41)
# Small focused experiment, not the original 10,000-row mixed-task release.
GATES = dict(minimum_rows=dict(train=500, validation=100, test=100),
             minimum_targets=dict(train=60, validation=15, test=15),
             minimum_families=dict(train=16, validation=4, test=4))


def crop_box(query):
    q = np.asarray(query, float)
    require(q.shape == (2,) and np.isfinite(q).all() and
            ((q >= 0) & (q < [848, 408])).all(), 'Invalid query')
    size = POLICY['crop_source_size']
    x, y = np.clip(np.floor(q-size/2), 0, [848-size, 408-size]).astype(int)
    return [int(x), int(y), int(x+size), int(y+size)]


def crop_image(rgb, query):
    require(rgb.mode == 'RGB' and rgb.size == (848, 408), 'Original RGB required')
    return rgb.crop(crop_box(query)).resize((768, 768), Image.Resampling.BICUBIC)


def screen(label, rgb, mask):
    """Mask radius is a legibility PROXY, never a physical diameter estimate.

    No RGB modifications. Labels/depth certify synthetic visibility only;
    visual review remains necessary for anatomical clarity.
    """
    require(rgb.shape == (408, 848, 3) and mask.shape == (408, 848), 'Invalid image shape')
    line = np.asarray(label['accepted_interval_uv'], float)
    require(line.ndim == 2 and line.shape[1] == 2 and len(line) >= 2 and
            np.isfinite(line).all(), 'Invalid interval')
    valid = ((line >= 0) & (line < [848, 408])).all(axis=1)
    require(valid.all(), 'Interval out of frame')
    xy = np.floor(line).astype(int); x, y = xy.T
    length = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
    widths = 2*distance_transform_edt(mask)[y, x]
    # Only actual target pixels near the proximal interval enter exposure checks.
    x0,y0 = np.maximum(xy.min(axis=0)-8, 0)
    x1,y1 = np.minimum(xy.max(axis=0)+9, [848,408])
    pixels = rgb[y0:y1,x0:x1][mask[y0:y1,x0:x1]].astype(float)
    luma = pixels @ np.asarray([.2126,.7152,.0722]) if len(pixels) else np.asarray([0.])
    metrics = dict(interval_length_px=length, width_proxy_px=float(np.median(widths)),
                   local_median_luma=float(np.median(luma)),
                   local_dark_fraction=float(np.mean(luma < 30)),
                   all_interval_samples_on_target=bool(mask[y,x].all()))
    reasons=[]
    if label['answer']['status'] != 'localized' or label['answer']['visibility'] != 'clear': reasons.append('not_clear')
    if not label.get('query_cut_visible_connection_verified'): reasons.append('unverified_query_connection')
    if label['proximal_visible_pixel_fraction'] < .95: reasons.append('proximal_visibility')
    if not metrics['all_interval_samples_on_target']: reasons.append('interval_not_fully_visible')
    for metric, bound in [('interval_length_px','minimum_interval_px'),
                          ('width_proxy_px','minimum_width_proxy_px'),
                          ('local_median_luma','minimum_local_median_luma')]:
        if metrics[metric] < POLICY[bound]: reasons.append(metric)
    if metrics['local_dark_fraction'] > POLICY['maximum_local_dark_fraction']: reasons.append('local_dark_fraction')
    return dict(passed=not reasons,reasons=reasons,**metrics)


def select_views(rows):
    """Deterministic farthest-view sampling per target; never use cut coordinates."""
    groups={}
    for row in rows: groups.setdefault((row['source_plant_family'],row['target_id']),[]).append(row)
    result=[]
    for key, group in sorted(groups.items()):
        group=sorted(group,key=lambda r:hashlib.sha256(f"41:{r['id']}".encode()).hexdigest())
        features=np.asarray([r['selection_features'] for r in group],float)
        scale=np.ptp(features,axis=0); scale[scale < 1e-9]=1.
        features=(features-features.mean(axis=0))/scale
        chosen=[0]
        while len(chosen) < min(len(group),POLICY['views_per_target']):
            distances=((features[:,None]-features[chosen])**2).sum(axis=2).min(axis=1)
            distances[chosen]=-1
            chosen.append(int(np.argmax(distances)))
        result.extend(group[i] for i in chosen)
    return sorted(result,key=lambda r:r['id'])
