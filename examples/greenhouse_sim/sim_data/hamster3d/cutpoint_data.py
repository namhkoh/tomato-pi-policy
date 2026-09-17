"""Tomato cut-point task in 3D HAMSTER's prompt/answer format (RGB + metric depth in, point_3d out).

Bridge experiment (2026-09-16): reuse the released 3D HAMSTER checkpoint (Qwen3-VL-8B + frozen
LingBot-Depth encoder + geometry merger) on the task-v3 release. Coordinates are 0-1000 normalized
(x'=1000u/848, y'=1000v/408, 1 decimal); depth is metres (2 decimals) taken from the native optical-Z
map at the label pixel. Occluded targets answer with a null point and label "occluded".
"""
import json, re, sys
import numpy as np
from pathlib import Path
from PIL import Image
sys.path.insert(0, '/workspace/nhkoh/tomato-vlm/greenhouse_training_code/examples/greenhouse_sim')
from sim_data.training_export import read_jsonl            # noqa: E402
from sim_data.training_contract import validate_answer      # noqa: E402
from hamster3d.inference.preprocessing import resize_to_target  # noqa: E402

W, H = 848, 408
CONTRACT = 'hamster3d_cutpoint_point3d.v1'
CONTRACT_NO_QUERY = 'hamster3d_cutpoint_point3d_noquery.v1'
SUFFIX = 'Report the point_3d location in JSON.'
# Single-target variant (v2 data design, 2026-09-17): the image contains exactly one deleafable petiole,
# so no query pixel is given; the model must find the stalk, trace it to the main stem and point to the cut.
INSTRUCTION_NO_QUERY = ('Tomato plant seen from the robot head camera. Exactly one leaf-bearing petiole in this view is the deleafing target. '
                        'Find it, trace it to its junction with the main stem, and if the junction and the cut region are visible, point to the '
                        'nominal cut point 10 mm along the petiole from the junction, with its depth. If the cut region is hidden behind a leaf, '
                        'fruit or stem, report it as occluded instead of guessing.')
INSTRUCTION = ('Tomato plant seen from the robot head camera. The target petiole passes through the point ({x:.1f}, {y:.1f}); '
               'that is NOT the cut point. Trace the petiole to its junction with the main stem. If the junction and the cut '
               'region are visible, point to the nominal cut point 10 mm along the petiole from the junction, with its depth. '
               'If the cut region is hidden behind a leaf, fruit or stem, report it as occluded instead of guessing.')


def to_norm(u, v): return 1000. * u / W, 1000. * v / H
def to_px(x, y): return W * x / 1000., H * y / 1000.


def depth_at(depth, valid, u, v):
    x, y = int(min(max(np.floor(u), 0), W - 1)), int(min(max(np.floor(v), 0), H - 1))
    return float(depth[y, x]) if valid[y, x] else None


def load_frame(root, row):
    """Original RGB (PIL), native depth (float32 metres, invalid -> 0) and validity."""
    rgb = Image.open(Path(root) / row['files']['rgb']).convert('RGB')
    depth = np.load(Path(root) / row['files']['depth']).astype(np.float32)
    valid = np.asarray(Image.open(Path(root) / row['files']['validity'])) == 255
    return rgb, np.where(valid, depth, 0.).astype(np.float32), valid


def answer_text(row, depth, valid):
    a = row['answer']
    if a['status'] == 'localized':
        u, v = a['cut_point_uv']; d = depth_at(depth, valid, u, v)
        if d is None: raise ValueError('No valid native depth at the label pixel for ' + row['id'])
        x, y = to_norm(u, v)
        body = json.dumps([{'point_3d': [round(x, 1), round(y, 1), round(d, 2)], 'label': 'cut point'}])
    else:
        body = json.dumps([{'point_3d': None, 'label': 'occluded'}])
    return '```json\n' + body + '\n```'


def user_text(row, query=True):
    if not query: return INSTRUCTION_NO_QUERY + '\n' + SUFFIX
    x, y = to_norm(*row['query_pixel_uv'])
    return INSTRUCTION.format(x=x, y=y) + '\n' + SUFFIX


def messages(row, depth=None, valid=None, include_answer=False, query=True):
    m = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': user_text(row, query)}]}]
    if include_answer: m.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer_text(row, depth, valid)}]})
    return m


def prepare(rgb, depth, longest_edge=640):
    """Resize like hamster3d.inference.preprocessing (RGB bilinear, depth nearest, float16 round-trip)."""
    import torch
    rgb_r, _ = resize_to_target(np.asarray(rgb), longest_edge)
    depth_r, _ = resize_to_target(depth, longest_edge, interp=__import__('cv2').INTER_NEAREST)
    depth_r = np.asarray(depth_r).astype(np.float16).astype(np.float32)
    geo = torch.from_numpy(rgb_r).float().permute(2, 0, 1).unsqueeze(0) / 255.
    return Image.fromarray(rgb_r), geo, torch.from_numpy(depth_r).float().unsqueeze(0)


def encode(row, root, processor, longest_edge=640, maximum_tokens=2048, query=True):
    """Supervised encoding: labels on the assistant answer only, verified by prefix equality."""
    import torch
    rgb, depth, valid = load_frame(root, row)
    img, geo, dmap = prepare(rgb, depth, longest_edge)
    full_m = messages(row, depth, valid, include_answer=True, query=query)
    full_t = processor.apply_chat_template(full_m, tokenize=False, add_generation_prompt=False)
    pre_t = processor.apply_chat_template(full_m[:1], tokenize=False, add_generation_prompt=True)
    full = processor(text=[full_t], images=[img], return_tensors='pt')
    pre = processor(text=[pre_t], images=[img], return_tensors='pt')
    ids, pids = full['input_ids'], pre['input_ids']
    n = pids.shape[1]
    if not (0 < n < ids.shape[1] <= maximum_tokens): raise ValueError('Empty answer or token budget exceeded')
    if not torch.equal(ids[:, :n], pids): raise ValueError('Chat-template prefix mismatch; cannot mask loss safely')
    if not torch.equal(full['image_grid_thw'], pre['image_grid_thw']): raise ValueError('Image grid mismatch')
    labels = ids.clone(); labels[:, :n] = -100
    full['labels'] = labels
    full['geometry_encoder_inputs'] = [geo]
    full['depth_maps'] = [dmap]
    return full


def token_weights(labels, tokenizer, row, class_weights=None, status_weight=1.):
    """1 on answer tokens, status_weight through the point_3d value start ('[' or 'null'), times class weight."""
    import torch
    status = row['answer']['status']
    sup = labels != -100
    w = torch.zeros(labels.shape, dtype=torch.float32); w[sup] = 1.
    if status_weight != 1.:
        ids = labels[sup].tolist(); count = None
        for k in range(1, min(len(ids), 40) + 1):
            t = tokenizer.decode(ids[:k])
            if '"point_3d": [' in t or '"point_3d": null' in t: count = k; break
        if count is None: raise ValueError('point_3d value start not found in the answer prefix')
        pos = sup.nonzero()[:count]; w[pos[:, 0], pos[:, 1]] = float(status_weight)
    if class_weights: w = w * float(class_weights[status])
    return w


def parse_output(raw):
    """Model text -> canonical task-v3 answer dict (pixels) + predicted depth, or (None, reason)."""
    m = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
    text = m.group(1) if m else raw.strip()
    try: arr = json.loads(text)
    except Exception: return None, None, 'invalid_json'
    if not isinstance(arr, list) or not arr or not isinstance(arr[0], dict): return None, None, 'invalid_structure'
    e = arr[0]; pt = e.get('point_3d', e.get('point_2d'))
    if pt is None:
        ans = dict(status='abstain', cut_point_uv=None, visibility='occluded', next_action='change_viewpoint'); d = None
    else:
        if not isinstance(pt, list) or len(pt) < 2 or any(not isinstance(c, (int, float)) for c in pt[:2]): return None, None, 'invalid_point'
        x, y = pt[0], pt[1]
        if not (0 <= x < 1000 and 0 <= y < 1000): return None, None, 'point_out_of_frame'
        u, v = to_px(x, y); d = float(pt[2]) if len(pt) >= 3 and isinstance(pt[2], (int, float)) else None
        ans = dict(status='localized', cut_point_uv=[u, v], visibility='clear', next_action='inspect_cut_region')
    try: validate_answer(ans)
    except Exception as err: return None, None, 'invalid_answer:' + type(err).__name__
    return ans, d, None
