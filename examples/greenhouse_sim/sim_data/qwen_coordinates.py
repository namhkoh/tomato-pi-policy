"""Versioned, reversible Qwen grounding coordinates; never rewrite a release.

Qwen3-VL grounding uses relative 0..1000 coordinates. Canonical task-v3 labels
remain original 848x408 pixels. This adapter converts BOTH query and answer.
"""
import json
import math
import re
from .training_contract import SYSTEM_PROMPT,user_prompt,validate_answer

COORDINATE_ADAPTER='qwen3_grounding_normalized_1000.v1'
PIXEL_RULE='Coordinates are continuous pixels in the ORIGINAL 848x408 image, top-left edge origin, x right and y down. '
NORMALIZED_RULE=('Coordinates (query and cut_point_uv) are normalized continuous coordinates in [0,1000), '
    'top-left edge origin, x right and y down: x_norm=1000*x/848 and y_norm=1000*y/408 '
    'for the ORIGINAL 848x408 image, independent of processor resizing. ')
if SYSTEM_PROMPT.count(PIXEL_RULE)!=1: raise RuntimeError('Task pixel convention changed; re-audit the Qwen adapter')
NORMALIZED_SYSTEM_PROMPT=(SYSTEM_PROMPT.replace(PIXEL_RULE,NORMALIZED_RULE)
    .replace('The user supplies a query pixel on the target petiole, NOT the cut point.',
             'The user supplies a normalized query coordinate on the target petiole, NOT the cut point.')
    .replace('cut_point_uv (a two-number pixel array or null)',
             'cut_point_uv (a two-number normalized coordinate array or null)'))


def convert_point(point,*,to_normalized):
    if (not isinstance(point,list) or len(point)!=2
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in point)):
        raise ValueError('Finite two-number coordinates required')
    bounds=[848.,408.] if to_normalized else [1000.,1000.]
    if any(not 0<=v<bound for v,bound in zip(point,bounds)):
        raise ValueError('Coordinates outside their declared frame; no clipping')
    numerators=[1000.,1000.] if to_normalized else [848.,408.]
    # Preserve continuous coordinates rather than introducing integer rounding.
    result=[v*n/b for v,n,b in zip(point,numerators,bounds)]
    out_bounds=[1000.,1000.] if to_normalized else [848.,408.]
    if any(not 0<=v<bound for v,bound in zip(result,out_bounds)):
        raise ValueError('Coordinate conversion reached the excluded image boundary')
    return result


def answer_to_pixels(answer):
    """Strict decoder for THIS adapter, not an automatic scale guess."""
    if not isinstance(answer,dict): raise ValueError('Expected a JSON object, not an array of pairs')
    result=dict(answer)
    if result.get('cut_point_uv') is not None:
        result['cut_point_uv']=convert_point(result['cut_point_uv'],to_normalized=False)
    return validate_answer(result)


def adapt_messages(messages):
    if ([m.get('role') for m in messages] not in (['system','user'],['system','user','assistant'])
            or messages[0]['content']!=[dict(type='text',text=SYSTEM_PROMPT)]):
        raise ValueError('Expected original task-v3 messages, not already normalized messages')
    content=messages[1]['content']
    if len(content)!=2 or content[0].get('type')!='image' or content[1].get('type')!='text':
        raise ValueError('Expected unaltered RGB plus query text')
    text=content[1]['text']
    match=re.match(r'The target petiole passes through pixel \(([0-9]+\.[0-9]), ([0-9]+\.[0-9])\)\.',text)
    if not match: raise ValueError('Unrecognized original query convention')
    query=list(map(float,match.groups()))
    if text!=user_prompt(query): raise ValueError('Original query prompt changed')
    normalized=convert_point(query,to_normalized=True)
    query_text=(f'The target petiole passes through normalized coordinates ({normalized[0]}, {normalized[1]}). '
        'Locate its nominal cut point if the junction and cut region are visually '
        'distinguishable; otherwise abstain. Use the original full image.')
    result=[dict(role='system',content=[dict(type='text',text=NORMALIZED_SYSTEM_PROMPT)]),
        dict(role='user',content=[dict(content[0]),dict(type='text',text=query_text)])]
    if len(messages)==3:
        answer=validate_answer(json.loads(messages[2]['content'][0]['text']))
        answer=dict(answer)
        if answer['cut_point_uv'] is not None:
            answer['cut_point_uv']=convert_point(answer['cut_point_uv'],to_normalized=True)
        result.append(dict(role='assistant',content=[dict(type='text',text=json.dumps(answer,separators=(',',':')))]))
    return result
