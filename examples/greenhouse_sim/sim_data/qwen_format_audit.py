"""Read-only Qwen conversion audit of every frozen split (no processor/model).

Complements, never replaces, the normal release/image/depth validator. Checks
hash-bound chats, exact roles, both coordinate conversions and answer roundtrip.
No labels, images, reviews or splits are written.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .dataset_review import read_json,require,safe_file,write_json
from .depth_preview import sha256
from .training_export import read_jsonl
from .training_contract import SYSTEM_PROMPT,validate_answer
from .qwen_coordinates import adapt_messages,answer_to_pixels,COORDINATE_ADAPTER


def audit(root):
    root=Path(root).resolve();manifest=read_json(root/'manifest.json')
    require(manifest.get('state')=='complete_visible_occluded_baseline_release'
        and manifest.get('release_profile')=='visible_occluded_v1','Expected completed narrowed baseline')
    counts={};seen=set();maximum_error=0.
    for split in ('train','validation','test'):
        name=f'splits/{split}.jsonl';path=safe_file(root,name)
        require(sha256(path)==manifest['files_sha256'][name],'Split bytes differ from frozen manifest')
        result=dict(rows=0,localized=0,abstain=0)
        for row in read_jsonl(path):
            require(row['id'] not in seen,'Duplicate row ID across frozen splits');seen.add(row['id'])
            turns=row['messages'];images=row['images']
            require([t['role'] for t in turns]==['system','user','assistant'],'Incorrect chat roles')
            require(turns[0]['content']==SYSTEM_PROMPT,'Changed system prompt')
            require(len(images)==1 and images[0].startswith('images/') and images[0].endswith('.png'),'Expected one RGB path')
            require(turns[1]['content'].startswith('<image>\n') and turns[1]['content'].count('<image>')==1,'Incorrect image placeholder')
            canonical=validate_answer(json.loads(turns[2]['content']))
            text_messages=[dict(role='system',content=[dict(type='text',text=turns[0]['content'])]),
                dict(role='user',content=[dict(type='image',image=images[0]),dict(type='text',text=turns[1]['content'][8:])]),
                dict(role='assistant',content=[dict(type='text',text=turns[2]['content'])])]
            normalized=adapt_messages(text_messages)
            require(adapt_messages(text_messages[:2])==normalized[:2],'Training/inference prompts diverge')
            decoded=answer_to_pixels(json.loads(normalized[-1]['content'][0]['text']))
            require(all(decoded[k]==canonical[k] for k in ('status','visibility','next_action')),'Answer semantics changed')
            if canonical['status']=='localized':
                error=max(abs(a-b) for a,b in zip(decoded['cut_point_uv'],canonical['cut_point_uv']))
                maximum_error=max(maximum_error,error);require(error<1e-9,'Coordinate roundtrip changed labels')
            else:require(decoded['cut_point_uv'] is None,'Hidden cut leaked into abstention')
            result['rows']+=1;result[canonical['status']]+=1
        require(result['rows']==manifest['acceptance']['rows'][split],'Frozen split count changed')
        counts[split]=result
    return dict(state='passed_qwen_chat_coordinate_audit_not_model_runtime_validation',
        coordinate_adapter=COORDINATE_ADAPTER,manifest_sha256=sha256(root/'manifest.json'),
        files_sha256={p.name:sha256(p) for p in (Path(__file__),Path(__file__).with_name('qwen_coordinates.py'),Path(__file__).with_name('qwen_adapter.py'))},
        counts=counts,maximum_pixel_roundtrip_error=maximum_error,
        canonical_release_changed=False,images_transformed=False,model_input_is_RGB_only=True,
        normal_release_validator_still_required=True,image_depth_bytes_rechecked=False,
        actual_Qwen_processor_tested=False,model_weights_loaded=False,training_started=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    require(not args.output.resolve().is_relative_to(args.dataset.resolve()),'Keep audit outside immutable dataset')
    require(not args.output.exists(),'Choose a new audit output')
    result=audit(args.dataset);write_json(args.output,result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
