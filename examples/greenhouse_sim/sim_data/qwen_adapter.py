"""Portable task-v3 RGB adapter for Qwen3-VL; never starts training or motion.

Preserves the release's system/user/assistant messages. In particular, do not
convert a system turn to a "gpt" turn in an upstream conversations loader.
No geometry, native depth, query masks or review overlays enter this adapter.
"""
import json
from pathlib import Path

from PIL import Image

from .dataset_review import require,safe_file
from .training_contract import SYSTEM_PROMPT,validate_answer
from .training_export import read_jsonl,validate

MODEL_ID='Qwen/Qwen3-VL-8B-Instruct'


def model_messages(row,root,*,include_answer=False):
    """The same prompt/RGB conversion for supervised training and inference."""
    turns=row.get('messages',[])
    require(len(turns)==3 and [t.get('role') for t in turns]==['system','user','assistant'],
        'Expected exact system/user/assistant turns')
    require(turns[0]['content']==SYSTEM_PROMPT,'Do not discard or replace the task system prompt')
    user=turns[1]['content']
    require(isinstance(user,str) and user.startswith('<image>\n') and user.count('<image>')==1,
        'Expected exactly one RGB placeholder')
    require(len(row.get('images',[]))==1,'Expected exactly one RGB image')
    relative=Path(row['images'][0])
    require(relative.parts[0]=='images' and relative.suffix.lower()=='.png',
        'Only release RGB images may enter the model')
    image_path=safe_file(Path(root).resolve(),row['images'][0])
    with Image.open(image_path) as source:
        require(source.mode=='RGB' and source.size==(848,408),'Expected original robot-head RGB')
        rgb=source.copy()
    messages=[dict(role='system',content=[dict(type='text',text=SYSTEM_PROMPT)]),
        dict(role='user',content=[dict(type='image',image=rgb),dict(type='text',text=user[len('<image>\n'):])])]
    # Validate even for inference, but do not expose the answer in its prompt.
    validate_answer(json.loads(turns[2]['content']))
    if include_answer:
        messages.append(dict(role='assistant',content=[dict(type='text',text=turns[2]['content'])]))
    return messages


def encode_supervised(row,root,processor,*,maximum_tokens=2048):
    """Mask every prompt/image token using a verified generation-prefix match.

    Two processor calls intentionally favor an auditable boundary over guessed
    token IDs. They share the same in-memory original RGB. Processor resizing is
    internal; the answer remains coordinates in the original 848x408 frame.
    """
    import torch
    require(type(maximum_tokens) is int and maximum_tokens>0,'Invalid token budget')
    messages=model_messages(row,root,include_answer=True)
    full=processor.apply_chat_template(messages,tokenize=True,return_dict=True,return_tensors='pt')
    prefix=processor.apply_chat_template(messages[:-1],tokenize=True,return_dict=True,
        return_tensors='pt',add_generation_prompt=True)
    ids=full['input_ids'];prompt=prefix['input_ids']
    require(ids.ndim==2 and ids.shape[0]==1 and prompt.ndim==2 and prompt.shape[0]==1,
        'Expected one unpadded sequence')
    count=prompt.shape[1]
    require(0<count<ids.shape[1]<=maximum_tokens,'Empty answer or token budget exceeded; never truncate silently')
    require(torch.equal(ids[:,:count],prompt),'Processor template prefix changed; cannot safely mask loss')
    require('image_grid_thw' in full and 'image_grid_thw' in prefix
        and torch.equal(full['image_grid_thw'],prefix['image_grid_thw']),'Image token grid mismatch')
    require('attention_mask' in full and full['attention_mask'].shape==ids.shape
        and bool(torch.all(full['attention_mask']==1)),'Unexpected padding')
    labels=ids.clone();labels[:,:count]=-100
    full['labels']=labels
    return full


class GroundingDataset:
    """Only a complete validated portable release can become training input."""
    def __init__(self,root,*,split='train'):
        require(split in ('train','validation','test'),'Unknown split')
        self.root=Path(root).resolve()
        self.validation=validate(self.root)  # No incomplete-release override.
        self.rows=list(read_jsonl(safe_file(self.root,f'splits/{split}.jsonl')))
        require(bool(self.rows),'Empty split')

    def __len__(self): return len(self.rows)

    def messages(self,index):
        return model_messages(self.rows[index],self.root)

    def encode(self,index,processor,*,maximum_tokens=2048):
        return encode_supervised(self.rows[index],self.root,processor,maximum_tokens=maximum_tokens)
