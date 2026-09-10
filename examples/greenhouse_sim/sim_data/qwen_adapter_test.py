import json

import pytest
from PIL import Image
from sim_data.qwen_adapter import model_messages,encode_supervised,GroundingDataset
from sim_data.training_export import chat_row


@pytest.fixture
def example(tmp_path):
    (tmp_path/'images').mkdir()
    Image.new('RGB',(848,408),(35,75,20)).save(tmp_path/'images/rgb.png')
    row=chat_row('example','images/rgb.png',dict(query_pixel_uv=[300.,200.],
        answer=dict(status='localized',cut_point_uv=[200.,220.],visibility='clear',
            next_action='inspect_cut_region')))
    return tmp_path,row


def test_shared_input_preserves_roles_and_does_not_expose_answer(example):
    root,row=example
    messages=model_messages(row,root)
    assert [m['role'] for m in messages]==['system','user']
    assert messages[1]['content'][0]['image'].size==(848,408)
    assert messages[1]['content'][1]['text']==row['messages'][1]['content'][8:]
    assert row['messages'][2]['content'] not in str(messages)
    full=model_messages(row,root,include_answer=True)
    assert full[-1]['content'][0]['text']==row['messages'][-1]['content']


@pytest.mark.parametrize('change',[
    lambda r:r['images'].__setitem__(0,'review/overlay.png'),
    lambda r:r['images'].__setitem__(0,'images/../../secret.png'),
    lambda r:r['messages'][0].__setitem__('content','Default prompt'),
    lambda r:r['messages'][1].__setitem__('content','<image>\n<image>\nTarget'),
    lambda r:r['messages'][2].__setitem__('content','{"joint_command":[1,2,3]}')])
def test_invalid_inputs_fail_closed(example,change):
    root,row=example;change(row)
    with pytest.raises(ValueError): model_messages(row,root)


def test_mask_boundary_without_hardcoded_token_ids(example):
    torch=pytest.importorskip('torch')
    root,row=example
    class Processor:
        def apply_chat_template(self,messages,**kwargs):
            ids=[7,8,9] if kwargs.get('add_generation_prompt') else [7,8,9,10,11]
            return dict(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),
                image_grid_thw=torch.tensor([[1,13,27]]))
    out=encode_supervised(row,root,Processor())
    assert out['labels'].tolist()==[[-100,-100,-100,10,11]]
    with pytest.raises(ValueError,match='never truncate'):
        encode_supervised(row,root,Processor(),maximum_tokens=4)
    class Broken(Processor):
        def apply_chat_template(self,messages,**kwargs):
            result=super().apply_chat_template(messages,**kwargs)
            if kwargs.get('add_generation_prompt'): result['input_ids'][0,0]=111
            return result
    with pytest.raises(ValueError,match='prefix changed'): encode_supervised(row,root,Broken())


def test_incomplete_release_cannot_become_training_dataset(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema_version='greenhouse.grounding_training_release.v1',
        state='incomplete_engineering_export_do_not_claim_release')))
    with pytest.raises(ValueError,match='Incomplete release'): GroundingDataset(tmp_path)
