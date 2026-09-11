import json
import pytest
from sim_data.h200_train import choose_rows,check_paths,SingleImageCollator,Rows,arguments


def records(n=20):
    return [dict(id=f'{status}_{i}',messages=[{}, {}, dict(content=json.dumps(dict(status=status)))])
        for status in ('localized','abstain') for i in range(n)]


def test_smoke_and_overfit_are_balanced_deterministic_train_subsets():
    rows=records();original=list(rows)
    assert [r['id'] for r in choose_rows(rows,'smoke')]==['localized_0','abstain_0']
    assert len(choose_rows(rows,'overfit'))==32
    assert choose_rows(rows,'train') is rows and rows==original
    assert len(Rows(rows))==40 and Rows(rows)[0] is rows[0]
    with pytest.raises(ValueError):choose_rows(records(1),'overfit')
    with pytest.raises(ValueError):choose_rows(rows,'unknown')


def test_local_model_only_and_no_dataset_model_or_output_overwrite(tmp_path):
    dataset=tmp_path/'release';dataset.mkdir();(dataset/'manifest.json').write_text('{}')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text(json.dumps(dict(
        model_type='qwen3_vl',text_config=dict(hidden_size=4096,num_hidden_layers=36))))
    output=tmp_path/'run'
    assert check_paths(dataset,model,output)==(dataset,model,output)
    for bad in (dataset/'run',model/'run',tmp_path):
        with pytest.raises(ValueError):check_paths(dataset,model,bad)
    output.mkdir()
    with pytest.raises(ValueError):check_paths(dataset,model,output)
    with pytest.raises(ValueError):check_paths(dataset,tmp_path/'missing',tmp_path/'new')
    (model/'config.json').write_text('{"model_type":"qwen3_vl_moe"}')
    with pytest.raises(ValueError):check_paths(dataset,model,tmp_path/'new')


def test_collator_keeps_exact_existing_adapter_contract(monkeypatch):
    import sim_data.qwen_adapter as adapter
    calls=[]
    monkeypatch.setattr(adapter,'encode_supervised',lambda *a,**k:calls.append((a,k)) or {'encoded':True})
    collator=SingleImageCollator('release','processor');row=records()[0]
    assert collator([row])=={'encoded':True}
    assert calls==[((row,'release','processor'),{'maximum_tokens':2048,'coordinates':'normalized_1000'})]
    with pytest.raises(ValueError):collator([row,row])


@pytest.mark.parametrize('flag,value',[('--epochs','nan'),('--epochs','0'),('--learning-rate','inf'),('--gradient-accumulation','0')])
def test_invalid_config_fails_before_any_model_import(flag,value):
    with pytest.raises(SystemExit):arguments(['--dataset','d','--model','m','--output','o','--mode','smoke',flag,value])
