"""Explicit server-only Qwen3-VL LoRA recipe; never downloads model weights.

Run only after the user transfers the validated release and installs a local
Qwen3-VL-8B snapshot on the training server. Real-model/DDP qualification must
start with --mode smoke. Unit tests are not evidence of an H200 training run.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def choose_rows(rows,mode):
    """Smoke/overfit use only deterministic train examples of both statuses."""
    if mode=='train': return rows
    if mode not in ('smoke','overfit'): raise ValueError('Unknown server mode')
    per_status=1 if mode=='smoke' else 16
    groups={s:[] for s in ('localized','abstain')}
    for row in rows:
        status=json.loads(row['messages'][2]['content'])['status']
        if status not in groups: raise ValueError('Unsupported training status')
        if len(groups[status])<per_status: groups[status].append(row)
    if any(len(group)!=per_status for group in groups.values()):
        raise ValueError('Insufficient train-only examples for both statuses')
    return [row for pair in zip(*groups.values()) for row in pair]


def check_paths(dataset,model,output):
    dataset,model,output=(Path(p).resolve() for p in (dataset,model,output))
    if not (dataset/'manifest.json').is_file(): raise ValueError('Local release required')
    if not model.is_dir() or not (model/'config.json').is_file():
        raise ValueError('Local model snapshot required; this recipe never downloads Qwen')
    config=json.loads((model/'config.json').read_text())
    text=config.get('text_config',{})
    if (config.get('model_type')!='qwen3_vl' or text.get('hidden_size')!=4096
            or text.get('num_hidden_layers')!=36 or config.get('quantization_config')):
        raise ValueError('Expected dense, nonquantized Qwen3-VL-8B snapshot')
    if (output==dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output)
            or output==model or output.is_relative_to(model) or model.is_relative_to(output)):
        raise ValueError('Keep outputs separate from model and immutable dataset')
    if output.exists(): raise ValueError('Choose a NEW output; automatic resume/overwrite is disabled')
    return dataset,model,output


class Rows:
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]


class SingleImageCollator:
    """Batch one avoids unverified multi-image padding/grid concatenation."""
    def __init__(self,root,processor,coordinates='normalized_1000'):
        self.root=root;self.processor=processor;self.coordinates=coordinates
    def __call__(self,rows):
        from .qwen_adapter import encode_supervised
        if len(rows)!=1: raise ValueError('Verified single-image microbatch only')
        return encode_supervised(rows[0],self.root,self.processor,maximum_tokens=2048,coordinates=self.coordinates)


def arguments(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True,help='Already downloaded LOCAL snapshot')
    parser.add_argument('--output',type=Path,required=True,help='New run directory outside dataset/model')
    parser.add_argument('--mode',choices=('smoke','overfit','train'),required=True)
    parser.add_argument('--epochs',type=float,default=1.)
    parser.add_argument('--learning-rate',type=float,default=1e-4)
    parser.add_argument('--gradient-accumulation',type=int,default=8)
    parser.add_argument('--seed',type=int,default=41)
    parser.add_argument('--coordinates',choices=('normalized_1000','pixels'),default='normalized_1000')
    args=parser.parse_args(argv)
    import math
    if (not math.isfinite(args.epochs) or not 0<args.epochs<=10
            or not math.isfinite(args.learning_rate) or not 0<args.learning_rate<=1e-3
            or not 1<=args.gradient_accumulation<=128):
        parser.error('Require epochs (0,10], finite learning rate (0,0.001], accumulation [1,128]')
    return args


def main(argv=None):
    args=arguments(argv)
    # All model loading is local-only, even if the server has HF credentials.
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    import torch
    from accelerate import PartialState
    from accelerate.utils import broadcast_object_list
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,Trainer,TrainingArguments,TrainerCallback,set_seed
    from peft import LoraConfig,get_peft_model
    from .dataset_review import write_json
    from .training_export import validate,read_jsonl
    from .qwen_adapter import model_messages
    from .qwen_coordinates import COORDINATE_ADAPTER
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This explicit training recipe requires a BF16 CUDA training server')
    state=PartialState(timeout=timedelta(hours=2))
    if args.mode!='train' and state.num_processes!=1:
        raise ValueError('Run smoke and overfit on ONE GPU before distributed training')
    # Rank zero validates once, before any weights are loaded. Errors propagate
    # to every rank; other ranks must not silently continue with a partial run.
    status=[None]
    if state.is_main_process:
        try:
            root,model_path,output=check_paths(args.dataset,args.model,args.output)
            validated=validate(root,progress=lambda row:print(json.dumps(row),flush=True))
            digest=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()
            output.mkdir(parents=True,exist_ok=False)
            status[0]=dict(ok=True,validation=validated,manifest_sha256=digest)
        except Exception as error: status[0]=dict(ok=False,error=str(error))
    broadcast_object_list(status,from_process=0)
    if not status[0]['ok']: raise RuntimeError('Release/output preflight failed: '+status[0]['error'])
    root=args.dataset.resolve();model_path=args.model.resolve();output=args.output.resolve()
    if hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()!=status[0]['manifest_sha256']:
        raise RuntimeError('Dataset manifest differs between processes')
    set_seed(args.seed)
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True,trust_remote_code=False)
    all_train=list(read_jsonl(root/'splits/train.jsonl'))
    train_rows=choose_rows(all_train,args.mode)
    # Exercise real template/image-grid/assistant-loss boundary before weights.
    collator=SingleImageCollator(root,processor,args.coordinates)
    processor_samples=[]
    for row in choose_rows(all_train,'smoke'):
        encoded=collator([row])
        n=int((encoded['labels']!=-100).sum())
        text=processor.tokenizer.decode(encoded['labels'][encoded['labels']!=-100],skip_special_tokens=True).strip()
        expected=model_messages(row,root,include_answer=True,coordinates=args.coordinates)[-1]['content'][0]['text']
        if json.loads(text)!=json.loads(expected):
            raise RuntimeError('Supervised tokens do not decode to exactly the dataset answer')
        processor_samples.append(dict(id=row['id'],tokens=encoded['input_ids'].shape[1],
            supervised_tokens=n,image_grid_thw=encoded['image_grid_thw'].tolist()))
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    # Exact named language-attention projections only: vision/mergers frozen.
    targets=[name for name,_ in model.named_modules() if '.language_model.' in name
        and name.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')]
    if not targets: raise RuntimeError('Language attention module names changed; inspect before training')
    model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
        target_modules=targets,bias='none',task_type='CAUSAL_LM'))
    trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or '.language_model.' not in n for n,_ in trainable):
        raise RuntimeError('Unexpected non-language or non-LoRA trainable weights')

    class FiniteTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            # Deliberate per-microbatch mean loss, not an unverified global
            # assistant-token normalization across variable-length answers.
            outputs=model(**inputs);loss=outputs.loss
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite/scalar loss check failed')
            return (loss,outputs) if return_outputs else loss

    class GradientGuard(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if (not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients)
                    or not any(bool(torch.any(g!=0)) for g in gradients)):
                raise RuntimeError('Missing, nonfinite or entirely zero LoRA gradients')

    max_steps={'smoke':2,'overfit':100,'train':-1}[args.mode]
    accumulation=args.gradient_accumulation if args.mode=='train' else 1
    training=TrainingArguments(output_dir=str(output),num_train_epochs=args.epochs,max_steps=max_steps,
        learning_rate=args.learning_rate,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,bf16=True,optim='adamw_torch',
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False,ddp_timeout=7200,dataloader_num_workers=0,
        remove_unused_columns=False,label_names=['labels'],prediction_loss_only=True,
        eval_strategy='epoch' if args.mode=='train' else 'no',save_strategy='epoch' if args.mode=='train' else 'no',
        save_total_limit=2,logging_steps=1 if args.mode!='train' else 10,
        logging_nan_inf_filter=False,report_to=[],seed=args.seed,data_seed=args.seed,
        lr_scheduler_type='constant' if args.mode!='train' else 'cosine',
        warmup_ratio=0. if args.mode!='train' else .03,max_grad_norm=1.)
    eval_rows=list(read_jsonl(root/'splits/validation.jsonl')) if args.mode=='train' else None
    trainer=FiniteTrainer(model=model,args=training,train_dataset=Rows(train_rows),
        eval_dataset=Rows(eval_rows) if eval_rows else None,data_collator=collator,
        callbacks=[GradientGuard()])
    trainer.model_accepts_loss_kwargs=False
    if state.is_main_process:
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','accelerate','numpy','pillow')}
        write_json(output/'run_contract.json',dict(mode=args.mode,model_path=str(model_path),
            model_config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
            release=status[0],versions=versions,processor_samples=processor_samples,
            train_ids=[r['id'] for r in train_rows],world_size=state.num_processes,
            coordinates=args.coordinates,coordinate_adapter=COORDINATE_ADAPTER if args.coordinates=='normalized_1000' else 'canonical_task_v3_pixels',
            effective_examples_per_step=state.num_processes*accumulation,
            trainable_parameters=sum(p.numel() for _,p in trainable),lora_targets=targets,
            loss_normalization='mean_of_single_example_assistant_token_losses',
            test_split_used=False,model_download_allowed=False,training_arguments=training.to_dict()))
    result=trainer.train()
    trainer.save_model(str(output/'adapter'));trainer.save_state()
    if state.is_main_process:
        processor.save_pretrained(output/'adapter')
        write_json(output/'adapter'/'grounding_adapter.json',dict(coordinates=args.coordinates,
            coordinate_adapter=COORDINATE_ADAPTER if args.coordinates=='normalized_1000' else 'canonical_task_v3_pixels',
            canonical_resolution=[848,408],release_manifest_sha256=status[0]['manifest_sha256']))
        write_json(output/'completed.json',dict(state='completed_server_training_recipe',mode=args.mode,
            metrics=result.metrics,checkpoint_reload_verified=False,generation_accuracy_measured=False,
            simulator_actions_validated=False))
    state.wait_for_everyone()


if __name__=='__main__': main()
