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


def answer_token_weights(labels,tokenizer,row,class_weights=None,status_token_weight=1.):
    """Per-token loss weights: 1 on supervised answer tokens, status_token_weight on the
    answer prefix through the status value, all scaled by the example's class weight."""
    import torch
    status=json.loads(row['messages'][2]['content'])['status']
    supervised=labels!=-100
    weights=torch.zeros(labels.shape,dtype=torch.float32);weights[supervised]=1.
    if status_token_weight!=1.:
        ids=labels[supervised].tolist();target='"status":"'+status;count=None
        for k in range(1,min(len(ids),32)+1):
            if target in tokenizer.decode(ids[:k]): count=k;break
        if count is None: raise ValueError('Status value not found at the start of the supervised answer')
        positions=supervised.nonzero()[:count];weights[positions[:,0],positions[:,1]]=float(status_token_weight)
    if class_weights: weights=weights*float(class_weights[status])
    return weights


class SingleImageCollator:
    """Batch one avoids unverified multi-image padding/grid concatenation."""
    def __init__(self,root,processor,coordinates='normalized_1000',decimals=None,class_weights=None,status_token_weight=1.,depth_input=False):
        self.root=root;self.processor=processor;self.coordinates=coordinates;self.decimals=decimals
        self.class_weights=class_weights;self.status_token_weight=status_token_weight;self.depth_input=depth_input
    def __call__(self,rows):
        from .qwen_adapter import encode_supervised
        if len(rows)!=1: raise ValueError('Verified single-image microbatch only')
        encoded=encode_supervised(rows[0],self.root,self.processor,maximum_tokens=2048,coordinates=self.coordinates,decimals=self.decimals,depth_input=self.depth_input)
        if self.class_weights or self.status_token_weight!=1.:
            encoded['token_weights']=answer_token_weights(encoded['labels'],self.processor.tokenizer,rows[0],self.class_weights,self.status_token_weight)
        return encoded


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
    # Abstention-decision fixes (2026-09-12): shorter coordinate text, class-balanced loss, upweighted status token.
    parser.add_argument('--coordinate-decimals',type=int,default=None,help='Fixed decimals for normalized coordinate text (default: full float precision)')
    parser.add_argument('--class-balance',action='store_true',help='Weight each example loss by inverse train-split status frequency')
    parser.add_argument('--status-token-weight',type=float,default=1.,help='Extra loss weight on answer tokens through the status value')
    parser.add_argument('--depth-input',action='store_true',help='RGB-D variant: native depth rendered as a second image (explicit experiment outside the RGB-only contract)')
    # Full-parameter fine-tuning (user-requested 2026-09-12) as an explicit alternative to the LoRA recipe.
    parser.add_argument('--method',choices=('lora','full'),default='lora',help='lora: language-attention LoRA (default recipe); full: every weight trainable')
    parser.add_argument('--deepspeed',type=Path,default=None,help='DeepSpeed JSON (ZeRO-2 recommended for --method full on 4xH200)')
    parser.add_argument('--vision-learning-rate',type=float,default=None,help='--method full only: learning rate for the vision tower blocks/embeddings (default 0.1 x learning rate)')
    parser.add_argument('--freeze-vision',action='store_true',help='--method full only: keep the vision tower frozen (mergers still train)')
    parser.add_argument('--save-total-limit',type=int,default=2)
    # Optional experiment tracking. Credentials come from the server's wandb
    # login (netrc), never from run arguments.
    parser.add_argument('--wandb-project',default=None,help='Report metrics to this Weights & Biases project')
    parser.add_argument('--wandb-entity',default=None,help='Weights & Biases entity/team for --wandb-project')
    parser.add_argument('--run-name',default=None,help='Tracker run name; defaults to the output directory name')
    args=parser.parse_args(argv)
    import math
    if (not math.isfinite(args.epochs) or not 0<args.epochs<=10
            or not math.isfinite(args.learning_rate) or not 0<args.learning_rate<=1e-3
            or not 1<=args.gradient_accumulation<=128):
        parser.error('Require epochs (0,10], finite learning rate (0,0.001], accumulation [1,128]')
    if args.vision_learning_rate is None: args.vision_learning_rate=args.learning_rate*.1
    if not math.isfinite(args.vision_learning_rate) or not 0<=args.vision_learning_rate<=1e-3:
        parser.error('Require finite vision learning rate in [0,0.001]')
    if args.method=='lora' and (args.deepspeed or args.freeze_vision):
        parser.error('--deepspeed and --freeze-vision apply to --method full')
    if args.deepspeed and not args.deepspeed.is_file(): parser.error('DeepSpeed config file not found')
    if not 1<=args.save_total_limit<=20: parser.error('Require save-total-limit in [1,20]')
    if args.coordinate_decimals is not None and (args.coordinates!='normalized_1000' or not 0<=args.coordinate_decimals<=6):
        parser.error('--coordinate-decimals requires normalized coordinates and a value in [0,6]')
    if not math.isfinite(args.status_token_weight) or not 1<=args.status_token_weight<=20: parser.error('Require status token weight in [1,20]')
    return args


def is_vision_tower(name):
    """Vision encoder blocks/embeddings; the mergers project into the language model and follow its LR."""
    return name.startswith('model.visual.') and '.merger' not in name and 'deepstack_merger' not in name


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
    from .qwen_coordinates import COORDINATE_ADAPTER,adapter_name
    from .depth_input import DEPTH_INPUT
    from collections import Counter
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This explicit training recipe requires a BF16 CUDA training server')
    state=PartialState(timeout=timedelta(hours=2))
    if args.mode!='train' and state.num_processes!=1 and not (args.method=='full' and args.deepspeed):
        # Full fine-tuning with ZeRO-2 only exists as a sharded multi-GPU configuration
        # (fp32 master weights + Adam states of 8.8B parameters do not fit one GPU), so its
        # smoke/overfit qualification runs in exactly that configuration instead.
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
    counts=Counter(json.loads(r['messages'][2]['content'])['status'] for r in all_train)
    class_weights={s:len(all_train)/(len(counts)*n) for s,n in counts.items()} if args.class_balance else None
    # Exercise real template/image-grid/assistant-loss boundary before weights.
    collator=SingleImageCollator(root,processor,args.coordinates,args.coordinate_decimals,class_weights,args.status_token_weight,args.depth_input)
    processor_samples=[]
    for row in choose_rows(all_train,'smoke'):
        encoded=collator([row])
        n=int((encoded['labels']!=-100).sum())
        text=processor.tokenizer.decode(encoded['labels'][encoded['labels']!=-100],skip_special_tokens=True).strip()
        expected=model_messages(row,root,include_answer=True,coordinates=args.coordinates,decimals=args.coordinate_decimals,depth_input=args.depth_input)[-1]['content'][0]['text']
        if json.loads(text)!=json.loads(expected):
            raise RuntimeError('Supervised tokens do not decode to exactly the dataset answer')
        processor_samples.append(dict(id=row['id'],tokens=encoded['input_ids'].shape[1],
            supervised_tokens=n,image_grid_thw=encoded['image_grid_thw'].tolist(),
            weighted_tokens=(int((encoded['token_weights']>1).sum()) if 'token_weights' in encoded else None)))
    model=Qwen3VLForConditionalGeneration.from_pretrained(str(model_path),local_files_only=True,
        trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa')
    model.config.use_cache=False
    if args.method=='lora':
        # Exact named language-attention projections only: vision/mergers frozen.
        targets=[name for name,_ in model.named_modules() if '.language_model.' in name
            and name.rsplit('.',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')]
        if not targets: raise RuntimeError('Language attention module names changed; inspect before training')
        model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,
            target_modules=targets,bias='none',task_type='CAUSAL_LM'))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not trainable or any('lora_' not in n or '.language_model.' not in n for n,_ in trainable):
            raise RuntimeError('Unexpected non-language or non-LoRA trainable weights')
    else:
        # Full-parameter fine-tuning: language model, head and mergers always train;
        # the vision tower trains at its own (lower) LR unless frozen.
        targets=None
        for name,parameter in model.named_parameters():
            parameter.requires_grad=not (args.freeze_vision and is_vision_tower(name))
        trainable=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        if not any('.language_model.' in n for n,_ in trainable) or not any('lm_head' in n for n,_ in trainable):
            raise RuntimeError('Full fine-tuning expects trainable language and head weights')
        if not any('merger' in n for n,_ in trainable): raise RuntimeError('Full fine-tuning expects trainable mergers')
        if args.freeze_vision and any(is_vision_tower(n) for n,_ in trainable): raise RuntimeError('Vision tower should be frozen')
    vision_parameters=[p for n,p in trainable if is_vision_tower(n)]

    class FiniteTrainer(Trainer):
        def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
            # Deliberate per-microbatch mean loss, not an unverified global
            # assistant-token normalization across variable-length answers.
            weights=inputs.pop('token_weights',None)
            if weights is None or not model.training:
                outputs=model(**inputs);loss=outputs.loss  # evaluation stays an unweighted token mean
            else:
                labels=inputs.pop('labels');outputs=model(**inputs)
                logits=outputs.logits[:,:-1].float();targets=labels[:,1:];w=weights[:,1:].to(logits.device)
                mask=(targets!=-100)
                ce=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1).clamp(min=0),reduction='none').reshape(targets.shape)
                loss=(ce*w*mask).sum()/mask.sum()
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite/scalar loss check failed')
            return (loss,outputs) if return_outputs else loss
        def create_optimizer(self):
            # Full mode: explicit parameter groups so the vision tower can use a lower LR.
            if self.optimizer is None and args.method=='full':
                cls,kwargs=Trainer.get_optimizer_cls_and_kwargs(self.args,self.model)
                ids={id(p) for p in vision_parameters}
                rest=[p for _,p in self.model.named_parameters() if p.requires_grad and id(p) not in ids]
                groups=[dict(params=rest,lr=args.learning_rate,weight_decay=self.args.weight_decay)]
                if vision_parameters: groups.append(dict(params=vision_parameters,lr=args.vision_learning_rate,weight_decay=self.args.weight_decay))
                kwargs={k:v for k,v in kwargs.items() if k!='lr'}
                self.optimizer=cls(groups,lr=args.learning_rate,**kwargs)
            return super().create_optimizer()

    class GradientGuard(TrainerCallback):
        def on_pre_optimizer_step(self,args,state,control,model=None,**kwargs):
            gradients=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            if (not gradients or any(not bool(torch.isfinite(g).all()) for g in gradients)
                    or not any(bool(torch.any(g!=0)) for g in gradients)):
                raise RuntimeError('Missing, nonfinite or entirely zero LoRA gradients')

    class LoggedGradientGuard(TrainerCallback):
        """DeepSpeed partitions/clears per-parameter grads, so check the logged global norm instead."""
        def __init__(self): self.seen=0
        def on_log(self,args_,state,control,logs=None,**kwargs):
            import math
            norm=(logs or {}).get('grad_norm')
            if norm is None: return
            norm=float(norm);self.seen+=1
            if not math.isfinite(norm) or norm<=0.: raise RuntimeError('Nonfinite or zero global gradient norm')
        def on_train_end(self,args_,state,control,**kwargs):
            if not self.seen: raise RuntimeError('Global gradient norm was never logged; cannot confirm finite gradients')
    guard=LoggedGradientGuard() if args.deepspeed else GradientGuard()

    max_steps={'smoke':2,'overfit':100,'train':-1}[args.mode]
    report_to=[]
    if args.wandb_project:
        os.environ['WANDB_PROJECT']=args.wandb_project
        if args.wandb_entity: os.environ['WANDB_ENTITY']=args.wandb_entity
        os.environ.setdefault('WANDB_WATCH','false')
        report_to=['wandb']
    run_name=args.run_name or output.name
    accumulation=args.gradient_accumulation if args.mode=='train' else 1
    training=TrainingArguments(output_dir=str(output),num_train_epochs=args.epochs,max_steps=max_steps,
        learning_rate=args.learning_rate,per_device_train_batch_size=1,per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation,bf16=True,optim='adamw_torch',
        gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
        ddp_find_unused_parameters=False,ddp_timeout=7200,dataloader_num_workers=0,
        remove_unused_columns=False,label_names=['labels'],prediction_loss_only=True,
        eval_strategy='epoch' if args.mode=='train' else 'no',save_strategy='epoch' if args.mode=='train' else 'no',
        save_total_limit=args.save_total_limit,save_only_model=args.method=='full',
        deepspeed=str(args.deepspeed) if args.deepspeed else None,
        logging_steps=1 if args.mode!='train' else 10,
        logging_nan_inf_filter=False,report_to=report_to,run_name=run_name,seed=args.seed,data_seed=args.seed,
        lr_scheduler_type='constant' if args.mode!='train' else 'cosine',
        warmup_ratio=0. if args.mode!='train' else .03,max_grad_norm=1.)
    eval_rows=list(read_jsonl(root/'splits/validation.jsonl')) if args.mode=='train' else None
    trainer=FiniteTrainer(model=model,args=training,train_dataset=Rows(train_rows),
        eval_dataset=Rows(eval_rows) if eval_rows else None,data_collator=collator,
        callbacks=[guard])
    trainer.model_accepts_loss_kwargs=False
    if state.is_main_process:
        versions={name:importlib.metadata.version(name) for name in ('torch','transformers','peft','accelerate','numpy','pillow')}
        write_json(output/'run_contract.json',dict(mode=args.mode,model_path=str(model_path),
            model_config_sha256=hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
            release=status[0],versions=versions,processor_samples=processor_samples,
            train_ids=[r['id'] for r in train_rows],world_size=state.num_processes,
            coordinates=args.coordinates,coordinate_adapter=adapter_name(args.coordinate_decimals) if args.coordinates=='normalized_1000' else 'canonical_task_v3_pixels',
            coordinate_decimals=args.coordinate_decimals,class_weights=class_weights,status_token_weight=args.status_token_weight,
            depth_input=DEPTH_INPUT if args.depth_input else None,
            train_status_counts=dict(counts),effective_examples_per_step=state.num_processes*accumulation,
            trainable_parameters=sum(p.numel() for _,p in trainable),lora_targets=targets,method=args.method,
            vision_tower_parameters=sum(p.numel() for p in vision_parameters),vision_learning_rate=args.vision_learning_rate if args.method=='full' else None,
            freeze_vision=args.freeze_vision,deepspeed_config=json.loads(args.deepspeed.read_text()) if args.deepspeed else None,
            trainable_prefixes=sorted({n.split('.')[0] if not n.startswith('model.') else '.'.join(n.split('.')[:2]) for n,_ in trainable}),
            loss_normalization='mean_of_single_example_assistant_token_losses'+('_with_class_and_status_token_weights' if (class_weights or args.status_token_weight!=1.) else ''),
            test_split_used=False,model_download_allowed=False,training_arguments=training.to_dict()))
    result=trainer.train()
    artifact=output/('adapter' if args.method=='lora' else 'model')
    trainer.save_model(str(artifact));trainer.save_state()
    tracker=None
    if report_to and state.is_main_process:
        import wandb
        if wandb.run is not None:
            tracker=dict(project=wandb.run.project,entity=wandb.run.entity,id=wandb.run.id,url=wandb.run.url)
    if state.is_main_process:
        processor.save_pretrained(artifact)
        write_json(artifact/'grounding_adapter.json',dict(method=args.method,coordinates=args.coordinates,
            coordinate_adapter=adapter_name(args.coordinate_decimals) if args.coordinates=='normalized_1000' else 'canonical_task_v3_pixels',
            coordinate_decimals=args.coordinate_decimals,depth_input=DEPTH_INPUT if args.depth_input else None,
            canonical_resolution=[848,408],release_manifest_sha256=status[0]['manifest_sha256']))
        write_json(output/'completed.json',dict(state='completed_server_training_recipe',mode=args.mode,
            metrics=result.metrics,experiment_tracker=tracker,checkpoint_reload_verified=False,generation_accuracy_measured=False,
            simulator_actions_validated=False))
    state.wait_for_everyone()


if __name__=='__main__': main()
