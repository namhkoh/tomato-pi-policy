"""Fine-tune the released 3D HAMSTER checkpoint on the tomato cut-point task (bridge experiment).

Full fine-tune with DeepSpeed ZeRO-2 (language model, vision tower at 0.1x LR, geometry merger, head);
LingBot-Depth geometry encoder stays frozen (as in the release). Loss = per-example mean token CE on the
assistant answer with class weights and an up-weighted status prefix (same recipe as our full-02).
Modes: smoke (2 steps, 2 examples), overfit (32 examples, 60 steps), train (full split, epochs).
"""
import argparse, hashlib, importlib.metadata, json, math, os, sys, time
from collections import Counter
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cutpoint_data as cd


def choose_rows(rows, mode):
    if mode == 'train': return rows
    per = {'smoke': 1, 'overfit': 16}[mode]; g = {'localized': [], 'abstain': []}
    for r in rows:
        s = r['answer']['status']
        if len(g[s]) < per: g[s].append(r)
    return [r for pair in zip(*g.values()) for r in pair]


class Rows:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class Collator:
    def __init__(self, root, processor, longest_edge, class_weights, status_weight):
        self.root, self.processor, self.le, self.cw, self.sw = root, processor, longest_edge, class_weights, status_weight
    def __call__(self, rows):
        if len(rows) != 1: raise ValueError('single-example microbatch only')
        enc = cd.encode(rows[0], self.root, self.processor, self.le)
        enc['token_weights'] = cd.token_weights(enc['labels'], self.processor.tokenizer, rows[0], self.cw, self.sw)
        return dict(enc)


def is_vision_tower(n): return n.startswith('model.visual.') and 'merger' not in n
def is_geometry_encoder(n): return n.startswith('model.geometry_encoder.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True); p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--mode', choices=('smoke', 'overfit', 'train'), required=True)
    p.add_argument('--epochs', type=float, default=3.); p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--vision-learning-rate', type=float, default=1e-6); p.add_argument('--gradient-accumulation', type=int, default=8)
    p.add_argument('--longest-edge', type=int, default=640); p.add_argument('--class-balance', action='store_true')
    p.add_argument('--status-token-weight', type=float, default=1.); p.add_argument('--deepspeed', type=Path, default=None)
    p.add_argument('--seed', type=int, default=41); p.add_argument('--wandb-project', default=None); p.add_argument('--wandb-entity', default=None)
    p.add_argument('--run-name', default=None); p.add_argument('--save-total-limit', type=int, default=3)
    a = p.parse_args(argv)
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from accelerate import PartialState
    from transformers import AutoProcessor, AutoModelForImageTextToText, Trainer, TrainingArguments, TrainerCallback, set_seed
    from hamster3d.model import register_qwen3_vl_geometry
    register_qwen3_vl_geometry()
    state = PartialState(timeout=timedelta(hours=2))
    root, model_path, output = a.dataset.resolve(), a.model.resolve(), a.output.resolve()
    if state.is_main_process:
        if output.exists(): raise ValueError('Choose a NEW output directory')
        output.mkdir(parents=True)
    set_seed(a.seed)
    manifest_sha = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    all_train = list(cd.read_jsonl(root / 'splits/train.jsonl'))
    index = {r['id']: r for r in cd.read_jsonl(root / 'index.jsonl')}
    all_train = [index[r['id']] for r in all_train]
    train_rows = choose_rows(all_train, a.mode)
    counts = Counter(r['answer']['status'] for r in all_train)
    cw = {s: len(all_train) / (len(counts) * n) for s, n in counts.items()} if a.class_balance else None
    collator = Collator(root, processor, a.longest_edge, cw, a.status_token_weight)
    samples = []
    for r in choose_rows(all_train, 'smoke'):
        enc = collator([r]); sup = enc['labels'] != -100
        text = processor.tokenizer.decode(enc['labels'][sup], skip_special_tokens=True).strip()
        expected = cd.messages(r, *cd.load_frame(root, r)[1:], include_answer=True)[-1]['content'][0]['text']
        if text != expected: raise RuntimeError('Supervised tokens do not decode to the answer:\n' + text + '\n' + expected)
        samples.append(dict(id=r['id'], tokens=int(enc['input_ids'].shape[1]), supervised=int(sup.sum()), grid=enc['image_grid_thw'].tolist(),
                            weighted=int((enc['token_weights'] > (cw[r['answer']['status']] if cw else 1)).sum())))
    model = AutoModelForImageTextToText.from_pretrained(str(model_path), dtype=torch.bfloat16, trust_remote_code=True, attn_implementation='sdpa')
    model.config.use_cache = False
    for n, prm in model.named_parameters(): prm.requires_grad = not is_geometry_encoder(n)
    trainable = [(n, q) for n, q in model.named_parameters() if q.requires_grad]
    vision = [q for n, q in trainable if is_vision_tower(n)]
    if not any('language_model' in n for n, _ in trainable) or not any('geometry_merger' in n for n, _ in trainable): raise RuntimeError('unexpected trainable set')

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            w = inputs.pop('token_weights', None); dt = next(model.parameters()).dtype
            inputs['geometry_encoder_inputs'] = [t.to(dt) for t in inputs['geometry_encoder_inputs']]
            inputs['depth_maps'] = [t.to(dt) for t in inputs['depth_maps']]
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and torch.is_floating_point(v): inputs[k] = v.to(dt)
            if w is None or not model.training:
                out = model(**inputs); loss = out.loss
            else:
                labels = inputs.pop('labels'); out = model(**inputs)
                logits = out.logits[:, :-1].float(); tgt = labels[:, 1:]; ww = w[:, 1:].to(logits.device); mask = tgt != -100
                ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1).clamp(min=0), reduction='none').reshape(tgt.shape)
                loss = (ce * ww * mask).sum() / mask.sum()
            if loss.ndim != 0 or not bool(torch.isfinite(loss)): raise RuntimeError('non-finite loss')
            return (loss, out) if return_outputs else loss
        def create_optimizer(self):
            if self.optimizer is None:
                cls, kw = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model); ids = {id(q) for q in vision}
                rest = [q for _, q in self.model.named_parameters() if q.requires_grad and id(q) not in ids]
                kw = {k: v for k, v in kw.items() if k != 'lr'}
                groups = [dict(params=rest, lr=a.learning_rate, weight_decay=self.args.weight_decay)]
                if vision: groups.append(dict(params=vision, lr=a.vision_learning_rate, weight_decay=self.args.weight_decay))
                self.optimizer = cls(groups, lr=a.learning_rate, **kw)
            return super().create_optimizer()

    class Guard(TrainerCallback):
        def __init__(self): self.seen = 0
        def on_log(self, args_, st, ctl, logs=None, **k):
            g = (logs or {}).get('grad_norm')
            if g is not None:
                self.seen += 1
                if not math.isfinite(float(g)) or float(g) <= 0: raise RuntimeError('non-finite or zero grad norm')
        def on_train_end(self, args_, st, ctl, **k):
            if not self.seen: raise RuntimeError('grad norm never logged')

    max_steps = {'smoke': 2, 'overfit': 60, 'train': -1}[a.mode]; accum = a.gradient_accumulation if a.mode == 'train' else 1
    report = []
    if a.wandb_project:
        os.environ['WANDB_PROJECT'] = a.wandb_project
        if a.wandb_entity: os.environ['WANDB_ENTITY'] = a.wandb_entity
        os.environ.setdefault('WANDB_WATCH', 'false'); report = ['wandb']
    targs = TrainingArguments(output_dir=str(output), num_train_epochs=a.epochs, max_steps=max_steps, learning_rate=a.learning_rate,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=accum, bf16=True, optim='adamw_torch',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False}, ddp_find_unused_parameters=False, ddp_timeout=7200,
        dataloader_num_workers=0, remove_unused_columns=False, label_names=['labels'], prediction_loss_only=True,
        eval_strategy='epoch' if a.mode == 'train' else 'no', save_strategy='epoch' if a.mode == 'train' else 'no', save_total_limit=a.save_total_limit,
        save_only_model=True, deepspeed=str(a.deepspeed) if a.deepspeed else None, logging_steps=1 if a.mode != 'train' else 10, logging_nan_inf_filter=False,
        report_to=report, run_name=a.run_name or output.name, seed=a.seed, data_seed=a.seed, lr_scheduler_type='constant' if a.mode != 'train' else 'cosine',
        warmup_ratio=0. if a.mode != 'train' else .03, max_grad_norm=1.)
    eval_rows = [index[r['id']] for r in cd.read_jsonl(root / 'splits/validation.jsonl')] if a.mode == 'train' else None
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=Rows(train_rows), eval_dataset=Rows(eval_rows) if eval_rows else None,
                              data_collator=collator, callbacks=[Guard()])
    trainer.model_accepts_loss_kwargs = False
    if state.is_main_process:
        (output / 'run_contract.json').write_text(json.dumps(dict(mode=a.mode, base=str(model_path), contract=cd.CONTRACT, longest_edge=a.longest_edge,
            manifest_sha256=manifest_sha, class_weights=cw, status_token_weight=a.status_token_weight, learning_rate=a.learning_rate,
            vision_learning_rate=a.vision_learning_rate, epochs=a.epochs, effective_examples_per_step=state.num_processes * accum,
            trainable_parameters=sum(q.numel() for _, q in trainable), frozen_geometry_encoder=True, processor_samples=samples,
            train_ids=[r['id'] for r in train_rows], versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'deepspeed', 'accelerate')},
            instruction=cd.INSTRUCTION, suffix=cd.SUFFIX), indent=1))
    result = trainer.train()
    trainer.save_model(str(output / 'model')); trainer.save_state()
    if state.is_main_process:
        processor.save_pretrained(output / 'model')
        (output / 'model' / 'grounding_adapter.json').write_text(json.dumps(dict(contract=cd.CONTRACT, longest_edge=a.longest_edge, canonical_resolution=[848, 408],
            release_manifest_sha256=manifest_sha), indent=1))
        tracker = None
        if report:
            import wandb
            if wandb.run: tracker = dict(project=wandb.run.project, id=wandb.run.id, url=wandb.run.url)
        (output / 'completed.json').write_text(json.dumps(dict(mode=a.mode, metrics=result.metrics, tracker=tracker), indent=1))
    state.wait_for_everyone()


if __name__ == '__main__': main()
