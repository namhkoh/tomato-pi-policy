"""Reference-only delivery for original and generated native9mm datasets.

The bundle never changes a source row's schema. Its loader dispatches explicitly
to the corresponding frozen sensor loader; labels stay separate from inputs.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import hashlib, json
from . import native848_unique_petiole_dataset_v8 as original
from . import native848_generated_dataset_v1 as generated

SCHEMA='greenhouse.native848_reference_delivery_bundle.v1'
ENTRY_SCHEMA='greenhouse.native848_reference_delivery_entry.v1'
LOADERS={'original':original,'generated':generated}
LOADER_PINS={'original':'f3934b4f89b717866e58415c6700184d77ec497b6886005ff0c7e642ea41414a',
 'generated':'1b7e397469cdd3872c60350f91526f4175cca6c306dd631d08600f93ed9122ef'}
MANIFEST_SCHEMAS={'original':'greenhouse.native848_fully_labeled_verified_checkpoint_union.v6',
 'generated':'greenhouse.native848_reviewed_generated_supplement_union.v1'}
require=original.require
pin=original.pin
write=original.write


def _read(spec):return json.loads(original.read_pin(spec).read_text())


def _component(spec):
    require(set(spec)=={'component_id','kind','root','result','manifest','index'},'Exact component reference fields required')
    kind=spec['kind'];require(kind in LOADERS,'Unsupported source dataset kind')
    loader=LOADERS[kind];require(original.digest(loader.__file__)==LOADER_PINS[kind],'Source input loader changed')
    require(Path(spec['root']).is_absolute(),'Absolute source release required');root=Path(spec['root']).resolve()
    for key,name in [('result','result.json'),('manifest','manifest.json'),('index','index.jsonl')]:
        require(Path(spec[key]['path']).resolve()==root/name,'Component evidence escapes its release')
    require(not (root/'failure.json').exists() and not (root/'TRAINING_HOLD.json').exists(),'Incomplete or held source release')
    result=_read(spec['result']);manifest=_read(spec['manifest']);index=original.read_pin(spec['index'])
    require(result['state']=={'original':'accepted_fully_populated_checkpoint_union','generated':'complete_reviewed_generated_supplement'}[kind],
        'Source release is not complete')
    require(manifest['schema']==MANIFEST_SCHEMAS[kind] and manifest['task_id']==original.TASK
        and manifest['annotation_epoch']==original.EPOCH and manifest['fully_populated_greenhouse'] is True
        and manifest['plant_instance_count']==144 and manifest['training_started'] is False,'Typed complete native9mm release required')
    require(result['manifest']['sha256']==spec['manifest']['sha256'] and manifest['index']['sha256']==spec['index']['sha256'],
        'Result/manifest/index linkage differs')
    rows=[json.loads(line) for line in index.read_text().splitlines()]
    require(len(rows)==manifest['unique_images']==result['unique_images'],'Source count differs')
    require(len({r['frame_id'] for r in rows})==len(rows),'Repeated source frame ID')
    for row in rows:
        require(row['schema']==loader.FRAME_SCHEMA and row['task_id']==original.TASK and row['annotation_epoch']==original.EPOCH
            and row['individual_full_frame_review'] is True,'Typed individually reviewed source row required')
        require(set(row['inputs'])==original.INPUT_KEYS and row['model_instruction']==original.INSTRUCTION,'Source model input contract differs')
    return root,rows


def entry_for(component_id,kind,row_number,row):
    require(kind in LOADERS and row['schema']==LOADERS[kind].FRAME_SCHEMA,'Row schema does not match source loader')
    donor=row['source_family'];modified=kind=='generated'
    if modified:
        geometry=row['geometry_source_id']
        require(row['donor_source_family']==donor and row['morphology_id']==geometry!=donor
            and row['new_independent_source_family'] is False and row['source_geometry_modified'] is True,
            'Generated geometry must preserve original donor lineage')
    else:geometry=donor
    require(row['source_target_id'].startswith(donor+'/'),'Target must retain original donor family')
    return dict(schema=ENTRY_SCHEMA,component_id=component_id,source_kind=kind,source_row_number=row_number,
        source_frame_id=row['frame_id'],source_row_sha256=original.canonical_sha(row),split=row['split'],
        donor_source_family=donor,source_target_id=row['source_target_id'],geometry_source_id=geometry,
        generated_geometry=modified,new_independent_source_family=False,
        answer=deepcopy(row['answer']),decoded_rgb_sha256=row['decoded_rgb_sha256'],
        geometry_camera_signature=row['conservative_camera_signature'],
        donor_camera_signature=row.get('donor_camera_signature',row['conservative_camera_signature']))


def counts(entries):
    require(entries,'Empty delivery bundle')
    require(len({e['source_frame_id'] for e in entries})==len(entries),'Repeated source frame ID across components')
    require(len({e['decoded_rgb_sha256'] for e in entries})==len(entries),'Repeated decoded RGB across components')
    require(len({(e['geometry_source_id'],e['geometry_camera_signature']) for e in entries})==len(entries),
        'Repeated camera within one source geometry')
    donor_splits=defaultdict(set);camera_geometries=defaultdict(set)
    for e in entries:
        donor_splits[e['donor_source_family']].add(e['split'])
        camera_geometries[(e['donor_source_family'],e['donor_camera_signature'])].add(e['geometry_source_id'])
    require(all(len(v)==1 for v in donor_splits.values()),'Original donor leaks across splits')
    target_counts=Counter(e['source_target_id'] for e in entries)
    require(max(target_counts.values())<=200,'Existing global original-source-target cap exceeded')
    return dict(unique_images=len(entries),split_counts=dict(Counter(e['split'] for e in entries)),
        original_images=sum(not e['generated_geometry'] for e in entries),generated_images=sum(e['generated_geometry'] for e in entries),
        original_donor_count=len(donor_splits),source_target_count=len(target_counts),
        original_plant_counts=dict(Counter(e['donor_source_family'] for e in entries)),source_target_counts=dict(target_counts),
        geometry_counts=dict(Counter(e['geometry_source_id'] for e in entries)),
        original_geometry_count=len({e['geometry_source_id'] for e in entries if not e['generated_geometry']}),
        generated_geometry_count=len({e['geometry_source_id'] for e in entries if e['generated_geometry']}),
        new_independent_donors_from_generated=0,exact_rgb_duplicates=0,exact_camera_within_geometry_duplicates=0,
        donor_camera_positions_reused_across_geometries=sum(len(v)>1 for v in camera_geometries.values()))


class DeliveryDataset:
    """Open once; verify component indexes once; verify sensor bytes on each load."""
    def __init__(self,bundle):
        self.root=Path(bundle).resolve()
        require(not (self.root/'failure.json').exists() and not (self.root/'TRAINING_HOLD.json').exists(),'Failed or held delivery bundle')
        result=json.loads((self.root/'result.json').read_text())
        require(result['state']=='complete_verified_reference_delivery','Incomplete bundle')
        manifest_pin=dict(path=str(self.root/'manifest.json'),sha256=result['manifest']['sha256'])
        self.manifest=_read(manifest_pin)
        require(self.manifest['schema']==SCHEMA and self.manifest['loader_sha256']==original.digest(__file__),'Bundle schema or loader changed')
        self.components={}
        for spec in self.manifest['components']:
            require(spec['component_id'] not in self.components,'Duplicate component ID')
            self.components[spec['component_id']]=(*_component(spec),spec['kind'])
        spec=self.manifest['index'];require(Path(spec['path']).resolve()==self.root/'index.jsonl','Bundle index escapes root')
        self.entries=[json.loads(line) for line in original.read_pin(spec).read_text().splitlines()]
        for entry in self.entries:
            root,rows,kind=self.components[entry['component_id']];n=entry['source_row_number']
            require(type(n) is int and 0<=n<len(rows) and entry==entry_for(entry['component_id'],kind,n,rows[n]),'Normalized row differs from pinned typed source')
        require(Counter(e['component_id'] for e in self.entries)==Counter({k:len(v[1]) for k,v in self.components.items()}),
            'Bundle must include every row of each pinned release exactly once')
        require(counts(self.entries)==self.manifest['counts'] and len(self.entries)==result['unique_images'],'Bundle counts differ')
    def __len__(self):return len(self.entries)
    def load(self,index):
        entry=self.entries[index];root,rows,kind=self.components[entry['component_id']]
        row=rows[entry['source_row_number']]
        return dict(inputs=LOADERS[kind].load_model_inputs(row,root),answer=deepcopy(entry['answer']))


def create_bundle(components,output,*,expected_images):
    output=Path(output).resolve()
    require(output.parent in tuple(p/'training_exports' for p in generated.roots.DATA_ROOTS)
        and not output.exists(),'Fresh direct-child delivery path required')
    require(len({s['component_id'] for s in components})==len(components),'Duplicate component IDs')
    entries=[];source_rows={}
    for spec in components:
        root,rows=_component(spec);source_rows[spec['component_id']]=(root,rows,spec['kind'])
        for n,row in enumerate(rows):
            entries.append(entry_for(spec['component_id'],spec['kind'],n,row))
            LOADERS[spec['kind']].load_model_inputs(row,root)
    total=counts(entries);require(total['unique_images']==expected_images,'Unexpected delivery count')
    # Inputs were loaded and their bytes checked. No private label re-evaluation,
    # USD asset reload, native request, or source asset copy occurs here.
    output.mkdir()
    try:
        for name,chosen in [('index.jsonl',entries)]+[(s+'_index.jsonl',[e for e in entries if e['split']==s]) for s in ('train','validation','test')]:
            with (output/name).open('x',encoding='utf-8') as stream:
                for e in chosen:stream.write(json.dumps(e,allow_nan=False)+'\n')
        manifest=dict(schema=SCHEMA,task_id=original.TASK,annotation_epoch=original.EPOCH,
            components=components,index=pin(output/'index.jsonl'),split_indexes={s:pin(output/(s+'_index.jsonl')) for s in ('train','validation','test')},
            counts=total,loader='sim_data.native848_delivery_bundle_v1.DeliveryDataset',loader_sha256=original.digest(__file__),
            source_loader_pins=LOADER_PINS,native_resolution=[848,408],nominal_arc_m=.009,exactly_one_eligible_petiole_per_image=True,
            fully_populated_greenhouse=True,plant_instance_count=144,input_query=False,input_target_id=False,input_ground_truth_crop=False,input_ground_truth_mask=False,
            source_schema_dispatch_explicit=True,source_assets_copied=0,source_annotations_recomputed=False,all_model_inputs_loaded_and_verified=True,
            morphology_diversity_approved=False,source_release_directories_must_be_retained=True,training_started=False,stable_pointers_modified=False)
        write(output/'manifest.json',manifest)
        write(output/'result.json',dict(schema=SCHEMA,state='complete_verified_reference_delivery',unique_images=expected_images,
            manifest=pin(output/'manifest.json'),training_started=False))
        dataset=DeliveryDataset(output);require(len(dataset)==expected_images,'Delivery roundtrip count differs')
        for kind in ('original','generated'):
            i=next((i for i,e in enumerate(dataset.entries) if e['source_kind']==kind),None)
            if i is not None:dataset.load(i)
        donor_lines=[]
        for donor,n in sorted(total['original_plant_counts'].items()):
            own=[e for e in entries if e['donor_source_family']==donor]
            donor_lines.append(f"| {donor} | {own[0]['split']} | {sum(not e['generated_geometry'] for e in own)} | {sum(e['generated_geometry'] for e in own)} | {n} |")
        text='# Native greenhouse cut-point dataset\n\n'+str(expected_images)+' verified images in one reference bundle. '
        text+=f"{total['original_images']} original-geometry views plus {total['generated_images']} generated-geometry views.\n\n"
        text+='Task: from full848×408 RGB, aligned float32 optical-Z depth, validity and sensor calibration, locate one 9mm cut point on the single eligible petiole. All 144 greenhouse planting slots remain populated. Measured robot context supports a separate workspace filter.\n\n'
        text+='## Load a sample\n\n```python\nfrom sim_data.native848_delivery_bundle_v1 import DeliveryDataset\ndata = DeliveryDataset(r"'+str(output)+'")\nsample = data.load(0)\ninputs = sample["inputs"]  # sensors, measured workspace context, instruction\nlabel = sample["answer"]   # {"cut_point_uv": [u, v]}\n```\n\n'
        text+='Put `examples/greenhouse_sim` on PYTHONPATH and use the same Python environment as the supplied dataset loaders. Pass only `inputs` to the model. Index identities, private annotations, labels, renderer masks and ground-truth crops are not inference inputs.\n\n'
        text+='## Counts by original plant\n\n| Original donor | Split | Original views | Generated views | Total |\n|---|---|---:|---:|---:|\n'+'\n'.join(donor_lines)+'\n\n'
        text+='Detailed target and geometry counts are in `manifest.json`. Generated geometry IDs retain their donor split and do not create independent donors. The two modified geometries are limited donor-derived subtree augmentations; broad morphology diversity is not claimed.\n\n'
        text+='## Files and storage\n\n`index.jsonl` is the normalized training index; split indexes select TRAIN, validation and TEST. The manifest pins both source releases and dispatches their distinct schemas explicitly. Source assets are referenced, not copied: retain the absolute source release folders listed in the manifest. No annotations were rerun, no latest pointer was changed, and no training was launched.\n'
        (output/'README.md').write_text(text,encoding='utf-8')
        return manifest
    except BaseException as exc:
        write(output/'failure.json',dict(error=repr(exc),partial_bundle_not_admissible=True));raise
