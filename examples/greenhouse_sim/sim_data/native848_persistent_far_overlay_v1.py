"""Install a proven far overlay once; copy only foreground from full collision USD.

Caller owns the renderer, ordinary collision-stage foreground transactions and
fresh collision caches. This module never renders, warms RTX or approves labels.
"""
from copy import deepcopy
from pathlib import Path
import time
import numpy as np
from pxr import Sdf, Usd, UsdGeom

from . import native848_farmerge_overlay_v1 as frozen
from . import native848_persistent_foreground_v1 as foreground
from . import native848_persistent_far_population_v1 as variable
from .native848_bulk_workspace_v1 import BulkWorkspace
from .capture_contract import fingerprint
from .dataset_review import require, verify_bindings
from .depth_preview import sha256

SCHEMA='greenhouse.persistent_far_overlay_transaction.v1'
PINS={variable.__file__:'8f97ebe986b9f4dfaad36562002981726c6106cb06f70a35f1d851a7f18e7f1a',
      frozen.__file__:'a095f37d0dafa62a36f7b937c9ff2e7fa26ac54f7866430ce23b920b3045491e',
      frozen.finite.__file__:'502ad385dbaf347a07060c12425bade9c0ca6f9faa6559677ac70a15c0c61c9a',
      foreground.__file__:'a4fcdc3166b3c552fcba6d050a4b46b6e3b7d3a742b361160a5cdfc424df592f'}


def recipe_installer(schema):
    if schema==frozen.finite.SCHEMA:return frozen.install_overlay
    require(schema==variable.RECIPE_SCHEMA,'Unsupported persistent far population recipe')
    return variable.install_overlay


def clone_original_collision_scene(scene):
    """Call before generating the first morphology, after renderer setup.

    Snapshot the unchanged foreground helper on this returned complete stage,
    never on the subsequently packed renderer stage.
    """
    stage,layer,before,root_before=frozen.clone_collision_stage(scene['stage'])
    result=dict(scene,stage=stage)
    result['persistent_collision_keepalive']=(scene,stage,layer,scene['stage'].GetRootLayer())
    require(stage.GetSessionLayer().ExportToString()==before
            and scene['stage'].GetRootLayer().ExportToString()==root_before,'Original collision clone differs')
    return result


def camera_signature(record):
    return fingerprint({k:record[k] for k in ('calibration','joint_degrees',
        'robot_root_to_world_usd_row_vectors','camera_to_head_column_vectors')})


def authenticate_camera_bank(bank,selection):
    """Recompute positive coarse exclusions from pinned original CPU robot states."""
    require(isinstance(bank,list) and 1<=len(bank)<=16,'Finite pinned camera bank required')
    inputs=frozen.annotation.Inputs();checker=BulkWorkspace();records={};proofs=[];total=0
    try:
        for item in bank:
            require(set(item)=={'candidates','original_camera_cpu'},'Exact camera bank source pins required')
            candidates=inputs.read(item['candidates']);cpu=inputs.read(item['original_camera_cpu'])
            plan=inputs.read(candidates['source_camera_plan'])
            verify_bindings(candidates['source_bindings']);verify_bindings(cpu['source_bindings'])
            for source in (candidates['source_bindings'],cpu['source_bindings']):
                for path,h in source.items():
                    require(path not in inputs.bindings or inputs.bindings[path]==h,'Conflicting camera-bank source')
                    inputs.bindings[path]=h
            total+=len(candidates['records']);require(1<=total<=1024,'Camera bank frame bound exceeded')
            bounds=frozen.finite.candidate_bounds(candidates,cpu,plan,checker)
            for record,proof in zip(candidates['records'],bounds):
                positive=frozen.annotation.actual_coarse_bounds(selection,proof['bounds'])
                require(positive['all_components_outside_both_actual_arm_bounds'],
                        'Packed geometry enters a finite camera arm bound')
                key=camera_signature(record)
                records.setdefault(key,deepcopy(record))
                proofs.append(dict(camera_signature=key,source=item,sample_id=record['sample_id'],
                    bounds=proof['bounds'],coarse_exclusion=positive))
        inputs.bindings.update(checker.bindings)
    finally:checker.finish()
    verify_bindings(inputs.bindings)
    return dict(records=records,proofs=proofs,source_bindings=inputs.bindings)


def _meshes(stage,root):
    cache=UsdGeom.XformCache()
    return [frozen.snapshot_mesh(p,cache) for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.IsA(UsdGeom.Mesh)]


def _background_catalogue(rows,primary):
    return {r['prim_path']:{k:v for k,v in r.items() if k!='component_index'}
            for r in rows if not r['prim_path'].startswith(primary+'/')}


class PersistentFarOverlay:
    def __init__(self,render_scene,collision_scene,population,catalogue,recipe_pin,*,camera_bank):
        started=time.perf_counter();verify_bindings(PINS)
        self.scene=render_scene;self.collision_scene=collision_scene
        self.stage=render_scene['stage'];self.collision_stage=collision_scene['stage']
        require(self.stage!=self.collision_stage and self.stage.GetSessionLayer()!=self.collision_stage.GetSessionLayer(),
                'Distinct complete collision and render session stages required')
        self.inputs=frozen.annotation.Inputs();recipe=self.inputs.read(recipe_pin)
        selection=self.inputs.read(recipe['selection'])
        self.base_catalogue=self.inputs.read(recipe['base_catalogue'])
        self.primary=recipe['primary_root'];self.selection=selection
        self.camera_bank=authenticate_camera_bank(camera_bank,
            dict(selected={r['component_index']:r for r in selection['selected_components']}))
        self.bindings={**PINS,**self.inputs.bindings,**self.camera_bank['source_bindings'],
            str(Path(__file__).resolve()):sha256(__file__)}
        self.reference=deepcopy(population['scene_policy_evidence'])
        self.background_catalogue=_background_catalogue(catalogue,self.primary)
        self.sequence=0;self.failed=False;self._keepalive={};self._templates=[]
        self._validate_population(population,catalogue)
        self._copy_foreground()
        installer=recipe_installer(recipe['schema'])
        self.overlay=installer(render_scene,population,catalogue,recipe_pin)
        for path,h in self.overlay['source_bindings'].items():
            require(path not in self.bindings or self.bindings[path]==h,'Conflicting installed overlay source')
            self.bindings[path]=h
        # The frozen install's one-time clone is valid, but later collision must
        # follow the caller's full-stage swaps, never that stale first-morph clone.
        self.overlay['original_collision_stage']=self.collision_stage
        self.overlay['original_collision_session_keepalive']=self.collision_stage.GetSessionLayer()
        self.whole_retained=[r for r in self.overlay['retained_meshes'] if not r['path'].startswith(self.primary+'/')]
        self.initial_groups=deepcopy(self.overlay['render_groups'])
        self._protect=self._protected_layers(self.stage)
        self._collision_protect=self._protected_layers(self.collision_stage)
        self._bind(population)
        self._refresh_metadata(population,catalogue)
        self.timings=dict(install_seconds=time.perf_counter()-started)

    def _protected_layers(self,stage):
        omitted=(self.primary,self.scene['robot']['root'],*foreground.MUTABLE_RENDER_ROOTS)
        return {l.identifier:(l,foreground._hash_layer(l,omitted)) for l in stage.GetUsedLayers() if l.anonymous}

    def _check_layers(self,snapshot):
        omitted=(self.primary,self.scene['robot']['root'],*foreground.MUTABLE_RENDER_ROOTS)
        for layer,expected in snapshot.values():
            require(foreground._hash_layer(layer,omitted)==expected,'Persistent background/environment layer changed')

    def _validate_population(self,population,catalogue):
        census=population['scene_policy_evidence'];copy=deepcopy(census);h=copy.pop('deterministic_census_sha256')
        require(fingerprint(copy)==h and census['foreground_root']==self.primary
                and census['unchanged_background_count']==143 and population['counts']['component_plants']==144,
                'Complete current generated logical population required')
        frozen.finite.background_match(census,self.reference,self.primary)
        require(len(catalogue)==population['counts']['components']
                and len({r['prim_path'] for r in catalogue})==len(catalogue)
                and _background_catalogue(catalogue,self.primary)==self.background_catalogue,
                'Background component identity or complete current catalogue differs')
        require(population['persistent_swap_evidence']['outside_foreground_opinions_preserved'] is True
                and population['persistent_swap_evidence']['unchanged_background_count']==143,
                'Current collision-stage foreground transaction evidence required')
        require(all(self.collision_stage.GetPrimAtPath(r['prim_path']).IsActive() for r in catalogue),
                'Current full collision component missing')

    def _copy_foreground(self):
        require(self.collision_stage.GetSessionLayer().GetPrimAtPath(self.primary),
                'Collision foreground must have an authored session spec')
        before={l.identifier:(l,foreground._hash_layer(l,(self.primary,)))
                for l in self.stage.GetUsedLayers() if l.anonymous}
        old=foreground._copy_slot(self.stage.GetSessionLayer(),self.primary)
        target=self.stage.GetEditTarget()
        try:
            self.stage.SetEditTarget(self.stage.GetSessionLayer())
            require(self.stage.RemovePrim(self.primary) and not self.stage.GetPrimAtPath(self.primary),
                    'Foreground render slot has unsupported weaker opinions')
            require(Sdf.CopySpec(self.collision_stage.GetSessionLayer(),self.primary,
                    self.stage.GetSessionLayer(),self.primary),'Current full-collision foreground copy failed')
            require(_meshes(self.stage,self.primary)==_meshes(self.collision_stage,self.primary),
                    'Foreground arrays/materials/placement differ between collision and renderer')
            for layer,h in before.values():
                require(foreground._hash_layer(layer,(self.primary,))==h,
                        'Robot/camera/product/background opinions changed during foreground copy')
        except BaseException:
            foreground._restore_slot(self.stage,old,self.primary);raise
        finally:self.stage.SetEditTarget(target)

    def _bind(self,population):
        for path,h in population['source_bindings'].items():
            require(path not in self.bindings or self.bindings[path]==h,'Conflicting morphology source')
            self.bindings[path]=h
        verify_bindings(self.bindings)
        self._templates.append(population['template_stages'])
        for stage in (self.stage,self.collision_stage):
            self._keepalive.update({l.identifier:l for l in stage.GetUsedLayers()})

    def _refresh_metadata(self,population,catalogue):
        groups=[]
        for initial in self.initial_groups:
            group=deepcopy(initial)
            mapping=frozen.remap_members(catalogue,self.base_catalogue,self.selection['selected_components'],group['plant_root'])
            group['original_to_current_component_indices']=mapping;group['member_component_indices']=sorted(mapping.values())
            groups.append(group)
        self.overlay.update(logical_census=deepcopy(population['scene_policy_evidence']),logical_catalogue=deepcopy(catalogue),
            retained_meshes=deepcopy(self.whole_retained)+_meshes(self.stage,self.primary),render_groups=groups,
            source_bindings=dict(self.bindings),collision_snapshot=dict(current_full_collision_stage=True,
                distinct_session_layer=True,original_plant_mesh_count=sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(self.collision_stage.GetPrimAtPath('/World/PackPlants'))),
                current_foreground_signature_sha256=fingerprint(_meshes(self.collision_stage,self.primary)),
                collision_background_not_packed=True,generated_collision_cache_must_be_rebuilt=True))
        self.overlay['render_census']=frozen.verify_overlay(self.overlay)
        self.sequence+=1
        self.overlay['persistent_overlay_evidence']=dict(schema=SCHEMA,sequence=self.sequence,background_overlay_installations=1,
            unchanged_packed_background_count=len(self.overlay['render_groups']),
            whole_background_count=len(self.overlay['whole_roots']),background_shapes_materials_and_placements_preserved=True,
            collision_foreground_copied_exactly=True,current_coarse_component_indices_remapped=True,
            finite_camera_bank_sha256=fingerprint(self.camera_bank['proofs']),source_bindings=dict(self.bindings),
            native_validated=False,training_approved=False,accepted_training_increment=0)

    def refresh(self,population,catalogue):
        """Call only with drained writer/sink and after a full collision-stage swap."""
        started=time.perf_counter();require(not self.failed,'Failed persistent overlay cannot be reused')
        self._check_layers(self._protect);self._check_layers(self._collision_protect)
        frozen.verify_overlay(self.overlay);self._validate_population(population,catalogue)
        try:
            self._copy_foreground();self._bind(population);self._refresh_metadata(population,catalogue)
            self._check_layers(self._protect);self._check_layers(self._collision_protect)
            self.timings=dict(refresh_seconds=time.perf_counter()-started)
            self.overlay['persistent_overlay_evidence']['timings']=self.timings
            return self.overlay
        except BaseException:
            self.failed=True;raise

    def check_camera(self,record):
        require(camera_signature(record) in self.camera_bank['records'],'Camera outside authenticated finite overlay bank')

    def verify_backgrounds(self):
        require(not self.failed,'Failed persistent overlay cannot be verified')
        self._check_layers(self._protect);self._check_layers(self._collision_protect)
        require(not any(not l.anonymous and l.dirty for stage in (self.stage,self.collision_stage) for l in stage.GetUsedLayers()),
                'Protected external USD layer dirtied')
        verify_bindings(self.bindings)
        return frozen.verify_overlay(self.overlay)


def install_once(render_scene,collision_scene,population,catalogue,recipe_pin,*,camera_bank):
    return PersistentFarOverlay(render_scene,collision_scene,population,catalogue,recipe_pin,camera_bank=camera_bank)
