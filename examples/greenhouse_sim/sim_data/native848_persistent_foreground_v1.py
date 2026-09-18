"""CPU/USD foreground transactions for a finite persistent native scene.

This helper does not render or approve samples. The producer must drain its sink,
leave the writer inactive, close the old static monitor, and establish fresh
collision, catalogue, census and render evidence after every successful swap.
Only compatible variants of one original donor/slot may share this context.
"""
from copy import deepcopy
import hashlib
import difflib
import json
from pathlib import Path
import time

import numpy as np

from . import native848_generated_morphology_scene_v2 as generated
from .capture_contract import fingerprint
from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256

SCHEMA = 'greenhouse.persistent_foreground_transaction.v1'
GENERATED_ADAPTER_SHA256 = '30a5d90ae2bdf0eb59a303079c5029dbd5c3a26cf735b564f16c53288b4cbdda'
MUTABLE_RENDER_ROOTS = ('/Render', '/Replicator', '/Orchestrator')


def _pin(spec):
    require(set(spec) == {'path', 'sha256'}, 'Exact immutable pin required')
    path = Path(spec['path']).resolve()
    require(path.is_file() and sha256(path) == spec['sha256'], 'Changed pinned source: '+str(path))
    return dict(path=str(path), sha256=spec['sha256'])


def _delete(layer, path):
    from pxr import Sdf
    if layer.GetPrimAtPath(path):
        edit = Sdf.BatchNamespaceEdit()
        edit.Add(path, Sdf.Path.emptyPath)
        require(layer.Apply(edit), 'Cannot remove foreground snapshot spec')


def _copy_slot(layer, path):
    from pxr import Sdf
    require(bool(layer.GetPrimAtPath(path)), 'Foreground needs an authored session spec')
    result = Sdf.Layer.CreateAnonymous('persistent-foreground-slot')
    Sdf.CreatePrimInLayer(result, Sdf.Path(path).GetParentPath())
    require(Sdf.CopySpec(layer, path, result, path), 'Cannot copy foreground spec')
    return result


def _hash_layer(layer, omitted=()):
    from pxr import Sdf
    if omitted:
        copy = Sdf.Layer.CreateAnonymous('persistent-protected-opinions')
        copy.TransferContent(layer)
        for path in omitted:
            _delete(copy, path)
        layer = copy
    return hashlib.sha256(layer.ExportToString().encode('utf-8')).hexdigest()


def _restore_slot(stage, layer, path):
    from pxr import Sdf
    session = stage.GetSessionLayer()
    stage.SetEditTarget(session)
    require(stage.RemovePrim(path) and not stage.GetPrimAtPath(path),
            'Foreground has unsupported weaker opinions')
    require(Sdf.CopySpec(layer, path, session, path), 'Foreground restore failed')


class PersistentForeground:
    """Retain one original population and strong USD layers; no process lifecycle."""

    def __init__(self, scene, anchor, *, profile_pin, diagnostic_dir=None):
        started = time.perf_counter()
        require(sha256(generated.__file__) == GENERATED_ADAPTER_SHA256, 'Frozen generated adapter changed')
        self.stage = scene['stage']
        self.anchor = deepcopy(anchor)
        self.profile_pin = _pin(profile_pin)
        self.robot_root = scene['robot']['root']
        require(self.robot_root.startswith('/World/') and self.stage.GetPrimAtPath(self.robot_root),
                'Existing robot root required')
        self.original = dict(scene)
        for key in ('records', 'variants', 'counts', 'scene_policy_evidence', 'original_population_bindings'):
            self.original[key] = deepcopy(scene[key])
        self.original['template_stages'] = dict(scene['template_stages'])
        self._template_stages = dict(scene['template_stages'])
        self.scene = self.original
        self.scene_plan_pin = _pin(scene['scene_policy_evidence']['source_collection_plan'])
        require(str(Path(anchor['source_collection_plan']).resolve()) == self.scene_plan_pin['path']
                and anchor['source_bindings'].get(self.scene_plan_pin['path']) == self.scene_plan_pin['sha256'],
                'Anchor actual scene plan is not authenticated')
        self.plan = read_json(self.scene_plan_pin['path'])
        self.primary, self.backgrounds = generated.predecessor._population_snapshot(
            self.stage, self.original, self.anchor, self.plan)
        require(not (self.primary == self.robot_root or self.primary.startswith(self.robot_root+'/')),
                'Foreground and robot roots overlap')
        self.session = self.stage.GetSessionLayer()
        self.root = self.stage.GetRootLayer()
        require(self.root.anonymous and self.session.anonymous, 'Anonymous original scene required')
        self.original_slot = _copy_slot(self.session, self.primary)
        self.current_slot = self.original_slot
        self._current_template_hashes = {}
        self._keepalive = {layer.identifier:layer for layer in self.stage.GetUsedLayers()}
        self._baseline_layers = {key:layer for key,layer in self._keepalive.items() if layer.anonymous}
        self._omitted = (self.primary, self.robot_root, *MUTABLE_RENDER_ROOTS)
        self._baseline_hashes = {key:_hash_layer(layer, self._omitted)
                                 for key,layer in self._baseline_layers.items()}
        self.diagnostic_dir = None if diagnostic_dir is None else Path(diagnostic_dir).resolve()
        self._diagnostic_baselines = {}
        if self.diagnostic_dir is not None:
            self.diagnostic_dir.mkdir(parents=True, exist_ok=False)
            for key,layer in self._baseline_layers.items():
                path = self.diagnostic_dir/(hashlib.sha256(key.encode('utf-8')).hexdigest()+'.before.usda')
                raw = self._protected_text(layer).encode('utf-8')
                require(hashlib.sha256(raw).hexdigest() == self._baseline_hashes[key],
                        'Diagnostic baseline differs from protected layer hash')
                with path.open('xb') as stream: stream.write(raw)
                self._diagnostic_baselines[key] = path
        self.source_bindings = dict(self.original['original_population_bindings'])
        for pin in (self.scene_plan_pin, self.profile_pin):
            self.source_bindings[pin['path']] = pin['sha256']
        self.source_bindings[str(Path(__file__).resolve())] = sha256(__file__)
        verify_bindings(self.source_bindings)
        self.compatibility = dict(scene_plan=self.scene_plan_pin, profile=self.profile_pin,
            source_family=anchor['source_family'], split=anchor['split'],
            original_variant=deepcopy(anchor['original_variant']),
            foreground_world=generated.predecessor._world(self.stage, self.primary),
            background_sha256=fingerprint(self.backgrounds),
            family_assignments=deepcopy(self.plan['family_assignments']),
            protected_anonymous_layers=deepcopy(self._baseline_hashes))
        self.compatibility_sha256 = fingerprint(self.compatibility)
        self.sequence = 0
        self.current_identity = dict(geometry_source_id=anchor['source_family'])
        self.failed = False
        self.timings = dict(snapshot_seconds=time.perf_counter()-started)

    def _compatible(self, candidate, anchor, profile_pin):
        require(not self.failed, 'Failed foreground context cannot be reused')
        require(self.stage.GetRootLayer() == self.root and self.stage.GetSessionLayer() == self.session,
                'Persistent scene or session layer changed')
        require(_pin(profile_pin) == self.profile_pin and _pin(candidate['profile']) == self.profile_pin,
                'Incompatible renderer profile')
        require(anchor['split'] == 'train' and anchor['source_family'] == self.anchor['source_family']
                and candidate['source_family'] == self.anchor['source_family'], 'Incompatible donor or split')
        require(anchor['original_variant'] == self.anchor['original_variant'], 'Incompatible foreground slot or identity')
        require(str(Path(anchor['source_collection_plan']).resolve()) == self.scene_plan_pin['path']
                and anchor['source_bindings'].get(self.scene_plan_pin['path']) == self.scene_plan_pin['sha256'],
                'Incompatible actual scene plan')
        plan_pin = _pin(candidate['source_plan'])
        plan = read_json(plan_pin['path'])
        require(plan['family_assignments'] == self.plan['family_assignments']
                and plan['source_bindings_sha256'] == self.plan['source_bindings_sha256'],
                'Incompatible source assets or background assignments')
        qualification = read_json(_pin(candidate['qualification'])['path'])
        require(qualification.get('version') == generated.CONTROLLED_VERSION
                and qualification.get('source_family') == self.anchor['source_family']
                and qualification.get('split') == 'train', 'Incompatible generated qualification')

    def _protected_text(self, layer):
        # Diagnostic clone uses exactly the same omitted roots as _hash_layer.
        from pxr import Sdf
        copy = Sdf.Layer.CreateAnonymous('persistent-protected-diagnostic')
        copy.TransferContent(layer)
        for path in self._omitted:
            _delete(copy, path)
        return copy.ExportToString()

    def _dump_protected_changes(self):
        """Diagnostic only: preserve before/current text; never authorize a delta."""
        if self.diagnostic_dir is None:
            return
        try:
            changed = []
            for key,layer in self._baseline_layers.items():
                raw = self._protected_text(layer).encode('utf-8')
                current_hash = hashlib.sha256(raw).hexdigest()
                if current_hash == self._baseline_hashes[key]:
                    continue
                before = self._diagnostic_baselines[key]
                after = before.with_name(before.name.replace('.before.usda','.after.usda'))
                diff = before.with_name(before.name.replace('.before.usda','.diff'))
                with after.open('xb') as stream: stream.write(raw)
                before_raw = before.read_bytes()
                require(hashlib.sha256(before_raw).hexdigest() == self._baseline_hashes[key],
                        'Saved diagnostic baseline changed')
                text = ''.join(difflib.unified_diff(before_raw.decode('utf-8').splitlines(True),
                    raw.decode('utf-8').splitlines(True),fromfile=str(before),tofile=str(after)))
                with diff.open('x',encoding='utf-8') as stream: stream.write(text)
                changed.append(dict(layer_identifier=key,expected_sha256=self._baseline_hashes[key],
                    current_sha256=current_hash,before=str(before),after=str(after),diff=str(diff)))
            with (self.diagnostic_dir/'protected_changes.json').open('x',encoding='utf-8') as stream:
                json.dump(dict(sequence=self.sequence,omitted_roots=list(self._omitted),changed_layers=changed,
                    diagnostic_only=True,protected_guard_unchanged=True),stream,indent=2)
        except Exception as exc:
            # Diagnostics cannot turn an integrity rejection into success or hide it.
            self.diagnostic_error = repr(exc)

    def _check_protected(self):
        require(_hash_layer(_copy_slot(self.session, self.primary)) == _hash_layer(self.current_slot),
                'Current foreground was edited outside the transaction')
        for key,layer in self._baseline_layers.items():
            current_hash = _hash_layer(layer, self._omitted)
            if current_hash != self._baseline_hashes[key]:
                self._dump_protected_changes()
            require(current_hash == self._baseline_hashes[key],
                    'Protected background/environment opinions changed')
        for key,expected in self._current_template_hashes.items():
            require(_hash_layer(self._keepalive[key]) == expected, 'Current generated template changed')
        require(len(self.stage.GetPrimAtPath('/World/PackPlants').GetChildren()) == 144, 'Plant slot count changed')
        for path,row in self.backgrounds.items():
            prim = self.stage.GetPrimAtPath(path)
            require(prim and prim.IsActive() and not prim.IsInstanceable(), 'Background slot changed')
            require(np.array_equal(generated.predecessor._world(self.stage, path), row['world']),
                    'Background transform changed')
            require(all(self.stage.GetPrimAtPath(p) and self.stage.GetPrimAtPath(p).IsActive()
                        for p in row['record']['component_paths'].values()), 'Background component missing')
        verify_bindings(self.source_bindings)
        require(not any(not layer.anonymous and layer.dirty for layer in self.stage.GetUsedLayers()),
                'Protected source layer dirtied')

    def replace(self, candidate, *, anchor=None, profile_pin=None):
        """Restore original foreground, run frozen v4 adapter, return its new population.

        Robot/camera/render opinions may change between calls, but must not change
        during this transaction. Incompatible requests fail before any USD edit.
        """
        started = time.perf_counter()
        anchor = self.anchor if anchor is None else anchor
        profile_pin = self.profile_pin if profile_pin is None else profile_pin
        self._compatible(candidate, anchor, profile_pin)
        self._check_protected()
        checked = time.perf_counter()
        previous_slot = _copy_slot(self.session, self.primary)
        previous_identity = deepcopy(self.current_identity)
        before_layers = {layer.identifier:(layer, _hash_layer(layer, (self.primary,)))
                         for layer in self.stage.GetUsedLayers() if layer.anonymous}
        edit_target = self.stage.GetEditTarget()
        try:
            _restore_slot(self.stage, self.original_slot, self.primary)
            restored = time.perf_counter()
            result = generated.replace_controlled_foreground(self.stage, self.original,
                qualification_pin=candidate['qualification'], donor_plan_pin=candidate['source_plan'], anchor=anchor)
            replaced = time.perf_counter()
            for layer,expected in before_layers.values():
                require(_hash_layer(layer, (self.primary,)) == expected,
                        'Robot/camera/render or other outside-foreground opinions changed during swap')
            require(result['unchanged_background_count'] == 143 and result['counts'] == self.original['counts'],
                    'Replacement full population differs')
            for key in ('records', 'variants'):
                require([r for r in result[key] if r['plant_root'] != self.primary]
                        == [r for r in self.original[key] if r['plant_root'] != self.primary],
                        'Replacement changed background records')
            for path,h in result['source_bindings'].items():
                require(path not in self.source_bindings or self.source_bindings[path] == h, 'Conflicting source pin')
            verify_bindings(result['source_bindings'])
            self._keepalive.update({layer.identifier:layer for layer in self.stage.GetUsedLayers()})
            template = result['template_stages'][result['foreground_identity']['morphology_id']][0]
            self._template_stages.update(result['template_stages'])
            self._keepalive.update({layer.identifier:layer for layer in template.GetUsedLayers()})
            self._current_template_hashes = {layer.identifier:_hash_layer(layer)
                                            for layer in template.GetUsedLayers() if layer.anonymous}
            self.source_bindings.update(result['source_bindings'])
            self.current_slot = _copy_slot(self.session, self.primary)
            self.current_identity = deepcopy(result['foreground_identity'])
            self.sequence += 1
            timings = dict(precheck_seconds=checked-started, restore_seconds=restored-checked,
                frozen_adapter_seconds=replaced-restored, postcheck_seconds=time.perf_counter()-replaced,
                total_seconds=time.perf_counter()-started)
            evidence = dict(schema=SCHEMA, sequence=self.sequence, compatibility_sha256=self.compatibility_sha256,
                source_profile=self.profile_pin, qualification=deepcopy(candidate['qualification']),
                previous_geometry=previous_identity, current_geometry=deepcopy(self.current_identity),
                original_foreground_restored_before_adapter=True, unchanged_background_count=143,
                outside_foreground_opinions_preserved=True, timings=timings,
                fresh_collision_catalogue_static_and_native_evidence_required=True,
                native_capture_validated=False, training_approved=False, accepted_training_increment=0)
            result['persistent_swap_evidence'] = evidence
            result['source_bindings'][str(Path(__file__).resolve())] = sha256(__file__)
            self.timings = timings
            return result
        except BaseException:
            self.failed = True
            _restore_slot(self.stage, previous_slot, self.primary)
            raise
        finally:
            self.stage.SetEditTarget(edit_target)


    def verify_backgrounds(self):
        """Close the immutable source union; allow only between-frame robot/render edits."""
        require(not self.failed, 'Failed foreground context cannot be verified')
        started = time.perf_counter()
        self._check_protected()
        return dict(schema=SCHEMA, sequence=self.sequence, compatibility_sha256=self.compatibility_sha256,
                    unchanged_background_count=143, protected_source_union_rehashed=True,
                    source_file_count=len(self.source_bindings), elapsed_seconds=time.perf_counter()-started)


def snapshot_original(scene, anchor, *, profile_pin, diagnostic_dir=None):
    return PersistentForeground(scene, anchor, profile_pin=profile_pin, diagnostic_dir=diagnostic_dir)
