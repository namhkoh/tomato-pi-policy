"""CPU-only exact TRAIN67/36 far-component packing experiment; no native API.

The original collision evidence remains immutable. This creates an explicit
render representation with coarse far-group ownership and complete face lineage;
it cannot be consumed by the existing native/component annotation contracts.
"""
from pathlib import Path
from collections import defaultdict,Counter
from copy import deepcopy
import hashlib,json,time,traceback
import numpy as np
import psutil
from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf,Vt
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
from .native848_bulk_io_v1 import save_json
from . import native848_fully_labeled_coverage_v2 as coverage
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from .cut_regions import _oriented_chain,_sample

D=Path('D:/research/tomato-pi-policy/data/sim_data/diagnostics')
OUT=D/'native848_far_component_merge67_36_CPU_20260917_v1'
B=D/'native848_instanced_full144_CPU_parity_20260917_v3'
CPU=D/'native848_reset6_production_train67_36_prepare_20260917_v1/cpu_preflight.json'
PLAN=CPU.parent/'plan.json'
EPS=1e-6
PINS={str(CPU):'f9702b2a4982792f444f94a078f0813a434250543ca007d370b36d3971d149b8',
    str(PLAN):'5280df09194d841774acf8c984f996fc980e027ef43b7fafd28e7be3f520e4cc',
    str(B/'ordinary_census.json'):'0fca928b34b70ff701302b6f7d140156cdd4a247acadb3414cae2c49d056d12c',
    str(B/'ordinary_catalogue.json'):'e98bdc1224d6bf69ae5c5af5b0e6034fa8c75501c1b9b7a2fd10226fb4aba8da',
    str(B/'ordinary_meshes.json'):'0fd69b2d220d8060a148cd73398ae6dc21ce2783411c0cc2d7675772fef7e9c7',
    str(B/'ordinary_bounds.json'):'c239bc9e10fdd27abc60b3e94ab5bf5f0a70af75e9471cdbf6873cd74c58c2a8'}


def array_pin(value):
    a=np.asarray(value);require(a.dtype.kind in 'biuf' and np.isfinite(a).all(),'Finite numeric array required')
    return dict(dtype=a.dtype.str,shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def budget(start):
    require(time.perf_counter()-start<1800,'Finite thirty-minute CPU prototype budget expired')
    require(not (OUT/'STOP').exists(),'CPU prototype STOP requested')
    require(psutil.virtual_memory().available>=6*2**30,'CPU prototype physical headroom below6GiB')


def selected_components(cpu,census,catalogue,meshes,bindings):
    paths={r['prim_path']:i for i,r in enumerate(catalogue)};byid={(r['variant_id'],r['component_id']):i for i,r in enumerate(catalogue)}
    low=np.full((len(catalogue),3),np.inf);high=-low.copy();owned=defaultdict(list)
    for mesh in meshes:
        path=mesh['path']
        while path not in paths and path!='/':path=path.rsplit('/',1)[0] or '/'
        require(path in paths,'Mesh outside complete144 component catalogue');i=paths[path]
        low[i]=np.minimum(low[i],mesh['world_min']);high[i]=np.maximum(high[i],mesh['world_max']);owned[i].append(mesh)
    require(len(owned)==len(catalogue),'Logical component missing original mesh evidence')
    checker=WorkspaceChecker();bounds=[]
    try:
        for row in cpu['poses']:
            proof=row['proof'];snap=deepcopy(proof['actual_robot_snapshot']);require(proof['screen']['passed'] is True,'Original CPU collision screen failed')
            snap['visual_bound_screen']=proof['screen']
            bounds.append(coverage.robot_bounds(dict(robot_snapshot=snap,calibration=proof['actual_calibration']),checker))
        bindings.update(checker.bindings)
    finally:checker.finish()
    centers=np.asarray([v['shoulder_world_m'] for b in bounds for v in b['arms'].values()])
    radii=np.asarray([v['conservative_probe_reach_m']+b['margin_m'] for b in bounds for v in b['arms'].values()])
    clear=np.full(len(catalogue),np.inf)
    for center,radius in zip(centers,radii):
        clear=np.minimum(clear,np.linalg.norm(np.maximum(np.maximum(low-EPS-center,center-high-EPS),0),axis=1)-radius)
    far=clear>0;plants={r['variant_id']:r for r in census['all_plant_roots']};manifests={};children={}
    for plant in plants.values():
        family=plant['source_family']
        if family in manifests:continue
        mp=Path(plant['manifest_path']);require(sha256(mp)==plant['manifest_sha256'],'Changed source manifest')
        bindings[str(mp)]=plant['manifest_sha256'];cs=read_json(mp)['components'];manifests[family]={c['id']:c for c in cs};children[family]=defaultdict(list)
        for c in cs:children[family][c['parent']].append(c['id'])
    protected=set(np.flatnonzero(~far).tolist());petiole_proofs={};unknown=[]
    for i,row in enumerate(catalogue):
        if row['organ_type']!='sub_stem':continue
        plant=plants[row['variant_id']];family=plant['source_family'];src=manifests[family][row['component_id']];pointfar=False
        try:
            component=dict(type=src['type'],deleafed=src.get('deleafed'),translation_plant_m=src['transform']['translate'],attachment_plant_m=src['attach_point'],axis_plant=src['axis'],capsules_local_m=src['capsules'])
            chain,lengths,_,_=_oriented_chain(component,1e-6);require(lengths[-1]>=.030,'Petiole too short')
            point=_sample(chain,lengths,.009,component['translation_plant_m'])['point_plant_m']
            world=(np.asarray(plant['plant_to_world_usd_row_vectors']).T@np.r_[point,1])[:3]
            distances=np.linalg.norm(centers-world,axis=1);pointfar=bool((distances>radii+EPS).all())
            if pointfar:petiole_proofs[str(i)]=dict(nominal_arc_m=.009,nominal_world_m=world.tolist(),minimum_clearance_m=float((distances-radii-EPS).min()),every_pose_both_arms=True,visibility_assumed=False)
        except (ValueError,KeyError,TypeError) as exc:unknown.append(dict(component_index=i+1,reason=str(exc)))
        if pointfar and far[i]:continue
        pending=[row['component_id']]
        if src['parent'] in manifests[family]:protected.add(byid[(row['variant_id'],src['parent'])])
        while pending:
            key=pending.pop();protected.add(byid[(row['variant_id'],key)]);pending.extend(children[family][key])
    primary=[p['variant_id'] for p in plants.values() if p['decision']=='retained_exact_foreground'];require(len(primary)==1,'Unique original foreground required')
    protected.update(i for i,r in enumerate(catalogue) if r['variant_id']==primary[0])
    selected={i for i in range(len(catalogue)) if i not in protected};require(all(far[i] for i in selected),'Unproved merged component')
    require(all(str(i) in petiole_proofs for i in selected if catalogue[i]['organ_type']=='sub_stem'),'Merged petiole lacks positive exact9mm exclusion')
    proof=dict(schema='greenhouse.native848_far_component_selection_CPU.v1',exact_plan=dict(path=str(PLAN),sha256=PINS[str(PLAN)]),
        pose_count=len(bounds),all_pose_robot_bounds=bounds,world_point_error_bound_m=EPS,component_AABBs_expanded_per_axis_m=EPS,
        selected_components=[dict(**catalogue[i],original_world_min=low[i].tolist(),original_world_max=high[i].tolist(),
            minimum_expanded_AABB_clearance_m=float(clear[i]),petiole_exclusion=petiole_proofs.get(str(i))) for i in sorted(selected)],
        protected_component_indices=[i+1 for i in sorted(protected)],unknown_centerline_geometry_retained=unknown,
        original_foreground=primary[0],all_unknown_near_parent_descendant_dependencies_retained=True,native_authorized=False)
    return selected,owned,plants,manifests,proof


def material_value(value,material_path,bindings):
    if isinstance(value,Sdf.AssetPath):
        require(bool(value.resolvedPath),'Unresolved material asset cannot be preserved')
        p=Path(value.resolvedPath).resolve();require(p.is_file(),'Material asset missing');h=bindings.get(str(p))
        if h is None:h=sha256(p);bindings[str(p)]=h
        return dict(resolved_asset_path=str(p),sha256=h)
    if isinstance(value,Sdf.Path):
        require(value.HasPrefix(material_path),'External material connection unsupported');return '#MAT'+str(value)[len(str(material_path)):]
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,dict):return {str(k):material_value(v,material_path,bindings) for k,v in value.items()}
    try:return [material_value(v,material_path,bindings) for v in value]
    except TypeError:return str(value)


def graph(material,bindings):
    root=material.GetPath();rows=[]
    for prim in Usd.PrimRange(material.GetPrim()):
        attrs=[];rels=[]
        for attr in prim.GetAttributes():
            require(attr.GetNumTimeSamples()==0,'Animated material unsupported')
            if not attr.HasAuthoredValueOpinion() and not attr.GetConnections():continue
            attrs.append(dict(name=attr.GetName(),type=str(attr.GetTypeName()),value=material_value(attr.Get(),root,bindings),
                connections=[material_value(v,root,bindings) for v in attr.GetConnections()],custom=attr.IsCustom(),
                metadata=material_value({k:v for k,v in attr.GetAllAuthoredMetadata().items() if k not in ('default','connectionPaths')},root,bindings)))
        for rel in prim.GetRelationships():
            if rel.GetTargets():rels.append(dict(name=rel.GetName(),targets=[material_value(v,root,bindings) for v in rel.GetTargets()]))
        rows.append(dict(path=material_value(prim.GetPath(),root,bindings),type=prim.GetTypeName(),attrs=attrs,relationships=rels,
            metadata=material_value(prim.GetAllAuthoredMetadata(),root,bindings)))
    return rows


class Materials:
    def __init__(self,output,bindings):
        self.path=output/'materials.usdc';self.stage=Usd.Stage.CreateNew(str(self.path));UsdGeom.Scope.Define(self.stage,'/Materials')
        self.bindings=bindings;self.known={};self.proofs={}
    def add(self,material):
        before=graph(material,self.bindings);key=fingerprint(before)
        if key in self.known:return key
        old=material.GetPath();new=Sdf.Path('/Materials/M_'+key);flat=material.GetPrim().GetStage().Flatten()
        require(Sdf.CopySpec(flat,old,self.stage.GetRootLayer(),new),'Cannot copy complete material subtree')
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(new)):
            for attr in prim.GetAttributes():
                value=attr.Get()
                if isinstance(value,Sdf.AssetPath):
                    source=material.GetPrim().GetStage().GetAttributeAtPath(Sdf.Path(str(attr.GetPath()).replace(str(new),str(old),1)))
                    original=source.Get();require(original.resolvedPath,'Cannot resolve original copied material asset')
                    attr.Set(Sdf.AssetPath(str(Path(original.resolvedPath).resolve())))
                connections=attr.GetConnections()
                if connections:
                    require(all(p.HasPrefix(old) or p.HasPrefix(new) for p in connections),'External material link unsupported');attr.SetConnections([p.ReplacePrefix(old,new) if p.HasPrefix(old) else p for p in connections])
            for rel in prim.GetRelationships():
                targets=rel.GetTargets()
                if targets:
                    require(all(p.HasPrefix(old) or p.HasPrefix(new) for p in targets),'External material relation unsupported');rel.SetTargets([p.ReplacePrefix(old,new) if p.HasPrefix(old) else p for p in targets])
        after=graph(UsdShade.Material(self.stage.GetPrimAtPath(new)),self.bindings)
        require(after==before,'Material graph or resolved texture changed during copy')
        self.known[key]=str(new);self.proofs[key]=dict(source_material_path=str(old),canonical_graph=before,copied_graph_exact=True)
        return key
    def finish(self):self.stage.GetRootLayer().Save()


def load_sources(catalogue,selected,owned,plants,manifests,cpu,materials,bindings,start):
    # A donor component is loaded once; every actual clone still validates its
    # saved world matrix/path and receives its own face provenance below.
    import test_native848_instanced_labeled_scene_v3 as audit
    bindings[str(Path(audit.__file__).resolve())]=sha256(audit.__file__)
    needed=sorted({(catalogue[i]['source_plant_id'],catalogue[i]['component_id']) for i in selected});cache={}
    familyplants={p['source_family']:p for p in plants.values()}
    for j,(family,cid) in enumerate(needed):
        if j%100==0:budget(start)
        src=manifests[family][cid];asset=(Path(familyplants[family]['manifest_path']).parent/src['file']).resolve()
        expected=cpu['source_bindings'].get(str(asset));require(expected is not None and sha256(asset)==expected,'Source asset unbound or changed');bindings[str(asset)]=expected
        stage=Usd.Stage.Open(str(asset));root=stage.GetDefaultPrim().GetPath();parts=[]
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):continue
            mesh=UsdGeom.Mesh(prim)
            require(all(a.GetNumTimeSamples()==0 for a in prim.GetAttributes()),'Animated source mesh unsupported')
            require(mesh.GetSubdivisionSchemeAttr().Get()=='none' and mesh.GetOrientationAttr().Get()=='rightHanded'
                and mesh.GetDoubleSidedAttr().Get() is True and mesh.GetNormalsInterpolation()=='faceVarying'
                and not list(mesh.GetHoleIndicesAttr().Get() or []) and not UsdGeom.Subset.GetAllGeomSubsets(mesh),
                'Source topology/normal/material layout unsupported')
            points=np.asarray(mesh.GetPointsAttr().Get(),np.float32);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),np.int32)
            indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),np.int32);normals=np.asarray(mesh.GetNormalsAttr().Get(),np.float32)
            require(len(normals)==int(counts.sum())==len(indices) and np.isfinite(points).all() and np.isfinite(normals).all(),'Complete finite face-corner normals required')
            authored=[v for v in UsdGeom.PrimvarsAPI(prim).GetPrimvars() if v.HasValue()]
            require(len(authored)==1 and authored[0].GetName()=='primvars:st' and authored[0].GetInterpolation()=='faceVarying'
                and not authored[0].IsIndexed() and authored[0].GetElementSize()==1,'Only exact unindexed face-varying UV layout supported')
            uv=np.asarray(authored[0].Get(),np.float32);require(uv.shape==(len(indices),2) and np.isfinite(uv).all(),'UV count differs')
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial();require(bool(material),'Missing source material')
            topology={name:audit.usd_value(prim.GetAttribute(name).Get()) for name in ['faceVertexCounts','faceVertexIndices','holeIndices','subdivisionScheme','faceVaryingLinearInterpolation','interpolateBoundary','triangleSubdivisionRule','cornerIndices','cornerSharpnesses','creaseIndices','creaseLengths','creaseSharpnesses','orientation','doubleSided','normals']}
            topology['normalsInterpolation']=str(mesh.GetNormalsInterpolation())
            primvars={str(p.GetAttr().GetName()):dict(value=audit.usd_value(p.Get()),interpolation=str(p.GetInterpolation()),indices=audit.usd_value(p.GetIndices())) for p in UsdGeom.PrimvarsAPI(prim).GetPrimvars()}
            parts.append(dict(relative_path=str(prim.GetPath())[len(str(root)):],source_asset=str(asset),source_asset_sha256=expected,
                points=points,counts=counts,indices=indices,normals=normals,uv=uv,material=materials.add(material),
                points_pin=audit.array_hash(points,np.float64),topology_sha256=fingerprint(topology),primvars_sha256=fingerprint(primvars),
                normals_pin=array_pin(normals),uv_pin=array_pin(uv),source_material_path=str(material.GetPath()),default_root=str(root)))
        require(parts,'Source component has no meshes');cache[(family,cid)]=parts
        if (j+1)%500==0:print(json.dumps(dict(stage='source_mesh_material_validation',donor_components=j+1,total=len(needed),seconds=time.perf_counter()-start)),flush=True)
    materials.finish();return cache


def pack_plant(plant,entries,selected,owned,cache,manifests,materials,folder):
    root='/Plant';path=folder/(plant['plant_root'].rsplit('/',1)[1]+'.usdc');stage=Usd.Stage.CreateNew(str(path));rp=UsdGeom.Xform.Define(stage,root).GetPrim();stage.SetDefaultPrim(rp)
    family=plant['source_family'];manifest=manifests[family];plant_matrix=np.asarray(plant['plant_to_world_usd_row_vectors'],float).T
    require(np.array_equal(plant_matrix[:3,:3],np.eye(3)),'Prototype only supports original translation-only slots')
    by={row['component_id']:(i,row) for i,row in entries};original_prefix=plant['plant_root'];point_parts=[];count_parts=[];index_parts=[];normal_parts=[];uv_parts=[]
    facegroups=defaultdict(list);provenance=[];point_start=face_start=corner_start=0;maxerror=0.;selected_mesh_paths=set();near_meshes=0
    for i,row in sorted(entries,key=lambda item:(item[1]['prim_path'].count('/'),item[1]['prim_path'])):
        src=manifest[row['component_id']];relative=row['prim_path'][len(original_prefix):];prim=UsdGeom.Xform.Define(stage,root+relative)
        parent=src['parent'];offset=np.asarray(src['transform']['translate'],float)-(np.asarray(manifest[parent]['transform']['translate'],float) if parent else 0)
        prim.AddTranslateOp().Set(Gf.Vec3d(*offset));asset=(Path(plant['manifest_path']).parent/src['file']).resolve();prim.GetPrim().GetReferences().AddReference(asset.as_posix())
        if i not in selected:near_meshes+=len(owned[i]);continue
        parts=cache[(family,row['component_id'])];expected={m['path']:m for m in owned[i]}
        require(len(parts)==len(expected),'Source and actual mesh count differ')
        for part in parts:
            actual_path=row['prim_path']+part['relative_path'];require(actual_path in expected,'Exact original component mesh path missing');old=expected[actual_path]
            require(old['points']==part['points_pin'] and old['topology_sha256']==part['topology_sha256'] and old['primvars_sha256']==part['primvars_sha256'],'Original full-array parity cache differs')
            expected_material=row['prim_path']+part['source_material_path'][len(part['default_root']):]
            require(old['material_targets']==[expected_material] and old['purpose']=='default' and old['visibility']=='inherited','Actual material/visibility scope differs')
            world=np.asarray(old['world_matrix'],float).T;require(np.array_equal(world[:3,:3],np.eye(3)),'Normal-preserving translation-only mesh required')
            source_world=part['points'].astype(np.float64)+world[:3,3]
            packed=(part['points'].astype(np.float64)+world[:3,3]-plant_matrix[:3,3]).astype(np.float32)
            new_world=packed.astype(np.float64)+plant_matrix[:3,3];error=float(np.linalg.norm(new_world-source_world,axis=1).max());require(error<=EPS,'World point bake error exceeds explicit1micrometre bound');maxerror=max(maxerror,error)
            require(np.all(new_world.min(0)>=np.asarray(old['world_min'])-EPS) and np.all(new_world.max(0)<=np.asarray(old['world_max'])+EPS),
                'Packed source span escapes conservative original collision bounds')
            nfaces=len(part['counts']);npoints=len(packed);ncorners=len(part['indices'])
            point_parts.append(packed);count_parts.append(part['counts']);index_parts.append(part['indices']+point_start);normal_parts.append(part['normals']);uv_parts.append(part['uv'])
            facegroups[part['material']].extend(range(face_start,face_start+nfaces))
            provenance.append(dict(original_component_index=i+1,original_component_id=row['component_id'],original_mesh_path=actual_path,
                source_asset=part['source_asset'],source_asset_sha256=part['source_asset_sha256'],source_world_matrix=old['world_matrix'],
                points_begin=point_start,point_count=npoints,faces_begin=face_start,face_count=nfaces,corners_begin=corner_start,corner_count=ncorners,
                original_points=part['points_pin'],original_topology_sha256=part['topology_sha256'],original_primvars_sha256=part['primvars_sha256'],
                original_normals=part['normals_pin'],original_uv=part['uv_pin'],material_graph_sha256=part['material'],
                world_coordinate_max_euclidean_error_m=error,original_world_bounds=dict(min=old['world_min'],max=old['world_max']),
                packed_span_world_bounds=dict(min=new_world.min(0).tolist(),max=new_world.max(0).tolist())))
            point_start+=npoints;face_start+=nfaces;corner_start+=ncorners
            mesh_prim=stage.GetPrimAtPath(root+relative+part['relative_path']);require(mesh_prim.IsA(UsdGeom.Mesh),'Source reference mesh absent');mesh_prim.SetActive(False);selected_mesh_paths.add(actual_path)
    if provenance:
        packed_mesh=UsdGeom.Mesh.Define(stage,'/Plant/__FarCoarseGroup_0');points=np.concatenate(point_parts);counts=np.concatenate(count_parts);indices=np.concatenate(index_parts);normals=np.concatenate(normal_parts);uv=np.concatenate(uv_parts)
        packed_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points));packed_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(counts));packed_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices))
        packed_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals));packed_mesh.SetNormalsInterpolation('faceVarying');packed_mesh.GetSubdivisionSchemeAttr().Set('none');packed_mesh.GetOrientationAttr().Set('rightHanded');packed_mesh.GetDoubleSidedAttr().Set(True)
        packed_mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)],np.float32)))
        pv=UsdGeom.PrimvarsAPI(packed_mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying');pv.Set(Vt.Vec2fArray.FromNumpy(uv))
        for key,faces in sorted(facegroups.items()):
            mat=UsdShade.Material.Define(stage,'/Plant/__Materials/M_'+key);mat.GetPrim().GetReferences().AddReference(materials.path.as_posix(),materials.known[key])
            subset=UsdShade.MaterialBindingAPI(packed_mesh).CreateMaterialBindSubset('M_'+key,Vt.IntArray(faces));UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
        UsdShade.MaterialBindingAPI(packed_mesh).SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
        require(sum(map(len,facegroups.values()))==face_start,'Material face partition incomplete')
        packed_mesh.GetPrim().CreateAttribute('prototype:coarseOwnership',Sdf.ValueTypeNames.String,custom=True).Set('certified_far_original_components_complete_face_provenance')
    stage.GetRootLayer().Save()
    # Verify the serialized artifact, not only arrays before authoring.
    stage=None;verified=Usd.Stage.Open(str(path));active=[p for p in verified.Traverse() if p.IsA(UsdGeom.Mesh)]
    require(len(active)==near_meshes+bool(provenance),'Resulting visible mesh inventory differs')
    if provenance:
        m=UsdGeom.Mesh(verified.GetPrimAtPath('/Plant/__FarCoarseGroup_0'))
        require(np.array_equal(np.asarray(m.GetPointsAttr().Get()),points) and np.array_equal(np.asarray(m.GetFaceVertexCountsAttr().Get()),counts)
            and np.array_equal(np.asarray(m.GetFaceVertexIndicesAttr().Get()),indices) and np.array_equal(np.asarray(m.GetNormalsAttr().Get()),normals)
            and np.array_equal(np.asarray(UsdGeom.PrimvarsAPI(m).GetPrimvar('st').Get()),uv),'Serialized topology/points/normals/UV changed')
        subsets=UsdShade.MaterialBindingAPI(m).GetMaterialBindSubsets();require(len(subsets)==len(facegroups),'Material subset count differs')
        for subset in subsets:
            mat,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial();key=subset.GetPrim().GetName()[2:]
            require(np.array_equal(np.asarray(subset.GetIndicesAttr().Get()),facegroups[key]) and graph(mat,materials.bindings)==materials.proofs[key]['canonical_graph'],'Serialized material graph/face assignment changed')
    pp=folder/(path.stem+'.provenance.json');save_json(pp,dict(schema='greenhouse.native848_far_render_group_provenance.v1',plant=plant,
        merged_mesh_path='/Plant/__FarCoarseGroup_0' if provenance else None,faces=provenance,source_meshes_deactivated_only_after_geometry_replacement=sorted(selected_mesh_paths),
        logical_components_preserved=True,collision_original_representation_unchanged=True,native_component_ID_qualification_pending=True))
    return dict(plant_root=plant['plant_root'],artifact=dict(path=str(path),sha256=sha256(path)),provenance=dict(path=str(pp),sha256=sha256(pp)),
        merged_original_meshes=len(provenance),render_meshes=len(active),coarse_groups=int(bool(provenance)),material_draw_groups=len(facegroups),
        point_count=point_start,face_count=face_start,max_world_coordinate_euclidean_error_m=maxerror,normal_values_and_UVs_exact=True)


def run():
    start=time.perf_counter();require(not OUT.exists(),'Create-only prototype output required');verify_bindings(PINS);OUT.mkdir();bindings=dict(PINS)
    for path in (__file__,coverage.__file__):bindings[str(Path(path).resolve())]=sha256(path)
    try:
        budget(start);cpu=read_json(CPU);plan=read_json(PLAN);require(cpu['plan_sha256']==PINS[str(PLAN)] and plan['max_frames']==len(cpu['poses'])==36,'Exact finite36 scene only')
        require(cpu['full_scene_census']['sha256']==PINS[str(B/'ordinary_census.json')] and cpu['all_selected_pose_screens_passed'] is True,'CPU scene/geometry mismatch')
        census=read_json(B/'ordinary_census.json');catalogue=read_json(B/'ordinary_catalogue.json');meshes=read_json(B/'ordinary_meshes.json')
        selected,owned,plants,manifests,proof=selected_components(cpu,census,catalogue,meshes,bindings);save_json(OUT/'selection.json',proof)
        print(json.dumps(dict(stage='selection',selected_components=len(selected),logical_components=len(catalogue),protected_components=len(catalogue)-len(selected))),flush=True)
        materials=Materials(OUT,bindings);cache=load_sources(catalogue,selected,owned,plants,manifests,cpu,materials,bindings,start)
        save_json(OUT/'material_graph_provenance.json',materials.proofs);folder=OUT/'plants';folder.mkdir();entries=defaultdict(list)
        for i,row in enumerate(catalogue):entries[row['variant_id']].append((i,row))
        results=[]
        for n,plant in enumerate(sorted(plants.values(),key=lambda p:p['plant_root'])):
            budget(start);results.append(pack_plant(plant,entries[plant['variant_id']],selected,owned,cache,manifests,materials,folder))
            if (n+1)%6==0:print(json.dumps(dict(stage='packed_plants',plants=n+1,total=144,seconds=time.perf_counter()-start)),flush=True)
        population=Usd.Stage.CreateNew(str(OUT/'population_render_only.usda'));UsdGeom.Xform.Define(population,'/World');UsdGeom.Xform.Define(population,'/World/PackPlants')
        byroot={p['plant_root']:p for p in plants.values()}
        for entry in results:
            x=UsdGeom.Xform.Define(population,entry['plant_root']);x.AddTransformOp().Set(Gf.Matrix4d(byroot[entry['plant_root']]['plant_to_world_usd_row_vectors']))
            x.GetPrim().GetReferences().AddReference(entry['artifact']['path'],'/Plant')
        population.GetRootLayer().Save();require(len(population.GetPrimAtPath('/World/PackPlants').GetChildren())==144,'All144 slot roots required')
        actual_meshes=sum(p.IsA(UsdGeom.Mesh) for p in population.Traverse());require(actual_meshes==sum(r['render_meshes'] for r in results),'Actual composed render mesh count differs')
        verify_bindings(bindings);require(sha256(B/'ordinary_bounds.json')==PINS[str(B/'ordinary_bounds.json')],'Original collision cache changed')
        result=dict(schema='greenhouse.native848_far_component_merge_CPU_prototype.v1',passed=True,exact_plan=dict(path=str(PLAN),sha256=PINS[str(PLAN)]),
            population_scope='all144_complete_plants_render_overlay_only_existing_greenhouse_robot_and_original_collision_representation_retained_separately',
            original_meshes=len(meshes),actual_composed_render_meshes=actual_meshes,coarse_render_groups=sum(r['coarse_groups'] for r in results),
            material_draw_groups_for_merged_meshes=sum(r['material_draw_groups'] for r in results),unique_material_graphs=len(materials.known),
            actual_GPU_draw_call_count_not_measured=True,logical_components=len(catalogue),merged_components=len(selected),plants=results,
            max_world_coordinate_euclidean_error_m=max(r['max_world_coordinate_euclidean_error_m'] for r in results),permitted_world_coordinate_error_m=EPS,
            geometry_bitwise_equal=False,topology_no_welding_no_decimation=True,normal_and_UV_values_exact=True,material_graphs_and_resolved_textures_exact=True,
            original_collision_cache_unchanged=True,original36_CPU_collision_proof_reused_not_recomputed=True,each_packed_source_span_within_original_bounds_expanded_m=EPS,
            source_bindings=bindings,output_population=dict(path=str(OUT/'population_render_only.usda'),sha256=sha256(OUT/'population_render_only.usda')),
            selection=dict(path=str(OUT/'selection.json'),sha256=sha256(OUT/'selection.json')),materials=dict(path=str(materials.path),sha256=sha256(materials.path)),
            material_provenance=dict(path=str(OUT/'material_graph_provenance.json'),sha256=sha256(OUT/'material_graph_provenance.json')),
            native_capture_authorized=False,native_ID_and_RGBD_qualification_pending=True,current_annotation_export_eligible=False,
            training_approved=False,accepted_training_increment=0,elapsed_seconds=time.perf_counter()-start)
        save_json(OUT/'result.json',result);print(json.dumps(dict(stage='COMPLETE',result_sha256=sha256(OUT/'result.json'),render_meshes=actual_meshes,max_error_m=result['max_world_coordinate_euclidean_error_m'],seconds=result['elapsed_seconds'])),flush=True)
        return result
    except BaseException:
        save_json(OUT/'failure.json',dict(error=traceback.format_exc(),source_bindings=bindings,native_launched=False,partial_artifacts_not_qualified=True,elapsed_seconds=time.perf_counter()-start));raise


if __name__=='__main__':run()
