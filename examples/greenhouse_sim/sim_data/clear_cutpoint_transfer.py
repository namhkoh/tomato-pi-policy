"""Transfer data + matching lightweight code, preserving draft/ready status.

Unlike release_archive, an explicit --inspection-only can transport a draft.
It NEVER changes the dataset state or makes the trainer accept it.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from .clear_cutpoint_release import validate
from .dataset_review import read_json,require,safe_file
from .depth_preview import sha256


def package(dataset,output,*,inspection_only=False):
    dataset,output=Path(dataset).resolve(),Path(output).resolve()
    require(output.suffix=='.zip' and not output.is_relative_to(dataset),'ZIP outside dataset required')
    if inspection_only:require(output.name.endswith('.inspection.zip'),'Draft transport must be named .inspection.zip')
    partial=output.with_suffix('.zip.partial');checksum=output.with_suffix('.zip.sha256')
    require(not any(p.exists() for p in (output,partial,checksum)),'Choose new archive paths')
    validation=validate(dataset,allow_draft=inspection_only)
    manifest=read_json(dataset/'manifest.json')
    expected={**manifest['files_sha256'],'manifest.json':sha256(dataset/'manifest.json')}
    entries={f'grounding_release/{p}':safe_file(dataset,p) for p in expected}
    hashes={f'grounding_release/{p}':h for p,h in expected.items()}
    code=Path(__file__).parent
    for path in sorted(code.iterdir()):
        if path.is_file() and path.suffix in ('.py','.md','.json'):
            name='code/examples/greenhouse_sim/sim_data/'+path.name;entries[name]=path;hashes[name]=sha256(path)
    entries['README_TRAINING.md']=code/'CLEAR_CUTPOINT.md';hashes['README_TRAINING.md']=sha256(entries['README_TRAINING.md'])
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=code,capture_output=True,text=True,check=True).stdout.strip()
    receipt=dict(schema='greenhouse.clear_transfer.v1',git_commit=commit,validation=validation,
                 training_ready=manifest['state']=='complete_clear_cutpoint_release',inspection_only=inspection_only,
                 source_manifest_sha256=sha256(dataset/'manifest.json'),files_sha256=hashes,
                 no_model_weights=True,no_training_started=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(partial,'x',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for name,path in entries.items():z.write(path,name)
        z.writestr('TRANSFER_STATUS.json',json.dumps(receipt,indent=2))
    with zipfile.ZipFile(partial) as z:
        require(set(z.namelist())==set(entries)|{'TRANSFER_STATUS.json'},'Archive member mismatch')
        for name,wanted in hashes.items():
            h=hashlib.sha256()
            with z.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            require(h.hexdigest()==wanted,'Archive content changed: '+name)
        require(json.loads(z.read('TRANSFER_STATUS.json'))==receipt,'Transfer status changed')
    digest=sha256(partial);os.link(partial,output);partial.unlink()
    with checksum.open('x',encoding='ascii') as f:f.write(digest+'  '+output.name+'\n')
    return dict(archive=str(output),sha256=digest,bytes=output.stat().st_size,
                training_ready=receipt['training_ready'],git_commit=commit)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--inspection-only',action='store_true')
    a=p.parse_args(argv);print(json.dumps(package(a.dataset,a.output,inspection_only=a.inspection_only),indent=2))


if __name__=='__main__':main()
