"""Package only a normally validated complete release; never waive QA gates."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import os
import shutil
import zipfile

from .dataset_review import read_json,require,safe_file
from .depth_preview import sha256
from .training_export import validate


def archive(release,destination):
    release=Path(release).resolve();destination=Path(destination).resolve()
    checksum=destination.with_suffix(destination.suffix+'.sha256')
    partial=destination.with_suffix(destination.suffix+'.partial')
    require(destination.suffix.lower()=='.zip','Expected .zip output')
    require(not destination.is_relative_to(release),'Keep archive outside release')
    require(all(not p.exists() for p in (destination,checksum,partial)),'Never overwrite an archive/checksum')
    # No incomplete flag. Incomplete engineering previews fail before writing.
    validation=validate(release)
    manifest=read_json(release/'manifest.json')
    expected={**manifest['files_sha256'],'manifest.json':sha256(release/'manifest.json')}
    for name in expected:
        path=PurePosixPath(name)
        require(not path.is_absolute() and '\\' not in name and '..' not in path.parts
            and path.as_posix()==name,'Nonportable archive member')
        safe_file(release,name)
    destination.parent.mkdir(parents=True,exist_ok=True)
    # A failed/interrupted build remains visibly .partial, never a final ZIP.
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as bundle:
        for name in sorted(expected):
            with safe_file(release,name).open('rb') as source,bundle.open('grounding_release/'+name,'w',force_zip64=True) as target:
                shutil.copyfileobj(source,target,1024*1024)
    # Validate the bytes IN the archive (also checks ZIP CRC on stream reads),
    # not merely hashes of a source file which may have changed during copy.
    with zipfile.ZipFile(partial) as bundle:
        require(set(bundle.namelist())=={'grounding_release/'+p for p in expected},'Archive member mismatch')
        for name,wanted in expected.items():
            digest=hashlib.sha256()
            with bundle.open('grounding_release/'+name) as source:
                for chunk in iter(lambda:source.read(1024*1024),b''): digest.update(chunk)
            require(digest.hexdigest()==wanted,'Archive source changed during copy: '+name)
    digest=sha256(partial)
    # Exclusive hard-link publication on the same filesystem does not clobber
    # an output created by another process after our initial existence check.
    os.link(partial,destination)
    partial.unlink()  # only our own completed temporary archive
    with checksum.open('x',encoding='ascii') as stream:
        stream.write(digest+'  '+destination.name+'\n')
    return dict(state='complete_release_archive',archive=str(destination),sha256=digest,
        checksum=str(checksum),bytes=destination.stat().st_size,rows=validation['rows'],
        normal_release_validator_passed=True,archive_members_hash_verified=True,training_started=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(archive(args.release,args.output),indent=2),flush=True)


if __name__=='__main__': main()
