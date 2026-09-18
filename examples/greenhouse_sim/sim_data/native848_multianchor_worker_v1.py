"""Same native bootstrap and frozen per-frame capture for authenticated multi-anchor plans."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import shutil
import traceback
from . import native848_multianchor_plan_v1 as api
from . import native848_bulk_worker_v1 as producer
from .native848_bulk_io_v1 import save_json
from .dataset_review import require,read_json
from .depth_preview import sha256
FROZEN_PRODUCER_SHA='5306fe59b579aa8e7bfc021ac39639ef7e0e62b06a851b12a6ab79a96daa204a'

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    require(sha256(producer.__file__)==FROZEN_PRODUCER_SHA,'Frozen per-frame renderer changed')
    require(sha256(args.plan)==args.plan_sha256,'Changed bulk plan')
    plan=read_json(args.plan);output=args.output.resolve()
    require(plan['schema']==api.SCHEMA and not output.exists(),'New bulk output required')
    from sim_physics.host_memory import preflight
    from .native_generated_pair import windows_worker_admission
    memory=preflight();disk=shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,
        'Native memory/disk reserves unavailable')
    process=windows_worker_admission(None)
    output.mkdir(parents=True)
    save_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(args.plan.resolve()),plan_sha256=args.plan_sha256,host_memory_preflight=memory,
        available_disk_bytes=disk,process_admission=process,automatic_retries=False,training_approved=False))
    app=None;succeeded=False
    try:
        # No plan evaluation or native USD imports before SimulationApp owns ABI.
        from isaacsim import SimulationApp
        require(sha256(plan['profile_evidence']['path'])==plan['profile_evidence']['sha256'],'Changed actual profile')
        profile=read_json(plan['profile_evidence']['path'])
        if profile['schema']==api.ORIGINAL_PROFILE_SCHEMA:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
                sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false'])
        else:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RealTimePathTracing',
                anti_aliasing=2,sync_loads=False,disable_viewport_updates=True,
                extra_args=['--/app/settings/persistent=false','--/rtx/rtpt/enabled=true'])
        app=SimulationApp(config)
        checked=api.check(plan)
        result=producer.capture(app,output,args.plan,plan,checked)
        require(sha256(args.plan)==args.plan_sha256,'Bulk plan changed during capture')
        result['plan_sha256']=args.plan_sha256
        save_json(output/'result.json',result);succeeded=True
    except BaseException:
        save_json(output/'failure.json',dict(state='original848_bulk_failed',error=traceback.format_exc(),
            automatic_retries=False,training_approved=False,accepted_training_increment=0))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__=='__main__':
    main()
