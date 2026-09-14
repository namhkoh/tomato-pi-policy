"""Keep startup failures observable even when SimulationApp never returns."""
import json
import traceback
from datetime import datetime,timezone


def checkpoint(output,report,phase):
    """Exclusive diagnostic snapshot before/after an owned blocking startup API.

    Separate from report.json: neither an initializing snapshot nor successful
    parsing grants physical task authority. No record or decision is mutated.
    """
    if phase not in ('before_native_parse','before_reset','after_reset'):
        raise ValueError('Known bounded startup phase required')
    payload=dict(state='startup_checkpoint_not_qualification',phase=phase,
        observed_utc=datetime.now(timezone.utc).isoformat(),motion_authorized=False,
        training_eligible=False,report_snapshot=report)
    encoded=json.dumps(payload,indent=2,allow_nan=False)
    with (output/('startup_'+phase+'.json')).open('x',encoding='utf-8') as stream:
        stream.write(encoded)
    print('PHYSICS_STARTUP_CHECKPOINT '+phase,flush=True)


def start(factory,configuration,output,report):
    try:
        return factory(configuration)
    except Exception:
        report.update(state='failed_application_startup',error=traceback.format_exc(),
            task_scene_constructed=False,training_eligible=False)
        # Benchmark already requires a new, exclusively owned output directory.
        # Never turn a failed engine startup into an empty or successful run.
        with (output/'report.json').open('x',encoding='utf-8') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        raise
