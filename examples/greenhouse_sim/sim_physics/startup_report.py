"""Keep startup failures observable even when SimulationApp never returns."""
import json
import traceback


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
