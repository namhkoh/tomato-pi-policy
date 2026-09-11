from pathlib import Path
import json
from .benchmark import parser,report_configuration


def test_proposal_path_survives_success_and_failure_report_serialization():
    args=parser().parse_args(['--output','new_run','--cut-proposal-json','proposal.json'])
    config=report_configuration(args,Path('resolved_run'))
    assert config['cut_proposal_json']=='proposal.json'
    assert config['output']=='resolved_run'
    assert isinstance(args.cut_proposal_json,Path)
    for state in ('failed_bimanual_qualification','error','initializing'):
        assert json.loads(json.dumps(dict(state=state,configuration=config)))['configuration']==config


def test_none_proposal_stays_json_null():
    args=parser().parse_args(['--output','new_run'])
    assert report_configuration(args,Path('new_run'))['cut_proposal_json'] is None
