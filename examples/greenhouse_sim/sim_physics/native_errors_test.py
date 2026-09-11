"""CPU mocks only: no SimulationApp, physics solver, or logger configuration."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Thread
from types import ModuleType

import pytest

from .native_errors import CHANNEL_ROOTS, NativeError, NativeErrors, _MAX_COUNT


class FakeLogging:
    def __init__(self):
        self.enabled=True;self.threshold=0;self.read_error=None
        self.foreign=object();self.owned=object();self.console=[]
        self.loggers={self.foreign:lambda *args:self.console.append(args)}
        self.added=[];self.removed=[];self.on_add=None;self.on_remove=None
        self.add_error=None;self.remove_error=None

    def is_log_enabled(self):
        if self.read_error is not None:raise self.read_error
        return self.enabled

    def get_level_threshold(self):return self.threshold

    def add_logger(self, callback):
        self.added.append(callback)
        if self.add_error is not None:raise self.add_error
        if self.owned is not None:self.loggers[self.owned]=callback
        if self.on_add is not None:self.on_add(callback)
        return self.owned

    def remove_logger(self, handle):
        assert handle is self.owned and handle is not self.foreign
        self.removed.append(handle)
        if self.on_remove is not None:self.on_remove(self.loggers[handle])
        if self.remove_error is not None:raise self.remove_error
        del self.loggers[handle]

    def emit(self, source='omni.physx.plugin', level=1, filename='ExtD6Joint.cpp',
             line=305, message='PxD6Joint::setLinearLimit: limit invalid'):
        for callback in tuple(self.loggers.values()):
            callback(source,level,filename,line,message)

    # Any logging setting/default-logger mutation is forbidden.
    def forbidden(self, *args, **kwargs):raise AssertionError('Not a passive observer')
    set_level_threshold=forbidden
    set_level_threshold_for_source=forbidden
    set_log_enabled=forbidden
    set_log_enabled_for_source=forbidden
    reset=forbidden


@pytest.fixture
def monitor():
    logging=FakeLogging();observer=NativeErrors(logging_interface=logging)
    try:yield observer,logging
    finally:observer.close()


def test_import_has_no_carb_or_simulation_side_effects():
    package_root=str(Path(__file__).resolve().parents[1])
    script="""
import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('carb','omni','isaacsim','pxr'):
            raise AssertionError('Native import at module load: '+fullname)
sys.meta_path.insert(0,BlockNative())
sys.path.insert(0,sys.argv[1])
from sim_physics.native_errors import NativeErrors
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',script,package_root],
                          capture_output=True,text=True,timeout=15)
    assert result.returncode==0,result.stderr


def test_immediate_registration_is_passive_and_removal_is_owned_only(monitor):
    observer,logging=monitor
    assert len(logging.added)==1 and observer.report()['registered']
    assert observer.report()['last_checkpoint']=='registered_before_authoring'
    assert logging.enabled and logging.threshold==0
    logging.emit(level=0,message='ordinary warning')
    observer.check('after_reset')
    assert len(logging.console)==1
    observer.close();observer.close()
    assert logging.removed==[logging.owned]
    assert list(logging.loggers)==[logging.foreign]
    assert observer.report()['closed'] and not observer.report()['registered']
    with pytest.raises(NativeError,match='not actively registered'):observer.check()


@pytest.mark.parametrize('source',[root+suffix for root in CHANNEL_ROOTS for suffix in ('','.plugin')]
                         +['Omni.PhysX.Plugin','PHYSX','omni.physics.tensors.plugin'])
@pytest.mark.parametrize('level',[1,2])
def test_error_and_fatal_channel_families_are_latched(monitor,source,level):
    observer,logging=monitor
    logging.emit(source=source,level=level)
    with pytest.raises(NativeError,match='limit invalid'):observer.check('after_reset')
    report=observer.report()
    assert report['error_count']==1 and report['faulted']
    assert report['errors'][0]['source']==source
    assert report['errors'][0]['level']==level
    assert report['errors'][0]['line']==305
    assert len(logging.console)==1  # Capture does not suppress the other logger.


@pytest.mark.parametrize('source,level',[
    ('omni.physx.plugin',-2),('omni.physx.plugin',-1),('omni.physx.plugin',0),
    ('omni.physxExtra',1),('omni.physicsExtra.plugin',1),('physx_fake',1),
    ('other.omni.physx.plugin',1),('carb.physicsExtra',2),('omni.renderer',1),('',1)])
def test_channel_boundary_and_severity_filter_not_message_keywords(monitor,source,level):
    observer,logging=monitor
    logging.emit(source=source,level=level,message='PhysX error: setLinearLimit invalid')
    observer.check()
    assert observer.report()['error_count']==0


def test_empty_native_error_latches_permanently_and_is_json_safe(monitor):
    observer,logging=monitor
    logging.emit(filename=None,line=0,message='')
    for phase in ('after_reset','before_step','after_step','finalize'):
        with pytest.raises(NativeError,match='empty native error message'):observer.check(phase)
    logging.emit(level=0,message='later successful step cannot clear a latch')
    report=observer.report()
    assert report['errors'][0]['message']=='' and report['errors'][0]['filename']==''
    assert report['error_count']==1 and report['faulted']
    json.dumps(report,allow_nan=False)
    observer.close()
    assert observer.report()['faulted']


@pytest.mark.parametrize('field,value',[
    ('max_records',0),('max_records',129),('max_records',True),('max_records',2.0),
    ('max_message_chars',31),('max_message_chars',4097),('max_message_chars',False),
    ('max_message_chars',float('nan'))])
def test_invalid_storage_bounds_never_register(field,value):
    logging=FakeLogging()
    with pytest.raises(ValueError):NativeErrors(logging_interface=logging,**{field:value})
    assert not logging.added and not logging.removed


def test_records_are_capped_text_truncated_and_report_is_detached():
    logging=FakeLogging()
    observer=NativeErrors(logging_interface=logging,max_records=2,max_message_chars=32)
    try:
        logging.emit(source='omni.physx.'+'x'*200,filename='f'*700,message='m'*100)
        logging.emit(message='')
        for unused in range(20):logging.emit(message='dropped')
        report=observer.report()
        assert report['error_count']==22 and report['dropped_record_count']==20
        assert len(report['errors'])==2 and report['errors'][1]['message']==''
        row=report['errors'][0]
        assert len(row['source'])==128 and len(row['filename'])==512
        assert row['message']=='m'*32 and row['truncated']
        row['message']='changed';report['errors'].clear();report['channel_roots'].clear()
        assert observer.report()['errors'][0]['message']=='m'*32
        assert observer.report()['channel_roots']==list(CHANNEL_ROOTS)
        with pytest.raises(NativeError):observer.check()
    finally:observer.close()


@pytest.mark.parametrize('args',[
    (),('omni.physx',1,'file',1),('omni.physx',1,'file',1,'','extra'),
    (None,1,'file',1,''),('omni.physx',True,'file',1,''),
    ('omni.physx',float('nan'),'file',1,''),('omni.physx',2**40,'file',1,''),
    ('omni.physx',1,object(),1,''),('omni.physx',1,'file',True,''),
    ('omni.physx',1,'file',2**40,''),('omni.physx',1,'file',1,None)])
def test_malformed_callback_never_raises_and_latches_fault(monitor,args):
    observer,logging=monitor
    assert logging.added[0](*args) is None
    report=observer.report()
    assert report['faulted'] and report['observer_fault']
    with pytest.raises(NativeError,match='observer fault'):observer.check()


def test_empty_callback_exception_and_broken_exception_string_latch(monitor):
    observer,logging=monitor
    class BrokenSource(str):
        def __getitem__(self, key):raise ValueError('')
    logging.added[0](BrokenSource('omni.physx'),1,'file',1,'')
    assert 'ValueError:' in observer.report()['observer_fault']
    with pytest.raises(NativeError):observer.check()
    # The first callback failure remains bounded and cannot be overwritten.
    fault=observer.report()['observer_fault']
    logging.added[0]()
    assert observer.report()['observer_fault']==fault


@pytest.mark.parametrize('enabled,threshold',[(False,0),(None,0),(1,0),(True,2),
                                            (True,True),(True,0.0)])
def test_global_error_logging_unavailable_fails_before_registration(enabled,threshold):
    logging=FakeLogging();logging.enabled=enabled;logging.threshold=threshold
    with pytest.raises(NativeError,match='cannot deliver'):NativeErrors(logging_interface=logging)
    assert not logging.added and not logging.removed


@pytest.mark.parametrize('change',['disabled','threshold','empty_exception','broken_exception'])
def test_global_logging_failure_latches_even_after_configuration_restored(monitor,change):
    observer,logging=monitor
    class BrokenException(Exception):
        def __str__(self):raise ValueError('')
    if change=='disabled':logging.enabled=False
    elif change=='threshold':logging.threshold=2
    elif change=='empty_exception':logging.read_error=ValueError('')
    else:logging.read_error=BrokenException()
    with pytest.raises(NativeError):observer.check('before_step')
    assert observer.report()['observer_fault']
    logging.enabled=True;logging.threshold=0;logging.read_error=None
    with pytest.raises(NativeError):observer.check('after_step')


def test_missing_handle_is_rejected_without_removing_unknown_or_foreign_logger():
    logging=FakeLogging();logging.owned=None
    with pytest.raises(NativeError,match='LoggerHandle'):NativeErrors(logging_interface=logging)
    assert not logging.removed and list(logging.loggers)==[logging.foreign]


def test_falsey_but_non_null_owned_handle_is_valid():
    class Handle:
        def __bool__(self):return False
    logging=FakeLogging();logging.owned=Handle()
    with NativeErrors(logging_interface=logging) as observer:observer.check()
    assert logging.removed==[logging.owned]


def test_error_during_registration_is_detected_and_owned_handle_removed():
    logging=FakeLogging();logging.on_add=lambda callback:callback('omni.physx',1,None,0,'')
    with pytest.raises(NativeError,match='empty native'):NativeErrors(logging_interface=logging)
    assert logging.removed==[logging.owned] and list(logging.loggers)==[logging.foreign]


def test_registration_exception_is_not_masked():
    logging=FakeLogging();failure=ValueError('');logging.add_error=failure
    with pytest.raises(ValueError) as caught:NativeErrors(logging_interface=logging)
    assert caught.value is failure and not logging.removed


def test_failed_context_entry_cleans_up_its_handle(monitor):
    observer,logging=monitor
    logging.emit(message='')
    with pytest.raises(NativeError):
        with observer:pytest.fail('Must not enter body after a native error')
    assert observer.report()['closed'] and logging.removed==[logging.owned]


def test_error_during_removal_is_captured_without_holding_callback_lock():
    logging=FakeLogging()
    def final_callback(callback):
        worker=Thread(target=lambda:callback('omni.physx',1,None,0,''),daemon=True)
        worker.start();worker.join(timeout=2)
        assert not worker.is_alive(),'Removal blocked a native callback on the observer lock'
    logging.on_remove=final_callback
    with pytest.raises(NativeError,match='after_owned_logger_removal'):
        with NativeErrors(logging_interface=logging) as observer:observer.check('finalize')
    assert observer.report()['closed'] and observer.report()['error_count']==1


def test_normal_context_exit_checks_unchecked_final_error(monitor):
    observer,logging=monitor
    with pytest.raises(NativeError,match='context_exit'):
        with observer:logging.emit(message='late native error')
    assert observer.report()['closed']


def test_body_exception_is_preserved_even_when_native_error_also_exists(monitor):
    observer,logging=monitor;original=ValueError('')
    with pytest.raises(ValueError) as caught:
        with observer:
            logging.emit(message='')
            raise original
    assert caught.value is original and observer.report()['faulted']
    assert observer.report()['closed']


def test_removal_failure_preserves_original_and_allows_owned_retry(monitor):
    observer,logging=monitor;original=ValueError('')
    logging.remove_error=RuntimeError('')
    with pytest.raises(ValueError) as caught:
        with observer:raise original
    assert caught.value is original and any('Logger cleanup:' in note for note in original.__notes__)
    assert observer.report()['registered'] and observer.report()['observer_fault']
    logging.remove_error=None
    observer.close();observer.close()
    assert logging.removed==[logging.owned,logging.owned]
    assert list(logging.loggers)==[logging.foreign]


def test_removal_failure_cannot_mask_pending_native_error(monitor):
    observer,logging=monitor
    logging.remove_error=RuntimeError('remove failed')
    try:
        with pytest.raises(NativeError,match='empty native error') as caught:
            with observer:logging.emit(message='')
        assert any('Logger cleanup:' in note for note in caught.value.__notes__)
    finally:logging.remove_error=None


def test_removal_failure_without_prior_error_fails_exit(monitor):
    observer,logging=monitor;logging.remove_error=RuntimeError('')
    try:
        with pytest.raises(NativeError,match='Could not remove'):
            with observer:pass
    finally:logging.remove_error=None


def test_delivered_errors_are_thread_safe_and_storage_bounded():
    logging=FakeLogging();observer=NativeErrors(logging_interface=logging,max_records=3)
    try:
        callback=logging.added[0]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda unused:callback('omni.physx',1,None,0,''),range(1000)))
        report=observer.report()
        assert report['error_count']==1000 and report['dropped_record_count']==997
        assert len(report['errors'])==3
        with pytest.raises(NativeError):observer.check()
    finally:observer.close()


def test_counters_saturate_without_unbounded_integer_growth(monitor):
    observer,logging=monitor
    logging.emit()
    observer.max_records=1
    observer._total=_MAX_COUNT-1;observer._dropped=_MAX_COUNT-1
    logging.emit();logging.emit()
    report=observer.report()
    assert report['error_count']==report['dropped_record_count']==_MAX_COUNT
    assert report['counts_saturated']


def test_closed_callback_does_not_mutate_final_report(monitor):
    observer,logging=monitor
    callback=logging.added[0];observer.close();before=observer.report()
    callback('omni.physx',1,None,0,'too late')
    assert observer.report()==before


def test_report_never_claims_complete_native_error_or_physics_coverage(monitor):
    report=monitor[0].report()
    for field in ('logging_configuration_changed','per_source_thresholds_verified',
                  'asynchronous_delivery_flushed','prior_errors_observed',
                  'native_response_validated','training_eligible'):
        assert report[field] is False
    assert report['logging_interface_injected'] is True
    json.dumps(report,allow_nan=False)


def install_fake_carb(monkeypatch,logging,*,error_level=1,fatal_level=2):
    carb=ModuleType('carb');carb.__path__=[]
    module=ModuleType('carb.logging');module.LEVEL_ERROR=error_level;module.LEVEL_FATAL=fatal_level
    module.acquire_logging=lambda:logging
    carb.logging=module
    monkeypatch.setitem(sys.modules,'carb',carb)
    monkeypatch.setitem(sys.modules,'carb.logging',module)


def test_lazy_installed_api_path_uses_owned_handle(monkeypatch):
    logging=FakeLogging();install_fake_carb(monkeypatch,logging)
    with NativeErrors() as observer:
        assert not observer.report()['logging_interface_injected']
        observer.check('after_reset')
    assert logging.removed==[logging.owned]


@pytest.mark.parametrize('error,fatal',[(0,2),(1,1)])
def test_unexpected_installed_severity_contract_is_rejected(monkeypatch,error,fatal):
    logging=FakeLogging();install_fake_carb(monkeypatch,logging,error_level=error,fatal_level=fatal)
    with pytest.raises(NativeError,match='severity contract'):NativeErrors()
    assert not logging.added


@pytest.mark.parametrize('failure_phase',['author','activate','reset','coefficient_update','step'])
def test_documented_pre_author_reset_and_step_wiring_stops_before_acceptance(failure_phase):
    logging=FakeLogging();operations=[]
    def operation(name):
        assert len(logging.added)==1
        operations.append(name)
        if name==failure_phase:logging.emit(message='')
    with pytest.raises(NativeError):
        with NativeErrors(logging_interface=logging) as observer:
            for phase in ('author','activate','reset'):
                operation(phase);observer.check('after_'+phase)
            observer.check('before_coefficient_update')
            operation('coefficient_update');observer.check('before_step')
            operation('step');observer.check('after_step')
            operations.append('accepted')
    assert operations[-1]==failure_phase and 'accepted' not in operations
    assert observer.report()['faulted'] and observer.report()['closed']
