"""Bounded passive Carb error observer; no physics, launch, log edits or approval.

Construct BEFORE authoring/activation/reset. Call check('after_reset'), then
check before AND after every physics step (including after coefficient writes
but before solving). Keep it alive through final acceptance; prefer ``with``
so exit also checks errors delivered during removal. Main owns this wiring.

Installed primary API verified with plain Python, without SimulationApp:
  Isaac Sim 6.0.1 kit/kernel/py/carb/_carb.cp312-win_amd64.pyd
  carb.logging.acquire_logging().add_logger(callback) -> LoggerHandle
  callback(source: str, level: int, filename: str, line: int, message: str)
  remove_logger(the_owned_handle) -> None; LEVEL_ERROR=1, LEVEL_FATAL=2.
  kit/dev/include/carb/logging/{ILogging,Logger,Log}.h

The observed setLinearLimit failure uses [Error] [omni.physx.plugin]. Capture
that family plus physics/tensor channels, not message-keyword heuristics.
Carb may defer delivery or filter messages before observers. Python exposes no
per-source getters or async flush here: this helper does NOT prove complete or
synchronous error coverage, detect past errors, or validate any physics law.
It never changes thresholds, disables/removes another logger, suppresses the
normal console log, raises from a callback, or calls logging inside a callback.
"""
from threading import RLock


ERROR_LEVEL = 1
CHANNEL_ROOTS = ('omni.physx', 'omni.physics', 'physx', 'carb.physics')
_MAX_COUNT = 2**63-1


class NativeError(RuntimeError):
    """A delivered native error, observer fault, or unavailable error monitor."""


def _exception_text(error):
    try:
        message=str(error)
    except Exception:
        message='<exception text unavailable>'
    return (type(error).__name__+': '+message)[:1024]


class NativeErrors:
    """Registration is immediate; logging_interface injection is for mock tests.

    close() releases ONLY this monitor's handle, is idempotent, and preserves
    recorded faults. It does not itself re-raise an already latched physics
    error (so finally cleanup cannot mask the original). The context manager
    checks at exit, including faults delivered while its handle is removed.
    """
    def __init__(self, *, max_records=16, max_message_chars=2048, logging_interface=None):
        if (type(max_records) is not int or not 1<=max_records<=128
                or type(max_message_chars) is not int or not 32<=max_message_chars<=4096):
            raise ValueError('Bounded integer record/text limits required')
        self.max_records=max_records;self.max_message_chars=max_message_chars
        self._lock=RLock();self._records=[];self._total=0;self._dropped=0
        self._fault=None;self._handle=None;self._closed=False;self._closing=False
        self._checks=0;self._last_checkpoint='construction'
        self._injected=logging_interface is not None
        if logging_interface is None:
            import carb.logging as logging
            if logging.LEVEL_ERROR!=ERROR_LEVEL or logging.LEVEL_FATAL<=ERROR_LEVEL:
                raise NativeError('Unexpected installed Carb severity contract')
            logging_interface=logging.acquire_logging()
        self._logging=logging_interface
        try:
            self._check_logging()
            self._handle=self._logging.add_logger(self._receive)
            if self._handle is None:
                raise NativeError('Carb did not return an owned LoggerHandle')
            self.check('registered_before_authoring')
        except BaseException as original:
            try:self.close()
            except Exception as cleanup:original.add_note('Logger cleanup: '+_exception_text(cleanup))
            raise

    def _latch_fault(self, message):
        with self._lock:
            if self._fault is None:self._fault=message[:1024]

    def _check_logging(self):
        # Read-only global checks; no per-source/async coverage claim is made.
        enabled=self._logging.is_log_enabled()
        threshold=self._logging.get_level_threshold()
        if enabled is not True or type(threshold) is not int or threshold>ERROR_LEVEL:
            raise NativeError('Global Carb logging cannot deliver error-level records')

    def _receive(self, *args):
        # C++ may deliver concurrently, or while remove_logger is executing.
        # Latch before formatting; never log/raise/unsubscribe in this callback.
        try:
            with self._lock:
                if self._closed:return
                if len(args)!=5:raise ValueError('Unexpected Carb logger callback arity')
                source,level,filename,line,message=args
                if (not isinstance(source,str) or type(level) is not int
                        or not -(2**31)<=level<2**31):
                    raise ValueError('Invalid Carb logger source/severity')
                prefix=source[:128].lower()
                relevant=any(prefix==root or prefix.startswith(root+'.') for root in CHANNEL_ROOTS)
                if not relevant or level<ERROR_LEVEL:return
                self._total=min(self._total+1,_MAX_COUNT)
                if len(self._records)>=self.max_records:
                    self._dropped=min(self._dropped+1,_MAX_COUNT)
                    return
                if (not isinstance(filename,(str,type(None))) or not isinstance(message,str)
                        or type(line) is not int or not -(2**31)<=line<2**31):
                    raise ValueError('Invalid Carb logger record fields')
                filename='' if filename is None else filename
                self._records.append(dict(source=source[:128],level=level,
                    filename=filename[:512],line=line,message=message[:self.max_message_chars],
                    truncated=(len(source)>128 or len(filename)>512 or len(message)>self.max_message_chars),
                    last_checkpoint=self._last_checkpoint))
        except Exception as error:
            self._latch_fault('Logger callback failure: '+_exception_text(error))

    def _raise_latched(self, phase):
        with self._lock:
            if self._fault is not None:
                raise NativeError('Native error observer fault at '+phase+': '+self._fault)
            if self._total:
                first=self._records[0] if self._records else None
                detail=('<record unavailable>' if first is None else
                    first['source']+': '+(first['message'] or '<empty native error message>'))
                raise NativeError('Native PhysX/physics error at '+phase+': '+detail)

    def check(self, phase='native_check'):
        """Fail on ANY captured error, including empty text; never reset a latch."""
        if not isinstance(phase,str):raise ValueError('Checkpoint label must be text')
        phase=phase[:128]
        with self._lock:
            if self._closed or self._closing or self._handle is None:
                raise NativeError('Native error monitor is not actively registered')
            self._checks=min(self._checks+1,_MAX_COUNT);self._last_checkpoint=phase
        try:self._check_logging()
        except Exception as error:self._latch_fault(_exception_text(error))
        self._raise_latched(phase)

    def report(self):
        """Independent JSON-safe copy; truncation does not weaken the error latch."""
        with self._lock:
            return dict(method='passive_carb_physx_error_observer',channel_roots=list(CHANNEL_ROOTS),
                minimum_level=ERROR_LEVEL,error_count=self._total,dropped_record_count=self._dropped,
                counts_saturated=self._total==_MAX_COUNT or self._dropped==_MAX_COUNT,
                errors=[dict(row) for row in self._records],observer_fault=self._fault,
                faulted=self._total>0 or self._fault is not None,
                max_records=self.max_records,max_message_chars=self.max_message_chars,
                checks=self._checks,last_checkpoint=self._last_checkpoint,
                registered=self._handle is not None and not self._closed,
                closed=self._closed,logging_configuration_changed=False,
                logging_interface_injected=self._injected,per_source_thresholds_verified=False,
                asynchronous_delivery_flushed=False,prior_errors_observed=False,
                native_response_validated=False,training_eligible=False)

    def close(self):
        with self._lock:
            if self._closed:return
            if self._closing:raise NativeError('Concurrent logger close is unsupported')
            self._closing=True;handle=self._handle
        try:
            # Never hold our lock while calling Carb: removal can dispatch a
            # final callback on another thread. Never remove an unknown handle.
            if handle is not None:self._logging.remove_logger(handle)
        except Exception as error:
            self._latch_fault('Owned logger removal failed: '+_exception_text(error))
            with self._lock:self._closing=False
            raise NativeError('Could not remove owned native-error logger') from error
        with self._lock:
            self._handle=None;self._closed=True;self._closing=False

    def __enter__(self):
        try:self.check('context_enter')
        except BaseException as original:
            # A failed __enter__ does not invoke __exit__ automatically.
            try:self.close()
            except Exception as cleanup:original.add_note('Logger cleanup: '+_exception_text(cleanup))
            raise
        return self

    def __exit__(self, kind, original, traceback):
        pending=None
        if original is None:
            try:self.check('context_exit')
            except BaseException as error:pending=error
        try:self.close()
        except Exception as cleanup:
            if original is not None:original.add_note('Logger cleanup: '+_exception_text(cleanup))
            elif pending is not None:pending.add_note('Logger cleanup: '+_exception_text(cleanup))
            else:raise
        if original is None:
            if pending is not None:raise pending
            self._raise_latched('after_owned_logger_removal')
        return False
