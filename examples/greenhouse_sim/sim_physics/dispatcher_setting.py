"""Process-local scheduler experiment, applied before owned scene parsing.

No solver, contacts, timesteps or body parameters are changed. Readback proves
the configured preference, not which native task implementation actually ran.
https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/settings.html
"""

SETTING='/physics/physxDispatcher'


def startup_configuration(configuration,mode):
    """Configure a requested scheduler before Kit initializes its interfaces.

    Post-start readback/restoration remains separate. Never mutate the caller's
    launcher options or append contradictory native arguments.
    """
    if mode not in (None,'carb','physx'):
        raise ValueError('Explicit carb/physx dispatcher comparison required')
    result=dict(configuration)
    if mode is not None:
        extra=list(result.get('extra_args',[]))
        if any(not isinstance(v,str) or v.startswith('--'+SETTING) for v in extra):
            raise ValueError('Invalid or conflicting dispatcher startup argument')
        extra.append('--'+SETTING+'='+('true' if mode=='physx' else 'false'))
        result['extra_args']=extra
    return result


class DispatcherSetting:
    def __init__(self,settings,mode):
        if mode not in (None,'carb','physx'):
            raise ValueError('Explicit carb/physx dispatcher comparison required')
        self.settings=settings;self.mode=mode;self.previous=settings.get(SETTING)
        self.changed=False;self.closed=False

    def apply(self):
        if self.closed:raise RuntimeError('Dispatcher setting scope is closed')
        if self.mode is not None:
            self.changed=True  # Restore even if setting/readback throws.
            self.settings.set_bool(SETTING,self.mode=='physx')
        return self.verify()

    def verify(self):
        if self.closed:raise RuntimeError('Dispatcher setting scope is closed')
        observed=self.settings.get(SETTING)
        if self.mode is not None and observed is not (self.mode=='physx'):
            raise RuntimeError('Requested CPU dispatcher setting changed')
        return dict(requested=self.mode,configured_physx_dispatcher=observed,
            previous=self.previous,process_only=True,native_effective_scheduler_verified=False,
            solver_iterations_or_contacts_changed=False)

    def close(self):
        if self.closed:return
        if self.changed:
            if self.previous is None:self.settings.destroy_item(SETTING)
            else:self.settings.set(SETTING,self.previous)
        self.closed=True
