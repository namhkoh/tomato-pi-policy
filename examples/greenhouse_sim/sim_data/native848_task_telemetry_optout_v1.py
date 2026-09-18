"""CPU-reviewable, process-local telemetry opt-out configuration.

No native import, installation write, environment mutation, process control, or
process/resource exemption. Call child_environment before spawning the future
worker, then prepare_config before constructing its SimulationApp. Runtime
checks report settings only; process absence and capture quality need a smoke.
"""
from copy import deepcopy

ENV_KEY='OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA'
DISABLE_ARG='--/structuredLog/enable=false'
EXCLUDE_ARG='--/app/extensions/excluded/0=omni.kit.telemetry'
SCHEMA='greenhouse.native848_task_telemetry_optout.v1'


def require(value,message):
    if not value:raise ValueError(message)


def child_environment(environment):
    """Copy a subprocess environment; never modify the parent or user settings."""
    require(isinstance(environment,dict),'Explicit child environment required')
    require(all(isinstance(k,str) and isinstance(v,str) for k,v in environment.items()),'Environment must contain strings')
    result=dict(environment);result[ENV_KEY]='1';return result


def prepare_config(config,*,argv,environment,exclude_extension=False):
    """Preserve all capture settings; add only reviewed telemetry launch args.

    Structured logging disabled is the default to preserve the installed hard
    telemetry extension dependencies. Exclusion is an explicit alternate for
    separate native dependency/startup verification, not a fallback or retry.
    """
    require(type(exclude_extension) is bool,'Explicit boolean exclusion mode required')
    require(environment.get(ENV_KEY)=='1','Anonymous telemetry opt-out must be in the child environment before native imports')
    require(isinstance(config,dict) and isinstance(config.get('extra_args',[]),list),'Explicit launch config/extra_args list required')
    require(isinstance(argv,(list,tuple)),'Explicit future worker argv required')
    original=config.get('extra_args',[])
    for argument in [*original,*argv]:
        require(isinstance(argument,str),'Launch arguments must be strings')
        lower=argument.lower()
        require(not lower.startswith(('--/structuredlog','--/telemetry','--/app/extensions'))
            and 'omni.kit.telemetry' not in lower,
            'Conflicting telemetry/extension overrides require separate review')
        require(not lower.startswith(('--merge-config','--/app/settings/load')),
            'Additional configuration sources require separate review')
    result=deepcopy(config)
    result['extra_args']=[*original,DISABLE_ARG,*([EXCLUDE_ARG] if exclude_extension else [])]
    return result


def validate_runtime(get_setting,*,environment,exclude_extension=False,extension_enabled=None):
    """Read-only post-start/pre-close settings evidence, not native qualification."""
    require(environment.get(ENV_KEY)=='1','Child anonymous telemetry override missing')
    actual=get_setting('/structuredLog/enable')
    require(actual is False,'Structured telemetry logging remained enabled or unverified')
    if exclude_extension:
        require(extension_enabled is False,'Requested telemetry extension exclusion not observed')
    return dict(schema=SCHEMA,structured_log_enabled=actual,environment_override={ENV_KEY:'1'},
        telemetry_extension_exclusion_requested=exclude_extension,
        telemetry_extension_enabled=extension_enabled,
        anonymous_mode_setting=get_setting('/telemetry/enableAnonymousData'),
        normal_log_file=get_setting('/log/file'),normal_log_level=get_setting('/log/level'),
        only_settings_verified=True,transmitter_absence_verified=False,capture_qualified=False,
        process_or_resource_exemption=False,training_approved=False,accepted_increment=0)
