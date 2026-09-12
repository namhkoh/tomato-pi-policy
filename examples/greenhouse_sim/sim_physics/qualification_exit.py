"""Fail-closed process exit for owned native diagnostics with fast Kit shutdown."""


def exit_code(result, *, passed_state):
    # SimulationApp.close can exit the process before main returns. The state
    # string alone is insufficient: preserve an error/cleanup/assessment fault.
    assessment = result.get('assessment')
    cleanup = result.get('cleanup') or {}
    return 0 if (result.get('state') == passed_state and not result.get('error')
                 and isinstance(assessment, dict) and assessment.get('passed') is True
                 and not result.get('cleanup_failures') and not cleanup.get('failures')) else 2
