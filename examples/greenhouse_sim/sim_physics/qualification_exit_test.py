import pytest
from .qualification_exit import exit_code


def test_complete_assessed_pass_exits_zero():
    assert exit_code(dict(state='passed_coupon',assessment=dict(passed=True)),passed_state='passed_coupon') == 0


@pytest.mark.parametrize('change', [dict(state='failed_coupon'),dict(state='passed_OTHER'),
    dict(error='native fault'),dict(assessment=None),dict(assessment={'passed':False}),
    dict(assessment={'passed':1}),dict(cleanup_failures=['detach failed']),
    dict(cleanup={'failures':['export observer failed']})])
def test_failed_or_incomplete_result_cannot_exit_success(change):
    result=dict(state='passed_coupon',assessment=dict(passed=True));result.update(change)
    assert exit_code(result,passed_state='passed_coupon') == 2
