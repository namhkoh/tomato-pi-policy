from __future__ import annotations

from greenhouse_sim.isaac_rl import _advance_one_physics_step


class _Context:
    def __init__(self):
        self.steps = []
        self.renders = 0

    def step(self, *, render):
        self.steps.append(render)

    def render(self):
        self.renders += 1


def test_rendered_rl_tick_advances_one_nonrendering_physics_step_then_renders():
    context = _Context()
    _advance_one_physics_step(context, render=True)
    assert context.steps == [False]
    assert context.renders == 1


def test_headless_rl_tick_does_not_render():
    context = _Context()
    _advance_one_physics_step(context, render=False)
    assert context.steps == [False]
    assert context.renders == 0
