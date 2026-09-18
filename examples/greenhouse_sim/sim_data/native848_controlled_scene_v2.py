"""Original scene plus an explicitly replayed controlled catalogue; no substitution here."""
from .native848_original_scene_v2 import prepare_native_scene as prepare_original_scene
from . import native848_controlled_pair_plan_v2 as api


def prepare_native_scene(app, plan):
    generated = api.check(plan, full=True)
    scene = prepare_original_scene(app, plan['reference_anchor'])
    scene['generated'] = generated
    return scene
