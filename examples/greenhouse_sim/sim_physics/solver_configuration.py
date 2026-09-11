"""Explicit pre-parse solver controls; never change a running physics scene."""


def author_contact_order(scene, *, enabled):
    """Author the documented custom attribute; readback is USD, not native.

    Caller must own a stopped/unparsed scene. NVIDIA's articulation stability
    guide recommends this numerical comparison for dynamic grasp contacts:
    https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html
    No material, iteration, collision filtering or force limit is altered.
    """
    from pxr import Sdf
    if type(enabled) is not bool:
        raise ValueError('Explicit boolean contact solver order required')
    attr=scene.GetAttribute('physxScene:solveArticulationContactLast')
    previous=attr.Get() if attr else None
    if enabled:
        scene.CreateAttribute('physxScene:solveArticulationContactLast',Sdf.ValueTypeNames.Bool).Set(True)
    observed=scene.GetAttribute('physxScene:solveArticulationContactLast').Get()
    if enabled and observed is not True:
        raise RuntimeError('Contact-last USD authoring failed')
    return dict(requested=enabled,previous_usd_value=previous,observed_usd_value=observed,
        authored=enabled,scope='owned stopped scene before native parse',
        native_effective_value_verified=False,source='NVIDIA articulation stability guide')
