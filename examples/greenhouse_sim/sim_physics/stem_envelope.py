"""Continuous internal capsule envelope; no collider crosses a separable seam.

USD capsule height is the cylinder/spine length, excluding the hemispheres.
Shortening EVERY span by two radii creates a zero-radius neck at every joint.
Interior spines instead reach their joint anchors. Physical ends and the cut
interface retain flush rounded caps; this is not a flat tissue/wound model.
"""
import math


def capsule_span(length, radius, *, flush_start, flush_end):
    if (not all(math.isfinite(x) and x > 0 for x in (length, radius))
            or type(flush_start) is not bool or type(flush_end) is not bool):
        raise ValueError('Positive shaft dimensions and explicit endpoint flags required')
    start = -length / 2 + (radius if flush_start else 0.)
    end = length / 2 - (radius if flush_end else 0.)
    if end <= start:
        raise ValueError('Shaft endpoint span cannot contain its physical radius')
    return dict(height_m=end-start, center_z_m=(start+end)/2, radius_m=radius)
