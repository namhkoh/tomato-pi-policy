"""Conservative RGB/native-mask query usability; never edits sensor arrays.

Engineering thresholds, not a learned/readability guarantee. Target IDs establish
ownership; segment size, interior support, exposure and local contrast establish
minimum visible evidence. No dilation joins target islands.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label

POLICY = {
    'version': 'native_visible_query_usability.v1',
    'scope': 'engineering_thresholds_not_human_or_vlm_readability_validation',
    'connectivity': 8,
    'minimum_island_pixels': 100,
    'local_radius_px': 16,
    'minimum_local_pixels': 64,
    'minimum_local_extent_px': 20.,
    'minimum_interior_radius_px': 1.5,
    'minimum_frame_margin_px': 16.,
    'dark_luminance_8bit': 20.,
    'maximum_local_dark_fraction': .5,
    'minimum_local_contrast_8bit': 6.,
    'contrast_ring_width_px': 3,
    'minimum_contrast_ring_pixels': 16,
}


class QueryVisibility:
    def __init__(self, rgb, target_mask):
        rgb, mask = np.asarray(rgb), np.asarray(target_mask)
        if (rgb.shape != (408, 848, 3) or rgb.dtype != np.uint8
                or mask.shape != rgb.shape[:2] or mask.dtype != bool):
            raise ValueError('Expected original uint8 RGB and exact boolean native target mask')
        self.mask = mask.copy()
        self.components, _ = label(mask, structure=np.ones((3, 3), bool))
        self.sizes = np.bincount(self.components.ravel())
        self.interior = distance_transform_edt(mask)
        self.luminance = rgb.astype(float) @ np.array([.2126, .7152, .0722])

    def inspect(self, query):
        if (len(query) != 2 or not np.isfinite(query).all()
                or not 0 <= query[0] < 848 or not 0 <= query[1] < 408):
            raise ValueError('Query outside original image')
        x, y = np.floor(query).astype(int)
        component = int(self.components[y, x])
        result = dict(policy_version=POLICY['version'], passed=False,
                      query_pixel_uv=list(query), reasons=[])
        if not component:
            return {**result, 'reasons': ['query_not_native_target']}
        radius = POLICY['local_radius_px']
        region = np.s_[max(0, y-radius):min(408, y+radius+1),
                       max(0, x-radius):min(848, x+radius+1)]
        local = self.components[region] == component
        yy, xx = np.nonzero(local)
        values = self.luminance[region][local]
        # Dilation constructs a BACKGROUND measurement ring only. It never
        # changes a target mask, identity, query connectivity or depth.
        ring = binary_dilation(local, iterations=POLICY['contrast_ring_width_px']) & ~self.mask[region]
        background = self.luminance[region][ring]
        contrast = abs(float(np.median(values)-np.median(background))) if len(background) else None
        result.update(island_pixels=int(self.sizes[component]), local_pixels=int(local.sum()),
                      local_extent_px=float(np.hypot(np.ptp(xx), np.ptp(yy))),
                      interior_radius_px=float(self.interior[y, x]),
                      frame_margin_px=float(min(query[0], query[1], 848-query[0], 408-query[1])),
                      local_dark_fraction=float(np.mean(values < POLICY['dark_luminance_8bit'])),
                      local_median_luminance=float(np.median(values)),
                      contrast_ring_pixels=int(ring.sum()), local_contrast_8bit=contrast)
        for metric, threshold in (
                ('island_pixels', 'minimum_island_pixels'),
                ('local_pixels', 'minimum_local_pixels'),
                ('local_extent_px', 'minimum_local_extent_px'),
                ('interior_radius_px', 'minimum_interior_radius_px'),
                ('frame_margin_px', 'minimum_frame_margin_px'),
                ('contrast_ring_pixels', 'minimum_contrast_ring_pixels')):
            if result[metric] < POLICY[threshold]: result['reasons'].append(threshold)
        if result['local_dark_fraction'] > POLICY['maximum_local_dark_fraction']:
            result['reasons'].append('local_target_too_dark')
        if contrast is None or contrast < POLICY['minimum_local_contrast_8bit']:
            result['reasons'].append('insufficient_local_contrast')
        result['passed'] = not result['reasons']
        return result
