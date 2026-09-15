"""Resolution-aware native query usability; does not change legacy v3 rules."""
from copy import deepcopy
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label
from .capture_sensor import checked_resolution
from .query_visibility import POLICY as LEGACY_POLICY

POLICY = dict(deepcopy(LEGACY_POLICY), version="native_query_usability.explicit_resolution.v1",
              pixel_threshold_basis="absolute_native_pixels_not_resampled_legacy_pixels")


class NativeQueryVisibility:
    def __init__(self, rgb, mask, resolution):
        self.width, self.height = checked_resolution(resolution)
        rgb, mask = np.asarray(rgb), np.asarray(mask)
        if (rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8
                or mask.shape != rgb.shape[:2] or mask.dtype != bool):
            raise ValueError("Original native uint8 RGB and exact boolean mask required")
        self.mask = mask.copy()
        self.components, _ = label(mask, structure=np.ones((3, 3), bool))
        self.sizes = np.bincount(self.components.ravel())
        self.interior = distance_transform_edt(mask)
        self.luminance = rgb.astype(float) @ np.array([.2126, .7152, .0722])

    def inspect(self, query):
        if (not isinstance(query, (list, tuple)) or len(query) != 2
                or any(type(v) not in (int, float) for v in query)
                or not np.isfinite(query).all()
                or not 0 <= query[0] < self.width or not 0 <= query[1] < self.height):
            raise ValueError("Query outside declared native frame")
        x, y = np.floor(query).astype(int)
        component = int(self.components[y, x])
        result = dict(policy_version=POLICY["version"], passed=False,
                      query_pixel_uv=list(query), reasons=[])
        if not component:
            return dict(result, reasons=["query_not_native_target"])
        r = POLICY["local_radius_px"]
        region = np.s_[max(0,y-r):min(self.height,y+r+1), max(0,x-r):min(self.width,x+r+1)]
        local = self.components[region] == component
        yy, xx = np.nonzero(local)
        values = self.luminance[region][local]
        # Background contrast ring only. Never join disconnected target regions.
        ring = binary_dilation(local, iterations=POLICY["contrast_ring_width_px"]) & ~self.mask[region]
        background = self.luminance[region][ring]
        contrast = abs(float(np.median(values)-np.median(background))) if len(background) else None
        result.update(island_pixels=int(self.sizes[component]), local_pixels=int(local.sum()),
            local_extent_px=float(np.hypot(np.ptp(xx),np.ptp(yy))),
            interior_radius_px=float(self.interior[y,x]),
            frame_margin_px=float(min(query[0],query[1],self.width-query[0],self.height-query[1])),
            local_dark_fraction=float(np.mean(values < POLICY["dark_luminance_8bit"])),
            local_median_luminance=float(np.median(values)),
            contrast_ring_pixels=int(ring.sum()), local_contrast_8bit=contrast)
        for metric, bound in (
            ("island_pixels","minimum_island_pixels"),("local_pixels","minimum_local_pixels"),
            ("local_extent_px","minimum_local_extent_px"),("interior_radius_px","minimum_interior_radius_px"),
            ("frame_margin_px","minimum_frame_margin_px"),("contrast_ring_pixels","minimum_contrast_ring_pixels")):
            if result[metric] < POLICY[bound]:
                result["reasons"].append(bound)
        if result["local_dark_fraction"] > POLICY["maximum_local_dark_fraction"]:
            result["reasons"].append("local_target_too_dark")
        if contrast is None or contrast < POLICY["minimum_local_contrast_8bit"]:
            result["reasons"].append("insufficient_local_contrast")
        result["passed"] = not result["reasons"]
        return result
