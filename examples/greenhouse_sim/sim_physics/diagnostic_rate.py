"""Explicit bounded step-rate selection; no material or safety-limit tuning."""
import math


def frequency(value=240):
    if type(value) is not int or value not in (240,480):
        raise ValueError('Explicit 240 Hz baseline or 480 Hz convergence diagnostic required')
    return value


def matches(dt, physics_hz=240):
    hz=frequency(physics_hz)
    return type(dt) in (int,float) and math.isfinite(dt) and abs(dt-1/hz)<=1e-12
