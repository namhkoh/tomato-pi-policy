"""Bounded passive contact capture, NOT a geometry compiler or force controller.

The caller supplies an exact plant COLLIDER inventory and owns physics-step,
episode/generation and pose binding. Only observed rows are counted: neither an
empty stream nor a populated one certifies native reporting completeness.
Original collider order and signed normal/friction impulses are retained. A
friction anchor has no inferred normal or separation. Impulses are trace-only;
no contact law, classification, load gate or actuation is implemented here.
"""
from collections.abc import Mapping
from copy import deepcopy
from itertools import islice
import math
from numbers import Real
import re


MAX_ROWS=256
MAX_PLANT_COLLIDERS=4096
MAX_PATH_CHARS=2048
_PRIM_PATH=re.compile(r'/[A-Za-z_][A-Za-z0-9_]*(?:/[A-Za-z_][A-Za-z0-9_]*)*\Z')


def _path(value):
    if (not isinstance(value,str) or not 1<len(value)<=MAX_PATH_CHARS
            or _PRIM_PATH.fullmatch(value) is None):
        raise ValueError('Bounded absolute plain collider prim path required')
    return value


def _number(value):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('Finite numeric contact scalar required')
    return float(value)


def _vector(value):
    try:
        if len(value)!=3:raise ValueError('Three-component contact vector required')
        return [_number(v) for v in value]
    except TypeError as exc:
        raise ValueError('Three-component contact vector required') from exc


class PlantContactStream:
    """ContactEvents hook: begin_step(), add_contact(row), invalidate(error).

    Reset clears healthy rows only. Any fault permanently latches this object;
    construct a new collector after repairing/rebinding the caller's source.
    snapshot() and rows return copies, including partial rows for fault audits.
    No caller step IDs are generated or accepted as native evidence here.
    """
    def __init__(self,plant_colliders):
        if isinstance(plant_colliders,(str,bytes,Mapping)):
            raise ValueError('Exact bounded plant collider inventory required')
        try:paths=list(islice(iter(plant_colliders),MAX_PLANT_COLLIDERS+1))
        except TypeError as exc:raise ValueError('Iterable collider inventory required') from exc
        if not 1<=len(paths)<=MAX_PLANT_COLLIDERS:
            raise ValueError('Plant collider inventory exceeds bounds or is empty')
        paths=[_path(p) for p in paths]
        if len(set(paths))!=len(paths):raise ValueError('Duplicate plant collider path')
        self._plant_colliders=frozenset(paths)
        self._rows=[]
        self._error=None

    @property
    def error(self):return self._error

    @property
    def rows(self):return deepcopy(self._rows)

    def invalidate(self,error):
        if self._error is None:
            try:message=str(error)
            except Exception:message='<unprintable error>'
            self._error=(type(error).__name__+': '+message)[:1024]

    def _healthy(self):
        if self._error is not None:raise RuntimeError('Plant contact stream fault latched: '+self._error)

    def begin_step(self):
        self._healthy()
        self._rows=[]

    def add_contact(self,row):
        self._healthy()
        try:
            if not isinstance(row,Mapping):raise ValueError('Explicit contact row mapping required')
            kind=row.get('kind')
            required={'collider0','collider1','kind','point_world_m','impulse_on_0_ns'}
            if kind=='normal':required|={'normal_on_0','separation_m'}
            elif kind!='friction':raise ValueError('Explicit normal or friction row kind required')
            if set(row)!=required:raise ValueError('Missing or unexpected contact row fields')
            first,second=_path(row['collider0']),_path(row['collider1'])
            if first==second:raise ValueError('Contact requires two distinct collider paths')
            copied=dict(collider0=first,collider1=second,kind=kind,
                point_world_m=_vector(row['point_world_m']),impulse_on_0_ns=_vector(row['impulse_on_0_ns']))
            if kind=='normal':
                # Finite raw evidence only: no normalization, collinearity fit,
                # sign clamp or geometry/contact-law acceptance at capture time.
                copied.update(normal_on_0=_vector(row['normal_on_0']),separation_m=_number(row['separation_m']))
            if first not in self._plant_colliders and second not in self._plant_colliders:return
            if len(self._rows)>=MAX_ROWS:raise ValueError('Plant contact row limit exceeded (256)')
            self._rows.append(copied)  # One row even when BOTH endpoints are plant colliders.
        except Exception as exc:
            self.invalidate(exc)
            raise

    def snapshot(self):
        return dict(rows=self.rows,row_count=len(self._rows),maximum_rows=MAX_ROWS,
            plant_collider_paths=sorted(self._plant_colliders),error=self._error,
            impulse_basis='original_order_impulse_on_collider0_ns_trace_only',
            normal_geometry_basis='normal_on_collider0_and_signed_separation_without_modification',
            native_completeness_verified=False,step_and_pose_binding_verified=False,
            friction_normal_inferred=False,actuation_authorized=False,training_eligible=False)
