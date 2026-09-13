"""Optional faster diagnostic encoding, with the original finite-JSON contract.

orjson maps NaN/infinity to null, so it MUST NOT be used without this explicit
recursive finite check. Unsupported values/key types fall back to the original
stdlib encoder, including its errors and arbitrary-precision integer handling.
No record is rounded, sampled, omitted or changed in place.
"""
import json
import math

try:
    import orjson as _fast
except ImportError:
    _fast=None


def _finite(value,parents):
    kind=type(value)
    if kind in (str,int,bool,type(None)) or isinstance(value,(str,int)):return
    if isinstance(value,float):
        if not math.isfinite(value):raise ValueError('Out of range float values are not JSON compliant')
    elif isinstance(value,(list,tuple,dict)):
        identity=id(value)
        if identity in parents:raise ValueError('Circular reference detected')
        parents.add(identity)
        try:
            if isinstance(value,dict):
                for key,item in value.items():
                    _finite(key,parents);_finite(item,parents)
            else:
                for item in value:_finite(item,parents)
        finally:parents.remove(identity)
    else:
        # orjson also accepts dataclasses/datetimes that stdlib JSON rejects.
        # Do not let those bypass finite validation or change the contract.
        raise TypeError('Object of type '+kind.__name__+' is not JSON serializable')


def encode(record):
    if _fast is not None:
        _finite(record,set())
        try:return _fast.dumps(record)+b'\n'
        except _fast.JSONEncodeError:pass
    return json.dumps(record,allow_nan=False,separators=(',',':')).encode('utf-8')+b'\n'


def backend():return 'orjson_finite_checked_with_stdlib_fallback' if _fast is not None else 'stdlib_finite_json'
