"""Conservative finite sphere UNION covering a complete capsule plus margin.

For adjacent centres separated by h, every interior capsule point is at most
sqrt((r+margin)^2+(h/2)^2) from its nearest centre. End spheres cover both caps.
These are NOT point samples with gaps. Quantization is added to the radius.
"""
import math
import numpy as np


def cover(start,end,radius,margin,*,extra_m=.00025,max_spheres=256):
    a,b=np.asarray(start,float),np.asarray(end,float)
    if (a.shape!=(3,) or b.shape!=(3,) or not np.isfinite([a,b]).all()
            or any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not np.isfinite(v)
                for v in (radius,margin,extra_m)) or not 0<radius<=1
            or not .001<=margin<=.05 or not 0<extra_m<=.001
            or type(max_spheres) is not int or not 1<=max_spheres<=256):
        raise ValueError('Bounded finite capsule, >=1 mm margin and sphere budget required')
    length=math.hypot(*(b-a));r=radius+margin
    if not math.isfinite(length):raise ValueError('Unrepresentable capsule length')
    spacing=2*math.sqrt(2*r*extra_m+extra_m**2)
    intervals=max(1,int(math.ceil(length/spacing))) if length else 0
    count=intervals+1
    if count>max_spheres:raise ValueError('Conservative capsule cover exceeds sphere budget')
    centres=np.linspace(a,b,count)
    quantized=centres.astype(np.float32).astype(float)
    shift=float(np.linalg.norm(quantized-centres,axis=1).max())
    step=length/intervals if intervals else 0.
    # Upward float32 radius plus centre-rounding allowance protects native input.
    radius32=float(np.nextafter(np.float32(math.hypot(r,step/2)+shift+1e-9),np.float32(np.inf)))
    if not np.isfinite(radius32) or not np.isfinite(quantized).all():
        raise ValueError('Unrepresentable conservative native sphere cover')
    return quantized,radius32
