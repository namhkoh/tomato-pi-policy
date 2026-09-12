"""Read-only Windows x64 pool-tag diagnostic; no driver/process/OS changes.

NtQuerySystemInformation class 22 is an undocumented ABI, so unsupported
layouts/statuses fail closed. Tags identify allocations, NOT their unique
driver/application owner. Raw tag totals need not equal committed pool pages.
"""
import ctypes
from datetime import datetime,timezone
import json
import struct
import sys

ENTRY=struct.Struct('<4sII4xQIIQ')
MAX_BUFFER_BYTES=16*1024*1024


def decode(buffer,returned_bytes):
    if (type(returned_bytes) is not int or not 8<=returned_bytes<=len(buffer)
            or returned_bytes>MAX_BUFFER_BYTES):
        raise ValueError('Bounded returned x64 pool buffer required')
    count=struct.unpack_from('<I',buffer)[0]
    if 8+count*ENTRY.size>returned_bytes:
        raise ValueError('Pool entry count exceeds returned native bytes')
    rows=[];tags=set()
    for index in range(count):
        tag,pa,pf,pb,na,nf,nb=ENTRY.unpack_from(buffer,8+index*ENTRY.size)
        if tag in tags:raise ValueError('Duplicate native pool tag')
        tags.add(tag)
        rows.append(dict(tag=tag.decode('ascii',errors='backslashreplace'),tag_hex=tag.hex(),
            paged_bytes=pb,nonpaged_bytes=nb,paged_allocs=pa,paged_frees=pf,
            nonpaged_allocs=na,nonpaged_frees=nf))
    return rows


def snapshot():
    if sys.platform!='win32' or ctypes.sizeof(ctypes.c_void_p)!=8:
        raise RuntimeError('Only the explicitly bound Windows x64 ABI is supported')
    query=ctypes.WinDLL('ntdll').NtQuerySystemInformation
    query.argtypes=[ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
    query.restype=ctypes.c_int32
    size=1024*1024
    for _ in range(6):
        if size>MAX_BUFFER_BYTES:raise RuntimeError('Pool query exceeded bounded buffer')
        buf=ctypes.create_string_buffer(size);used=ctypes.c_uint32()
        status=query(22,buf,size,ctypes.byref(used))&0xffffffff
        if status==0:
            rows=decode(buf.raw,used.value);break
        if status!=0xc0000004:raise RuntimeError('Read-only pool query failed: '+hex(status))
        size=max(size*2,used.value+4096)
    else:raise RuntimeError('Pool query did not stabilize')
    from .host_memory import memory_snapshot
    return dict(sampled_utc=datetime.now(timezone.utc).isoformat(),
        source='NtQuerySystemInformation_SystemPoolTagInformation_x64',
        undocumented_abi=True,read_only=True,driver_owner_verified=False,
        tag_count=len(rows),raw_paged_bytes=sum(r['paged_bytes'] for r in rows),
        raw_nonpaged_bytes=sum(r['nonpaged_bytes'] for r in rows),
        memory=memory_snapshot(),
        largest_paged=sorted(rows,key=lambda r:r['paged_bytes'],reverse=True)[:20],
        largest_nonpaged=sorted(rows,key=lambda r:r['nonpaged_bytes'],reverse=True)[:20])


if __name__=='__main__':
    print(json.dumps(snapshot(),indent=2,allow_nan=False))
