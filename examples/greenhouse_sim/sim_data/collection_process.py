"""Observe an exact live Windows worker through a retained kernel handle.

PID, full command, executable and creation time must match the saved launch.
No exit code is inferred from files/logs. An exited/reused PID is not recoverable.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes as wt
from datetime import datetime, timezone
import os
from pathlib import Path
import time

import psutil

from .dataset_review import read_json, require, write_json
from .depth_preview import sha256

EXIT_SCHEMA='greenhouse.observed_worker_exit.v1'


class VerifiedProcess:
    def __init__(self, launch):
        require(os.name=='nt','Windows process handles required')
        self.api=ctypes.WinDLL('kernel32',use_last_error=True)
        signatures={
            'OpenProcess':([wt.DWORD,wt.BOOL,wt.DWORD],wt.HANDLE),
            'CloseHandle':([wt.HANDLE],wt.BOOL),
            'GetProcessTimes':([wt.HANDLE]+[ctypes.POINTER(wt.FILETIME)]*4,wt.BOOL),
            'WaitForSingleObject':([wt.HANDLE,wt.DWORD],wt.DWORD),
            'GetExitCodeProcess':([wt.HANDLE,ctypes.POINTER(wt.DWORD)],wt.BOOL),
            'QueryFullProcessImageNameW':([wt.HANDLE,wt.DWORD,wt.LPWSTR,ctypes.POINTER(wt.DWORD)],wt.BOOL),
            'TerminateProcess':([wt.HANDLE,wt.UINT],wt.BOOL)}
        for name,(args,result) in signatures.items():
            f=getattr(self.api,name); f.argtypes=args; f.restype=result
        self.handle=self.api.OpenProcess(0x100000|0x1000|0x0001,False,launch['pid'])
        if not self.handle: raise ctypes.WinError(ctypes.get_last_error())
        try:
            require(self.api.WaitForSingleObject(self.handle,0)==258,'Worker already exited; no adoption')
            times=[wt.FILETIME() for _ in range(4)]
            if not self.api.GetProcessTimes(self.handle,*map(ctypes.byref,times)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.created_epoch=((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)/1e7-11644473600
            started=datetime.fromisoformat(launch['started_utc']).timestamp()
            require(-.1<=started-self.created_epoch<=10,'PID creation does not match recorded launch')
            buffer=ctypes.create_unicode_buffer(32768); size=wt.DWORD(len(buffer))
            if not self.api.QueryFullProcessImageNameW(self.handle,0,buffer,ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.executable=buffer.value
            require(Path(self.executable).resolve()==Path(launch['command'][0]).resolve(),'Wrong worker executable')
            actual=psutil.Process(launch['pid']).cmdline()
            require(len(actual)==len(launch['command']) and actual[1:]==launch['command'][1:], 'Worker command does not match launch')
            require(self.api.WaitForSingleObject(self.handle,0)==258,'Worker exited during adoption')
        except BaseException:
            self.close(); raise

    def close(self):
        if self.handle: self.api.CloseHandle(self.handle); self.handle=None

    def wait(self, seconds):
        result=self.api.WaitForSingleObject(self.handle,max(0,min(1000,int(seconds*1000))))
        if result==258: return None
        if result!=0: raise ctypes.WinError(ctypes.get_last_error())
        code=wt.DWORD()
        if not self.api.GetExitCodeProcess(self.handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value

    def terminate(self):
        if not self.api.TerminateProcess(self.handle,1):
            raise ctypes.WinError(ctypes.get_last_error())


def observe(launch_path, output):
    launch_path,output=Path(launch_path).resolve(),Path(output).resolve()
    require(not output.exists(),'Choose a new observer output')
    launch_hash=sha256(launch_path); launch=read_json(launch_path)
    require(launch['command'][1:4]==['-u','-m','sim_data.collection_worker'],'Only native capture workers can be adopted')
    require(type(launch['timeout_s']) is int and 60<=launch['timeout_s']<=14400,'Invalid saved worker deadline')
    process=VerifiedProcess(launch)
    try:
        require(sha256(launch_path)==launch_hash,'Launch changed during adoption')
        output.mkdir(parents=True)
        bound=dict(schema_version=EXIT_SCHEMA,launch_path=str(launch_path),launch_sha256=launch_hash,
            pid=launch['pid'],command=launch['command'],created_epoch=process.created_epoch,
            executable=process.executable,observer_pid=os.getpid(),
            observer_started_utc=datetime.now(timezone.utc).isoformat(),
            method='retained_verified_windows_process_handle',training_approved=False)
        write_json(output/'attached.json',bound)
        deadline=datetime.fromisoformat(launch['started_utc']).timestamp()+launch['timeout_s']
        timed_out=False
        while True:
            code=process.wait(min(1,max(0,deadline-time.time())))
            if code is not None: break
            if time.time()>=deadline:
                timed_out=True; process.terminate()
                stop=time.monotonic()+20
                while code is None and time.monotonic()<stop: code=process.wait(1)
                require(code is not None,'Exact worker did not terminate after its original deadline')
                break
        result=dict(**bound,returncode=code,timed_out=timed_out,
                    observed_exit_utc=datetime.now(timezone.utc).isoformat(),
                    state='observed_exit_not_independent_audit_or_visual_approval')
        write_json(output/'exit.json',result)
        print('OBSERVED_WORKER_EXIT',launch['pid'],code,timed_out,flush=True)
        return result
    finally: process.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--launch',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv); observe(a.launch,a.output)


if __name__=='__main__': main()
