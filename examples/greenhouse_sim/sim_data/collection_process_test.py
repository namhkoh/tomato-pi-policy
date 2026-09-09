from datetime import datetime, timezone, timedelta
import os
import subprocess
import sys

import pytest

from sim_data.collection_process import VerifiedProcess


@pytest.mark.skipif(os.name!='nt',reason='Windows handle integration')
def test_retained_handle_observes_real_nonzero_exit_and_rejects_bad_identity():
    command=[sys.executable,'-c','import sys; sys.stdin.readline(); sys.exit(7)']
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    launch=dict(pid=process.pid,command=command,started_utc=datetime.now(timezone.utc).isoformat())
    handle=None
    try:
        wrong={**launch,'command':[command[0],'-c','different command']}
        with pytest.raises(ValueError,match='command'): VerifiedProcess(wrong)
        wrong={**launch,'started_utc':(datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()}
        with pytest.raises(ValueError,match='creation'): VerifiedProcess(wrong)
        handle=VerifiedProcess(launch)
        assert handle.wait(.01) is None
        process.communicate(b'exit\n',timeout=10)
        assert process.returncode==7 and handle.wait(1)==7
        assert handle.wait(0)==7  # Still available after the original process exits.
        with pytest.raises((ValueError,OSError)): VerifiedProcess(launch)
    finally:
        if handle: handle.close()
        if process.poll() is None: process.terminate(); process.wait(timeout=10)
