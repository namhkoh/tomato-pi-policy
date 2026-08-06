"""Make ``pxr`` importable without starting a simulator.

Isaac Sim's ``python.bat`` does not put USD on the import path; only a running
Kit application does. Asset conversion needs USD but not a simulator, and
booting Kit costs tens of seconds and a GPU context per invocation, so this
locates the ``omni.usd.libs`` extension that ships with Isaac Sim and wires it
up directly.

Under a live SimulationApp ``pxr`` is already imported and this is a no-op, so
the same modules work from both entry points.
"""

from __future__ import annotations

import glob
import os
import pathlib
import sys

_ENV_VAR = "ISAAC_SIM_PATH"


class UsdUnavailableError(RuntimeError):
    """Raised when USD cannot be located in any known Isaac Sim installation."""


def isaac_sim_root() -> pathlib.Path | None:
    """Best guess at the Isaac Sim installation directory."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return pathlib.Path(override)

    # Running under Isaac's own interpreter: .../isaac-sim/kit/python/kit.exe
    for parent in pathlib.Path(sys.executable).resolve().parents:
        if (parent / "isaac-sim.bat").exists() or (parent / "isaac-sim.sh").exists():
            return parent

    for candidate in (pathlib.Path("D:/isaac-sim"), pathlib.Path.home() / "isaacsim", pathlib.Path("/isaac-sim")):
        if candidate.exists():
            return candidate
    return None


def ensure_pxr() -> None:
    """Import-enable ``pxr``, raising if no USD build can be found."""
    try:
        import pxr  # noqa: PLC0415  (probe: availability is the question)
    except ImportError:
        pass
    else:
        return

    root = isaac_sim_root()
    if root is None:
        raise UsdUnavailableError(
            f"Could not find an Isaac Sim installation; set {_ENV_VAR} or run under Isaac Sim's python."
        )

    packages = sorted(glob.glob(str(root / "extscache" / "omni.usd.libs-*")))
    if not packages:
        raise UsdUnavailableError(f"No omni.usd.libs extension under {root / 'extscache'}")

    package = packages[-1]
    binaries = pathlib.Path(package) / "bin"
    if binaries.is_dir():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(binaries))
        os.environ["PATH"] = f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}"
    sys.path.insert(0, package)

    try:
        import pxr  # noqa: F401, PLC0415  (probe only)
    except ImportError as error:
        raise UsdUnavailableError(f"Found {package} but importing pxr still failed: {error}") from error
