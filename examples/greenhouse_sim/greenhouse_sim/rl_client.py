"""Small JSON-lines client and optional Gymnasium wrapper for online RL."""

from __future__ import annotations

import json
import socket

import numpy as np


class DeleafClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766, timeout_s: float = 120.0):
        self._socket = socket.create_connection((host, int(port)), timeout=float(timeout_s))
        self._stream = self._socket.makefile("rwb")
        hello = self._read()
        if not hello.get("ok"):
            raise RuntimeError(hello.get("error", "RL server rejected connection"))
        self.action_size = int(hello["action_size"])
        self.observation_size = int(hello["observation_size"])

    def _read(self) -> dict:
        raw = self._stream.readline()
        if not raw:
            raise ConnectionError("RL server closed the connection")
        response = json.loads(raw)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "RL server error"))
        return response

    def _request(self, payload: dict) -> dict:
        self._stream.write((json.dumps(payload, separators=(",", ":")) + "\n").encode())
        self._stream.flush()
        return self._read()

    def spec(self) -> dict:
        return self._request({"command": "spec"})

    def reset(self, *, seed: int = 0) -> tuple[np.ndarray, dict]:
        response = self._request({"command": "reset", "seed": int(seed)})
        return np.asarray(response["observation"], dtype=np.float32), response["info"]

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        response = self._request(
            {"command": "step", "action": np.asarray(action, dtype=float).tolist()}
        )
        return (
            np.asarray(response["observation"], dtype=np.float32),
            float(response["reward"]),
            bool(response["terminated"]),
            bool(response["truncated"]),
            response["info"],
        )

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            self._request({"command": "close"})
        finally:
            self._stream.close()
            self._socket.close()
            self._stream = None
            self._socket = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def gymnasium_env(host: str = "127.0.0.1", port: int = 8766):
    """Construct a Gymnasium Env only when Gymnasium is installed."""
    import gymnasium

    class _GymDeleafEnv(gymnasium.Env):
        metadata = {"render_modes": []}

        def __init__(self):
            self.client = DeleafClient(host, port)
            self.action_space = gymnasium.spaces.Box(
                -1.0, 1.0, shape=(self.client.action_size,), dtype=np.float32
            )
            self.observation_space = gymnasium.spaces.Box(
                -np.inf,
                np.inf,
                shape=(self.client.observation_size,),
                dtype=np.float32,
            )

        def reset(self, *, seed=None, options=None):
            del options
            super().reset(seed=seed)
            return self.client.reset(seed=0 if seed is None else seed)

        def step(self, action):
            return self.client.step(action)

        def close(self):
            self.client.close()

    return _GymDeleafEnv()
