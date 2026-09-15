"""Opt-in, single-verification content reader; no cross-check digest cache.

Deduplicate only path resolution and decoded metadata. Every unique canonical
file is fully SHA256 checked initially AND at finish. Original path spellings
and their link topology are freshly checked at finish, including paths with ..
and symlinks/junctions introduced after admission. Stats never authorize bytes.
This is a fail-closed snapshot check, not an atomic filesystem lock.
"""
import hashlib
import os
from pathlib import Path
import re
import stat

from ..native_capture_v3.reference_bank import MAX_JSON_BYTES, _parse


class Reader:
    def __init__(self):
        self.bindings = {}
        self.documents = {}
        self._maps = {}
        self._paths = {}
        self._pins = {}
        self._topology = {}
        self._cwd = Path.cwd()
        self._state = "open"
        self._counts = dict(initial_hashes=0, final_hashes=0, metadata_hashes=0,
                            initial_resolutions=0, final_resolutions=0)

    def _require(self, condition, message):
        if not condition:
            self._state = "failed"
            raise ValueError(message)

    def _open(self):
        self._require(self._state == "open", "Reader is single-use or failed")

    def _spelling(self, path):
        raw = os.fspath(path)
        self._require(isinstance(raw, str) and raw and "\x00" not in raw, "Invalid path")
        # Do not use abspath/normpath: collapsing '..' loses the original route.
        return raw if os.path.isabs(raw) else str(self._cwd / raw)

    @staticmethod
    def _node(path):
        try:
            info = path.lstat()
        except FileNotFoundError:
            return ("missing",)
        is_link = stat.S_ISLNK(info.st_mode)
        reparse = bool(getattr(info, "st_file_attributes", 0) & 0x400)
        if is_link or reparse:
            # Include link identity and text even if two routes reach the same file.
            target = os.readlink(path)
            return ("link", info.st_dev, info.st_ino,
                    getattr(info, "st_reparse_tag", None), target)
        return ("ordinary", stat.S_IFMT(info.st_mode))

    def _remember_topology(self, *paths):
        pending = list(paths)
        while pending:
            original = pending.pop()
            for prefix in (original, *original.parents):
                key = str(prefix)
                if key in self._topology:
                    continue
                node = self._node(prefix)
                self._topology[key] = node
                if node[0] == "link":
                    target = Path(node[-1])
                    pending.append(target if target.is_absolute() else prefix.parent / target)

    def resolve(self, path):
        self._open()
        raw = self._spelling(path)
        if raw not in self._paths:
            try:
                original = Path(raw)
                resolved = original.resolve(strict=False)
                # Include indirect links: a changed intermediate alias may still
                # resolve to the same bytes and final canonical file.
                self._remember_topology(original, resolved)
            except (OSError, RuntimeError, ValueError) as exc:
                self._require(False, "Cannot resolve original path: " + raw + ": " + str(exc))
            self._paths[raw] = resolved
            self._counts["initial_resolutions"] += 1
        return self._paths[raw]

    def _hash(self, path, phase):
        self._counts[phase + "_hashes"] += 1
        try:
            with Path(path).open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as exc:
            self._require(False, "Cannot hash bound file: " + str(path) + ": " + str(exc))

    def bind(self, path, expected):
        self._open()
        self._require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                      "Missing/invalid original SHA256")
        raw = self._spelling(path)
        self._require(raw not in self._pins or self._pins[raw] == expected,
                      "Conflicting original hashes: " + raw)
        resolved = self.resolve(raw)
        key = str(resolved)
        self._require(key not in self.bindings or self.bindings[key] == expected,
                      "Conflicting original hashes through alias: " + raw)
        if key not in self.bindings:
            self._require(resolved.is_file() and self._hash(resolved, "initial") == expected,
                          "Changed bound file: " + key)
            self.bindings[key] = expected
        self._pins[raw] = expected
        return resolved

    def bindings_from(self, bindings):
        self._open()
        self._require(isinstance(bindings, dict) and bindings, "Missing existing source bindings")
        previous = self._maps.get(id(bindings))
        if previous is not None:
            self._require(previous[0] is bindings and previous[1] == bindings,
                          "Conflicting or mutated binding map")
            return
        for name, expected in bindings.items():
            self._require(isinstance(name, str) and os.path.isabs(name),
                          "Absolute original binding required")
            self.bind(name, expected)
        # Keep an immutable-by-ownership copy; identity alone could hide edited pins.
        self._maps[id(bindings)] = (bindings, dict(bindings))

    def json(self, path, expected):
        resolved = self.bind(path, expected)
        key = str(resolved)
        if key not in self.documents:
            try:
                with resolved.open("rb") as stream:
                    data = stream.read(MAX_JSON_BYTES + 1)
                self._require(len(data) <= MAX_JSON_BYTES, "Metadata size limit")
                self._counts["metadata_hashes"] += 1
                self._require(hashlib.sha256(data).hexdigest() == expected, "File changed during read")
                self.documents[key] = _parse(data)
            except (OSError, ValueError):
                self._state = "failed"
                raise
        return self.documents[key]

    def safe(self, root, name):
        root = self.resolve(root)
        self._require(isinstance(name, str) and name and not Path(name).is_absolute(),
                      "Relative file required")
        path = self.resolve(root / name)
        self._require(path.is_relative_to(root) and path != root, "File escapes source directory")
        return path

    def finish(self):
        self._open()
        try:
            for name, expected in self.bindings.items():
                self._require(self._hash(name, "final") == expected,
                              "Source changed during bank build: " + name)
            # After the hashes, not before: catch retargeting during the final sweep.
            for raw, expected in self._paths.items():
                self._counts["final_resolutions"] += 1
                self._require(Path(raw).resolve(strict=False) == expected,
                              "Original path target changed: " + raw)
            for raw, expected in self._topology.items():
                self._require(self._node(Path(raw)) == expected,
                              "Original path topology changed: " + raw)
            for original, snapshot in self._maps.values():
                self._require(original == snapshot, "Binding map changed during verification")
        except (OSError, RuntimeError, ValueError):
            self._state = "failed"
            raise
        self._state = "finished"

    def diagnostics(self):
        return dict(self._counts, unique_files=len(self.bindings),
                    original_paths=len(self._paths), topology_nodes=len(self._topology),
                    state=self._state, stat_only_byte_acceptance=False)
