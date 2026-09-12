"""Bounded-memory source integrity checks; same complete-file SHA-256."""
import hashlib
from pathlib import Path


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()
