"""Private filesystem primitives for atomic identity checkpoints."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def secure_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"checkpoint directory must be real: {path}")
    else:
        path.mkdir(parents=True)
    os.chmod(path, 0o700)


def locked_file(path: Path):
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    os.fchmod(descriptor, 0o600)
    handle = os.fdopen(descriptor, "r+")
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def read_private_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"checkpoint file must be regular: {path}")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError(f"checkpoint file permissions are too broad: {path}")
    return json.loads(path.read_text())


def write_private_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
