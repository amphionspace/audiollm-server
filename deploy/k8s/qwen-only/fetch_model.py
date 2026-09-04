from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

destination = Path("/models/amphion-spec")
marker = destination / ".model-sha256"
url = os.environ["AMPHION_SPEC_ARCHIVE_URL"]
expected = os.environ["AMPHION_SPEC_ARCHIVE_SHA256"].strip().lower()

if marker.is_file() and marker.read_text().strip().lower() == expected:
    raise SystemExit(0)

destination.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary:
    archive = Path(temporary.name)
    with urllib.request.urlopen(url, timeout=300) as response:
        shutil.copyfileobj(response, temporary)

hasher = hashlib.sha256()
with archive.open("rb") as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
if digest != expected:
    archive.unlink(missing_ok=True)
    raise RuntimeError(f"model archive sha256 mismatch: expected {expected}, got {digest}")

with tarfile.open(archive) as bundle:
    destination_root = destination.resolve()
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(destination_root):
            raise RuntimeError(f"unsafe model archive path: {member.name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"unsupported model archive entry: {member.name}")
    for child in destination.iterdir():
        if child.name == ".model-sha256":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    bundle.extractall(destination)
archive.unlink()
if not (destination / "config.json").is_file():
    raise RuntimeError("model archive must contain config.json at its root")
marker.write_text(expected + "\n")
