import hashlib
import hmac
import os
import tempfile
from pathlib import Path
from typing import Protocol

from .settings import settings


class ArtifactStore(Protocol):
    def put(self, key: str, data: bytes) -> str: ...
    def get(self, key: str, digest: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalArtifacts:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Invalid artifact key.")
        return path

    def put(self, key: str, data: bytes) -> str:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            tmp = Path(temporary.name)
            try:
                temporary.write(data)
                temporary.close()
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return hashlib.sha256(data).hexdigest()

    def get(self, key: str, digest: str) -> bytes:
        data = self.path(key).read_bytes()
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), digest):
            raise ValueError("Artifact integrity check failed.")
        return data

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)


store: ArtifactStore = LocalArtifacts(settings.artifact_root)
