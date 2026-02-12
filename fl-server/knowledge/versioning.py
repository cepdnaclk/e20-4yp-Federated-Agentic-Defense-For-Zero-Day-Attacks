from dataclasses import dataclass
from typing import Optional


@dataclass
class Versions:
    model_version: int = 0
    signature_version: int = 0


class VersionManager:
    def __init__(self):
        self._versions = Versions()
        self._history = []  # stack of previous versions for rollback

    @property
    def versions(self) -> Versions:
        return self._versions

    def bump_model(self) -> Versions:
        self._history.append((self._versions.model_version, self._versions.signature_version))
        self._versions.model_version += 1
        return self._versions

    def bump_signatures(self) -> Versions:
        self._history.append((self._versions.model_version, self._versions.signature_version))
        self._versions.signature_version += 1
        return self._versions

    def rollback(self) -> Optional[Versions]:
        if not self._history:
            return None
        mv, sv = self._history.pop()
        self._versions.model_version = mv
        self._versions.signature_version = sv
        return self._versions
