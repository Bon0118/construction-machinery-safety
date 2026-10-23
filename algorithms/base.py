"""Common contracts shared by every detection algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class AlgorithmDescriptor:
    code: str
    name: str
    config_version: int = 1
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "config_version": self.config_version,
            "description": self.description,
        }


class AlgorithmSettings(ABC):
    """Validated, immutable settings snapshot used by one algorithm run."""

    @abstractmethod
    def validate(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError


class DetectionAlgorithm(ABC):
    """Runtime adapter exposed to task and command-line runners."""

    descriptor: AlgorithmDescriptor
    settings: AlgorithmSettings

    @property
    def code(self) -> str:
        return self.descriptor.code

    @property
    def name(self) -> str:
        return self.descriptor.name

    def config_snapshot(self) -> Mapping[str, Any]:
        return self.settings.to_dict()

    @abstractmethod
    def process_video(self, source: str, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def process_image_array(self, image: Any, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError
