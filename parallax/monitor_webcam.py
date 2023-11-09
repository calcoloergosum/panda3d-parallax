"""Monitor and webcam ... real world environment stuff!"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from parallax.data import Webcam, Window


@dataclass
class RealWorldSetup:
    camera: Webcam
    window: Window

    @classmethod
    def from_json(cls, path: Path) -> RealWorldSetup:
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]) -> RealWorldSetup:
        key2value = key2value.copy()
        return cls(
            camera=Webcam.from_dict(key2value.pop("camera")),
            window=Window.from_dict(key2value.pop("window")),
            **key2value,
        )
