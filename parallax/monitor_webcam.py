"""Monitor and webcam ... real world environment stuff!"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from parallax.data import Webcam, Window


@dataclass
class RealWorldSetup:
    webcam: Webcam
    window: Window

    @classmethod
    def from_preset(cls, name: str) -> RealWorldSetup:
        configs = {
            "Macbook Pro 2021 M1": {
                "webcam": {
                    "fov_deg": 54.611362,
                    "focal_length": 35,
                    "sensor": {
                        "resolution": (1280, 720),
                    },
                    "pose": {
                        "xyz": (0, 0, 5 + .5 * 179.03),
                        "hpr": (0, -15, 0),
                        "relative_to": "window",
                    },
                },
                "window": {
                    "width": 1440,
                    "height": 850,
                    "pixels_per_mm": 5.027,
                    # (286.45, 179.03): Monitor size in mm
                    # (1440,  900): Pixel space - This is what we need
                    # (2880, 1800): With HiDPI 2x scaled
                    # (2560, 1600): Physical resolution
                    "pose": {
                        "xyz": (0, 0, 0),
                        "hpr": (0, 10, 0),
                    },
                },
            }
        }
        config = configs.get(name, None)
        if config is None:
            print("Supported names:")
            for _name in configs:
                print("  - " + _name)
            raise KeyError(_name)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]) -> RealWorldSetup:
        key2value = key2value.copy()
        return cls(
            webcam=Webcam.from_dict(key2value.pop("webcam")),
            window=Window.from_dict(key2value.pop("window")),
            **key2value,
        )
