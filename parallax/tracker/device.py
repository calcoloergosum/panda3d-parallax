from dataclasses import dataclass
from typing import List

import AVFoundation
from typing_extensions import Self


@dataclass
class Device:
    local_name: str
    id: str
    model_name: str

    @classmethod
    def list_all(cls) -> List[Self]:
        """Choose OpenCV device index by device name. I wish opencv supported this interface"""
        session = AVFoundation.AVCaptureSession.alloc().init()
        devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
        devs = [Device(
            local_name=dev.localizedName(),
            id=dev.uniqueID(),
            model_name=dev.modelID(),
        ) for dev in devices]
        session.release()
        print(f"{'local name': <20} {'id': <40} {'model name': <50}")
        for dev in devs:
            print(
                f"{dev.local_name[:20]: <20} " +
                f"{dev.id        [:40]: <40} " +
                f"{dev.model_name[:50]: <50} " +
                "")
        # Not sure about the below behavior
        return sorted(devs, key=lambda x: x.id)
