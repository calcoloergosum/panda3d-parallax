from __future__ import annotations

import functools
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, List, Optional, Tuple,
                    TypeVar, Union)
import itertools as it

import AVFoundation
import cv2
import numpy as np
import structlog
from typing_extensions import Self

from .data import Matrix3f, Pose, Size2i, Xywh, Xyxy, Point2f, Xyz, xyxy2xywh

LOGGER = structlog.get_logger(__name__)

T = TypeVar("T")
U = TypeVar("U")

@dataclass
class Device:
    local_name: str
    id: str
    model_name: str


def get_devices():
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


@dataclass
class Camera2D:
    resolution: Size2i
    camera_matrix: Matrix3f
    distort_coefficients: Tuple[float, ...]
    pose: Optional[Pose]
    device_id: str
    device_name: str = ''
    depth: Optional[float] = None

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]) -> Self:
        pose = key2value.pop("pose")
        if pose is not None:
            pose = Pose(**pose)
        return cls(pose=pose, **key2value)

    def pt2camera_ray(self, pt: Point2f) -> Tuple[Xyz]:
        """Return camera ray in gl coordinates"""
        cam_mat = np.array(self.camera_matrix)
        ray = (pt - cam_mat[:2, 2]) / np.diag(cam_mat[:2, :2])
        return np.array((ray[0], -ray[1], 1))

    def pt2line(self, pt: Point2f) -> Tuple[Xyz, Xyz]:
        return np.array(self.pose.xyz), self.pose.rot_mat() @ self.pt2camera_ray(pt)


def find_idx(a_list: List[T], predicate: Callable[[T], bool]) -> Optional[T]:
    for i, t in enumerate(a_list):
        if predicate(t):
            return i
    return None


@dataclass
class Tracker2DCameras:
    master: Camera2D
    slaves: List[Camera2D]

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]) -> Self:
        master = Camera2D.from_dict(key2value.pop("master"))
        slaves = [Camera2D.from_dict(s) for s in key2value.pop("slaves")]
        return cls(master=master, slaves=slaves, **key2value)

    def track_faces(self, ma: MovingAverageND) -> Tuple[threading.Thread, MovingAverageND]:
        """Simple face tracking"""
        face_cascade = cv2.CascadeClassifier(
            (Path(__file__).parent / 'resource/haarcascade_frontalface_default.xml').as_posix())
        
        id2camera = {s.device_id: s for s in (self.master, *self.slaves)}
        def on_frame(id2frame: Dict[str, np.ndarray]) -> np.ndarray:
            # For debugging ... nudge the face position from the center towards left-top of camera image
            # which is right-top of the monitor from user's point of view.
            # h, w = frame.shape[:2]
            # on_face(frame, [(int(w * 0.45), int(h * 0.45), int(w * 0.0), int(h * 0.2))])
            # return True
            LOGGER.info('Detecting face')

            id2face = {}
            for id, frame in id2frame.items():
                frame = cv2.undistort(frame, np.array(id2camera[id].camera_matrix), np.array(id2camera[id].distort_coefficients))
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                xywhs = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(xywhs) == 0:
                    print(f"No face from camera {id}")
                    return True

                # use only the biggest bbox
                x, y, w, h = sorted(xywhs, key=lambda xywh: xywh[2] * xywh[3])[-1]
                xy = ((x + 0.5 * w, y + 0.5 * h))
                id2face[id] = xy

            if len(id2face) == 0:
                print("no face found")
                return True
            if len(id2frame) == len(id2face) == 1:
                camera = id2camera[id]
                ray = camera.pt2camera_ray(id2face[id])
                coord = camera.pose.xyz + camera.pose.rot_mat() @ ray * self.master.depth
                ma.push(coord)
                return True

            mean_points = []
            for i, j in it.combinations(id2frame, 2):
                xi, xj = id2face[i], id2face[j]
                oi, vi = id2camera[i].pt2line(xi)
                oj, vj = id2camera[j].pt2line(xj)

                # perpendicular vector
                vz = np.cross(vi, vj)
                vz /= np.linalg.norm(vz)
                a, b, c = np.linalg.inv(np.hstack((vi[:, None], vj[:, None], vz[:, None]))) @ (oj - oi)
                xyz_l, xyz_r = oi + a*vi, oj - b*vj
                np.testing.assert_allclose(xyz_l + c * vz, xyz_r)
                xyz = np.mean((xyz_l, xyz_r), axis=0).tolist()
                error = abs(c)
                print(f"Intercamera error: {error}")
                mean_points.append(xyz)

            ma.push(np.mean(mean_points, axis=0))
            return True
        return self.loop_frame(on_frame)

    def loop_frame(self, on_frame: Callable[[np.ndarray], bool]) -> bool:
        # Pretty much the standard way of capturing video stream...
        if len(self.slaves) == 0:
            if self.master.depth is None:
                raise ValueError("Depth should be set to master camera when no slaves")

        devs = get_devices()
        master_idx = find_idx(devs, lambda dev: dev.id == self.master.device_id)
        if master_idx is None:
            raise KeyError(f"Device ID {self.master.device_id} not in", [d.id for d in devs])
        slave_idxs = []
        for s in self.slaves:
            idx = find_idx(devs, lambda dev: dev.id == s.device_id)
            if idx is None:
                raise KeyError(f"Device ID {s.device_id} not in", [d.id for d in devs])
            slave_idxs.append(idx)

        # OpenCV AVFoundation backend device index is sorted by device id
        id2caps = {}
        for i, d in enumerate(devs):
            if i not in (master_idx, *slave_idxs):
                continue
            cap = id2caps[d.id] = cv2.VideoCapture(i)
            if not cap.isOpened():
                LOGGER.critical(f"Cannot open camera for device with ID {d.id}")
                return False
        assert len(id2caps) > 0, "No capture device opened"

        try:
            t_last = 0
            while True:
                t = time.time()
                # To throttle
                if t - t_last < .01:
                    continue
                # Capture frame-by-frame
                LOGGER.info('Taking frame')
                id2frame = {}
                for id, cap in id2caps.items():
                    ret, frame = cap.read()
                    if not ret:
                        LOGGER.critical(f"Can't receive frame for device with ID {id}. Exiting ...")
                        break
                    id2frame[id] = frame
                t_last = t
                if not on_frame(id2frame):
                    break
            return True
        finally:
            for cap in id2caps.values():
                cap.release()

    @property
    def relative_to(self) -> str:
        return self.master.pose.relative_to


@dataclass
class TrackerRealsense:
    pose: Pose
    def track_faces(self, ma: MovingAverageND) -> Tuple[threading.Thread, MovingAverageND]:
        """Simple face tracking"""
        self.loop_face_track(functools.partial(self.on_face, ma=ma))

    def loop_face_track(self, on_face: Callable[[List[Xyxy], np.ndarray], bool]):
        """Much part is borrowed from github.com/kylelscott"""
        # pylint: disable=too-many-locals, import-outside-toplevel
        import pyrealsense2 as rs
        pipeline = rs.pipeline()

        # Configure streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)
        align_to = rs.stream.color
        align = rs.align(align_to)

        print("Loading model")
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt",
            "res10_300x300_ssd_iter_140000.caffemodel")
        try:
            while True:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                # Align the depth frame to color frame
                align               = rs.align(rs.stream.color)
                aligned_frames      = align.process(frames)
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame         = aligned_frames.get_color_frame()

                if not aligned_depth_frame or not color_frame:
                    continue

                # Convert images to numpy arrays
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Begin the detection portion
                h, w = color_image.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(
                        color_image, (300, 300)),
                        1.0, (300, 300), (104.0, 177.0, 123.0)
                    )
                net.setInput(blob, "data")
                detections = net.forward("detection_out")

                # loop over the detections
                boxes = [
                    detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    for i in range(0, detections.shape[2])
                    if detections[0, 0, i, 2] > 0.5
                ]
                on_face(xyxy2xywh(boxes), depth_image)
        finally:
            # Stop streaming
            pipeline.stop()

    def on_face(self, ma: MovingAverageND, xywhs: List[Xywh], depth_image: np.ndarray) -> bool:
        if len(xywhs) == 0:
            return
        x, y, w, h = sorted(xywhs, key=lambda xywh: xywh[2] * xywh[3])[-1]
        # print(f"{x:.2f} {y:.2f} {w:.2f} {h:.2f}")
        xy = ((x + 0.5 * w, y + 0.3 * h))

        xc, yc = int(xy[0]), int(xy[1])
        ds = depth_image[max(0, yc - 10): yc + 10, max(0, xc - 10): xc + 10].flatten()
        ds = ds[ds > 0]
        if len(ds) == 0:
            return
        d = np.median(ds)
        ma.push((*xy, d))
        return

    @property
    def relative_to(self) -> str:
        return self.pose.relative_to


_Tracker = Union[TrackerRealsense, Tracker2DCameras]


@dataclass
class Tracker:
    type: str
    inner: _Tracker

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]) -> Self:
        _type = key2value.pop("type")
        if _type == '2d_cameras':
            _tracker = Tracker2DCameras.from_dict(key2value)
        elif _type == 'realsense':
            _tracker = TrackerRealsense.from_dict(key2value)
        else:
            raise KeyError(_type)

        return cls(type=_type, inner=_tracker)

    @classmethod
    def from_json(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text()))

    def track_faces(self, n_frame_to_average: int) -> Tuple[threading.Thread, MovingAverageND]:
        """Simple face tracking"""
        moving_average = MovingAverageND.new(n_frame_to_average)
        thread = threading.Thread(target=self.inner.track_faces, args=(moving_average,))
        thread.start()
        return thread, moving_average

    @property
    def relative_to(self) -> str:
        return self.inner.relative_to

class MovingMeasure(Generic[T]):
    def __init__(self, values: List[T], max_length: int) -> None:
        self.values = values
        self.max_length = max_length

    def push(self, value: T) -> None:
        self.values.append(value)
        if len(self.values) > self.max_length:
            self.values = self.values[1:]

    def get(self) -> Optional[T]:
        if self.values == []:
            return None
        return self._get()

    def _get(self):
        raise NotImplementedError

    @classmethod
    def new(cls: U, max_length: int) -> U:
        return cls([], max_length)


class MovingAverageND(MovingMeasure[Tuple[float, float]]):
    def _get(self) -> T:
        return np.mean(self.values, axis=0)
