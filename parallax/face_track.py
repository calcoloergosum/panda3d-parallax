"""Simple face tracking"""
import threading
import time
from pathlib import Path
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import structlog

from .data import Xyxy, xyxy2xywh

LOGGER = structlog.get_logger(__name__)

T = TypeVar("T")
U = TypeVar("U")


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


def loop_frame(on_frame: Callable[[np.ndarray], bool]) -> bool:
    # Pretty much the standard way of capturing video stream...
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            LOGGER.critical("Cannot open camera")
            return False
        t_last = 0
        while True:
            t = time.time()
            # To throttle
            if t - t_last < .01:
                continue
            # Capture frame-by-frame
            LOGGER.info('Taking frame')
            ret, frame = cap.read()
            t_last = t
            if not ret:
                LOGGER.critical("Can't receive frame (stream end?). Exiting ...")
                break
            if not on_frame(frame):
                break
        return True
    finally:
        cap.release()


def loop_face_track_cv2(
    on_face: Callable[[np.ndarray, List[Tuple[int, int, int, int]]], bool],
) -> bool:
    face_cascade = cv2.CascadeClassifier(
        (Path(__file__).parent / 'resource' /
         'haarcascade_frontalface_default.xml').as_posix())
    # eye_cascade = cv2.CascadeClassifier('./haarcascade_eye.xml')
    def on_frame(frame: np.ndarray) -> np.ndarray:
        # For debugging ... nudge the face position from the center towards left-top of camera image
        # which is right-top of the monitor from user's point of view.
        # h, w = frame.shape[:2]
        # on_face(frame, [(int(w * 0.45), int(h * 0.45), int(w * 0.0), int(h * 0.2))])
        # return True
        LOGGER.info('Detecting face')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xywhs = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(xywhs) == 0:
            return True
        # use only the biggest bbox
        on_face(xywhs)
        return True
    return loop_frame(on_frame)


def loop_face_track_realsense(on_face: Callable[[List[Xyxy], np.ndarray], bool]):
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


def start_face_track(n_frame_to_average: int, method: str) -> MovingAverageND:
    moving_average = MovingAverageND.new(n_frame_to_average)

    def on_face_2d(
        xywhs: List[Tuple[int, int, int, int]],
    ) -> bool:
        if len(xywhs) == 0:
            return
        x, y, w, h = sorted(xywhs, key=lambda xywh: xywh[2] * xywh[3])[-1]
        xy = ((x + 0.5 * w, y + 0.5 * h))
        moving_average.push(xy)
        return

    def on_face_3d(
        xywhs: List[Tuple[int, int, int, int]],
        depth_image: np.ndarray,
    ) -> bool:
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
        moving_average.push((*xy, d))
        return

    if method == 'cv2':
        loop_face_track = loop_face_track_cv2
        on_face = on_face_2d
    elif method == 'realsense':
        loop_face_track = loop_face_track_realsense
        on_face = on_face_3d
    else:
        raise KeyError(f"Unknown method {method}")
    thread = threading.Thread(target=loop_face_track, args=(on_face,))
    thread.start()
    return moving_average
