"""Simple face tracking"""
import threading
import time
from pathlib import Path
from typing import Callable, Generic, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import structlog

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


class MovingAverage2D(MovingMeasure[Tuple[float, float]]):
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


def loop_face_track(
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
        on_face(frame, xywhs)
        return True
    return loop_frame(on_frame)


def start_face_track(n_frame_to_average: int, debug: bool = False):
    moving_average = MovingAverage2D.new(n_frame_to_average)

    def on_face(
        frame: np.ndarray, xywhs: List[Tuple[int, int, int, int]],
    ) -> bool:
        if len(xywhs) == 0:
            return
        x, y, w, h = sorted(xywhs, key=lambda xywh: xywh[2] * xywh[3])[-1]
        if debug:
            for (x,y,w,h) in xywhs:
                cv2.rectangle(frame, (x,y), (x+w, y+h),(255,0,0),2)
            cv2.imshow('frame', frame)
        xy = ((x + 0.5 * w, y + 0.3 * h))
        moving_average.push(xy)
    thread = threading.Thread(target=loop_face_track, args=(on_face,))
    thread.start()
    return moving_average
