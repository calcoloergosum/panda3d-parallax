"""Data types defined here!"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
import panda3d.core
from typing_extensions import Self

Size1f = float
Size2i = Tuple[int, int]
Size2f = Tuple[float, float]
Point2f = np.ndarray  # location in 2D; shape = (2,)
Tangent3f = np.ndarray  # tangent vector; shape = (3,)
Radian = float
Degree = float
Size1f = float
Xyz = Tuple[float, float, float]  # location vector; shape = (3,)
Hpr = Tuple[float, float, float]  # Yaw, Pitch, Roll in degrees
AxisAngle = Tuple[float, float, float]  # Rodrigues representation
Xyxy = Tuple[float, float, float, float]  # Bbox format
Xywh = Tuple[float, float, float, float]  # Bbox format
Matrix3f = Tuple[Xyz, Xyz, Xyz]


def xyxy2xywh(xyxys: Xyxy) -> Xywh:
    if len(xyxys) == 0:
        return []
    xmin, ymin, xmax, ymax = np.array(xyxys).T
    return np.array((xmin, ymin, xmax - xmin, ymax - ymin)).T


@dataclass
class Sensor:
    resolution: Size2i


@dataclass
class Webcam:
    fov_deg: Degree  # angle of max(sensor_resolution)
    sensor: Sensor
    pose: Pose

    def __post_init__(self) -> None:
        # pylint: disable=not-a-mapping
        if isinstance(self.sensor, dict):
            self.sensor = Sensor(**self.sensor)

    def angle2pixels(self, angle: Radian) -> Size1f:
        """angle from the center to pixels"""
        return max(self.sensor.resolution) * np.tan(angle) / np.tan(self.fov_rad / 2)

    def pixels2angle(self, pixels: Size1f) -> Radian:
        x = max(self.sensor.resolution)
        y = pixels * np.tan(self.fov_rad / 2)
        return np.arctan2(y, x)

    @property
    def fov_rad(self) -> Radian:
        return np.radians(self.fov_deg)

    def fov_mm_per_z(self) -> Size1f:
        return 2 * np.tan(self.fov_rad / 2)

    def mm_per_pixel(self, distance_to_plane: Size1f) -> Size1f:
        return self.fov_mm_per_z() * distance_to_plane / max(self.sensor.resolution)

    def image2cam(self, xy_on_image, distance: float):
        x_w, _z_w = self.mm_per_pixel(distance) * (
            np.array(xy_on_image) - .5 * np.array(self.sensor.resolution))
        *xyz, _= self.pose.as_mat() @ (x_w, distance, - _z_w, 1)
        return xyz

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]):
        return cls(
            sensor=Sensor(**key2value.pop("sensor")),
            pose=Pose.from_container(key2value),
            **key2value
        )


@dataclass
class Window:
    width: int
    height: int
    pose: Pose
    pixels_per_mm: Size1f

    @property
    def mms_per_pixel(self):
        return 1 / self.pixels_per_mm

    @property
    def size(self) -> Size2i:
        return (self.width, self.height)

    @property
    def size_mm(self) -> Size2f:
        return tuple(k / self.pixels_per_mm for k in self.size)

    @classmethod
    def from_dict(cls, key2value: Dict[str, Any]):
        return cls(
            pose=Pose.from_container(key2value),
            **key2value
        )

    @classmethod
    def from_json(cls, path: Path) -> Self:
        return cls.from_dict(json.loads(path.read_text()))

class Matrix4f:
    """Adapter for column-vector style to panda3d row-vector style"""
    def __init__(self, *args) -> None:
        """Nothing much"""
        assert len(args) == 16
        self.args = args

    @classmethod
    def from_np(cls, arr: np.ndarray) -> Matrix4f:
        """matrix"""
        return cls(*arr.T.flatten())

    def as_panda3d(self) -> panda3d.core.LMatrix4f:
        """matrix"""
        # mat[[2, 1]] = mat[[1, 2]]
        args = self.args[:4] + self.args[8:12] + self.args[4:8] + self.args[12:16]
        return panda3d.core.LMatrix4f(*args)


@dataclass
class Pose:
    """pose of a rigid body in 3D"""
    xyz: Xyz
    hpr: Optional[Hpr] = None
    rvec: Optional[AxisAngle] = None
    relative_to: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.xyz, tuple):
            self.xyz = tuple(self.xyz)
        if self.hpr is not None:
            if not isinstance(self.hpr, tuple):
                self.hpr = tuple(self.hpr)
        if self.rvec is not None:
            if not isinstance(self.rvec, tuple):
                self.rvec = tuple(self.rvec)

    def rot_mat(self):
        if self.hpr is not None:
            return hpr2mat(self.hpr)
        if self.rvec is not None:
            return cv2.Rodrigues(self.rvec)[0]

    def as_mat(self):
        """Returns 3d affine transformation that maps
        local coordinates to world coordinates"""
        mat = np.identity(4)
        mat[:3, :3] = hpr2mat(self.hpr)
        mat[:3, 3] = self.xyz
        return mat

    @classmethod
    def from_container(cls, a_dict):
        return cls(**a_dict.pop("pose", {"xyz": (0, 0, 0), "hpr": (0, 0, 0)}))


class ProjectionMatrix(Matrix4f):
    @classmethod
    def from_frustum(
        cls,
        left: float, right: float,
        bottom: float, top: float,
        near: float, far: float,
    ) -> ProjectionMatrix:
        """https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml"""
        x = 2.0 * near / (right - left)
        y = 2.0 * near / (top - bottom)
        a, b = - (right + left) / (right - left), - (top + bottom) / (top - bottom)
        c, d = (far + near) / (far - near), - (2 * far * near) / (far - near)
        e = 1.0

        mat = np.array((
            (x, 0, a, 0),
            (0, y, b, 0),
            (0, 0, c, d),
            (0, 0, e, 0),
        ))

        return cls.from_np(mat)


def hpr2quat(hpr: Hpr) -> panda3d.core.Quat():
    """HPR to panda3d Quaternion"""
    quat = panda3d.core.Quat()
    quat.setHpr(hpr)
    return quat


def hpr2mat(hpr: Hpr) -> np.ndarray:
    """HPR to column-vector matrix format"""
    quat = hpr2quat(hpr)
    return np.array((quat.get_right(), quat.get_forward(), quat.get_up())).T
