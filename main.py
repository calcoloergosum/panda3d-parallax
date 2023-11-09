"""Parallax Demo Program

To use with regular webcam, put `--track cv2`.
To use with realsenses cameras, put `--track realsense`.

TODO: Make camera settings configurable.
      Now is hardcoded in `parallax/monitor_webcam.py`
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import panda3d.core

import parallax
from parallax.panda3d_related import FakeWindowApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='(Base) path of model', type=Path)
    parser.add_argument('config', help='refer to configs/', type=Path)
    parser.add_argument('--distance', help='distance to the model in mm', type=float, default=2000)
    parser.add_argument('--model-hpr', help='yaw-pitch-roll of model in degrees',
                        type=ast.literal_eval, default=(0, 0, 0))
    parser.add_argument('--track', action='store', choices=('cv2', 'realsense'), default='cv2')
    args = parser.parse_args()

    cnf = parallax.monitor_webcam.RealWorldSetup.from_json(args.config)

    app = FakeWindowApp()
    app.setup(cnf.window)
    app.setup_lighting(at=cnf.window.pose.xyz,)
    model = app.load_model(args.model)
    app.place_model(
        model,
        at=cnf.window.pose.xyz,
        hpr=args.model_hpr,
        scale_to=max(cnf.window.size_mm) / 2,
    )
    app.add_box(window=cnf.window)
    app.reset_camera((0, args.distance, 0))

    moving_average = parallax.face_track.start_face_track(5, method=args.track)

    def on_update(task):
        xy = moving_average.get()
        if xy is None:
            return task.cont
        if len(xy) == 2:
            xyz_cam = cnf.camera.image2cam(xy, distance=args.distance)
        if len(xy) == 3:
            # print(xy)
            xyz_cam = cnf.camera.image2cam(xy[:2], distance=xy[2])
        if cnf.camera.pose.relative_to == 'window':
            *xyz_world, _ = cnf.window.pose.as_mat() @ (*xyz_cam, 1)
        else:
            xyz_world = xyz_cam
        app.apply_offset(xyz_world, cnf.window)
        return task.cont

    def on_window_change(win: panda3d.core.GraphicsWindow):
        if (cnf.window.width, cnf.window.height) == win.properties.size:
            return
        cnf.window.width, cnf.window.height = win.properties.size
        app.clear_objects()
        app.place_model(
            model,
            at=cnf.window.pose.xyz,
            hpr=args.model_hpr,
            scale_to=max(cnf.window.size_mm) / 2,
        )
        app.add_box(window=cnf.window)

    app.updateTask = app.taskMgr.add(on_update, "update")
    app.accept('window-event', on_window_change)
    app.run()

if __name__ == '__main__':
    import logging

    import structlog
    logging.basicConfig(level=logging.CRITICAL)
    structlog.configure(
        processors=[
            # If log level is too low, abort pipeline and throw away log entry.
            structlog.stdlib.filter_by_level
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    np.set_printoptions(precision=3)
    main()
