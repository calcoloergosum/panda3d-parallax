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
    parser.add_argument('tracker', help='refer to configs/tracker', type=Path)
    parser.add_argument('monitor', help='Monitor configuration. Refer to configs/monitor', type=Path)
    parser.add_argument('--model-hpr', help='yaw-pitch-roll of model in degrees',
                        type=ast.literal_eval, default=(0, 0, 0))
    args = parser.parse_args()

    monitor = parallax.data.Window.from_json(args.monitor)
    tracker = parallax.tracker.Tracker.from_json(args.tracker)

    app = FakeWindowApp()
    app.setup(monitor)
    app.setup_lighting(at=monitor.pose.xyz,)
    model = app.load_model(args.model)
    model_scale = np.mean(monitor.size_mm) / 3
    app.place_model(
        model,
        at=monitor.pose.xyz,
        hpr=args.model_hpr,
        scale_to=model_scale,
    )
    app.add_box(window=monitor)
    # app.reset_camera((0, args.distance, 0))

    thread_track, moving_average = tracker.track_faces(5)

    def on_update(task):
        xyz = moving_average.get()
        if xyz is None:
            return task.cont
        # convert opengl coordinates to pandas coordinates
        xyz = (xyz[0], xyz[2], xyz[1])

        if tracker.inner.master.pose.relative_to == 'window':
            *xyz_world, _ = monitor.pose.as_mat() @ (*xyz, 1)
        else:
            xyz_world = xyz
        app.apply_offset(xyz_world, monitor)
        return task.cont

    def on_window_change(win: panda3d.core.GraphicsWindow):
        if (monitor.width, monitor.height) == win.properties.size:
            return
        monitor.width, monitor.height = win.properties.size
        app.clear_objects()
        app.place_model(
            model,
            at=monitor.pose.xyz,
            hpr=args.model_hpr,
            scale_to=model_scale,
        )
        app.add_box(window=monitor)

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
