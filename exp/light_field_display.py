#!/usr/bin/env python

import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np

import cv2
import numpy as np
from pathlib import Path
import re
import itertools as it


def main():
    gui.Application.instance.initialize()
    fov = 90
    loc_normalized = np.array((0, 0, 0))
    l2r_normalized = np.array((.05, 0, 0))

    print("[1/3] Setting up window")
    w = gui.Application.instance.create_window("Two scenes", 1025, 512)
    scene1 = gui.SceneWidget()
    scene1.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
    scene2 = gui.SceneWidget()
    scene2.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

    w.add_child(scene1)
    w.add_child(scene2)

    def on_layout(_):
        r = w.content_rect
        scene1.frame = gui.Rect(r.x, r.y, r.width / 2, r.height)
        scene2.frame = gui.Rect(r.x + r.width / 2 + 1, r.y, r.width / 2, r.height)

    print("[2/3] Setting up scene")
    pcd, wh = load_pcd()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2
    scene1.scene.add_geometry("pcd", pcd, mat)
    scene2.scene.add_geometry("pcd", pcd, mat)

    print("[3/3] Setting up cameras")
    s = min(wh)
    l2r = s * l2r_normalized
    loc = s * loc_normalized
    pos1, pos2 = loc -.5 * l2r, loc + .5 * l2r
    scene1.setup_camera(fov, scene1.scene.bounding_box, pos1)
    scene2.setup_camera(fov, scene2.scene.bounding_box, pos2)

    def on_m_event(something):
        cam = scene2.scene.camera.get_model_matrix()
        m = cam
        drx = (m[:3, :3] @ (0, 0, -1)).squeeze()
        up  = (m[:3, :3] @ (0, 1,  0)).squeeze()
        pos = m[:3, 3] - l2r
        scene1.look_at(pos + drx, pos, up)
        scene1.force_redraw()
        if something.type == gui.MouseEvent.Type.BUTTON_DOWN:
            print(f"[debug] Current pos: {m[:3, 3]} {pos}")
            print("[debug] mouse:", (something.x, something.y))
        return gui.Widget.EventCallbackResult.IGNORED

    w.set_on_layout(on_layout)
    scene1.set_on_mouse(on_m_event)
    scene2.set_on_mouse(on_m_event)

    gui.Application.instance.run()


def load_color_depth(a: float, b: float):
    regex = re.compile("\\d+")

    idx1, color_path = sorted(it.chain.from_iterable(zip(map(int, regex.findall(p.stem)), it.repeat(p)) for p in Path("data/comfyui/").glob("rgb_*.png")))[-1]
    idx2, depth_path = sorted(it.chain.from_iterable(zip(map(int, regex.findall(p.stem)), it.repeat(p)) for p in Path("data/comfyui/").glob("depth_*.tiff")))[-1]
    color = cv2.imread(color_path.as_posix())[:, :, ::-1].copy()  # BGR -> RGB
    depth = cv2.imread(depth_path.as_posix(), cv2.IMREAD_ANYDEPTH)
    assert color is not None and depth is not None
    val, _ = cv2.threshold(depth, None, 256, type=cv2.THRESH_OTSU)
    med = np.median(depth[depth < val])
    h, w = color.shape[:2]
    depth = cv2.resize(depth, (w, h))
    if depth.dtype == np.uint16:
        depth = depth / 65535
        med = med / 65535
        val = val / 65535
    elif depth.dtype == np.uint8:
        depth = depth / 255
        med = med / 255
        val = val / 255
    else:
        raise TypeError

    # q1, q2 = np.quantile(depth[depth < val], [0.25, 0.75])
    # stdev = np.sqrt(.5 * ((med - q1) ** 2 + (med - q2) ** 2))
    depth = (b + a * (1 - depth)).astype(np.float32)
    depth[depth > 1000] = 1000
    return o3d.geometry.Image(color), o3d.geometry.Image(depth)
    # depth = (1000 / np.clip(depth, 1e-7, None)).astype(np.float32)
    # depth[depth > 10000] = 10000


def load_pcd():
    a, b = 1., 1.
    color_raw, depth_raw = load_color_depth(a, b)
    h, w = np.asarray(color_raw).shape[:2]
    f_ = min(w, h)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)
    intr = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=f_, fy=f_, cx=w/2, cy=h/2)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr)
    # HACK: issue with Open3D, default depth scale is 1000
    scale = f_ * 1000
    pcd.transform(np.diag((scale, scale, scale, 1)))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd, (w, h)


if __name__ == "__main__":
    main()
