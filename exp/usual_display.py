import open3d as o3d
print(o3d.__version__)

import cv2
import numpy as np
from pathlib import Path
import re
import itertools as it

regex = re.compile("\\d+")

idx1, color_path = sorted(it.chain.from_iterable(zip(map(int, regex.findall(p.stem)), it.repeat(p)) for p in Path("data/comfyui/").glob("rgb_*.png")))[-1]
idx2, depth_path = sorted(it.chain.from_iterable(zip(map(int, regex.findall(p.stem)), it.repeat(p)) for p in Path("data/comfyui/").glob("depth_*.tiff")))[-1]
print(idx1, idx2)
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

q1, q2 = np.quantile(depth[depth < val], [0.25, 0.75])
stdev = np.sqrt(.5 * ((med - q1) ** 2 + (med - q2) ** 2))
print(med)
depth = (0.7 + (1 - depth)).astype(np.float32)
depth[depth > 1000] = 1000
# depth = (1000 / np.clip(depth, 1e-7, None)).astype(np.float32)
# depth[depth > 10000] = 10000

color_raw = o3d.geometry.Image(color)
depth_raw = o3d.geometry.Image(depth)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)


intr = o3d.camera.PinholeCameraIntrinsic(width=w, height=h, fx=w, fy=h, cx=w/2, cy=h/2)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intr)
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
print("[1/3] Showing Point Cloud")
o3d.visualization.draw_geometries([pcd], width=2 * w, height=2 * h)

# tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
# for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
#     print(f"alpha={alpha:.3f}")
#     mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#         pcd, alpha, tetra_mesh, pt_map)
#     mesh.compute_vertex_normals()
# o3d.visualization.draw_geometries([tetra_mesh], mesh_show_back_face=True)