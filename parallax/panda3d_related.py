"""Panda3d rendering related stuff"""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import panda3d.core
import structlog
from direct.showbase.ShowBase import ShowBase

from parallax.data import Hpr, ProjectionMatrix, Window, Xyz, hpr2mat

LOGGER = structlog.get_logger()

class FakeWindowApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.objects = []

    def setup(self, window: Window) -> None:
        # setting windows up
        winprops = panda3d.core.WindowProperties()
        winprops.setSize(*window.size)
        self.win.requestProperties(winprops)
        self.render.setShaderAuto()
        self.setBackgroundColor(0,0,0)

    def load_model(self, path: Path) -> None:
        if path.suffix == '.dae':
            daepath = path
            compressed = False
            if not daepath.exists() and (daepath.parent / (daepath.stem +'.gz')).exists():
                subprocess.check_call(['gunzip', daepath+'.gz'])
                compressed = True
            (fd, eggpath) = tempfile.mkstemp(suffix='.egg')
            os.close(fd)
            subprocess.check_call(['dae2egg', '-o', eggpath, daepath])
            print(f'loading {daepath}')
            model = self.loader.loadModel(eggpath)
            os.remove(eggpath)
            if compressed:
                subprocess.check_call(['gzip', daepath])
            path = eggpath
        elif path.suffix in ('.wrl', '.egg', '.gltf'):
            print(f'loading {path}')
            model = self.loader.loadModel(path)
        else:
            raise NotImplementedError(path.suffix)
        return model

    def place_model(self,
        model: panda3d.core.NodePath,
        at: Xyz, hpr: Hpr,
        scale_to: float):
        # Determine scale and translation using bounding volume
        trans_center = np.identity(4)

        xs, ys, zs = get_quantile(model, (0.001, 0.5, 0.999))
        trans_center[:3, 3] -= np.array((xs[1], ys[1], zs[1]))
        factor = scale_to / max(xs[2] - xs[0], ys[2] - ys[0], zs[2] - zs[0])
        scale = np.diag((factor, factor, factor, 1))

        # Use given yaw-pitch-roll
        mat_rot = np.identity(4)
        mat_rot[:3, :3] = hpr2mat(hpr)

        trans_to_given = np.identity(4)
        trans_to_given[:3, 3] = at

        mat = trans_to_given @ mat_rot @ scale @ trans_center
        model.set_mat(panda3d.core.LMatrix4f(*mat.T.flatten()))
        model.setTwoSided(True)
        model.reparentTo(self.render)
        self.objects.append(model)

    def add_box(self, window: Window):
        # pylint: disable=too-many-locals
        hpr = window.pose.hpr
        assert hpr[2] % 90 == 0, "roll of the monitor should be a multiple of the right angle"
        assert -45 <= hpr[1] <= 45 and -45 <= hpr[0] < 45, "Still working on it ..."

        mpp = window.mms_per_pixel
        d = .5 * np.mean(window.size_mm) * mpp

        w, h = window.width, window.height
        xyz = window.pose.xyz
        *luf, _ = (window.pose.as_mat() @ (-.5*w, 0,  .5*h, 1) - (*xyz, 0)) * mpp
        *ruf, _ = (window.pose.as_mat() @ ( .5*w, 0,  .5*h, 1) - (*xyz, 0)) * mpp
        *ldf, _ = (window.pose.as_mat() @ (-.5*w, 0, -.5*h, 1) - (*xyz, 0)) * mpp
        *rdf, _ = (window.pose.as_mat() @ ( .5*w, 0, -.5*h, 1) - (*xyz, 0)) * mpp
        lub = (luf[0], -d, luf[2])
        rub = (ruf[0], -d, ruf[2])
        ldb = (ldf[0], -d, ldf[2])
        rdb = (rdf[0], -d, rdf[2])

        format_ = panda3d.core.GeomVertexFormat.getV3c4()
        data = panda3d.core.GeomVertexData("Data", format_, panda3d.core.Geom.UHStatic)
        vertices = panda3d.core.GeomVertexWriter(data, "vertex")
        # colors = panda3d.core.GeomVertexWriter(data, 'color')

        vertices.addData3f(*ldb)
        # colors.addData4f(0.0, 1.0, 0.0, .5)
        vertices.addData3f(*rdb)
        # colors.addData4f(1.0, 1.0, 0.0, .5)
        vertices.addData3f(*ldf)
        # colors.addData4f(0.0, 0.0, 0.0, .5)
        vertices.addData3f(*rdf)
        # colors.addData4f(1.0, 0.0, 0.0, .5)
        vertices.addData3f(*lub)
        # colors.addData4f(0.0, 1.0, 1.0, .5)
        vertices.addData3f(*rub)
        # colors.addData4f(1.0, 1.0, 1.0, .5)
        vertices.addData3f(*luf)
        # colors.addData4f(0.0, 0.0, 1.0, .5)
        vertices.addData3f(*ruf)
        # colors.addData4f(1.0, 0.0, 1.0, .5)

        triangles = panda3d.core.GeomTriangles(panda3d.core.Geom.UHStatic)
        def quad(v0, v1, v2, v3):
            triangles.addVertices(v0, v1, v2)
            triangles.addVertices(v0, v2, v3)
            triangles.closePrimitive()

        quad(4, 5, 7, 6) # Z+
        quad(0, 2, 3, 1) # Z-
        quad(3, 7, 5, 1) # X+
        quad(4, 6, 2, 0) # X-
        # addQuad(2, 6, 7, 3) # Y+
        quad(0, 1, 5, 4) # Y-

        geom = panda3d.core.Geom(data)
        geom.addPrimitive(triangles)

        node = panda3d.core.GeomNode("Box")
        node.addGeom(geom)

        nodepath = panda3d.core.NodePath(node)
        nodepath.reparentTo(self.render)
        nodepath.setTwoSided(True)
        self.objects.append(nodepath)

    def setup_lighting(self, at: Xyz) -> None:
        s = .3
        alight = panda3d.core.AmbientLight("ambient light")
        alight.setColor(panda3d.core.Vec4(s, s, s, 1))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        self.objects.append(alnp)

        def add_light(name, x, y, z, r, g, b):
            dlight = panda3d.core.PointLight(name)
            dlight.setColor(panda3d.core.VBase4(r, g, b, 1))
            dlnp = self.render.attachNewNode(dlight)
            dlnp.setPos(x, y, z)
            self.render.setLight(dlnp)
            self.objects.append(dlnp)
        x, y, z = at
        d = 80
        add_light('--', x-d, y-d, z, .5, .1, .3)
        add_light('+-', x+d, y-d, z, .1, .5, .1)
        add_light('++', x+d, y+d, z, .1, .1, .5)
        add_light('-+', x-d, y+d, z, .5, .2, .1)

    def apply_offset(self, xyz_cam: Xyz, window: Window) -> None:
        # pylint: disable=too-many-locals
        # xyz_cam described from monitor coordinates system
        window2world = window.pose.as_mat()
        world2window = np.linalg.inv(window2world)
        *xyz_cam_, _ = world2window @ (*xyz_cam, 1)

        # Camera's position projected to monitor plane
        flip_z = np.array((
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ))
        window2cam = flip_z @ np.array((
            (1, 0, 0, - xyz_cam_[0]),
            (0, 1, 0, - xyz_cam_[1]),
            (0, 0, 1, - xyz_cam_[2]),
            (0, 0, 0, 1),
        ))
        world2cam = window2cam @ world2window
        cam2world = np.linalg.inv(world2cam)
        np.testing.assert_almost_equal(world2cam @ (*xyz_cam, 1), (0, 0, 0, 1))
        np.testing.assert_almost_equal(cam2world @ (0, 0, 0, 1), (*xyz_cam, 1))

        # When up vector is fixed, the above is equivalent to:
        # *xyz_proj, _ = window2world @ (xyz_cam_[0], 0, xyz_cam_[2], 1)
        # np.testing.assert_almost_equal(world2cam @ (*xyz_proj, 1), (0, xyz_cam_[1], 0, 1))
        # self.cam.setPos(*xyz_camera)
        # self.cam.lookAt(*xyz_proj)

        # monitor's points described from camera
        w, h = window.size_mm

        _, y, _ = xyz_cam_
        if y == 0:
            LOGGER.critical("encountered depth 0!!")
            return
        # xyz_cam
        l, _, b, _ = world2cam @ window2world @ ( .5 * w, 0, -.5 * h, 1) / y
        r, _, t, _ = world2cam @ window2world @ (-.5 * w, 0,  .5 * h, 1) / y

        # 0.239, -0.239, 0.148, 0.429
        self.cam.setMat(panda3d.core.LMatrix4f(*cam2world.T.flatten()))
        np.testing.assert_almost_equal(self.cam.getPos(), xyz_cam, decimal=4)
        lens = panda3d.core.MatrixLens()
        lens.setCoordinateSystem(panda3d.core.CSYupRight)
        lens.setUserMat(
            ProjectionMatrix.from_frustum(
                l, r, b, t, 1, 10000,
            ).as_panda3d()
        )
        lens.setFilmSize(2)
        self.set_lens(lens)

    def set_lens(self, lens: panda3d.core.Lens) -> None:
        self.cam.node().setLens(lens)
        self.cam.node().setLens(lens)

    def reset_camera(self, xyz_camera: Tuple[float, float, float]) -> None:
        # spherical coordinates
        self.cam.setPos(*xyz_camera)
        self.cam.lookAt(panda3d.core.Point3(0, 0, 0))

    def clear_objects(self):
        for obj in self.objects:
            obj.detachNode()


def for_each_vertex(model, callback: Callable[[Xyz], None]):
    # pylint: disable=too-many-nested-blocks
    for node in model.node().get_children():
        if node.is_geom_node():
            for geom in node.get_geoms():
                vdata = geom.getVertexData()
                vertex = panda3d.core.GeomVertexReader(vdata, 'vertex')
                for prim in geom.get_primitives():
                    for p in range(prim.getNumPrimitives()):
                        s = prim.getPrimitiveStart(p)
                        e = prim.getPrimitiveEnd(p)
                        for i in range(s, e):
                            vi = prim.getVertex(i)
                            vertex.setRow(vi)
                            callback(vertex.getData3())


def get_quantile(model, qs):
    xs, ys, zs = [], [], []
    def update_list(xyz: Xyz) -> None:
        x, y, z = xyz
        xs.append(x)
        ys.append(y)
        zs.append(z)
    for_each_vertex(model, update_list)
    return (
        np.quantile(xs, qs),
        np.quantile(ys, qs),
        np.quantile(zs, qs),
    )


def get_bounding_volume(model) -> Tuple[float, float, float, float, float, float]:
    xmin, ymin, zmin = np.inf, np.inf, np.inf
    xmax, ymax, zmax = - np.inf, - np.inf, - np.inf

    def update_bvol(xyz: Xyz) -> None:
        nonlocal xmin, xmax, ymin, ymax, zmin, zmax
        x, y, z = xyz
        xmin, xmax = min(xmin, x), max(xmax, x)
        ymin, ymax = min(ymin, y), max(ymax, y)
        zmin, zmax = min(zmin, z), max(zmax, z)
    for_each_vertex(model, update_bvol)
    return xmin, xmax, ymin, ymax, zmin, zmax
