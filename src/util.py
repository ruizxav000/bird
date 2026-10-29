import bpy
import bmesh
import numpy as np


def srgb(r, g, b, a=1.0):
    """sRGB 0-1 -> linear RGBA (Blender colour sockets are scene-linear)."""
    def f(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    return (f(r), f(g), f(b), a)


def srgb_np(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def mesh_object(name, verts, faces, smooth=True, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(np.asarray(verts).tolist(), [], [list(map(int, f)) for f in faces])
    me.validate()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def set_point_attr(me, name, values, kind="FLOAT"):
    """kind: FLOAT, FLOAT_VECTOR, FLOAT_COLOR"""
    if name in me.attributes:
        me.attributes.remove(me.attributes[name])
    a = me.attributes.new(name, kind, "POINT")
    v = np.asarray(values, dtype=np.float32)
    if kind == "FLOAT":
        a.data.foreach_set("value", v.ravel())
    elif kind == "FLOAT_VECTOR":
        a.data.foreach_set("vector", v.ravel())
    else:
        a.data.foreach_set("color", v.ravel())
    return a


def set_corner_uv(me, name, uv_per_vertex):
    uvl = me.uv_layers.new(name=name)
    loops = np.empty(len(me.loops), dtype=np.int32)
    me.loops.foreach_get("vertex_index", loops)
    uvl.data.foreach_set("uv", np.asarray(uv_per_vertex, np.float32)[loops].ravel())
    return uvl


def get_verts(me):
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    return co.reshape(-1, 3).astype(float)


def get_normals(me):
    n = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("normal", n)
    return n.reshape(-1, 3).astype(float)


def smooth_mesh(ob, iterations=4, factor=0.5):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    for _ in range(iterations):
        bmesh.ops.smooth_vert(bm, verts=bm.verts, factor=factor,
                              use_axis_x=True, use_axis_y=True, use_axis_z=True)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def look_at(ob, target, up=(0, 0, 1)):
    from mathutils import Vector
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def node(nt, typ, loc=(0, 0), **props):
    n = nt.nodes.new(typ)
    n.location = loc
    for k, v in props.items():
        setattr(n, k, v)
    return n


def link(nt, a, b):
    nt.links.new(a, b)


def tube(points, radii, sides=12, cap=True, twist_ref=(0, 0, 1)):
    """Sweep a circle along a polyline. Returns verts, faces, uv(per vert),
    plus per-vertex (u_around, v_along_metres)."""
    P = np.asarray(points, float)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = np.asarray(twist_ref, float)
    # parallel transport frame
    N = np.zeros_like(P)
    v = np.cross(T[0], ref)
    if np.linalg.norm(v) < 1e-6:
        v = np.cross(T[0], (1, 0, 0))
    N[0] = v / np.linalg.norm(v)
    for i in range(1, n):
        b = np.cross(T[i - 1], T[i])
        if np.linalg.norm(b) < 1e-9:
            N[i] = N[i - 1]
        else:
            b /= np.linalg.norm(b)
            ang = np.arccos(np.clip(T[i - 1] @ T[i], -1, 1))
            nv = N[i - 1]
            nv = (nv * np.cos(ang) + np.cross(b, nv) * np.sin(ang)
                  + b * (b @ nv) * (1 - np.cos(ang)))
            N[i] = nv / np.linalg.norm(nv)
    B = np.cross(T, N)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    verts, uv = [], []
    for i in range(n):
        for j in range(sides + 1):
            a = 2 * np.pi * j / sides
            verts.append(P[i] + radii[i] * (np.cos(a) * N[i] + np.sin(a) * B[i]))
            uv.append((j / sides, seg[i]))
    faces = []
    w = sides + 1
    for i in range(n - 1):
        for j in range(sides):
            a = i * w + j
            faces.append((a, a + 1, a + w + 1, a + w))
    if cap:
        tip = len(verts)
        verts.append(P[-1] + T[-1] * radii[-1] * 0.6)
        uv.append((0.5, seg[-1] + radii[-1]))
        for j in range(sides):
            a = (n - 1) * w + j
            faces.append((a, a + 1, tip))
    return np.array(verts), faces, np.array(uv)
