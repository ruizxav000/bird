"""Build the Lilac-breasted Roller scene.

Run:  python src/build.py [--preview]   (with the bpy module installed)
Writes scene/roller.blend
"""
import os
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import bpy  # noqa: E402
import numpy as np  # noqa: E402
from mathutils import Vector  # noqa: E402

import sdf  # noqa: E402
import util  # noqa: E402
from util import srgb  # noqa: E402


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.scale_length = 1.0
    return sc


def clay(name, col):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = srgb(*col)
    bsdf.inputs["Roughness"].default_value = 0.6
    return m


def build_core(sc):
    import beak
    v, f = sdf.build_body()
    body = util.mesh_object("Body", v, f)
    util.smooth_mesh(body, 3, 0.5)
    v, f = sdf.build_head()
    head = util.mesh_object("Head", v - sdf.HEAD_PIVOT, f)
    util.smooth_mesh(head, 2, 0.5)

    pivot = bpy.data.objects.new("HeadPivot", None)
    sc.collection.objects.link(pivot)
    pivot.location = Vector(sdf.HEAD_PIVOT)
    pivot.rotation_mode = "XYZ"
    head.parent = pivot

    (uv, uf, uuv), (lv, lf, luv) = beak.build_bill()
    up = util.mesh_object("BillUpper", uv - sdf.HEAD_PIVOT, uf)
    lo = util.mesh_object("BillLower", lv - sdf.HEAD_PIVOT, lf)
    util.set_corner_uv(up.data, "UVMap", uuv)
    util.set_corner_uv(lo.data, "UVMap", luv)
    for o in (up, lo):
        o.parent = pivot

    eyes = []
    for sx in (1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=sdf.EYE_R)
        e = bpy.context.object
        e.name = "Eye_R" if sx > 0 else "Eye_L"
        bpy.ops.object.shade_smooth()
        e.location = Vector(sdf.EYE_C * (sx, 1, 1) - sdf.HEAD_PIVOT)
        # local +Z of the sphere = gaze axis (outward, a little forward)
        e.rotation_euler = Vector((0, math.radians(90 * sx), math.radians(-12 * sx)))
        e.parent = pivot
        eyes.append(e)
    return body, head, pivot, (up, lo), eyes


def build_camera_light(sc):
    cam_d = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Camera", cam_d)
    sc.collection.objects.link(cam)
    az = math.radians(18)
    dist = 5.4
    target = Vector((0, -0.03, 0.035))
    cam.location = target + Vector((math.cos(az) * dist, math.sin(az) * dist, -0.25))
    util.look_at(cam, target)
    cam_d.lens = 300
    cam_d.sensor_width = 36
    cam_d.sensor_fit = "AUTO"
    cam_d.shift_x = -0.07
    cam_d.shift_y = -0.02
    sc.camera = cam

    sun_d = bpy.data.lights.new("Sun", "SUN")
    sun_d.energy = 4.0
    sun_d.angle = math.radians(1.5)
    sun_d.color = (1.0, 0.80, 0.60)
    sun = bpy.data.objects.new("Sun", sun_d)
    sc.collection.objects.link(sun)
    # low golden-hour sun, front-right of the bird (it lights the face)
    sun.rotation_euler = (math.radians(74), 0, math.radians(120))
    return cam, sun


def build_world(sc):
    w = bpy.data.worlds.new("Sky")
    sc.world = w
    w.use_nodes = True
    nt = w.node_tree
    nt.nodes.clear()
    out = util.node(nt, "ShaderNodeOutputWorld", (800, 0))
    # what the camera sees: the flat, hazy pale-cyan sky of the reference
    tc = util.node(nt, "ShaderNodeTexCoord", (-800, 200))
    sep = util.node(nt, "ShaderNodeSeparateXYZ", (-600, 200))
    ramp = util.node(nt, "ShaderNodeValToRGB", (-400, 200))
    ramp.color_ramp.elements[0].position = 0.40
    ramp.color_ramp.elements[0].color = srgb(0.60, 0.77, 0.80)
    ramp.color_ramp.elements[1].position = 0.60
    ramp.color_ramp.elements[1].color = srgb(0.53, 0.72, 0.78)
    mr = util.node(nt, "ShaderNodeMapRange", (-500, 50))
    mr.inputs["From Min"].default_value = -0.3
    mr.inputs["From Max"].default_value = 0.3
    util.link(nt, tc.outputs["Generated"], sep.inputs[0])
    util.link(nt, sep.outputs["Z"], mr.inputs["Value"])
    util.link(nt, mr.outputs["Result"], ramp.inputs["Fac"])
    bg_cam = util.node(nt, "ShaderNodeBackground", (-100, 200))
    bg_cam.inputs["Strength"].default_value = 1.0
    util.link(nt, ramp.outputs["Color"], bg_cam.inputs["Color"])
    # what lights the bird: open sky dome + warm ground bounce
    sky = util.node(nt, "ShaderNodeValToRGB", (-400, -200))
    sky.color_ramp.elements[0].position = 0.45
    sky.color_ramp.elements[0].color = srgb(0.55, 0.47, 0.36)  # dry savanna bounce
    sky.color_ramp.elements[1].position = 0.55
    sky.color_ramp.elements[1].color = srgb(0.62, 0.76, 0.86)
    mr2 = util.node(nt, "ShaderNodeMapRange", (-600, -200))
    mr2.inputs["From Min"].default_value = -1
    mr2.inputs["From Max"].default_value = 1
    util.link(nt, sep.outputs["Z"], mr2.inputs["Value"])
    util.link(nt, mr2.outputs["Result"], sky.inputs["Fac"])
    bg_lit = util.node(nt, "ShaderNodeBackground", (-100, -200))
    bg_lit.inputs["Strength"].default_value = 0.9
    util.link(nt, sky.outputs["Color"], bg_lit.inputs["Color"])
    lp = util.node(nt, "ShaderNodeLightPath", (-100, 450))
    mix = util.node(nt, "ShaderNodeMixShader", (400, 0))
    util.link(nt, lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
    util.link(nt, bg_lit.outputs[0], mix.inputs[1])
    util.link(nt, bg_cam.outputs[0], mix.inputs[2])
    util.link(nt, mix.outputs[0], out.inputs["Surface"])
    # Generated coords on the world are the view direction
    tc.object = None


def setup_render(sc, preview):
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = 48 if preview else 256
    sc.cycles.use_denoising = True
    sc.render.resolution_x = 1080
    sc.render.resolution_y = 1920
    sc.render.resolution_percentage = 30 if preview else 100
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Base Contrast" if False else "None"
    sc.render.fps = 24


def main():
    preview = "--preview" in sys.argv
    sc = reset()
    body, head, pivot, bill, eyes = build_core(sc)
    body.data.materials.append(clay("ClayBody", (0.7, 0.5, 0.6)))
    head.data.materials.append(clay("ClayHead", (0.6, 0.7, 0.5)))
    for b in bill:
        b.data.materials.append(clay("ClayBill", (0.05, 0.05, 0.05)))
    for e in eyes:
        e.data.materials.append(clay("ClayEye", (0.2, 0.1, 0.05)))
    build_camera_light(sc)
    build_world(sc)
    setup_render(sc, preview)
    os.makedirs(os.path.join(ROOT, "scene"), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "scene", "roller.blend"))


if __name__ == "__main__":
    main()
