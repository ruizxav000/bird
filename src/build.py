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
    pivot.location = Vector(sdf.HEAD_PIVOT) + Vector((0, -0.001, -0.004))
    pivot.scale = (1.08, 1.08, 1.08)
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
        e.rotation_euler = Vector((0, math.radians(90 * sx), math.radians(12 * sx)))
        e.parent = pivot
        eyes.append(e)
        # cornea: clear dome over the iris
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=sdf.EYE_R * 1.035)
        c = bpy.context.object
        c.name = "Cornea_" + ("R" if sx > 0 else "L")
        bpy.ops.object.shade_smooth()
        c.parent = e
        c.scale = (1, 1, 1.04)
        # lids: dark bare-skin ring framing the eye
        bpy.ops.mesh.primitive_torus_add(major_radius=sdf.EYE_R * 0.88, minor_radius=sdf.EYE_R * 0.09,
                                         major_segments=48, minor_segments=12)
        lid = bpy.context.object
        lid.name = "Lid_" + ("R" if sx > 0 else "L")
        bpy.ops.object.shade_smooth()
        lid.parent = e
        lid.location = (0, 0, sdf.EYE_R * 0.47)
    return body, head, pivot, (up, lo), eyes


def build_camera_light(sc):
    cam_d = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Camera", cam_d)
    sc.collection.objects.link(cam)
    az = math.radians(2)
    dist = 5.4
    target = Vector((0, -0.03, 0.035))
    cam.location = target + Vector((math.cos(az) * dist, math.sin(az) * dist, -0.25))
    util.look_at(cam, target)
    cam_d.lens = 330
    cam_d.sensor_width = 36
    cam_d.sensor_fit = "AUTO"
    cam_d.shift_x = 0.02
    cam_d.shift_y = -0.02
    sc.camera = cam
    # inspection camera: tight on the bird, for look-dev only
    cd2 = bpy.data.cameras.new("CloseCam")
    cam2 = bpy.data.objects.new("CloseCam", cd2)
    sc.collection.objects.link(cam2)
    t2 = Vector((0, 0.01, 0.075))
    cam2.location = t2 + Vector((math.cos(az) * dist, math.sin(az) * dist, -0.1))
    util.look_at(cam2, t2)
    cd2.lens = 800
    cd3 = bpy.data.cameras.new("HeadCam")
    cam3 = bpy.data.objects.new("HeadCam", cd3)
    sc.collection.objects.link(cam3)
    t3 = Vector((0, 0.03, 0.125))
    cam3.location = t3 + Vector((math.cos(az) * dist, math.sin(az) * dist, 0.0))
    util.look_at(cam3, t3)
    cd3.lens = 2200
    cd4 = bpy.data.cameras.new("FeetCam")
    cam4 = bpy.data.objects.new("FeetCam", cd4)
    sc.collection.objects.link(cam4)
    t4 = Vector((0, 0.0, 0.0))
    cam4.location = t4 + Vector((math.cos(az) * dist, math.sin(az) * dist, -0.6))
    util.look_at(cam4, t4)
    cd4.lens = 1800

    sun_d = bpy.data.lights.new("Sun", "SUN")
    sun_d.energy = 12.0
    sun_d.angle = math.radians(1.5)
    sun_d.color = (1.0, 0.76, 0.52)
    sun = bpy.data.objects.new("Sun", sun_d)
    sc.collection.objects.link(sun)
    # low golden-hour sun, front-right of the bird (it lights the face)
    az_s, el_s = math.radians(68), math.radians(17)
    to_sun = Vector((math.cos(az_s) * math.cos(el_s), math.sin(az_s) * math.cos(el_s), math.sin(el_s)))
    sun.rotation_euler = (-to_sun).to_track_quat("-Z", "Y").to_euler()
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
    ramp.color_ramp.elements[0].color = srgb(0.56, 0.80, 0.86)
    ramp.color_ramp.elements[1].position = 0.60
    ramp.color_ramp.elements[1].color = srgb(0.48, 0.75, 0.84)
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
    bg_lit.inputs["Strength"].default_value = 0.5
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
    sc.view_settings.look = "AgX - Medium High Contrast"
    sc.render.fps = 24
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.015
    sc.cycles.transparent_max_bounces = 40
    sc.cycles.max_bounces = 8
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    # a real shutter: head snaps smear slightly, holds stay crisp
    sc.render.use_motion_blur = True
    sc.render.motion_blur_shutter = 0.5
    # long lens at ~5 m, f/5.6: the far side of the bird falls slightly soft
    cam = sc.camera
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 5.6
    cam.data.dof.focus_distance = (cam.location - Vector((0.0, 0.03, 0.125))).length - 0.02


def build_coats(sc, body, head, pivot):
    import plumage
    import materials as M
    fm = M.feather_material("FeatherContour", fray=0.60, translucency=0.3)
    down = M.skin_material()
    strand = M.strand_material()
    eye_avoid = [(sdf.EYE_C * (sx, 1, 1), sdf.EYE_R + 0.0009) for sx in (1, -1)]
    out = []
    for ob, is_head in ((body, False), (head, True)):
        me = ob.data
        off = sdf.HEAD_PIVOT if is_head else np.zeros(3)
        V = util.get_verts(me) + off
        N = util.get_normals(me)
        F = np.array([p.vertices[:] for p in me.polygons])
        if is_head:
            reject = lambda P: sdf.body_sdf(P) < -0.0015  # buried in the body
        else:
            reject = lambda P: sdf.head_sdf(P) < 0.0015   # under the head plug
        fv, ff, at = plumage.coat(V, F, N, is_head, reject, avoid=eye_avoid if is_head else None)
        co = util.mesh_object(ob.name + "Feathers", fv - off, ff)
        for k, kind in (("uvn", "FLOAT_VECTOR"), ("uvm", "FLOAT_VECTOR")):
            util.set_point_attr(co.data, k, np.c_[at[k], np.zeros(len(at[k]))], kind)
        for k in ("fcol", "scol"):
            util.set_point_attr(co.data, k, np.c_[at[k], np.ones(len(at[k]))], "FLOAT_COLOR")
        util.set_point_attr(co.data, "rnd", at["rnd"])
        co.data.materials.append(fm)
        co.parent = pivot if is_head else None
        # tint the under-surface so any gap reads as dark down, not skin
        f = plumage.head_fields if is_head else plumage.body_fields
        col = f(V, N)[0]
        util.set_point_attr(me, "fcol", np.c_[util.srgb_np(col), np.ones(len(col))], "FLOAT_COLOR")
        me.materials.append(down)
        out.append(co)
        n_fl = 55000 if is_head else 120000
        fp, fc = plumage.fluff(V, F, N, is_head, n_fl, reject, avoid=eye_avoid if is_head else None)
        fo = util.curves_object(ob.name + "Fluff", fp - off, 0.00003, 0.000008,
                                curve_attrs={"fcol": ("FLOAT_COLOR", np.c_[fc, np.ones(len(fc))])})
        fo.data.materials.append(strand)
        fo.parent = pivot if is_head else None
        print(ob.name, "feathers:", len(ff) // ((plumage.NU - 1) * (plumage.NV - 1)), flush=True)
    return out


def attach_cards(name, verts, faces, at, mat, parent=None):
    ob = util.mesh_object(name, verts, faces)
    for k in ("uvn", "uvm"):
        util.set_point_attr(ob.data, k, np.c_[at[k], np.zeros(len(at[k]))], "FLOAT_VECTOR")
    for k in ("fcol", "scol"):
        util.set_point_attr(ob.data, k, np.c_[at[k], np.ones(len(at[k]))], "FLOAT_COLOR")
    util.set_point_attr(ob.data, "rnd", at["rnd"])
    ob.data.materials.append(mat)
    ob.parent = parent
    return ob


def build_wings(sc):
    import wing
    import materials as M
    wm = M.feather_material("FeatherWing", rough=0.38, sheen=0.3, fray=0.16,
                            translucency=0.15, gloss=0.18, shaft_w=0.035, shaft_mix=0.8,
                            thin_film=380.0, bump_str=0.18)
    surf = wing.SideSurface(0.0060)
    obs = []
    for sx in (1, -1):
        v, f, at = wing.cards(wing.wing_specs(), surf, sx)
        obs.append(attach_cards("Wing_R" if sx > 0 else "Wing_L", v, f, at, wm))
    # contour feathers overlapping the wing's front and lower edges
    import plumage
    hi = wing.SideSurface(0.0105)
    cm = bpy.data.materials["FeatherContour"]
    for sx in (1, -1):
        roots, normals = wing.overlap_roots(hi, sx)
        if len(roots) == 0:
            continue
        col, streak, L, W, T = plumage.body_fields(roots, normals)
        flows = plumage.flow_field(roots, False)
        shaft = plumage.mix(col * 0.92, plumage.C["shaft"], np.clip(streak * 0.95, 0, 1))
        v, f, at = plumage.build_cards(roots, normals, flows, L * 0.85, W * 0.9, np.full(len(L), 0.0006),
                                       util.srgb_np(col), util.srgb_np(shaft), hi.drape, sink=0.0042)
        obs.append(attach_cards("WingOverlap_" + ("R" if sx > 0 else "L"), v, f, at, cm))
    tp = bpy.data.objects.new("TailPivot", None)
    sc.collection.objects.link(tp)
    tp.location = Vector(wing.TAIL_BASE)
    v, f, at = wing.rigid_cards(wing.tail_specs())
    obs.append(attach_cards("Tail", v - wing.TAIL_BASE, f, at, wm, parent=tp))
    return obs


def build_feet_branch(sc):
    import limbs
    import materials as M
    fm, cm = M.foot_material(), M.simple("Claw", (0.06, 0.05, 0.045), rough=0.35, spec=0.6)
    for sx in (1, -1):
        for i, (v, f, uv, kind) in enumerate(limbs.build_foot(sx)):
            ob = util.mesh_object(f"Foot{'R' if sx > 0 else 'L'}_{kind}{i}", v, f)
            util.set_corner_uv(ob.data, "UVMap", uv)
            ob.data.materials.append(fm if kind == "toe" else cm)
    bm = M.bark_material()
    for i, (v, f, uv) in enumerate(limbs.branch_parts()):
        ob = util.mesh_object(f"Branch{i}", v, f)
        util.set_corner_uv(ob.data, "UVMap", uv)
        ob.data.materials.append(bm)


def main():
    preview = "--preview" in sys.argv
    sc = reset()
    import materials as M
    body, head, pivot, bill, eyes = build_core(sc)
    build_coats(sc, body, head, pivot)
    build_wings(sc)
    bm = M.bill_material()
    for b in bill:
        b.data.materials.append(bm)
    im, cm = M.iris_material(sdf.EYE_R), M.cornea_material()
    lm = M.simple("Lid", (0.03, 0.027, 0.026), rough=0.55, spec=0.4)
    for e in eyes:
        e.data.materials.append(im)
        for ch in e.children:
            ch.data.materials.append(cm if ch.name.startswith("Cornea") else lm)
    build_feet_branch(sc)
    blinks = []
    for e in eyes:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=32, radius=sdf.EYE_R * 1.07)
        cap = bpy.context.object
        cap.name = "Blink_" + e.name[-1]
        me = cap.data
        import bmesh
        bm_ = bmesh.new()
        bm_.from_mesh(me)
        bmesh.ops.delete(bm_, geom=[v for v in bm_.verts if v.co.z < sdf.EYE_R * 0.15], context="VERTS")
        bm_.to_mesh(me)
        bm_.free()
        bpy.ops.object.shade_smooth()
        cap.parent = e
        cap.data.materials.append(lm)
        blinks.append(cap)
    rig = bpy.data.objects.new("BodyRig", None)
    sc.collection.objects.link(rig)
    rig.location = (0, 0, 0.03)
    bpy.context.view_layer.update()
    for name in ("Body", "BodyFeathers", "BodyFluff", "Wing_R", "Wing_L", "TailPivot", "HeadPivot"):
        ob = bpy.data.objects[name]
        mw = ob.matrix_world.copy()
        ob.parent = rig
        ob.matrix_parent_inverse = rig.matrix_world.inverted()
        ob.matrix_world = mw
    import anim
    anim.animate(sc, bpy.data.objects["HeadPivot"], rig, bpy.data.objects["TailPivot"], blinks)
    build_camera_light(sc)
    build_world(sc)
    setup_render(sc, preview)
    os.makedirs(os.path.join(ROOT, "scene"), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(ROOT, "scene", "roller.blend"))


if __name__ == "__main__":
    main()
