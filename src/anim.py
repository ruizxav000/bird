"""'Looking around': a perched roller scanning its surroundings.

Birds don't pan their heads smoothly. They hold the head still, then snap
it to a new direction in a few frames (a head saccade), often with a small
overshoot, and cock it sideways to look down with one eye. Rollers hunt
exactly like this, watching the ground from a vantage perch.
Yaw > 0 turns the bill away from camera, yaw < 0 toward it.
"""
import math
import bpy

FPS = 24
FRAMES = 216  # 9 s

# (frame the snap lands, yaw, pitch, roll) in degrees; each snap takes SNAP frames
POSES = [
    (1, 0, 0, 0),
    (34, -32, -6, 4),     # glance toward camera side, down at the grass
    (62, -38, -16, 14),   # cock the head: one eye to the ground
    (90, 38, 6, -3),      # snap away, scanning the horizon
    (118, 46, 2, -6),     # small correction
    (146, -8, -10, -8),   # back past centre, looking down-forward
    (176, -62, 0, 4),     # look almost straight at the camera
    (204, -4, 2, 0),      # settle forward
]
SNAP = 4
BLINKS = [78, 162]
TAIL_FLICKS = [128]


def _key(ob, path, frame, value, idx=None, interp="BEZIER", easing="AUTO"):
    if idx is None:
        setattr(ob, path, value)
        ob.keyframe_insert(path, frame=frame)
    else:
        getattr(ob, path)[idx] = value
        ob.keyframe_insert(path, index=idx, frame=frame)


def _set_interp(ob, mapping):
    """mapping: frame -> (interpolation, easing) applied to keys at frame."""
    ad = ob.animation_data
    if not ad or not ad.action:
        return
    for fc in _fcurves(ad):
        for kp in fc.keyframe_points:
            f = int(round(kp.co[0]))
            if f in mapping:
                kp.interpolation, kp.easing = mapping[f]
                if kp.interpolation == "BACK":
                    kp.back = 1.2


def _fcurves(ad):
    act = ad.action
    if hasattr(act, "fcurves"):
        return act.fcurves
    # layered actions (Blender 4.4+)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                out.extend(bag.fcurves)
    return out


def animate(sc, head_pivot, body_rig, tail_pivot, blink_objs):
    sc.frame_start, sc.frame_end = 1, FRAMES
    sc.render.fps = FPS
    head_pivot.rotation_mode = "XYZ"
    mapping = {}
    prev = None
    for (f, yaw, pitch, roll) in POSES:
        rot = (math.radians(pitch), math.radians(roll), math.radians(yaw))
        if prev is not None:
            # hold the previous pose right up to the snap
            start = f - SNAP
            head_pivot.rotation_euler = prev
            head_pivot.keyframe_insert("rotation_euler", frame=start)
            mapping[start] = ("SINE", "EASE_IN")
        head_pivot.rotation_euler = rot
        head_pivot.keyframe_insert("rotation_euler", frame=f)
        mapping[f] = ("LINEAR", "AUTO")  # dead-still hold until the next snap
        prev = rot
    # the snap itself: fast start, overshoot and settle
    for f in list(mapping):
        if mapping[f][0] == "SINE":
            mapping[f] = ("BACK", "EASE_OUT")
    _set_interp(head_pivot, mapping)

    # breathing ~36 breaths/min, very subtle chest expansion
    for f in range(1, FRAMES + 1, 3):
        ph = 2 * math.pi * f / FPS / 1.65
        s = 1 + 0.006 * math.sin(ph)
        body_rig.scale = (1 + 0.004 * math.sin(ph), s, s)
        body_rig.keyframe_insert("scale", frame=f)

    # blinks: lids sweep over the eye in ~3 frames
    for ob in blink_objs:
        # local X of the eye is vertical: lids meet across the middle
        ob.scale = (0.0, 1, 1)
        ob.keyframe_insert("scale", frame=1)
        for b in BLINKS:
            for df, v in ((-2, 0.0), (0, 1.0), (1, 1.0), (3, 0.0)):
                ob.scale = (max(v, 0.0), 1, 1)
                ob.keyframe_insert("scale", frame=b + df)

    # tail: a single flick down and back up, as when balancing
    tail_pivot.rotation_mode = "XYZ"
    for f, a in ((1, 0), (TAIL_FLICKS[0] - 2, 0), (TAIL_FLICKS[0] + 1, 7), (TAIL_FLICKS[0] + 6, -1.5), (TAIL_FLICKS[0] + 12, 0)):
        tail_pivot.rotation_euler = (math.radians(a), 0, 0)
        tail_pivot.keyframe_insert("rotation_euler", frame=f)
