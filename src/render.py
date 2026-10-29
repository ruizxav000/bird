"""Render stills or the animation from scene/roller.blend.

python src/render.py still OUT.png [--frame N] [--pct P] [--samples S] [--cam NAME]
python src/render.py anim OUTDIR [--start A] [--end B] [--pct P | --res WxH] [--samples S]
"""
import os
import sys
import time
import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def arg(name, default, cast=str):
    if name in sys.argv:
        return cast(sys.argv[sys.argv.index(name) + 1])
    return default


def main():
    mode, out = sys.argv[1], os.path.abspath(sys.argv[2])
    bpy.ops.wm.open_mainfile(filepath=arg("--blend", os.path.join(ROOT, "scene", "roller.blend")))
    sc = bpy.context.scene
    sc.render.resolution_percentage = arg("--pct", sc.render.resolution_percentage, int)
    res = arg("--res", None)
    if res:  # exact output size, e.g. 720x1280
        sc.render.resolution_x, sc.render.resolution_y = map(int, res.split("x"))
        sc.render.resolution_percentage = 100
    sc.cycles.samples = arg("--samples", sc.cycles.samples, int)
    sc.render.threads_mode = "AUTO"
    cam = arg("--cam", None)
    if cam:
        sc.camera = bpy.data.objects[cam]
    if mode == "still":
        sc.frame_set(arg("--frame", sc.frame_current, int))
        sc.render.filepath = out
        t = time.time()
        bpy.ops.render.render(write_still=True)
        print(f"rendered {out} in {time.time() - t:.1f}s")
    else:
        sc.frame_start = arg("--start", sc.frame_start, int)
        sc.frame_end = arg("--end", sc.frame_end, int)
        os.makedirs(out, exist_ok=True)
        for f in range(sc.frame_start, sc.frame_end + 1):
            p = os.path.join(out, f"f{f:04d}.png")
            if os.path.exists(p):
                continue
            sc.frame_set(f)
            sc.render.filepath = p
            t = time.time()
            bpy.ops.render.render(write_still=True)
            print(f"frame {f} {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
