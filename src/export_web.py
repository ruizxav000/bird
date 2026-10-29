"""Export the procedural roller for the interactive web viewer (three.js).

Runs the same generators as build.py at web-friendly resolution and packs
everything into one compact binary (int16 positions, uint8 colours), written
as web/model.js (base64) so the page can load it without a fetch.

python src/export_web.py
"""
import base64
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

import numpy as np  # noqa: E402

import sdf  # noqa: E402
import plumage  # noqa: E402
import wing  # noqa: E402
import beak  # noqa: E402
import limbs  # noqa: E402

HEAD_OFFSET = np.array([0.0, -0.001, -0.004])  # matches build.py pivot
HEAD_SCALE = 1.08
BOX = np.array([0.5, 0.5, 0.5])  # half extent for int16 quantisation (m)


def to_three(v):
    """bird space (X right, Y forward, Z up) -> three.js (Y up)."""
    v = np.asarray(v, float)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], axis=-1)


def lin_to_srgb(c):
    c = np.clip(np.asarray(c, float), 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def smooth(V, F, it, fac=0.5):
    n = len(V)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    deg = np.bincount(edges[:, 0], minlength=n).astype(float)
    for _ in range(it):
        acc = np.zeros_like(V)
        np.add.at(acc, edges[:, 0], V[edges[:, 1]])
        V = V + fac * (acc / np.maximum(deg, 1)[:, None] - V)
    return V


def vnormals(V, F):
    t = V[F]
    fn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    vn = np.zeros_like(V)
    for i in range(3):
        np.add.at(vn, F[:, i], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def tris(faces):
    out = []
    for f in faces:
        f = list(map(int, f))
        for k in range(1, len(f) - 1):
            out.append((f[0], f[k], f[k + 1]))
    return np.array(out, dtype=np.uint32)


class Pack:
    def __init__(self):
        self.buf = bytearray()
        self.meshes = []

    def _add(self, arr):
        arr = np.ascontiguousarray(arr)
        while len(self.buf) % 4:
            self.buf.append(0)
        off = len(self.buf)
        self.buf += arr.tobytes()
        return [off, int(arr.size), str(arr.dtype)]

    def mesh(self, name, group, kind, pos, index=None, grid=None, uv=None,
             fcol=None, scol=None, rnd=None, extra=None):
        q = np.round(np.clip(to_three(pos) / BOX, -1, 1) * 32767).astype(np.int16)
        m = {"name": name, "group": group, "kind": kind, "count": int(len(pos)),
             "pos": self._add(q)}
        if index is not None:
            idx = np.asarray(index, np.uint32)
            m["index"] = self._add(idx.astype(np.uint16) if len(pos) < 65536 else idx)
        if grid is not None:
            m["grid"] = grid  # [n_cards, nu, nv]: indices are rebuilt in JS
        if uv is not None:
            uv = np.asarray(uv, np.float32)
            if kind in ("contour", "flight"):  # feather-texture coords fit in a byte
                m["uv"] = self._add(np.round(np.clip(uv, 0, 1) * 255).astype(np.uint8))
            else:
                m["uv"] = self._add(uv)
        if fcol is not None:
            m["fcol"] = self._add(np.round(lin_to_srgb(fcol) * 255).astype(np.uint8))
        if scol is not None:
            m["scol"] = self._add(np.round(lin_to_srgb(scol) * 255).astype(np.uint8))
        if rnd is not None:
            m["rnd"] = self._add(np.round(np.asarray(rnd) * 255).astype(np.uint8))
        if extra:
            m.update(extra)
        self.meshes.append(m)
        print(f"{name:16s} verts={len(pos):7d}  bytes so far {len(self.buf)/1e6:.2f} MB", flush=True)


def cards_uv(at):
    """uvn (u 0..1 along, v -1..1 across) -> texture uv (u, v01)."""
    return np.c_[at["uvn"][:, 0], 0.5 + 0.5 * at["uvn"][:, 1]].astype(np.float32)


def main():
    pk = Pack()
    plumage.NU, plumage.NV = 6, 3
    eye_avoid = [(sdf.EYE_C * (sx, 1, 1), sdf.EYE_R + 0.0009) for sx in (1, -1)]

    # ---- skin surfaces + contour feathers + loose barbs
    for is_head in (False, True):
        if is_head:
            v, f = sdf.polygonize(sdf.head_sdf, (-0.035, -0.025, 0.07), (0.035, 0.065, 0.16), 0.0011)
        else:
            v, f = sdf.polygonize(sdf.body_sdf, (-0.05, -0.10, -0.03), (0.05, 0.08, 0.13), 0.0017)
        v = smooth(v, f, 2 if is_head else 3)
        n = vnormals(v, f)
        off = sdf.HEAD_PIVOT if is_head else np.zeros(3)
        grp = "head" if is_head else "body"
        if is_head:
            reject = lambda P: sdf.body_sdf(P) < -0.0015
        else:
            reject = lambda P: sdf.head_sdf(P) < 0.0015
        fields = plumage.head_fields if is_head else plumage.body_fields
        pk.mesh(grp + "_skin", grp, "skin", v - off, index=f.astype(np.uint32),
                fcol=plumage.srgb_np(fields(v, n)[0]))
        fv, ff, at = plumage.coat(v, f, n, is_head, reject, avoid=eye_avoid if is_head else None)
        k = plumage.NU * plumage.NV
        pk.mesh(grp + "_feathers", grp, "contour", fv - off, grid=[len(fv) // k, plumage.NU, plumage.NV],
                uv=cards_uv(at), fcol=at["fcol"], scol=at["scol"], rnd=at["rnd"])
        fp, fc = plumage.fluff(v, f, n, is_head, 14000 if is_head else 26000, reject,
                               avoid=eye_avoid if is_head else None, segs=4)
        pk.mesh(grp + "_fluff", grp, "strands", fp.reshape(-1, 3) - off,
                fcol=fc, extra={"segs": 4})  # one colour per strand

    # ---- folded wings and tail
    surf = wing.SideSurface(0.0060)
    for sx in (1, -1):
        v, f, at = wing.cards(wing.wing_specs(), surf, sx, nu=12, nv=3)
        pk.mesh("wing_" + ("R" if sx > 0 else "L"), "body", "flight", v, grid=[len(v) // 36, 12, 3],
                uv=cards_uv(at), fcol=at["fcol"], scol=at["scol"], rnd=at["rnd"])
    v, f, at = wing.rigid_cards(wing.tail_specs(), nu=18, nv=3)
    pk.mesh("tail", "tail", "flight", v - wing.TAIL_BASE, grid=[len(v) // 54, 18, 3],
            uv=cards_uv(at), fcol=at["fcol"], scol=at["scol"], rnd=at["rnd"])

    # ---- bill
    (uv_, uf, _), (lv, lf, _) = beak.build_bill()
    pk.mesh("bill_upper", "head", "bill", uv_ - sdf.HEAD_PIVOT, index=tris(uf))
    pk.mesh("bill_lower", "head", "bill", lv - sdf.HEAD_PIVOT, index=tris(lf))

    # ---- feet
    for sx in (1, -1):
        for i, (v, f, uv, kind) in enumerate(limbs.build_foot(sx)):
            pk.mesh(f"foot{sx}_{i}", "world", "toe" if kind == "toe" else "claw", v,
                    index=tris(f), uv=uv)

    # ---- branch
    for i, (v, f, uv) in enumerate(limbs.branch_parts()):
        pk.mesh(f"branch{i}", "world", "bark", v, index=tris(f), uv=uv)

    a, b = sdf.EYE_C, sdf.HEAD_PIVOT
    meta = {
        "box": BOX.tolist(),
        "headPivot": to_three(sdf.HEAD_PIVOT + HEAD_OFFSET).tolist(),
        "headScale": HEAD_SCALE,
        "tailPivot": to_three(wing.TAIL_BASE).tolist(),
        "rigPivot": to_three([0, 0, 0.03]).tolist(),
        "eye": {"centre": to_three(a - b).tolist(), "radius": sdf.EYE_R},
        "meshes": pk.meshes,
    }
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    js = ("window.ROLLER_META=" + json.dumps(meta, separators=(",", ":")) + ";\n"
          "window.ROLLER_BIN=\"" + base64.b64encode(bytes(pk.buf)).decode() + "\";\n")
    out = os.path.join(ROOT, "web", "model.js")
    with open(out, "w") as fh:
        fh.write(js)
    print(f"wrote {out}: {len(js)/1e6:.2f} MB (binary {len(pk.buf)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
