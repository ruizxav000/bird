"""Contour-feather coat: colour map, scattering and draped feather cards.

Plumage map for an adult Lilac-breasted Roller (Coracias caudatus caudatus),
from field descriptions and the reference frames:
  forehead + supercilium  creamy white
  lores / eye-stripe      black, from the gape through the eye
  crown -> nape           pale yellow-green -> greenish-turquoise
  ear-coverts / cheeks    rufous-orange, grading into lilac
  chin                    whitish
  throat + breast         lilac with white shaft streaks
  belly, flanks, vent     turquoise (paler toward the vent)
  mantle / scapulars      cinnamon-brown
  rump                    ultramarine-violet
"""
import numpy as np
from scipy.spatial import cKDTree

import sdf
from util import srgb_np

RNG = np.random.default_rng(7)

C = {k: np.array(v) for k, v in dict(
    crown=(0.66, 0.70, 0.46), crown2=(0.48, 0.66, 0.50), nape=(0.36, 0.60, 0.55),
    white=(0.93, 0.91, 0.84), stripe=(0.045, 0.04, 0.035), cheek=(0.86, 0.46, 0.30),
    chin=(0.91, 0.82, 0.82), lilac=(0.80, 0.39, 0.66), lilac2=(0.70, 0.36, 0.70),
    rufneck=(0.84, 0.50, 0.40), turq=(0.17, 0.60, 0.65), turq2=(0.38, 0.73, 0.73),
    brown=(0.63, 0.41, 0.21), rump=(0.22, 0.24, 0.66), shaft=(0.97, 0.94, 0.96),
).items()}


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = np.asarray(t)[..., None]
    return a * (1 - t) + b * t


# --------------------------------------------------------------- colour map
def body_fields(P, N):
    x, y, z = P.T
    lat = np.abs(N[:, 0])
    dorsal = N @ np.array([0, -0.819, 0.574])
    # lilac upper breast -> turquoise belly; turquoise climbs on the flanks
    zb = 0.071 + 0.010 * lat - 0.006 * ss(0.0, 0.03, -y)
    t_lt = ss(zb - 0.005, zb + 0.005, z)
    turq = mix(C["turq"], C["turq2"], ss(0.035, -0.005, z) * 0.8)
    lil = mix(C["lilac2"], C["lilac"], ss(0.08, 0.11, z))
    lil = mix(lil, C["rufneck"], ss(0.45, 0.85, lat) * ss(0.088, 0.105, z) * 0.6)
    col = mix(turq, lil, t_lt)
    streak = t_lt * (1 - ss(0.3, 0.6, dorsal))
    back = mix(C["brown"], C["nape"], ss(0.094, 0.112, z))
    back = mix(back, C["rump"], ss(-0.03, -0.05, y) * ss(0.04, 0.0, z))
    tb = ss(0.05, 0.35, dorsal)
    col = mix(col, back, tb)
    streak = streak * (1 - tb)
    # feather length / plumage depth (puffed, cool-morning posture)
    L = np.full(len(P), 0.0170)
    T = np.full(len(P), 0.0062)
    L = L * (1 - 0.15 * tb)
    T = T * (1 - 0.35 * tb)
    near_tail = ss(-0.03, -0.065, y)
    L = L * (1 - 0.15 * near_tail)
    T = T * (1 - 0.30 * near_tail)
    W = 0.55 * L
    return col, streak, L, W, T


def head_fields(P, N):
    x, y, z = P.T
    ax = np.abs(x)
    ey, ez = sdf.EYE_C[1], sdf.EYE_C[2]
    zl = np.where(y > ey, ez + (y - ey) * -0.32, ez + (y - ey) * 0.15)
    dz = z - zl
    span = ss(0.012, 0.018, y) * (1 - ss(0.050, 0.054, y))
    side = ss(0.004, 0.009, ax)
    half = 0.0030 + 0.0008 * ss(0.03, 0.02, np.abs(y - ey))
    stripe = (1 - ss(half - 0.0006, half + 0.0006, np.abs(dz))) * span * side
    sup = ss(0.0026, 0.0036, dz) * (1 - ss(0.0066, 0.0078, dz)) * ss(0.014, 0.020, y) * side
    forehead = ss(0.034, 0.040, y) * ss(0.126, 0.129, z) * (1 - stripe)
    crown = mix(C["nape"], C["crown2"], ss(-0.004, 0.012, y))
    crown = mix(crown, C["crown"], ss(0.016, 0.030, y))
    below = 1 - ss(-0.004, -0.002, dz)
    cheek = mix(C["cheek"], C["lilac"], ss(0.112, 0.100, z) + ss(0.012, 0.004, ax) * 0.6)
    col = mix(crown, cheek, below)
    chin = ss(0.034, 0.040, y) * ss(0.124, 0.118, z) * ss(0.011, 0.006, ax)
    col = mix(col, C["chin"], chin)
    col = mix(col, C["white"], np.clip(sup + forehead, 0, 1))
    col = mix(col, C["stripe"], stripe)
    streak = below * (1 - chin) * (1 - stripe) * ss(0.122, 0.112, z)
    # lower plug blends into the body map
    bcol, bstreak, bL, bW, bT = body_fields(P, N)
    tb = ss(0.110, 0.098, z) * ss(0.030, 0.012, y)
    col = mix(col, bcol, tb)
    streak = streak * (1 - tb) + bstreak * tb
    # small feathers on the face, longer and fluffier on crown and neck
    L = 0.0062 + 0.0030 * ss(0.03, 0.01, y) * ss(0.13, 0.15, z)
    L = L + 0.0050 * ss(0.120, 0.104, z)
    d_eye = np.linalg.norm(P - sdf.EYE_C * np.c_[np.sign(x) + (x == 0), np.ones_like(x), np.ones_like(x)], axis=1) - sdf.EYE_R
    L = L * (0.40 + 0.60 * ss(0.0008, 0.0085, d_eye))
    d_bill = np.linalg.norm(P - sdf.BILL_BASE, axis=1)
    L = L * (0.40 + 0.60 * ss(0.006, 0.016, d_bill))
    T = 0.0020 + 0.0018 * ss(0.03, 0.01, y) * ss(0.13, 0.15, z) + 0.0025 * ss(0.120, 0.104, z)
    T = T * (0.5 + 0.5 * ss(0.0008, 0.006, d_eye))
    L = mix(L[:, None], bL[:, None], tb)[:, 0]
    T = mix(T[:, None], bT[:, None], tb)[:, 0]
    W = 0.60 * L
    return col, streak, L, W, T


def flow_field(P, head):
    src_b = np.array([0, 0.060, 0.170])
    src_h = np.array([0, 0.095, 0.124])
    vb = P - src_b
    vh = P - src_h
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    vh /= np.linalg.norm(vh, axis=1, keepdims=True)
    w = ss(0.106, 0.124, P[:, 2]) if head else np.zeros(len(P))
    v = vb * (1 - w[:, None]) + vh * w[:, None]
    return v


# --------------------------------------------------------------- scattering
def scatter(V, F, fields, coverage, inside_reject=None):
    tri = V[F]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area[:, None], 1e-12)
    _, _, L, W, _ = fields(cen, nrm)
    lam = coverage * area / (0.62 * L * W)
    # oversample, then thin with variable-radius dart throwing (blue noise)
    cnt = RNG.poisson(lam * 6)
    fi = np.repeat(np.arange(len(F)), cnt)
    r1 = np.sqrt(RNG.random(len(fi)))
    r2 = RNG.random(len(fi))
    b = np.c_[1 - r1, r1 * (1 - r2), r1 * r2]
    pts = np.einsum("ij,ijk->ik", b, tri[fi])
    nn = nrm[fi]
    if inside_reject is not None:
        keep = ~inside_reject(pts)
        pts, nn, fi = pts[keep], nn[keep], fi[keep]
    _, _, Lp, Wp, _ = fields(pts, nn)
    rad = 0.5 * np.sqrt(0.62 * Lp * Wp / coverage)
    order = RNG.permutation(len(pts))
    cell = rad.max()
    grid = {}
    acc = []
    for i in order:
        p = pts[i]
        key = tuple((p // cell).astype(int))
        ok = True
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        if np.sum((pts[j] - p) ** 2) < (0.5 * (rad[i] + rad[j])) ** 2:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault(key, []).append(i)
            acc.append(i)
    acc = np.array(acc)
    return pts[acc], nn[acc]


class Drape:
    """Snap points onto a surface (smoothed k-NN of dense vertices)."""

    def __init__(self, V, N):
        self.V, self.N = V, N
        self.kd = cKDTree(V)

    def project(self, X, k=8):
        d, idx = self.kd.query(X, k=k)
        w = 1.0 / (d + 1e-5) ** 2
        w /= w.sum(axis=1, keepdims=True)
        S = np.einsum("ij,ijk->ik", w, self.V[idx])
        Nn = np.einsum("ij,ijk->ik", w, self.N[idx])
        Nn /= np.linalg.norm(Nn, axis=1, keepdims=True)
        # move to plane of the local surface through X's foot point
        S = S + Nn * np.einsum("ij,ij->i", X - S, Nn)[:, None] * 0.0
        return S, Nn


# ------------------------------------------------------------- feather card
NU, NV = 10, 7


def contour_outline(u):
    w = np.where(u < 0.55, 0.42 + 0.58 * np.sin(0.5 * np.pi * np.minimum(u / 0.55, 1)),
                 np.sqrt(np.clip(1 - ((u - 0.55) / 0.47) ** 2, 0, 1)))
    return np.maximum(w, 0.06)


def build_cards(roots, normals, flows, L, W, T, colors, shaft_cols, drape,
                outline=contour_outline, curl=0.16, sink=0.0006, tip_jitter=0.28,
                avoid=None):
    """Draped feather cards. Returns verts, faces, attribute dict."""
    n = len(roots)
    u = np.linspace(0, 1, NU)
    v = np.linspace(-1, 1, NV)
    U, Vv = np.meshgrid(u, v, indexing="ij")  # NU x NV
    U = U.ravel(); Vv = Vv.ravel()
    t = flows - normals * np.einsum("ij,ij->i", flows, normals)[:, None]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    # random twist of each feather about the normal
    ang = RNG.normal(0, 0.16, n)
    b = np.cross(normals, t)
    t = t * np.cos(ang)[:, None] + b * np.sin(ang)[:, None]
    b = np.cross(normals, t)
    sc = RNG.uniform(0.85, 1.15, n)
    Lr = L * sc
    Wr = W * RNG.uniform(0.9, 1.1, n)
    tip = T * RNG.uniform(1 - tip_jitter, 1 + 0.5 * tip_jitter, n)
    wo = outline(U)  # (k,)
    X = (roots[:, None, :]
         + t[:, None, :] * (U * Lr[:, None])[..., None]
         + b[:, None, :] * (Vv * wo * 0.5 * Wr[:, None])[..., None])
    flat = X.reshape(-1, 3)
    S, Nn = drape.project(flat)
    S = S.reshape(n, -1, 3); Nn = Nn.reshape(n, -1, 3)
    h = -sink + (tip[:, None] + sink) * (1 - (1 - U[None, :]) ** 1.7)
    h = h - curl * (Vv[None, :] ** 2) * (wo * 0.5 * Wr[:, None])
    P = S + Nn * h[..., None]
    if avoid is not None:
        for c, r in avoid:
            d = P - c
            dist = np.linalg.norm(d, axis=-1, keepdims=True)
            push = np.clip(r - dist, 0, None)
            P = P + d / np.maximum(dist, 1e-9) * push
    k = NU * NV
    verts = P.reshape(-1, 3)
    base = np.arange(n)[:, None, None] * k
    qi = np.arange(NU - 1)[:, None] * NV + np.arange(NV - 1)[None, :]
    q = np.stack([qi, qi + NV, qi + NV + 1, qi + 1], axis=-1)  # (NU-1,NV-1,4)
    faces = (base + q.reshape(1, -1, 4)).reshape(-1, 4)
    attrs = {
        "uvn": np.tile(np.c_[U, Vv], (n, 1)),
        "uvm": np.concatenate([np.c_[U * Lr[i] * 1000, Vv * wo * Wr[i] * 500] for i in range(n)]),
        "fcol": np.repeat(colors, k, axis=0) if colors.ndim == 2 else colors.reshape(-1, 3),
        "scol": np.repeat(shaft_cols, k, axis=0),
        "rnd": np.repeat(RNG.random(n), k),
    }
    return verts, faces, attrs


def coat(V, Fc, Nv, head, inside_reject=None, coverage=3.6, avoid=None):
    """V, Fc: surface mesh in bird space; Nv: vertex normals."""
    fields = head_fields if head else body_fields
    roots, normals = scatter(V, Fc, fields, coverage, inside_reject)
    col, streak, L, W, T = fields(roots, normals)
    flows = flow_field(roots, head)
    # per-feather colour jitter (value + slight hue drift), then to linear
    col = np.clip(col * RNG.normal(1.0, 0.05, (len(col), 1))
                  + RNG.normal(0, 0.015, col.shape), 0, 1)
    shaft = mix(col * 0.92, C["shaft"], np.clip(streak * 0.95, 0, 1))
    drape = Drape(V, Nv)
    verts, faces, attrs = build_cards(roots, normals, flows, L, W, T,
                                      srgb_np(col), srgb_np(shaft), drape, avoid=avoid)
    return verts, faces, attrs
