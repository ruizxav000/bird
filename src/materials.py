"""Shader networks: feathers, bill, eye, cornea, feet, bark."""
import bpy
from util import node, link, srgb


def _attr(nt, name, loc, kind="GEOMETRY"):
    a = node(nt, "ShaderNodeAttribute", loc, attribute_name=name)
    a.attribute_type = kind
    return a


def _math(nt, op, a=None, b=None, loc=(0, 0), clamp=False):
    m = node(nt, "ShaderNodeMath", loc, operation=op)
    m.use_clamp = clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            m.inputs[i].default_value = x
        else:
            link(nt, x, m.inputs[i])
    return m.outputs[0]


def _ss(nt, a, b, x, loc=(0, 0)):
    m = node(nt, "ShaderNodeMapRange", loc, interpolation_type="SMOOTHSTEP")
    m.inputs["From Min"].default_value = a
    m.inputs["From Max"].default_value = b
    link(nt, x, m.inputs["Value"])
    return m.outputs["Result"]


def _mixc(nt, fac, a, b, loc=(0, 0), blend="MIX"):
    m = node(nt, "ShaderNodeMix", loc, data_type="RGBA", blend_type=blend)
    for sock, val in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
        if isinstance(val, (int, float)):
            sock.default_value = val
        elif isinstance(val, tuple):
            sock.default_value = val
        else:
            link(nt, val, sock)
    return m.outputs[2]


def new_mat(name):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    return m, nt


def feather_material(name="Feather", rough=0.55, sheen=0.45, barb_mm=0.24,
                     barb_slope=0.9, fray=0.42, translucency=0.25, gloss=0.0,
                     shaft_w=0.065, shaft_mix=0.75, thin_film=0.0, bump_str=0.25):
    m, nt = new_mat(name)
    out = node(nt, "ShaderNodeOutputMaterial", (1600, 0))
    uvn = _attr(nt, "uvn", (-1600, 300))
    uvm = _attr(nt, "uvm", (-1600, 0))
    fcol = _attr(nt, "fcol", (-1600, -300))
    scol = _attr(nt, "scol", (-1600, -500))
    rnd = _attr(nt, "rnd", (-1600, -700))
    sep = node(nt, "ShaderNodeSeparateXYZ", (-1400, 300))
    link(nt, uvn.outputs["Vector"], sep.inputs[0])
    sepm = node(nt, "ShaderNodeSeparateXYZ", (-1400, 0))
    link(nt, uvm.outputs["Vector"], sepm.inputs[0])
    u = sep.outputs["X"]
    av = _math(nt, "ABSOLUTE", sep.outputs["Y"], loc=(-1200, 300))
    avm = _math(nt, "ABSOLUTE", sepm.outputs["Y"], loc=(-1200, 0))
    # barbs leave the rachis angled toward the tip
    s = _math(nt, "SUBTRACT", sepm.outputs["X"], _math(nt, "MULTIPLY", avm, barb_slope, loc=(-1050, 0)), loc=(-900, 0))
    sr = _math(nt, "ADD", s, _math(nt, "MULTIPLY", rnd.outputs["Fac"], 31.0), loc=(-800, -150))
    phase = _math(nt, "MULTIPLY", sr, 6.2832 / barb_mm, loc=(-750, 0))
    stripe = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "SINE", phase), 1.0), 0.5, loc=(-600, 0))
    # barb clumping / vane splits: 1D noise along barb index
    cv = node(nt, "ShaderNodeCombineXYZ", (-750, -250))
    link(nt, _math(nt, "MULTIPLY", sr, 1.3), cv.inputs[0])
    link(nt, rnd.outputs["Fac"], cv.inputs[1])
    nz = node(nt, "ShaderNodeTexNoise", (-550, -250))
    nz.noise_dimensions = "2D"
    nz.inputs["Scale"].default_value = 1.0
    nz.inputs["Detail"].default_value = 3.0
    link(nt, cv.outputs[0], nz.inputs["Vector"])
    n1 = nz.outputs["Fac"]
    # frayed outline: barb tips separate near the vane edge
    edge = _math(nt, "SUBTRACT", 1.0, _math(nt, "ADD",
                 _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", n1, 0.35), fray * 2.0),
                 _math(nt, "MULTIPLY", stripe, fray * 0.7)), loc=(-350, -200))
    alpha = _math(nt, "LESS_THAN", av, edge, loc=(-200, -200))
    # vane splits where barbs part
    split = _ss(nt, 0.66, 0.72, n1, loc=(-350, -400))
    split_band = _ss(nt, 0.25, 0.55, av, loc=(-350, -550))
    alpha = _math(nt, "MULTIPLY", alpha, _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", split, split_band)), loc=(-50, -300))
    # colour: display colour, downy base, rachis / shaft streak, barb shading
    down = _mixc(nt, 0.55, fcol.outputs["Color"], (0.05, 0.05, 0.055, 1), loc=(-600, -500))
    dfac = _math(nt, "SUBTRACT", 1.0, _ss(nt, 0.10, 0.42, u), loc=(-600, -650))
    col = _mixc(nt, dfac, fcol.outputs["Color"], down, loc=(-400, -650))
    rw = _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", 1.0, _math(nt, "MULTIPLY", u, 0.55)), shaft_w, loc=(-600, -800))
    rmask = _math(nt, "SUBTRACT", 1.0, _ss(nt, 0.0, 1.0, _math(nt, "DIVIDE", av, rw)), loc=(-400, -800))
    col = _mixc(nt, _math(nt, "MULTIPLY", rmask, shaft_mix), col, scol.outputs["Color"], loc=(-200, -650))
    shade = _math(nt, "ADD", 0.86, _math(nt, "MULTIPLY", stripe, 0.14), loc=(-200, -850))
    col = _mixc(nt, 1.0, col, shade, loc=(0, -650), blend="MULTIPLY")
    # micro relief from barbs
    bump = node(nt, "ShaderNodeBump", (100, -900))
    bump.inputs["Strength"].default_value = bump_str
    bump.inputs["Distance"].default_value = 0.00005
    link(nt, stripe, bump.inputs["Height"])
    bsdf = node(nt, "ShaderNodeBsdfPrincipled", (400, -400))
    link(nt, col, bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Specular IOR Level"].default_value = 0.35 + gloss
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Sheen Roughness"].default_value = 0.35
    bsdf.inputs["Coat Weight"].default_value = gloss
    bsdf.inputs["Coat Roughness"].default_value = 0.25
    if thin_film:
        # structural (non-pigment) blues: faint angle-dependent shimmer
        bsdf.inputs["Thin Film Thickness"].default_value = thin_film
        bsdf.inputs["Thin Film IOR"].default_value = 1.45
    link(nt, bump.outputs["Normal"], bsdf.inputs["Normal"])
    tr = node(nt, "ShaderNodeBsdfTranslucent", (400, -100))
    link(nt, col, tr.inputs["Color"])
    mixs = node(nt, "ShaderNodeMixShader", (800, -300))
    mixs.inputs[0].default_value = translucency
    link(nt, bsdf.outputs[0], mixs.inputs[1])
    link(nt, tr.outputs[0], mixs.inputs[2])
    transp = node(nt, "ShaderNodeBsdfTransparent", (800, 0))
    mixa = node(nt, "ShaderNodeMixShader", (1200, 0))
    link(nt, alpha, mixa.inputs[0])
    link(nt, transp.outputs[0], mixa.inputs[1])
    link(nt, mixs.outputs[0], mixa.inputs[2])
    link(nt, mixa.outputs[0], out.inputs["Surface"])
    return m


def skin_material(name="Down"):
    """Under-surface: dark downy base tinted by the plumage colour."""
    m, nt = new_mat(name)
    out = node(nt, "ShaderNodeOutputMaterial", (600, 0))
    a = _attr(nt, "fcol", (-400, 0))
    c = _mixc(nt, 0.6, a.outputs["Color"], (0.04, 0.04, 0.045, 1))
    b = node(nt, "ShaderNodeBsdfPrincipled", (200, 0))
    link(nt, c, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.9
    b.inputs["Sheen Weight"].default_value = 0.5
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m


def bill_material():
    m, nt = new_mat("Bill")
    out = node(nt, "ShaderNodeOutputMaterial", (900, 0))
    uv = node(nt, "ShaderNodeUVMap", (-1200, 0), uv_map="UVMap")
    sep = node(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    link(nt, uv.outputs[0], sep.inputs[0])
    tc = node(nt, "ShaderNodeTexCoord", (-1200, -300))
    nz = node(nt, "ShaderNodeTexNoise", (-900, -300))
    nz.inputs["Scale"].default_value = 900.0
    nz.inputs["Detail"].default_value = 6.0
    link(nt, tc.outputs["Object"], nz.inputs["Vector"])
    # fine longitudinal keratin grain
    mp = node(nt, "ShaderNodeMapping", (-1000, -550))
    mp.inputs["Scale"].default_value = (1.0, 6.0, 40.0)
    link(nt, uv.outputs[0], mp.inputs["Vector"])
    gr = node(nt, "ShaderNodeTexNoise", (-800, -550))
    gr.inputs["Scale"].default_value = 30.0
    gr.inputs["Detail"].default_value = 4.0
    link(nt, mp.outputs[0], gr.inputs["Vector"])
    base = (0.018, 0.018, 0.020, 1)
    worn = (0.075, 0.072, 0.070, 1)
    # slightly greyer, worn culmen base and cutting edges; glossier tip
    wear = _math(nt, "MULTIPLY", _ss(nt, 0.45, 0.85, nz.outputs["Fac"]), 0.45, loc=(-600, -300))
    col = _mixc(nt, wear, base, worn, loc=(-300, 0))
    rough = node(nt, "ShaderNodeMapRange", (-300, -300))
    rough.inputs["To Min"].default_value = 0.22
    rough.inputs["To Max"].default_value = 0.48
    link(nt, gr.outputs["Fac"], rough.inputs["Value"])
    b = node(nt, "ShaderNodeBsdfPrincipled", (300, 0))
    link(nt, col, b.inputs["Base Color"])
    link(nt, rough.outputs["Result"], b.inputs["Roughness"])
    b.inputs["Specular IOR Level"].default_value = 0.5
    b.inputs["Coat Weight"].default_value = 0.15
    b.inputs["Coat Roughness"].default_value = 0.3
    bump = node(nt, "ShaderNodeBump", (0, -500))
    bump.inputs["Strength"].default_value = 0.15
    bump.inputs["Distance"].default_value = 0.0002
    link(nt, gr.outputs["Fac"], bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], b.inputs["Normal"])
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m


def iris_material(radius):
    """Eye sphere whose local +Z is the gaze axis: dark-brown iris, round pupil."""
    m, nt = new_mat("Iris")
    out = node(nt, "ShaderNodeOutputMaterial", (1000, 0))
    tc = node(nt, "ShaderNodeTexCoord", (-1400, 0))
    sep = node(nt, "ShaderNodeSeparateXYZ", (-1200, 0))
    link(nt, tc.outputs["Object"], sep.inputs[0])
    rho = _math(nt, "DIVIDE", _math(nt, "SQRT", _math(nt, "ADD",
                _math(nt, "MULTIPLY", sep.outputs["X"], sep.outputs["X"]),
                _math(nt, "MULTIPLY", sep.outputs["Y"], sep.outputs["Y"]))), radius, loc=(-900, 0))
    front = _math(nt, "GREATER_THAN", sep.outputs["Z"], 0.0, loc=(-900, -200))
    ang = _math(nt, "ARCTAN2", sep.outputs["Y"], sep.outputs["X"], loc=(-900, -400))
    cc = node(nt, "ShaderNodeCombineXYZ", (-700, -400))
    link(nt, _math(nt, "MULTIPLY", ang, 9.0), cc.inputs[0])
    link(nt, _math(nt, "MULTIPLY", rho, 4.0), cc.inputs[1])
    fib = node(nt, "ShaderNodeTexNoise", (-500, -400))
    fib.noise_dimensions = "2D"
    fib.inputs["Scale"].default_value = 1.5
    fib.inputs["Detail"].default_value = 8.0
    link(nt, cc.outputs[0], fib.inputs["Vector"])
    iris_a = (0.030, 0.012, 0.005, 1)
    iris_b = (0.011, 0.004, 0.002, 1)
    iris = _mixc(nt, fib.outputs["Fac"], iris_a, iris_b, loc=(-300, -300))
    # warmer ring near the pupil, dark limbal ring
    iris = _mixc(nt, _math(nt, "MULTIPLY", _ss(nt, 0.60, 0.42, rho), 0.5), iris, (0.045, 0.018, 0.006, 1), loc=(-150, -300))
    iris = _mixc(nt, _ss(nt, 0.70, 0.86, rho), iris, (0.02, 0.012, 0.008, 1), loc=(0, -300))
    pupil = _ss(nt, 0.36, 0.33, rho, loc=(0, -100))
    col = _mixc(nt, pupil, iris, (0.003, 0.003, 0.003, 1), loc=(200, -200))
    col = _mixc(nt, front, (0.01, 0.01, 0.01, 1), col, loc=(400, -200))
    b = node(nt, "ShaderNodeBsdfPrincipled", (650, 0))
    link(nt, col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.45
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m


def cornea_material():
    m, nt = new_mat("Cornea")
    out = node(nt, "ShaderNodeOutputMaterial", (600, 0))
    g = node(nt, "ShaderNodeBsdfGlass", (0, 100))
    g.inputs["Roughness"].default_value = 0.0
    g.inputs["IOR"].default_value = 1.376
    t = node(nt, "ShaderNodeBsdfTransparent", (0, -100))
    lp = node(nt, "ShaderNodeLightPath", (0, 350))
    mx = node(nt, "ShaderNodeMixShader", (300, 0))
    link(nt, lp.outputs["Is Shadow Ray"], mx.inputs[0])
    link(nt, g.outputs[0], mx.inputs[1])
    link(nt, t.outputs[0], mx.inputs[2])
    link(nt, mx.outputs[0], out.inputs["Surface"])
    return m


def simple(name, col, rough=0.5, spec=0.5, coat=0.0, sss=0.0):
    m, nt = new_mat(name)
    out = node(nt, "ShaderNodeOutputMaterial", (400, 0))
    b = node(nt, "ShaderNodeBsdfPrincipled", (0, 0))
    b.inputs["Base Color"].default_value = srgb(*col)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Specular IOR Level"].default_value = spec
    b.inputs["Coat Weight"].default_value = coat
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m


def strand_material(name="Fluff"):
    m, nt = new_mat(name)
    out = node(nt, "ShaderNodeOutputMaterial", (600, 0))
    a = _attr(nt, "fcol", (-400, 0))
    hi = node(nt, "ShaderNodeHairInfo", (-400, -250))
    # strands darken toward the root (self-shadowed down)
    f = _ss(nt, 0.0, 0.3, hi.outputs["Intercept"], loc=(-200, -250))
    c = _mixc(nt, f, (0.03, 0.03, 0.035, 1), a.outputs["Color"], loc=(0, 0))
    b = node(nt, "ShaderNodeBsdfPrincipled", (250, 0))
    link(nt, c, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.5
    b.inputs["Sheen Weight"].default_value = 0.4
    tr = node(nt, "ShaderNodeBsdfTranslucent", (250, -300))
    link(nt, c, tr.inputs["Color"])
    mx = node(nt, "ShaderNodeMixShader", (450, 0))
    mx.inputs[0].default_value = 0.3
    link(nt, b.outputs[0], mx.inputs[1])
    link(nt, tr.outputs[0], mx.inputs[2])
    link(nt, mx.outputs[0], out.inputs["Surface"])
    return m


def foot_material():
    """Yellow-olive scutate toes: transverse scales along the toe."""
    m, nt = new_mat("Feet")
    out = node(nt, "ShaderNodeOutputMaterial", (900, 0))
    uv = node(nt, "ShaderNodeUVMap", (-1200, 0), uv_map="UVMap")
    sep = node(nt, "ShaderNodeSeparateXYZ", (-1000, 0))
    link(nt, uv.outputs[0], sep.inputs[0])
    # scutes ~1.1 mm apart along the toe (V is metres along)
    ph = _math(nt, "MULTIPLY", sep.outputs["Y"], 6.2832 / 0.0011, loc=(-800, 0))
    wob = node(nt, "ShaderNodeTexNoise", (-1000, -300))
    wob.inputs["Scale"].default_value = 3.0
    link(nt, uv.outputs[0], wob.inputs["Vector"])
    ph = _math(nt, "ADD", ph, _math(nt, "MULTIPLY", wob.outputs["Fac"], 2.0), loc=(-650, 0))
    band = _math(nt, "MULTIPLY", _math(nt, "ADD", _math(nt, "SINE", ph), 1.0), 0.5, loc=(-500, 0))
    groove = _ss(nt, 0.08, 0.35, band, loc=(-350, 0))
    tc = node(nt, "ShaderNodeTexCoord", (-1000, -500))
    nz = node(nt, "ShaderNodeTexNoise", (-800, -500))
    nz.inputs["Scale"].default_value = 1500.0
    nz.inputs["Detail"].default_value = 5.0
    link(nt, tc.outputs["Object"], nz.inputs["Vector"])
    base = _mixc(nt, nz.outputs["Fac"], srgb(0.72, 0.60, 0.30), srgb(0.55, 0.46, 0.24), loc=(-300, -300))
    col = _mixc(nt, _math(nt, "ADD", groove, 0.45, clamp=True), srgb(0.30, 0.24, 0.12), base, loc=(-100, -200))
    b = node(nt, "ShaderNodeBsdfPrincipled", (300, 0))
    link(nt, col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.55
    b.inputs["Subsurface Weight"].default_value = 0.15
    b.inputs["Subsurface Radius"].default_value = (0.001, 0.0006, 0.0003)
    bump = node(nt, "ShaderNodeBump", (100, -500))
    bump.inputs["Strength"].default_value = 0.6
    bump.inputs["Distance"].default_value = 0.0002
    link(nt, groove, bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], b.inputs["Normal"])
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m


def bark_material():
    """Sun-bleached dead acacia twig: grey-tan, split and cracked."""
    m, nt = new_mat("Bark")
    out = node(nt, "ShaderNodeOutputMaterial", (1200, 0))
    uv = node(nt, "ShaderNodeUVMap", (-1400, 0), uv_map="UVMap")
    mp = node(nt, "ShaderNodeMapping", (-1200, 0))
    mp.inputs["Scale"].default_value = (6.0, 12.0, 1.0)  # stretch features along the wood
    link(nt, uv.outputs[0], mp.inputs["Vector"])
    mp2 = node(nt, "ShaderNodeMapping", (-1200, -300))
    mp2.inputs["Scale"].default_value = (14.0, 45.0, 1.0)
    link(nt, uv.outputs[0], mp2.inputs["Vector"])
    big = node(nt, "ShaderNodeTexNoise", (-1000, 0))
    big.inputs["Scale"].default_value = 3.0
    big.inputs["Detail"].default_value = 8.0
    big.inputs["Roughness"].default_value = 0.6
    link(nt, mp.outputs[0], big.inputs["Vector"])
    vor = node(nt, "ShaderNodeTexVoronoi", (-1000, -300), feature="DISTANCE_TO_EDGE")
    vor.inputs["Scale"].default_value = 1.0
    vor.inputs["Randomness"].default_value = 1.0
    link(nt, mp2.outputs[0], vor.inputs["Vector"])
    crack = _ss(nt, 0.0, 0.035, vor.outputs["Distance"], loc=(-800, -300))
    fine = node(nt, "ShaderNodeTexNoise", (-1000, -600))
    fine.inputs["Scale"].default_value = 25.0
    fine.inputs["Detail"].default_value = 10.0
    link(nt, mp.outputs[0], fine.inputs["Vector"])
    light = srgb(0.62, 0.58, 0.53)
    warm = srgb(0.52, 0.43, 0.35)
    grey = srgb(0.42, 0.41, 0.40)
    c = _mixc(nt, big.outputs["Fac"], light, warm, loc=(-600, 0))
    c = _mixc(nt, _ss(nt, 0.45, 0.75, fine.outputs["Fac"]), c, grey, loc=(-400, 0))
    c = _mixc(nt, _math(nt, "ADD", crack, 0.35, clamp=True), srgb(0.25, 0.21, 0.17), c, loc=(-200, 0))
    b = node(nt, "ShaderNodeBsdfPrincipled", (400, 0))
    link(nt, c, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85
    h = _math(nt, "ADD", _math(nt, "MULTIPLY", crack, 0.6), _math(nt, "MULTIPLY", fine.outputs["Fac"], 0.4), loc=(-200, -400))
    bump = node(nt, "ShaderNodeBump", (100, -400))
    bump.inputs["Strength"].default_value = 0.9
    bump.inputs["Distance"].default_value = 0.0006
    link(nt, h, bump.inputs["Height"])
    link(nt, bump.outputs["Normal"], b.inputs["Normal"])
    link(nt, b.outputs[0], out.inputs["Surface"])
    return m
