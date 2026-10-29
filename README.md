# Lilac-breasted Roller — procedural 3D render

A photoreal Lilac-breasted Roller (*Coracias caudatus*) perched on a dead
branch in low golden-hour light, looking around. Everything is generated from
Python with Blender's `bpy` module and rendered in Cycles. There are no
hand-sculpted assets and no image textures.

## The story

A roller sits on its favourite vantage perch: the bare tip of a dead twig
against hazy sky. It does what rollers spend most of the day doing, which is
scanning the ground for insects and lizards. The head moves the way bird heads
really move. It holds dead still, snaps to a new direction in four frames with
a tiny overshoot, then holds again. It cocks its head to put one eye on the
ground, looks away along the horizon, and glances straight at the camera. It
blinks twice, breathes, and flicks its tail once.

## What the model is based on

From field descriptions (eBird, Forshaw's *Watchbird* account and others), a
field-guide plate, and the reference video frames:

| Region | Colour / detail |
|---|---|
| Forehead, supercilium, chin | creamy white |
| Lores + eye-stripe | black, from the gape through the eye |
| Crown → nape | pale yellow-green → greenish turquoise |
| Ear-coverts / cheeks | rufous-orange, grading into lilac |
| Throat + breast | lilac with **white shaft streaks** |
| Belly, flanks, vent | turquoise, paler toward the vent |
| Mantle, scapulars, tertials | cinnamon-brown |
| Bend of wing (lesser coverts) | deep ultramarine-violet |
| Median / greater coverts | light turquoise-azure, scalloped |
| Primaries | azure base → deep violet-blue → blackish tips |
| Rump | violet-blue |
| Tail | azure, darker central pair; outer pair elongated into black-tipped **streamers** (~8–9 cm beyond the tail; adults only) |
| Bill | black, stout, arched culmen, hooked tip |
| Eye | dark brown, set in the black stripe |
| Feet | yellow-olive, scutate, **syndactyl** (toes 3+4 fused at the base) |

Size is about 37 cm including the streamers (~104 g). Proportions were measured
off the reference frames. Sexes look alike.

## How it's built (`src/`)

- `sdf.py`: body and head as signed-distance fields (smooth-unioned
  ellipsoids and capsules), polygonised with marching cubes. The head is a
  separate rigid part with a cylindrical neck plug, so it can turn through any
  angle without opening a gap.
- `plumage.py`: about 8,000 contour feathers. They are blue-noise scattered,
  sized and coloured from an anatomical colour map, oriented along a
  bill→tail flow field, and draped onto the body as shingled, cross-curved
  cards. About 175k loose barb strands give the silky, puffed look.
- `wing.py`: the folded wing built feather by feather (primaries,
  secondaries, tertials, three covert rows, scapulars) and laid out in the
  wing's side profile. There is also the tail with 12 rectrices, coverts, and
  the streamers.
- `beak.py`: superellipse lofted bill with gape line, decurved culmen and hook.
- `limbs.py`: syndactyl feet wrapped around the twig, with claws and tarsi. It
  also builds the branch system.
- `materials.py`: the shaders. The feather shader has barbs angled off the
  rachis, frayed vane edges, vane splits, a downy base, shaft streaks,
  translucency, and thin-film sheen on the structural blues. The eye has a glass
  cornea over a fibrous iris. There are also bill keratin, scutate feet, and
  weathered bark.
- `anim.py`: head saccades, blinks, breathing, tail flick.
- `build.py`: assembles the scene: camera (330 mm, f/5.6, motion blur), low sun, sky.
- `render.py`: renders stills or the animation.

## Reproduce

```bash
python3 -m venv /opt/bpyenv && /opt/bpyenv/bin/pip install bpy==5.2.3 numpy scipy scikit-image pillow
```

```bash
/opt/bpyenv/bin/python src/build.py
```

```bash
/opt/bpyenv/bin/python src/render.py still renders/hero.png --frame 1
```

```bash
/opt/bpyenv/bin/python src/render.py anim renders/frames --pct 67 --samples 64
```

```bash
ffmpeg -framerate 24 -i renders/frames/f%04d.png -c:v libx264 -crf 16 -pix_fmt yuv420p renders/roller.mp4
```

The extra cameras `CloseCam`, `HeadCam` and `FeetCam` are for look-dev
(`--cam HeadCam`).
