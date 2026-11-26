#!/usr/bin/env python3
"""
mcrender.py - pixel-exact 3D Minecraft-style player renderer with studio lighting.

Why this exists: the #1 thing that makes a thumbnail look cheap is a sloppy skin
cutout (halo, missing overlay pixels, blurred texels, jagged edges). Instead of
cutting a character out of a screenshot, we RENDER it ourselves with a true
alpha channel, so the silhouette is mathematically exact and every skin texel
is a crisp square.

Features
  * classic (4px) and slim (3px) arms, 64x64 and legacy 64x32 skins
  * second layer (hat / jacket / sleeves / pants) inflated like in-game,
    alpha-tested so transparent overlay pixels never hide the base layer
  * posable rig (root, torso, head, arms, legs) + pose presets
  * held item as extruded voxel sprite (original pixel art, several palettes)
  * perspective camera with auto framing
  * lighting: key (sun) + sky fill + ground bounce + back/rim light,
    shadow-mapped self shadows (PCF soft), screen-space AO,
    screen-space silhouette rim highlight, subtle specular, filmic tone map
  * supersampled anti-aliasing (box downsample of premultiplied colour)
  * extra passes (alpha, depth, normal, part-id, rim mask) for compositing

Use as a module (compose.py does) or CLI:
  python3 mcrender.py --skin skin.png --pose run --yaw -30 --out char.png
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# math helpers
# ----------------------------------------------------------------------------

def rx(a):
    a = math.radians(a); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)

def ry(a):
    a = math.radians(a); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)

def rz(a):
    a = math.radians(a); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)

def norm(v):
    v = np.asarray(v, np.float64); n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v

def srgb_to_lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)

def hex_rgb(h):
    if isinstance(h, (list, tuple)):
        return np.array(h[:3], np.float32) / (255.0 if max(h[:3]) > 1 else 1.0)
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255.0

def filmic(x):
    """ACES-ish fitted curve (Narkowicz) - keeps highlights soft, like Blender Filmic."""
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

# ----------------------------------------------------------------------------
# skin loading
# ----------------------------------------------------------------------------

def load_skin(path_or_img) -> np.ndarray:
    """Return 64x64 RGBA float32 (sRGB, 0..1). Legacy 64x32 skins are upgraded."""
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    im = im.convert('RGBA')
    w, h = im.size
    if w != 64 and w % 64 == 0:  # HD skins: nearest downscale to 64 grid
        im = im.resize((64, h * 64 // w), Image.NEAREST); w, h = im.size
    if (w, h) == (64, 32):
        im = upgrade_legacy(im)
    if im.size != (64, 64):
        raise ValueError(f'unsupported skin size {im.size}')
    return np.asarray(im, np.float32) / 255.0

def upgrade_legacy(im: Image.Image) -> Image.Image:
    """64x32 -> 64x64: left limbs = mirrored right limbs (vanilla behaviour)."""
    new = Image.new('RGBA', (64, 64), (0, 0, 0, 0)); new.paste(im, (0, 0))
    def mirror_limb(src_u, src_v, dst_u, dst_v):
        # box w=4,h=12,d=4 ; faces: right,front,left,back (row v+4) top,bottom (row v)
        def cp(sx, sy, w, h, dx, dy):
            new.paste(im.crop((sx, sy, sx + w, sy + h)).transpose(Image.FLIP_LEFT_RIGHT), (dx, dy))
        u, v, U, V = src_u, src_v, dst_u, dst_v
        cp(u + 4, v, 4, 4, U + 4, V)      # top
        cp(u + 8, v, 4, 4, U + 8, V)      # bottom
        cp(u + 0, v + 4, 4, 12, U + 8, V + 4)   # right -> left
        cp(u + 4, v + 4, 4, 12, U + 4, V + 4)   # front
        cp(u + 8, v + 4, 4, 12, U + 0, V + 4)   # left -> right
        cp(u + 12, v + 4, 4, 12, U + 12, V + 4) # back
    mirror_limb(0, 16, 16, 48)   # leg
    mirror_limb(40, 16, 32, 48)  # arm
    return new

def detect_slim(skin: np.ndarray) -> bool:
    """Slim (Alex) skins leave the 4th column of the right-arm front transparent."""
    a = skin[..., 3]
    # slim layout leaves x 54-55 (rows 20-31) and x 50-51 (rows 16-19) unused
    return bool(a[20:32, 54:56].max() < 0.01 and a[16:20, 50:52].max() < 0.01)

# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------

class Mesh:
    def __init__(self):
        self.quads = []  # dict(p=(4,3) tl,tr,bl,br ; tex=idx ; rect=(u,v,w,h) ; part ; alpha_test ; double)
        self.textures = []

    def add_tex(self, arr):
        self.textures.append(arr.astype(np.float32)); return len(self.textures) - 1

def box_faces(x0, y0, z0, x1, y1, z1, u, v, w, h, d):
    """Six faces of a cuboid in vanilla box-UV layout.
    Returns list of (name, tl, tr, bl, br, (tu,tv,tw,th))."""
    P = lambda x, y, z: np.array([x, y, z], np.float64)
    f = []
    f.append(('front', P(x0, y1, z1), P(x1, y1, z1), P(x0, y0, z1), P(x1, y0, z1), (u + d, v + d, w, h)))
    f.append(('back', P(x1, y1, z0), P(x0, y1, z0), P(x1, y0, z0), P(x0, y0, z0), (u + 2 * d + w, v + d, w, h)))
    f.append(('right', P(x0, y1, z0), P(x0, y1, z1), P(x0, y0, z0), P(x0, y0, z1), (u, v + d, d, h)))
    f.append(('left', P(x1, y1, z1), P(x1, y1, z0), P(x1, y0, z1), P(x1, y0, z0), (u + d + w, v + d, d, h)))
    f.append(('top', P(x0, y1, z0), P(x1, y1, z0), P(x0, y1, z1), P(x1, y1, z1), (u + d, v, w, d)))
    f.append(('bottom', P(x0, y0, z1), P(x1, y0, z1), P(x0, y0, z0), P(x1, y0, z0), (u + d + w, v, w, d)))
    return f

# part id constants (used for masks in compositing)
PART_IDS = {'head': 1, 'body': 2, 'rarm': 3, 'larm': 4, 'rleg': 5, 'lleg': 6, 'item': 7, 'item2': 8}

def player_parts(slim: bool):
    aw = 3 if slim else 4
    # name: (box min, box max relative to pivot), pivot, base uv, overlay uv, size(w,h,d), inflate overlay
    return {
        'head': dict(pivot=(0, 24, 0), mn=(-4, 0, -4), mx=(4, 8, 4), uv=(0, 0), ov=(32, 0), s=(8, 8, 8), inf=0.5),
        'body': dict(pivot=(0, 24, 0), mn=(-4, -12, -2), mx=(4, 0, 2), uv=(16, 16), ov=(16, 32), s=(8, 12, 4), inf=0.25),
        'rarm': dict(pivot=(-5 if not slim else -5, 22, 0), mn=(-aw + 1 if not slim else -2, -10, -2),
                     mx=(1, 2, 2), uv=(40, 16), ov=(40, 32), s=(aw, 12, 4), inf=0.25),
        'larm': dict(pivot=(5, 22, 0), mn=(-1, -10, -2), mx=(aw - 1 if not slim else 2, 2, 2),
                     uv=(32, 48), ov=(48, 48), s=(aw, 12, 4), inf=0.25),
        'rleg': dict(pivot=(-1.9, 12, 0), mn=(-2.1, -12, -2), mx=(1.9, 0, 2), uv=(0, 16), ov=(0, 32), s=(4, 12, 4), inf=0.25),
        'lleg': dict(pivot=(1.9, 12, 0), mn=(-1.9, -12, -2), mx=(2.1, 0, 2), uv=(16, 48), ov=(0, 48), s=(4, 12, 4), inf=0.25),
    }

# ----------------------------------------------------------------------------
# poses
# ----------------------------------------------------------------------------
# Every limb: fwd (raise forward, deg), side (raise OUTWARD away from the body, deg; same sign for both sides), twist (deg)
# head: yaw (+ = turn to the character's left / viewer's right when facing camera),
#       pitch (+ = look down), roll
# torso: yaw, pitch (+ = lean forward), roll ; root: yaw, pitch, roll, lift (px)

POSES = {
    'idle':   dict(head=dict(yaw=8, pitch=4), rarm=dict(fwd=6, side=4), larm=dict(fwd=-4, side=4),
                   rleg=dict(fwd=-3), lleg=dict(fwd=3)),
    'stand_sword': dict(head=dict(yaw=-6, pitch=6), torso=dict(yaw=-6), rarm=dict(fwd=28, side=6, twist=-10),
                   larm=dict(fwd=-8, side=6), rleg=dict(fwd=-6, side=2), lleg=dict(fwd=6, side=-2)),
    'walk':   dict(head=dict(yaw=6, pitch=3), rarm=dict(fwd=-28, side=3), larm=dict(fwd=28, side=3),
                   rleg=dict(fwd=26), lleg=dict(fwd=-26), root=dict(lift=0.3)),
    'run':    dict(head=dict(pitch=-4, yaw=6), torso=dict(pitch=8), rarm=dict(fwd=-55, side=6), larm=dict(fwd=60, side=6),
                   rleg=dict(fwd=48), lleg=dict(fwd=-48), root=dict(pitch=6, lift=1.0)),
    'sprint_attack': dict(head=dict(pitch=-2, yaw=-10), torso=dict(pitch=10, yaw=-12), rarm=dict(fwd=95, side=12, twist=-25),
                   larm=dict(fwd=-45, side=10), rleg=dict(fwd=-42), lleg=dict(fwd=40), root=dict(pitch=8, lift=1.5)),
    'swing':  dict(head=dict(yaw=-14, pitch=4), torso=dict(yaw=-20, pitch=4), rarm=dict(fwd=130, side=-14, twist=-35),
                   larm=dict(fwd=-30, side=16), rleg=dict(fwd=-22, side=4), lleg=dict(fwd=24, side=-2)),
    'jump':   dict(head=dict(pitch=-10, yaw=6), torso=dict(pitch=-4), rarm=dict(fwd=-30, side=40), larm=dict(fwd=150, side=18),
                   rleg=dict(fwd=55), lleg=dict(fwd=-30), root=dict(lift=6, pitch=-4)),
    'leap':   dict(head=dict(pitch=-6, yaw=10), torso=dict(pitch=6, roll=-4), rarm=dict(fwd=-50, side=20),
                   larm=dict(fwd=75, side=14), rleg=dict(fwd=70), lleg=dict(fwd=-45), root=dict(lift=5, pitch=10, roll=-6)),
    'wave':   dict(head=dict(yaw=6, pitch=-4, roll=-4), rarm=dict(fwd=10, side=150, twist=10), larm=dict(fwd=-6, side=8),
                   rleg=dict(fwd=-4, side=3), lleg=dict(fwd=4, side=-3)),
    'cheer':  dict(head=dict(pitch=-14), rarm=dict(fwd=20, side=155), larm=dict(fwd=20, side=155),
                   rleg=dict(side=6), lleg=dict(side=-6), root=dict(lift=1)),
    'point':  dict(head=dict(yaw=-18, pitch=2), torso=dict(yaw=-10), rarm=dict(fwd=88, side=-14), larm=dict(fwd=-8, side=6),
                   rleg=dict(fwd=-8), lleg=dict(fwd=8)),
    'look_back': dict(head=dict(yaw=-28, pitch=10), torso=dict(yaw=-8, pitch=4), rarm=dict(fwd=34, side=10, twist=-20),
                   larm=dict(fwd=-6, side=8), rleg=dict(fwd=6), lleg=dict(fwd=-10), root=dict()),
    'crouch': dict(head=dict(pitch=10), torso=dict(pitch=28), rarm=dict(fwd=20, side=4), larm=dict(fwd=20, side=4),
                   rleg=dict(fwd=-12), lleg=dict(fwd=-12), root=dict(lift=-1.5)),
    'flex':   dict(head=dict(yaw=10, pitch=-4), torso=dict(yaw=8), rarm=dict(fwd=5, side=85, twist=0),
                   larm=dict(fwd=5, side=85), rleg=dict(side=8), lleg=dict(side=-8)),
    'dance':  dict(head=dict(roll=8, yaw=10), torso=dict(roll=-6), rarm=dict(fwd=30, side=110), larm=dict(fwd=-20, side=40),
                   rleg=dict(fwd=30, side=6), lleg=dict(fwd=-8, side=-6), root=dict(roll=4, lift=0.8)),
    't_pose': dict(rarm=dict(side=90), larm=dict(side=90)),
}

def resolve_pose(pose):
    if isinstance(pose, str):
        if pose not in POSES: raise ValueError(f'unknown pose {pose}; choose from {list(POSES)}')
        return json.loads(json.dumps(POSES[pose]))
    base = resolve_pose(pose.get('base', 'idle')) if 'base' in pose else {}
    for k, v in pose.items():
        if k == 'base': continue
        base.setdefault(k, {}).update(v)
    return base

# ----------------------------------------------------------------------------
# items (original pixel-art, 16x16) - legend: . empty, o outline, d dark, m mid, l light, w white/hilite, h handle dark, H handle light, g guard
# ----------------------------------------------------------------------------
SPRITES = {
'sword': [
".............ooo",
"............owlo",
"...........owlmo",
"..........owlmo.",
".........owlmo..",
"........owlmo...",
".......owlmo....",
"......owlmo.....",
".oo..owlmo......",
".ogooldmo.......",
"..oggddo........",
"...ogho.........",
"..ohHggo........",
".ohHo.ogo.......",
"ohHo...oo.......",
"ooo.............",
],
'pickaxe': [
"....ooooo.......",
"...ollllmo......",
"....oddmmlo.....",
".........omlo...",
"........ohoomo..",
".......ohHo.odo.",
"......ohHo..omo.",
".....ohHo....ldo",
"....ohHo.....omo",
"...ohHo.......o.",
"..ohHo..........",
".ohHo...........",
"ohHo............",
"oho.............",
"oo..............",
"................",
],
'axe': [
"......oooo......",
".....ollmmo.....",
"....olwlmmdo....",
"....ollmohdo....",
".....ommhHo.....",
"......ohHdo.....",
".....ohHo.o.....",
"....ohHo........",
"...ohHo.........",
"..ohHo..........",
".ohHo...........",
"ohHo............",
"oho.............",
"oo..............",
"................",
"................",
],
}

ITEM_PALETTES = {
    # outline, dark, mid, light, white, handle dark, handle light, guard
    'diamond':   dict(o='#0c3b3a', d='#1b8b85', m='#2fd0c4', l='#7ef5e6', w='#e6fffb', h='#3d2a14', H='#7a5528', g='#1b8b85'),
    'netherite': dict(o='#1a1418', d='#3a3036', m='#5b4f57', l='#8a7d86', w='#c9bcc5', h='#2a1f14', H='#5a4026', g='#3a3036'),
    'gold':      dict(o='#5a3a00', d='#c58a10', m='#f2c230', l='#ffe57a', w='#fffbe0', h='#3d2a14', H='#7a5528', g='#c58a10'),
    'iron':      dict(o='#2a2a2a', d='#8a8a8a', m='#c8c8c8', l='#e8e8e8', w='#ffffff', h='#3d2a14', H='#7a5528', g='#8a8a8a'),
    'amethyst':  dict(o='#2a0f4a', d='#6a2fb0', m='#a45cf0', l='#d4a8ff', w='#f6ecff', h='#2a1f3a', H='#5a3a7a', g='#6a2fb0'),
    'ruby':      dict(o='#3a0610', d='#9a1530', m='#e0304e', l='#ff8a9a', w='#ffe6ea', h='#2a1f14', H='#5a4026', g='#9a1530'),
    'emerald':   dict(o='#063a1a', d='#14904a', m='#2fdc74', l='#9cffc0', w='#eafff2', h='#3d2a14', H='#7a5528', g='#14904a'),
    'wood':      dict(o='#2a1a08', d='#6a4a20', m='#9a7034', l='#c49a58', w='#e0c48a', h='#3d2a14', H='#7a5528', g='#6a4a20'),
}

def sprite_rgba(kind='sword', palette='diamond', custom=None):
    rows = SPRITES[kind]; pal = dict(ITEM_PALETTES[palette]);
    if custom: pal.update(custom)
    a = np.zeros((16, 16, 4), np.float32)
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch == '.': continue
            a[j, i, :3] = hex_rgb(pal[ch]); a[j, i, 3] = 1
    return a

# ----------------------------------------------------------------------------
# build the posed mesh
# ----------------------------------------------------------------------------

def build_player(skin, slim=None, pose='idle', item=None, overlay=True):
    if slim is None: slim = detect_slim(skin)
    P = resolve_pose(pose)
    g = lambda part, k: float(P.get(part, {}).get(k, 0.0))
    mesh = Mesh(); tex = mesh.add_tex(skin)
    parts = player_parts(slim)

    # root transform (about the hips, y=12)
    hip = np.array([0, 12, 0.0])
    R_root = ry(g('root', 'yaw')) @ rx(-g('root', 'pitch')) @ rz(-g('root', 'roll'))
    T_root = np.array([0, g('root', 'lift'), 0.0])
    R_torso = ry(g('torso', 'yaw')) @ rx(-g('torso', 'pitch')) @ rz(-g('torso', 'roll'))

    def limb_R(name, sign):
        return rz(sign * g(name, 'side')) @ rx(-g(name, 'fwd')) @ ry(g(name, 'twist'))

    local_R = {
        'head': ry(g('head', 'yaw')) @ rx(g('head', 'pitch')) @ rz(-g('head', 'roll')),
        'body': np.eye(3),
        'rarm': limb_R('rarm', -1), 'larm': limb_R('larm', +1),
        'rleg': limb_R('rleg', -1), 'lleg': limb_R('lleg', +1),
    }

    def world_xf(name):
        pv = np.array(parts[name]['pivot'], np.float64)
        Rl = local_R[name]
        if name in ('head', 'body', 'rarm', 'larm'):   # children of torso (torso pivots at hip)
            def f(p):
                q = (Rl @ p.T).T + pv              # local -> model
                q = (R_torso @ (q - hip).T).T + hip  # torso lean around hips
                return (R_root @ (q - hip).T).T + hip + T_root
        else:
            def f(p):
                q = (Rl @ p.T).T + pv
                return (R_root @ (q - hip).T).T + hip + T_root
        return f

    xfs = {n: world_xf(n) for n in parts}
    for name, pp in parts.items():
        w, h, d = pp['s']
        layers = [(pp['uv'], 0.0, False)]
        if overlay: layers.append((pp['ov'], pp['inf'], True))
        for (u, v), inf, is_ov in layers:
            mn = np.array(pp['mn']) - inf; mx = np.array(pp['mx']) + inf
            for fname, tl, tr, bl, br, rect in box_faces(*mn, *mx, u, v, w, h, d):
                pts = xfs[name](np.stack([tl, tr, bl, br]))
                if is_ov:
                    tu, tv, tw, th = rect
                    if skin[tv:tv + th, tu:tu + tw, 3].max() < 0.5:
                        continue  # empty overlay face
                mesh.quads.append(dict(p=pts, tex=tex, rect=rect, part=PART_IDS[name],
                                       alpha_test=is_ov, double=is_ov))

    if item:
        add_item(mesh, item, xfs['rarm'] if item.get('hand', 'right') == 'right' else xfs['larm'], PART_IDS['item'])
    return mesh, slim

def add_item(mesh, item, arm_xf, part_id):
    kind = item.get('kind', 'sword'); pal = item.get('palette', 'diamond')
    spr = sprite_rgba(kind, pal, item.get('colors'))
    tex = mesh.add_tex(spr)
    s = float(item.get('scale', 0.68))      # world units per sprite pixel
    th = float(item.get('thickness', 1.0))   # voxel depth in sprite pixels
    grip = item.get('grip', {'sword': (3.5, 12.5), 'pickaxe': (3, 11.5), 'axe': (3, 10.5)}[kind])
    # sprite axes in arm-local space: tip direction (+col,-row) -> forward(+z), slightly downwards tilt adjustable
    a = norm([0, 1, 1]) * s     # +col
    b = norm([0, 1, -1]) * s    # +row
    n = np.array([1.0, 0, 0]) * s * th
    R_extra = ry(item.get('yaw', 0)) @ rx(-item.get('pitch', 0)) @ rz(item.get('roll', 0))
    hand = np.array(item.get('offset', (0, -9.0, 0.5)), np.float64)
    def to_arm(col, row, depth):
        p = (col - grip[0]) * a + (row - grip[1]) * b + (depth - 0.5) * n
        return (R_extra @ p) + hand
    alpha = spr[..., 3] > 0.5
    def opaque(i, j):
        return 0 <= i < 16 and 0 <= j < 16 and alpha[j, i]
    for j in range(16):
        for i in range(16):
            if not alpha[j, i]: continue
            rect = (i, j, 1, 1)
            # faces of voxel [i,i+1]x[j,j+1]x[0,1] in sprite space
            C = lambda c, r, d_: to_arm(c, r, d_)
            faces = []
            # sprite front (depth=0) & back (depth=1)
            faces.append((C(i, j, 0), C(i + 1, j, 0), C(i, j + 1, 0), C(i + 1, j + 1, 0)))
            faces.append((C(i + 1, j, 1), C(i, j, 1), C(i + 1, j + 1, 1), C(i, j + 1, 1)))
            if not opaque(i, j - 1): faces.append((C(i, j, 1), C(i + 1, j, 1), C(i, j, 0), C(i + 1, j, 0)))
            if not opaque(i, j + 1): faces.append((C(i, j + 1, 0), C(i + 1, j + 1, 0), C(i, j + 1, 1), C(i + 1, j + 1, 1)))
            if not opaque(i - 1, j): faces.append((C(i, j, 1), C(i, j, 0), C(i, j + 1, 1), C(i, j + 1, 0)))
            if not opaque(i + 1, j): faces.append((C(i + 1, j, 0), C(i + 1, j, 1), C(i + 1, j + 1, 0), C(i + 1, j + 1, 1)))
            for f in faces:
                pts = arm_xf(np.stack(f))
                mesh.quads.append(dict(p=pts, tex=tex, rect=rect, part=part_id, alpha_test=False, double=True))

# ----------------------------------------------------------------------------
# rasteriser (z-buffer, perspective-correct barycentrics, per-triangle loop)
# ----------------------------------------------------------------------------

def quads_to_tris(mesh):
    P = np.array([q['p'] for q in mesh.quads], np.float64)           # (Q,4,3)
    ST = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float64)
    tri_idx = [(0, 1, 2), (1, 3, 2)]
    V = np.concatenate([P[:, list(t)] for t in tri_idx], 0)           # (2Q,3,3)
    UV = np.concatenate([np.broadcast_to(ST[list(t)], (len(P), 3, 2)) for t in tri_idx], 0)
    qi = np.concatenate([np.arange(len(P))] * 2)
    return V, UV, qi

def rasterize(Vs, W, H, tex_lookup=None, cull=None):
    """Vs: (N,3,3) screen-space verts: x,y pixels, z = view depth (>0).
    tex_lookup(t, s, t_) -> bool mask for alpha test (or None).
    Returns depth (H,W), tri id (H,W) int32 (-1 empty), bary (H,W,3) perspective-correct."""
    zbuf = np.full((H, W), np.inf, np.float32)
    tid = np.full((H, W), -1, np.int32)
    bary = np.zeros((H, W, 3), np.float32)
    for t in range(len(Vs)):
        v = Vs[t]
        if cull is not None and not cull[t]: continue
        if (v[:, 2] <= 1e-3).any(): continue
        xmin = max(int(np.floor(v[:, 0].min())), 0); xmax = min(int(np.ceil(v[:, 0].max())), W - 1)
        ymin = max(int(np.floor(v[:, 1].min())), 0); ymax = min(int(np.ceil(v[:, 1].max())), H - 1)
        if xmin > xmax or ymin > ymax: continue
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = v
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9: continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)[:, None] + 0.5
        w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area
        w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any(): continue
        # perspective-correct
        q0, q1, q2 = w0 / z0, w1 / z1, w2 / z2
        qs = q0 + q1 + q2
        z = 1.0 / qs
        sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        m = inside & (z < sub)
        if not m.any(): continue
        b0, b1, b2 = q0 / qs, q1 / qs, q2 / qs
        if tex_lookup is not None:
            keep = tex_lookup(t, b0, b1, b2)
            if keep is not None: m &= keep
            if not m.any(): continue
        sub[m] = z[m]
        tid[ymin:ymax + 1, xmin:xmax + 1][m] = t
        bs = bary[ymin:ymax + 1, xmin:xmax + 1]
        bs[..., 0][m] = b0[m]; bs[..., 1][m] = b1[m]; bs[..., 2][m] = b2[m]
    return zbuf, tid, bary

# ----------------------------------------------------------------------------
# camera
# ----------------------------------------------------------------------------

class Camera:
    def __init__(self, yaw=0, pitch=5, fov=30, target=(0, 16, 0), dist=None, roll=0):
        self.fov = fov
        d = dist if dist is not None else 32 / (2 * math.tan(math.radians(fov) / 2)) * 1.45
        self.target = np.array(target, np.float64)
        # yaw 0 => camera on +z looking at the character's front. yaw>0 orbits to the character's left (viewer right)
        dirv = ry(yaw) @ rx(-pitch) @ np.array([0, 0, 1.0])
        self.pos = self.target + dirv * d
        f = norm(self.target - self.pos)
        r = norm(np.cross(f, [0, 1, 0]))
        u = np.cross(r, f)
        if roll:
            R = np.eye(3); c, s = math.cos(math.radians(roll)), math.sin(math.radians(roll))
            r, u = r * c + u * s, -r * s + u * c
        self.f, self.r, self.u = f, r, u

    def view(self, P):
        d = P - self.pos
        return np.stack([d @ self.r, d @ self.u, d @ self.f], -1)  # x right, y up, z forward

# ----------------------------------------------------------------------------
# main render
# ----------------------------------------------------------------------------

DEFAULT_LIGHT = dict(
    # Directions are CAMERA-relative: x = screen right, y = up, z = towards the viewer.
    # key_cam z<0 means the sun is BEHIND the character (backlight -> strong rim, softer front).
    key_cam=(0.62, 0.6, 0.35),     # direction TOWARDS the key light (sun)
    key_dir=None,                  # optional world-space override
    key_color='#fff1dc', key=3.2,
    sky_color='#bcd6ff', sky=0.55,
    ground_color='#d9c7a8', ground=0.25,
    rim_cam=None,                  # default: behind the character, same side as the key
    fill_cam=None,                 # default: camera side, opposite the key (soft studio fill)
    fill_color='#dfe9ff', fill=0.9,
    rim_color='#fff6e8', rim=1.6,
    rim_width=0.9,                 # screen-space silhouette rim width (in output px)
    spec=0.18, ao=0.55, shadow=0.85, exposure=1.0,
)

def render(skin, pose='idle', slim=None, item=None, overlay=True, yaw=-20, pitch=6, fov=32,
           height=1200, ss=3, margin=0.04, light=None, cam_roll=0, mesh_override=None, target=None, dist=None):
    """Returns dict(rgba=uint8 (H,W,4) sRGB straight alpha, alpha, depth, normal, part, rim, light_screen_dir)."""
    L = dict(DEFAULT_LIGHT); L.update(light or {})
    if skin is not None and (isinstance(skin, str) or isinstance(skin, Image.Image)): skin = load_skin(skin)
    if mesh_override is not None: mesh, slim = mesh_override, slim
    else: mesh, slim = build_player(skin, slim, pose, item, overlay)
    V, UV, qi = quads_to_tris(mesh)
    if target is None and mesh_override is not None:
        allp = np.array([qq['p'] for qq in mesh.quads]).reshape(-1, 3); target = (allp.min(0) + allp.max(0)) / 2
        if dist is None: dist = np.linalg.norm(allp.max(0) - allp.min(0)) / (2 * math.tan(math.radians(fov) / 2)) * 1.2
    cam = Camera(yaw=yaw, pitch=pitch, fov=fov, roll=cam_roll, target=target if target is not None else (0, 16, 0), dist=dist)
    Vv = cam.view(V.reshape(-1, 3)).reshape(V.shape)
    f = 1.0 / math.tan(math.radians(fov) / 2)
    px = Vv[..., 0] / Vv[..., 2] * f; py = Vv[..., 1] / Vv[..., 2] * f
    # auto framing
    x0, x1, y0, y1 = px.min(), px.max(), py.min(), py.max()
    span_y = (y1 - y0); span_x = (x1 - x0)
    Hs = int(height * ss); scale = Hs * (1 - 2 * margin) / span_y
    Ws = int(math.ceil(span_x * scale + 2 * margin * Hs)); Ws += (-Ws) % ss
    sx = (px - x0) * scale + margin * Hs
    sy = (y1 - py) * scale + margin * Hs
    S = np.stack([sx, sy, Vv[..., 2]], -1)

    # face normals in world space
    q = np.array([qq['p'] for qq in mesh.quads])
    N = np.cross(q[:, 2] - q[:, 0], q[:, 1] - q[:, 0]); N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    centers = q.mean(1)
    facing = ((cam.pos - centers) * N).sum(1) > 0
    double = np.array([qq['double'] for qq in mesh.quads])
    cull = np.concatenate([facing | double] * 2)

    rects = np.array([qq['rect'] for qq in mesh.quads], np.float32)
    texid = np.array([qq['tex'] for qq in mesh.quads])
    atest = np.array([qq['alpha_test'] for qq in mesh.quads])
    T = mesh.textures

    def sample(tri, b0, b1, b2, channel=None):
        uv = UV[tri]; qd = qi[tri]
        s = b0 * uv[0, 0] + b1 * uv[1, 0] + b2 * uv[2, 0]
        t = b0 * uv[0, 1] + b1 * uv[1, 1] + b2 * uv[2, 1]
        tu, tv, tw, th = rects[qd]
        ui = np.clip(np.floor(tu + s * tw), tu, tu + tw - 1).astype(np.int32)
        vi = np.clip(np.floor(tv + t * th), tv, tv + th - 1).astype(np.int32)
        return T[texid[qd]][vi, ui]

    def alpha_lookup(tri, b0, b1, b2):
        if not atest[qi[tri]]: return None
        return sample(tri, b0, b1, b2)[..., 3] > 0.5

    zbuf, tid, bary = rasterize(S, Ws, Hs, alpha_lookup, cull)
    cov = tid >= 0

    # --- resolve G-buffer
    t = np.where(cov, tid, 0); qd = qi[t]
    albedo = np.zeros((Hs, Ws, 3), np.float32)
    for k in np.unique(texid):
        mk = cov & (texid[qd] == k)
        if not mk.any(): continue
        tt = t[mk]; b = bary[mk]
        uv = UV[tt]
        s_ = (b[:, :1] * uv[:, 0] + b[:, 1:2] * uv[:, 1] + b[:, 2:3] * uv[:, 2])
        tu, tv, tw, th = rects[qi[tt]].T
        ui = np.clip(np.floor(tu + s_[:, 0] * tw), tu, tu + tw - 1).astype(np.int32)
        vi = np.clip(np.floor(tv + s_[:, 1] * th), tv, tv + th - 1).astype(np.int32)
        c = T[k][vi, ui]
        rgb = c[:, :3].copy()
        if k == 0 and mesh_override is None:  # base layer of a skin is opaque in-game: transparent texel => black
            rgb[c[:, 3] < 0.5] = 0.0
        albedo[mk] = rgb
    albedo = srgb_to_lin(albedo)
    nrm = N[qd]
    # flip normals of back-facing double-sided faces
    Vw = V[t]  # (H,W,3,3)
    wpos = (bary[..., :, None] * Vw).sum(2)
    toward = cam.pos[None, None] - wpos
    flip = (nrm * toward).sum(-1) < 0
    nrm = np.where(flip[..., None], -nrm, nrm)
    part = np.array([qq['part'] for qq in mesh.quads])[qd] * cov

    # --- lighting
    def cam2world(v):
        v = norm(v); return norm(v[0] * cam.r + v[1] * cam.u - v[2] * cam.f)
    kd = norm(L['key_dir']) if L.get('key_dir') is not None else cam2world(L['key_cam'])
    kc = srgb_to_lin(hex_rgb(L['key_color'])) * L['key']
    sc = srgb_to_lin(hex_rgb(L['sky_color'])) * L['sky']; gc = srgb_to_lin(hex_rgb(L['ground_color'])) * L['ground']
    kcam = norm(L['key_cam']) if L.get('key_dir') is None else np.array([kd @ cam.r, kd @ cam.u, -(kd @ cam.f)])
    rim_dir = cam2world(L['rim_cam'] if L.get('rim_cam') is not None else (np.sign(kcam[0] or 1) * 0.85, 0.4, -0.55))
    rc = srgb_to_lin(hex_rgb(L['rim_color'])) * L['rim']
    fill_dir = cam2world(L['fill_cam'] if L.get('fill_cam') is not None else (-np.sign(kcam[0] or 1) * 0.55, 0.35, 0.9))
    fcol = srgb_to_lin(hex_rgb(L['fill_color'])) * L['fill']
    ndf = np.clip((nrm * fill_dir).sum(-1), 0, 1)

    ndl = np.clip((nrm * kd).sum(-1), 0, 1)
    sh = shadow_map(V, cull_all=np.concatenate([np.ones(len(mesh.quads), bool)] * 2), kd=kd, wpos=wpos, cov=cov, nrm=nrm,
                    alpha_lookup=alpha_lookup)
    shadow = 1 - L['shadow'] * (1 - sh)
    hemi = 0.5 + 0.5 * nrm[..., 1]
    amb = sc[None, None] * hemi[..., None] + gc[None, None] * (1 - hemi[..., None])
    ao = ssao(zbuf, cov, radius=max(2, int(Hs * 0.012)), strength=L['ao'])
    rim_face = np.clip((nrm * rim_dir).sum(-1), 0, 1) ** 2
    vdir_ = norm_rows(toward)
    hvec = norm_rows(vdir_ + kd[None, None])
    spec = np.clip((nrm * hvec).sum(-1), 0, 1) ** 40 * L['spec'] * shadow

    col = albedo * (kc[None, None] * (ndl * shadow)[..., None] + (amb + fcol[None, None] * ndf[..., None]) * ao[..., None]) \
        + albedo * rc[None, None] * (rim_face * 0.35 * shadow)[..., None] \
        + kc[None, None] * spec[..., None] * 0.25

    # screen-space silhouette rim (the bright edge you see on pro renders)
    ldir2 = cam_dir_2d(cam, rim_dir)
    rim_mask = screen_rim(cov, zbuf, ldir2, width=max(1.0, L['rim_width'] * ss), depth_gap=2.0)
    col = col + (rim_mask[..., None] * rc[None, None] * 0.55) * (0.35 + albedo * 1.2)

    col = filmic(col * L['exposure'])
    srgb = lin_to_srgb(col)

    # --- downsample (premultiplied box filter)
    alpha = cov.astype(np.float32)
    pm = srgb * alpha[..., None]
    def down(a):
        h, w = a.shape[:2]
        return a.reshape(h // ss, ss, w // ss, ss, *a.shape[2:]).mean((1, 3))
    Hc, Wc = Hs // ss * ss, Ws // ss * ss
    pm, alpha_d = down(pm[:Hc, :Wc]), down(alpha[:Hc, :Wc])
    rgb = np.where(alpha_d[..., None] > 1e-4, pm / np.maximum(alpha_d[..., None], 1e-4), 0)
    out = np.dstack([rgb, alpha_d])
    depth_d = down(np.where(cov, zbuf, 0)[:Hc, :Wc]) / np.maximum(alpha_d, 1e-4)
    nrm_d = down((nrm * alpha[..., None])[:Hc, :Wc])
    rim_d = down(rim_mask[:Hc, :Wc])
    part_d = part[:Hc:ss, :Wc:ss]
    return dict(rgba=(np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), alpha=alpha_d, depth=depth_d,
                normal=nrm_d, part=part_d, rim=rim_d, light2d=ldir2, slim=slim)

def norm_rows(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)

def cam_dir_2d(cam, d):
    x = float(np.dot(d, cam.r)); y = float(np.dot(d, cam.u))
    v = np.array([x, -y]); n = np.linalg.norm(v)
    return v / n if n > 1e-6 else np.array([1.0, 0])

def shift(a, dx, dy, fill=0):
    """integer shift of 2D array by (dx,dy)."""
    out = np.full_like(a, fill)
    H, W = a.shape[:2]
    xs0, xs1 = max(0, -dx), min(W, W - dx); ys0, ys1 = max(0, -dy), min(H, H - dy)
    out[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx] = a[ys0:ys1, xs0:xs1]
    return out

def screen_rim(cov, zbuf, ldir, width=3.0, depth_gap=2.0):
    """Pixels whose neighbour in the light direction is background or much farther away."""
    acc = np.zeros(cov.shape, np.float32)
    steps = max(1, int(round(width)))
    for k in range(1, steps + 1):
        dx, dy = int(round(ldir[0] * k)), int(round(ldir[1] * k))
        nb_cov = shift(cov, -dx, -dy, False)
        nb_z = shift(np.where(cov, zbuf, 1e6), -dx, -dy, 1e6)
        edge = cov & (~nb_cov | (nb_z - zbuf > depth_gap))
        acc = np.maximum(acc, edge.astype(np.float32) * (1 - (k - 1) / steps * 0.6))
    return acc

def ssao(zbuf, cov, radius=8, strength=0.5, n=12):
    z = np.where(cov, zbuf, 1e6).astype(np.float32)
    occ = np.zeros(z.shape, np.float32)
    rng = np.random.default_rng(3)
    for i in range(n):
        ang = 2 * math.pi * i / n + rng.uniform(0, 0.4)
        r = radius * (0.35 + 0.65 * (i % 3) / 2)
        dx, dy = int(round(math.cos(ang) * r)), int(round(math.sin(ang) * r))
        nz = shift(z, dx, dy, 1e6)
        dz = z - nz
        occ += ((dz > 0.15) & (dz < 6)).astype(np.float32) * np.clip(1 - dz / 6, 0, 1)
    occ /= n
    return np.clip(1 - strength * occ * 1.6, 0.25, 1).astype(np.float32)

def shadow_map(V, cull_all, kd, wpos, cov, nrm, alpha_lookup, res=1400):
    """Orthographic shadow map from the key light with 5x5 PCF."""
    f = -kd; r = norm(np.cross(f, [0, 1, 0]) if abs(f[1]) < 0.99 else np.cross(f, [1, 0, 0])); u = np.cross(r, f)
    P = V.reshape(-1, 3)
    lx, ly, lz = P @ r, P @ u, P @ f
    pad = 2.0
    x0, x1, y0, y1 = lx.min() - pad, lx.max() + pad, ly.min() - pad, ly.max() + pad
    sc = res / max(x1 - x0, y1 - y0)
    zoff = lz.min() - 10
    Sx = (lx - x0) * sc; Sy = (y1 - ly) * sc; Sz = lz - zoff
    S = np.stack([Sx, Sy, Sz], -1).reshape(V.shape)
    depth = ortho_depth(S, res, alpha_lookup)  # orthographic => linear depth interpolation
    W = wpos
    qx = ((W @ r) - x0) * sc; qy = (y1 - (W @ u)) * sc; qz = (W @ f) - zoff
    bias = 0.35 + 0.6 * (1 - np.clip((nrm * kd).sum(-1), 0, 1))
    lit = np.zeros(qx.shape, np.float32)
    k = 0
    for oy in (-2, -1, 0, 1, 2):
        for ox in (-2, -1, 0, 1, 2):
            ix = np.clip((qx + ox * 0.9).astype(np.int32), 0, res - 1)
            iy = np.clip((qy + oy * 0.9).astype(np.int32), 0, res - 1)
            lit += (qz - bias <= depth[iy, ix]).astype(np.float32); k += 1
    lit /= k
    return np.where(cov, lit, 1.0)

def ortho_depth(S, res, alpha_lookup):
    zbuf = np.full((res, res), np.inf, np.float32)
    for t in range(len(S)):
        v = S[t]
        xmin = max(int(np.floor(v[:, 0].min())), 0); xmax = min(int(np.ceil(v[:, 0].max())), res - 1)
        ymin = max(int(np.floor(v[:, 1].min())), 0); ymax = min(int(np.ceil(v[:, 1].max())), res - 1)
        if xmin > xmax or ymin > ymax: continue
        (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) = v
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9: continue
        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32)[:, None] + 0.5
        w0 = ((x1 - xs) * (y2 - ys) - (x2 - xs) * (y1 - ys)) / area
        w1 = ((x2 - xs) * (y0 - ys) - (x0 - xs) * (y2 - ys)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any(): continue
        if alpha_lookup is not None:
            keep = alpha_lookup(t, w0, w1, w2)
            if keep is not None: inside &= keep
        z = w0 * z0 + w1 * z1 + w2 * z2
        sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        m = inside & (z < sub)
        sub[m] = z[m]
    return zbuf

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skin', required=True, help='skin PNG path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pose', default='idle', help=f'one of {list(POSES)} or JSON')
    ap.add_argument('--slim', choices=['auto', 'yes', 'no'], default='auto')
    ap.add_argument('--yaw', type=float, default=-22); ap.add_argument('--pitch', type=float, default=6)
    ap.add_argument('--fov', type=float, default=32); ap.add_argument('--height', type=int, default=1200)
    ap.add_argument('--ss', type=int, default=3)
    ap.add_argument('--item', default=None, help='e.g. sword:diamond, axe:netherite, pickaxe:gold')
    ap.add_argument('--light', default=None, help='JSON overrides of DEFAULT_LIGHT')
    ap.add_argument('--no-overlay', action='store_true')
    a = ap.parse_args()
    pose = json.loads(a.pose) if a.pose.strip().startswith('{') else a.pose
    item = None
    if a.item:
        k, _, p = a.item.partition(':'); item = dict(kind=k, palette=p or 'diamond')
    slim = None if a.slim == 'auto' else a.slim == 'yes'
    r = render(a.skin, pose, slim, item, not a.no_overlay, a.yaw, a.pitch, a.fov, a.height, a.ss,
               light=json.loads(a.light) if a.light else None)
    Image.fromarray(r['rgba']).save(a.out)
    print(f"saved {a.out} {r['rgba'].shape[1]}x{r['rgba'].shape[0]} slim={r['slim']}")

if __name__ == '__main__':
    main()

# ----------------------------------------------------------------------------
# procedural voxel props (original textures - no game assets are shipped)
# ----------------------------------------------------------------------------

def _noise_tex(base, var=0.08, seed=0, blotch=None):
    rng = np.random.default_rng(seed)
    t = np.ones((16, 16, 4), np.float32); c = hex_rgb(base)
    n = rng.normal(0, var, (16, 16, 1)).astype(np.float32)
    t[..., :3] = np.clip(c + n * (0.6 + c), 0, 1)
    if blotch:
        bc = hex_rgb(blotch)
        for _ in range(6):
            x, y = rng.integers(0, 15, 2); t[y:y + rng.integers(1, 3), x:x + rng.integers(1, 4), :3] = bc
    return t

def block_textures(seed=0):
    T = {}
    T['grass_top'] = _noise_tex('#5fae3c', 0.10, seed + 1)
    side = _noise_tex('#8a5f3a', 0.08, seed + 2)
    g = _noise_tex('#5fae3c', 0.10, seed + 3); rng = np.random.default_rng(seed + 4)
    for x in range(16):
        side[:3 + rng.integers(0, 3), x] = g[:1, x]
    T['grass_side'] = side
    T['dirt'] = _noise_tex('#8a5f3a', 0.08, seed + 5)
    T['stone'] = _noise_tex('#8c8c8c', 0.07, seed + 6, blotch='#6e6e6e')
    T['sand'] = _noise_tex('#e3d6a4', 0.05, seed + 7, blotch='#d2c38e')
    p = _noise_tex('#b08850', 0.05, seed + 8)
    p[[3, 7, 11, 15], :, :3] *= 0.62
    for r, x in ((0, 5), (4, 12), (8, 2), (12, 9)): p[r:r + 4, x, :3] *= 0.7
    T['planks'] = p
    T['obsidian'] = _noise_tex('#1c1229', 0.06, seed + 9, blotch='#3b2a5a')
    T['endstone'] = _noise_tex('#e8e6b0', 0.05, seed + 10, blotch='#d6d39a')
    for name, col in (('wool_red', '#c8312f'), ('wool_blue', '#3352c4'), ('wool_white', '#ececec'),
                      ('wool_lime', '#79cc2c'), ('wool_yellow', '#f0c21e'), ('wool_purple', '#8a37b8')):
        w = _noise_tex(col, 0.04, seed + hash(name) % 97)
        w[::4, :, :3] *= 0.93; T[name] = w
    T['bed_top'] = _noise_tex('#c8312f', 0.03, seed + 11); T['bed_top'][10:, :, :3] = hex_rgb('#f2f2f2')
    return T

def build_island(seed=3, radius=7, depth=6, theme='grass', bed='red', defense='wool', extras=True):
    """A bedwars-style floating island: top surface, tapered rocky underside, bed + defense pyramid."""
    rng = np.random.default_rng(seed)
    T = block_textures(seed)
    blocks = {}
    top = {'grass': ('grass_top', 'grass_side'), 'sand': ('sand', 'sand'), 'end': ('endstone', 'endstone')}[theme]
    for x in range(-radius - 1, radius + 2):
        for z in range(-radius - 1, radius + 2):
            r = max(0.0, math.hypot(x, z) + rng.normal(0, 0.6))
            if r > radius: continue
            blocks[(x, 0, z)] = top
            dmax = int(depth * (1 - (r / radius) ** 1.4) + rng.integers(0, 2))
            for y in range(1, dmax + 1):
                mat = 'dirt' if y <= 2 and theme == 'grass' else ('stone' if rng.random() < 0.8 else 'dirt')
                if theme != 'grass' and y <= 2: mat = top[0]
                blocks[(x, -y, z)] = (mat, mat)
    if bed:
        bx, bz = 0, 0
        defense_mat = {'wool': f'wool_{"white" if bed != "white" else "red"}', 'obsidian': 'obsidian',
                       'planks': 'planks', 'endstone': 'endstone'}[defense]
        # stepped defense pyramid around the bed: each ring a different material (reads as layers)
        rings = [ 'obsidian', 'endstone' if defense != 'endstone' else 'planks', 'planks', defense_mat]
        for x in range(-3, 4):
            for z in range(-3, 5):
                dz = -z if z < 0 else (z - 1 if z > 1 else 0)
                ring = max(abs(x), dz)
                if ring > 3: continue
                for y in range(1, 5 - ring):
                    blocks[(x, y, z)] = (rings[ring], rings[ring])
        if extras:
            for _ in range(3):
                x, z = rng.integers(-radius + 2, radius - 1, 2)
                if abs(x) > 4 or abs(z) > 4:
                    blocks[(int(x), 1, int(z))] = ('stone', 'stone')
    mesh = Mesh(); tid = {k: mesh.add_tex(v) for k, v in T.items()}
    S = 16.0
    dirs = {'front': (0, 0, 1), 'back': (0, 0, -1), 'right': (-1, 0, 0), 'left': (1, 0, 0), 'top': (0, 1, 0), 'bottom': (0, -1, 0)}
    for (x, y, z), (mtop, mside) in blocks.items():
        x0, y0, z0 = x * S - S / 2, y * S - S, z * S - S / 2
        for fname, tl, tr, bl, br, _ in box_faces(x0, y0, z0, x0 + S, y0 + S, z0 + S, 0, 0, 16, 16, 16):
            dx, dy, dz = dirs[fname]
            if (x + dx, y + dy, z + dz) in blocks: continue
            name = mtop if fname == 'top' else ('dirt' if fname == 'bottom' and mside == 'grass_side' else mside)
            mesh.quads.append(dict(p=np.stack([tl, tr, bl, br]), tex=tid[name], rect=(0, 0, 16, 16), part=9,
                                   alpha_test=False, double=False))
    return mesh
