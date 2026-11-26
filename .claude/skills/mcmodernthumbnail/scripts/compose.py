#!/usr/bin/env python3
"""
compose.py - build a modern "cinematic render" Minecraft thumbnail from a JSON spec.

  python3 compose.py spec.json            # writes spec["out"] (.png 1920x1080 + _720.jpg)

Pipeline (order matters - it mirrors how pro designers stack layers in Photoshop):
  1. background  : cover-fit, depth-of-field (disc bokeh in linear light), grade, sun glow
  2. lighting    : sun position drives the 3D light of every character (key/rim side match)
  3. characters  : pixel-exact renders from mcrender.py, colour matched to the scene,
                   contact + cast shadow, light wrap, rim bloom, optional glow/outline
  4. atmosphere  : volumetric god rays (radial blur w/ occlusion) + stylised beams,
                   haze / inverse vignette, particles (2 depth layers), bloom
  5. grade       : lifted blacks, soft S-curve, split-tone, saturation, film grain
  6. typography  : badge (LIVE style), titles (3D extrude + stroke), watermark
  7. finish      : thin inner white frame, export 1920x1080 PNG + 1280x720 JPG (<2 MB)

All coordinates in the spec are fractions of the canvas (0..1), so specs are
resolution independent.  See references/spec.md for every field.
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mcrender as mr  # noqa: E402

FONT_DIR = os.path.join(HERE, '..', 'assets', 'fonts')
FONTS = {
    'lilita': 'LilitaOne.ttf', 'luckiest': 'LuckiestGuy.ttf', 'titan': 'TitanOne.ttf', 'bungee': 'Bungee.ttf',
    'pixel': 'PressStart2P.ttf', 'silkscreen': 'Silkscreen-Bold.ttf', 'kaushan': 'KaushanScript.ttf',
    'pacifico': 'Pacifico.ttf',
}

# ----------------------------------------------------------------------------
# colour utils (float32 sRGB 0..1 unless noted)
# ----------------------------------------------------------------------------
lin = mr.srgb_to_lin
srgb = mr.lin_to_srgb
hexc = mr.hex_rgb

def lum(a):
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722

def screen(a, b):
    return 1 - (1 - a) * (1 - b)

def blur(a, sigma):
    if sigma <= 0.3: return a
    k = int(sigma * 3) * 2 + 1
    return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REFLECT).reshape(a.shape)

def disc_blur(a, radius):
    """Lens-like defocus: disc kernel (bokeh) instead of gaussian."""
    if radius < 1: return a
    r = int(math.ceil(radius)); y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.clip(radius + 0.5 - np.sqrt(x * x + y * y), 0, 1).astype(np.float32); k /= k.sum()
    return cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REFLECT).reshape(a.shape)

def to_f(im):
    return np.asarray(im, np.float32) / 255.0

def to_u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

def over(dst, src_rgb, src_a):
    a = src_a[..., None]
    return dst * (1 - a) + src_rgb * a

def cover_fit(im: Image.Image, W, H, focus=(0.5, 0.5)):
    w, h = im.size; s = max(W / w, H / h)
    im = im.resize((max(W, round(w * s)), max(H, round(h * s))), Image.LANCZOS)
    w, h = im.size
    x = int((w - W) * focus[0]); y = int((h - H) * focus[1])
    return im.crop((x, y, x + W, y + H))

def fbm(H, W, octaves=5, seed=0, base=4):
    rng = np.random.default_rng(seed); out = np.zeros((H, W), np.float32); amp = 1; tot = 0
    for o in range(octaves):
        n = base * 2 ** o
        g = rng.random((n + 1, int(n * W / H) + 1)).astype(np.float32)
        out += cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC) * amp; tot += amp; amp *= 0.5
    return out / tot

# ----------------------------------------------------------------------------
# background
# ----------------------------------------------------------------------------

def make_sky(W, H, top='#2f8fe8', horizon='#bfe8ff', seed=1, clouds=0.55):
    t = np.linspace(0, 1, H, dtype=np.float32)[:, None, None] ** 0.8
    a = hexc(top)[None, None]; b = hexc(horizon)[None, None]
    sky = a * (1 - t) + b * t
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    if clouds > 0:
        n = fbm(H, W, 6, seed, 3)
        c = np.clip((n - 0.48) * 3.2, 0, 1) ** 1.4 * clouds
        c *= np.linspace(0.4, 1, H, dtype=np.float32)[:, None]
        sky = screen(sky, c[..., None] * np.array([1, 1, 1], np.float32) * 0.9)
    return sky

def load_background(spec, W, H):
    bg = spec.get('background', {})
    if bg.get('image'):
        im = Image.open(bg['image']).convert('RGB')
        im = cover_fit(im, W, H, tuple(bg.get('focus', (0.5, 0.5))))
        a = to_f(im)
    else:
        a = make_sky(W, H, bg.get('top', '#2f8fe8'), bg.get('horizon', '#bfe8ff'), bg.get('seed', 1), bg.get('clouds', 0.55))
    if bg.get('flip'): a = a[:, ::-1].copy()
    # depth of field: blur everything, then optionally keep a focus band sharp
    r = float(bg.get('blur', 6)) * H / 1080
    if r > 0:
        L = lin(a)
        hi = np.clip(lum(L) - 0.75, 0, None)[..., None] * 3.0  # boost highlights -> real bokeh balls
        Lb = disc_blur(L * (1 + hi), r) / (1 + disc_blur(hi, r))
        b = srgb(Lb)
        mode = bg.get('dof', 'uniform')
        if mode == 'uniform':
            a = b
        else:
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            if mode == 'bottom':      # foreground (bottom) blurred, far part sharp-ish  (like ref #1 sand)
                m = np.clip((yy / H - bg.get('focus_y', 0.65)) / 0.25, 0, 1)
            elif mode == 'radial':    # sharp around focus point
                fx, fy = bg.get('focus_point', (0.5, 0.5))
                d = np.sqrt(((xx / W - fx) * 1.6) ** 2 + (yy / H - fy) ** 2)
                m = np.clip((d - bg.get('focus_radius', 0.25)) / 0.3, 0, 1)
            else:
                m = np.ones((H, W), np.float32)
            a = over(a, b, m)
    g = bg.get('grade', {})
    a = grade(a, exposure=g.get('exposure', 0.05), sat=g.get('saturation', 0.95), contrast=g.get('contrast', 0.9),
              lift=g.get('lift', 0.04), temp=g.get('temperature', 0.0))
    return a

# ----------------------------------------------------------------------------
# grading
# ----------------------------------------------------------------------------

def grade(a, exposure=0.0, sat=1.0, contrast=1.0, lift=0.0, temp=0.0, tint_shadows=None, tint_highlights=None,
          split=0.0, gamma=1.0):
    a = a * (2 ** exposure)
    if temp:
        a = a * np.array([1 + 0.08 * temp, 1 + 0.01 * temp, 1 - 0.08 * temp], np.float32)
    l = lum(a)[..., None]
    a = l + (a - l) * sat
    # soft S curve around 0.5
    a = 0.5 + (a - 0.5) * contrast
    a = np.clip(a, 0, 1)
    if gamma != 1: a = a ** (1 / gamma)
    if split:
        l = lum(a)[..., None]
        sh = hexc(tint_shadows or '#1e3a6e'); hl = hexc(tint_highlights or '#ffd9a8')
        a = a + split * ((1 - l) ** 2 * (sh - 0.5) * 0.5 + l ** 2 * (hl - 0.5) * 0.5)
    a = lift + a * (1 - lift)
    return np.clip(a, 0, 1)

PRESETS = {
    # measured on the 3 reference thumbnails: mean lum ~0.6, sat 0.23-0.44, p2 0.02-0.06, edges brighter than centre
    'airy':   dict(exposure=0.08, sat=0.92, contrast=1.04, lift=0.035, split=0.25, tint_shadows='#2a4f8a', tint_highlights='#fff0d8', haze=0.32, bloom=0.38, grain=0.035),
    'golden': dict(exposure=0.02, sat=1.02, contrast=1.06, lift=0.03, temp=0.6, split=0.35, tint_shadows='#3a2a5e', tint_highlights='#ffc98a', haze=0.28, bloom=0.45, grain=0.035),
    'punchy': dict(exposure=0.0, sat=1.18, contrast=1.12, lift=0.015, split=0.15, haze=0.12, bloom=0.3, grain=0.025),
    'night':  dict(exposure=-0.1, sat=1.1, contrast=1.1, lift=0.03, temp=-0.6, split=0.4, tint_shadows='#1a1050', tint_highlights='#b8a0ff', haze=0.15, bloom=0.6, grain=0.04),
    'frost':  dict(exposure=0.1, sat=0.8, contrast=1.0, lift=0.05, temp=-0.4, split=0.3, tint_shadows='#2a5a8a', tint_highlights='#eaf6ff', haze=0.4, bloom=0.4, grain=0.035),
}

# ----------------------------------------------------------------------------
# characters
# ----------------------------------------------------------------------------

def sample_env(bg, W, H):
    top = bg[: H // 3].reshape(-1, 3).mean(0)
    bottom = bg[2 * H // 3:].reshape(-1, 3).mean(0)
    return top, bottom

def char_light(ch, sun, bg_env):
    """Derive camera-relative 3D lights from where the sun sits in the 2D frame."""
    sx, sy = sun.get('pos', (0.85, 0.05))
    cx = ch.get('x', 0.5); cy = ch.get('y', 1.0) - ch.get('height', 0.9) * 0.6
    dx = sx - cx; dy = cy - sy
    behind = sun.get('behind', True)
    kz = -0.45 if behind else 0.55
    key = (dx * 1.6, max(0.25, dy * 1.4), kz)
    top, bottom = bg_env
    L = dict(key_cam=key, key_color=sun.get('color', '#fff1dc'), key=sun.get('key', 3.0),
             sky_color='#%02x%02x%02x' % tuple(int(c * 255) for c in np.clip(top * 1.1, 0, 1)),
             sky=sun.get('sky', 0.8), fill=ch.get('fill', 1.0 if sun.get('behind', True) else 0.5), ground_color='#%02x%02x%02x' % tuple(int(c * 255) for c in bottom),
             ground=sun.get('bounce', 0.3), rim=ch.get('rim', 1.8), rim_color=sun.get('rim_color', sun.get('color', '#fff6e8')),
             rim_width=ch.get('rim_width', 1.1), ao=ch.get('ao', 0.6), shadow=ch.get('self_shadow', 0.8),
             exposure=ch.get('exposure', 1.0), spec=ch.get('spec', 0.2))
    L.update(ch.get('light', {}))
    return L

def image_character(ch, H, sun):
    """User-supplied cut-out (see cutout.py). Rim is computed in screen space from the alpha."""
    im = Image.open(ch['image']).convert('RGBA')
    bb = im.getbbox(); im = im.crop(bb)
    hpx = int(ch.get('height', 0.9) * H); im = im.resize((max(1, int(im.width * hpx / im.height)), hpx), Image.LANCZOS)
    a = to_f(im)
    cov = a[..., 3] > 0.5
    sx, sy = sun.get('pos', (0.85, 0.05)); cx = ch.get('x', 0.5)
    ldir = np.array([np.sign(sx - cx) or 1.0, -0.6]); ldir /= np.linalg.norm(ldir)
    rim = mr.screen_rim(cov, np.zeros(cov.shape, np.float32), ldir, width=max(1.5, H / 540))
    return dict(rgba=(a * 255).astype(np.uint8), rim=rim.astype(np.float32), part=np.zeros(cov.shape, np.int32))

def place_character(canvas, ch, sun, env, W, H, scratch):
    if ch.get('image'):
        r = image_character(ch, H, sun); item = None
        return _place_rendered(canvas, ch, sun, env, W, H, r, item)
    skin = ch['skin']
    if not os.path.exists(skin):  # treat as username
        import fetch_skin
        info = fetch_skin.fetch(skin, os.path.join(scratch, f'{skin}_skin.png'))
        skin = info['path']
        if ch.get('slim') is None: ch['slim'] = info['slim']
    item = ch.get('item')
    if isinstance(item, str):
        k, _, p = item.partition(':'); item = dict(kind=k, palette=p or 'diamond')
    hpx = int(ch.get('height', 0.9) * H)
    margin = 0.04
    L = char_light(ch, sun, env)
    r = mr.render(skin, ch.get('pose', 'idle'), ch.get('slim'), item, not ch.get('no_overlay', False),
                  ch.get('yaw', -25), ch.get('pitch', 6), ch.get('fov', 34), height=int(hpx / (1 - 2 * margin)),
                  ss=ch.get('ss', 3), margin=margin, light=L, cam_roll=ch.get('roll', 0))
    return _place_rendered(canvas, ch, sun, env, W, H, r, item)

def _place_rendered(canvas, ch, sun, env, W, H, r, item):
    rgba = to_f(r['rgba']); a = rgba[..., 3]; rgb = rgba[..., :3]
    if ch.get('flip'):
        rgba = rgba[:, ::-1]; a = a[:, ::-1]; rgb = rgb[:, ::-1]; r['rim'] = r['rim'][:, ::-1]; r['part'] = r['part'][:, ::-1]
    h, w = a.shape
    # anchor: x = centre, y = feet line (bottom of the alpha bbox)
    ys = np.where(a.max(1) > 0.01)[0]; feet = ys.max() if len(ys) else h - 1
    xs = np.where(a.max(0) > 0.01)[0]; cxp = (xs.min() + xs.max()) / 2
    X = int(ch.get('x', 0.5) * W - cxp); Y = int(ch.get('y', 1.0) * H - feet)

    def paste(layer2d):  # place a 2D array into a canvas-size array
        out = np.zeros((H, W) + layer2d.shape[2:], np.float32)
        x0, y0 = max(X, 0), max(Y, 0); x1, y1 = min(X + w, W), min(Y + h, H)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = layer2d[y0 - Y:y1 - Y, x0 - X:x1 - X]
        return out

    A = paste(a); C = paste(rgb); RIM = paste(r['rim']); PART = paste(r['part'].astype(np.float32))

    # 1) colour match: pull the render slightly towards the scene's ambient tint
    env_top, env_bottom = env
    amb = (env_top * 0.6 + env_bottom * 0.4)
    m = ch.get('color_match', 0.12)
    C = C * (1 - m) + C * (amb / max(amb.mean(), 1e-3)) * m

    # 2) shadows on the ground
    if ch.get('ground_shadow', True):
        canvas = ground_shadow(canvas, A, sun, ch, W, H, Y + feet)

    # 3) back glow (coloured aura) - popular in pvp/tier thumbnails
    if ch.get('glow'):
        g = ch['glow']; col = hexc(g.get('color', '#7ad7ff')); rad = g.get('radius', 0.03) * H
        aura = blur(A, rad) * g.get('strength', 1.0)
        canvas = screen(canvas, np.clip(aura, 0, 1)[..., None] * col)
    if ch.get('outline'):
        o = ch['outline']; k = max(1, int(o.get('width', 0.006) * H))
        dil = cv2.dilate((A > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        dil = blur(dil.astype(np.float32), 0.8)
        canvas = over(canvas, np.broadcast_to(hexc(o.get('color', '#ffffff')), canvas.shape), dil)

    # 4) light wrap: blurred background bleeding over the silhouette edge (sells the integration)
    lw = ch.get('light_wrap', 0.35)
    if lw > 0:
        inner = A * (1 - blur(A, 0.012 * H))
        bgb = blur(canvas, 0.02 * H)
        C = screen(C, bgb * (inner * lw * 1.8)[..., None])

    # character "pop": local-contrast boost (unsharp on luminance) + a touch of saturation
    pop = ch.get('pop', 0.5)
    if pop > 0:
        l = lum(C)[..., None]; C = np.clip(l + (C - l) * (1 + 0.25 * pop), 0, 1)
        C = np.clip(C + (C - blur(C, 0.006 * H)) * pop * 1.2, 0, 1)

    canvas = over(canvas, C, A)

    # 5) rim bloom: soft glow hugging the lit edge
    rb = ch.get('rim_bloom', 0.8)
    if rb > 0:
        rc = hexc(sun.get('rim_color', sun.get('color', '#fff6e8')))
        glow = blur(RIM, 0.004 * H) * 0.8 + blur(RIM, 0.012 * H) * 0.9
        canvas = screen(canvas, np.clip(glow * rb, 0, 1)[..., None] * rc)

    # 6) enchant glint on the item
    if item and item.get('enchant'):
        im = (np.abs(PART - mr.PART_IDS['item']) < 0.5).astype(np.float32) * A
        col = hexc(item.get('glint', '#b36bff'))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        stripes = 0.5 + 0.5 * np.sin((xx + yy * 0.6) / (0.012 * H))
        canvas = screen(canvas, (im * (0.35 + 0.4 * stripes))[..., None] * col)
        canvas = screen(canvas, blur(im, 0.01 * H)[..., None] * col * 0.9)
    return canvas, A

def place_prop(canvas, pr, sun, env, W, H):
    """Procedural voxel prop (bedwars island). Anchored at its centre (x, y)."""
    mesh = mr.build_island(pr.get('seed', 3), pr.get('radius', 7), pr.get('depth', 6), pr.get('theme', 'grass'),
                           pr.get('bed', 'red'), pr.get('defense', 'wool'))
    L = char_light(dict(x=pr.get('x', 0.7), y=pr.get('y', 0.5) + pr.get('height', 0.6) * 0.6, height=pr.get('height', 0.6),
                        rim=pr.get('rim', 1.4)), sun, env)
    L['rim_width'] = pr.get('rim_width', 1.2)
    r = mr.render(None, mesh_override=mesh, yaw=pr.get('yaw', -30), pitch=pr.get('pitch', 26), fov=pr.get('fov', 35),
                  height=int(pr.get('height', 0.6) * H / 0.92), ss=pr.get('ss', 2), margin=0.04, light=L)
    rgba = to_f(r['rgba']); a = rgba[..., 3]; rgb = rgba[..., :3]
    if pr.get('flip'): a = a[:, ::-1]; rgb = rgb[:, ::-1]
    if pr.get('blur', 0) > 0:  # prop sits at another depth plane -> its own defocus
        rad = pr['blur'] * H / 1080
        pm = disc_blur(rgb * a[..., None], rad); a = disc_blur(a, rad); rgb = pm / np.maximum(a[..., None], 1e-4)
    h, w = a.shape
    X = int(pr.get('x', 0.7) * W - w / 2); Y = int(pr.get('y', 0.5) * H - h / 2)
    A = np.zeros((H, W), np.float32); C = np.zeros((H, W, 3), np.float32)
    x0, y0, x1, y1 = max(X, 0), max(Y, 0), min(X + w, W), min(Y + h, H)
    if x1 > x0 and y1 > y0:
        A[y0:y1, x0:x1] = a[y0 - Y:y1 - Y, x0 - X:x1 - X]; C[y0:y1, x0:x1] = rgb[y0 - Y:y1 - Y, x0 - X:x1 - X]
    # atmospheric perspective: distant props pick up haze/sky colour
    fog = pr.get('atmosphere', 0.12)
    C = C * (1 - fog) + np.array(env[0]) * fog
    canvas = over(canvas, C, A)
    return canvas, A

def ground_shadow(canvas, A, sun, ch, W, H, feet_y):
    """Contact shadow (tight, dark) + cast shadow (long, soft) pointing away from the sun."""
    ys, xs = np.where(A > 0.5)
    if len(xs) == 0: return canvas
    cx = xs.mean(); feet_y = min(feet_y, H - 1)
    width = (xs.max() - xs.min())
    sh = np.zeros((H, W), np.float32)
    # contact ellipse
    cv2.ellipse(sh, (int(cx), int(feet_y)), (int(width * 0.42), max(2, int(width * 0.07))), 0, 0, 360, 1.0, -1)
    sh = blur(sh, width * 0.05) * ch.get('contact_shadow', 0.55)
    # cast: squash the silhouette onto the ground plane and skew away from the sun
    sx, sy = sun.get('pos', (0.85, 0.05))
    dirx = -np.sign(sx * W - cx) if abs(sx * W - cx) > 1 else 1
    # sun behind the subject -> shadow falls towards the viewer (down on screen), else away (up)
    s = ch.get('cast_squash', 0.22) * (1 if sun.get('behind', True) else -0.6)
    k = dirx * ch.get('cast_skew', 1.1) * abs(s)
    # forward map: x' = x + k*(feet-y), y' = feet + s*(feet-y)
    M = np.float32([[1, -k, k * feet_y], [0, -s, feet_y * (1 + s)]])
    cast = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR)
    cast = blur(cast, H * 0.008) * ch.get('cast_shadow', 0.35)
    sh = np.clip(sh + cast, 0, 0.85)
    tint = hexc(ch.get('shadow_tint', '#1b2440'))
    return canvas * (1 - sh[..., None]) + canvas * tint * sh[..., None] * 0.6

# ----------------------------------------------------------------------------
# atmosphere
# ----------------------------------------------------------------------------

def god_rays(canvas, occl, sun, W, H):
    sx, sy = sun.get('pos', (0.85, 0.05)); P = (sx * W, sy * H)
    col = hexc(sun.get('color', '#fff1dc'))
    strength = sun.get('rays', 0.55)
    if strength <= 0: return canvas
    # (a) volumetric: bright sky minus occluders, radially blurred towards the sun
    src = np.clip(lum(canvas) - 0.55, 0, 1) * (1 - occl)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - P[0]) ** 2 + (yy - P[1]) ** 2) / H
    src = src * np.clip(1.2 - d, 0, 1)
    acc = np.zeros_like(src); n = 28
    for i in range(n):
        s = 1 - i * 0.022
        M = np.float32([[s, 0, P[0] * (1 - s)], [0, s, P[1] * (1 - s)]])
        acc += cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR) * (0.965 ** i)
    acc = blur(acc / n * 3.2, 2)
    # (b) stylised beams: angular noise fanning out from the sun (the diagonal light shafts in the refs)
    ang = np.arctan2(yy - P[1], xx - P[0])
    rng = np.random.default_rng(sun.get('seed', 7))
    beams = np.zeros((H, W), np.float32)
    for k in range(sun.get('beams', 9)):
        a0 = rng.uniform(-math.pi, math.pi); wdt = rng.uniform(0.015, 0.06); amp = rng.uniform(0.35, 1)
        da = np.angle(np.exp(1j * (ang - a0)))
        beams += amp * np.exp(-(da / wdt) ** 2)
    beams *= np.clip(1 - d / 1.4, 0, 1) ** 1.5
    beams = blur(beams, H * 0.004)
    light = (acc * 0.9 + beams * 0.32 * sun.get('beam_strength', 1.0)) * strength
    light = light * (1 - blur(occl, H * 0.004) * sun.get('rays_on_subject', 0.75))
    canvas = screen(canvas, np.clip(light, 0, 1)[..., None] * col)
    return canvas

def sun_glow(canvas, sun, W, H):
    """Soft sun disc + wide glow. Drawn BEFORE the characters so they silhouette against it."""
    sx, sy = sun.get('pos', (0.85, 0.05)); col = hexc(sun.get('color', '#fff1dc'))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt((xx - sx * W) ** 2 + (yy - sy * H) ** 2) / H
    glow = np.exp(-(d / sun.get('glow_size', 0.38)) ** 2) * sun.get('glow', 0.8) + np.exp(-(d / 0.07) ** 2) * 0.7
    return screen(canvas, np.clip(glow, 0, 1)[..., None] * col)

def haze(canvas, amount, W, H, color='#ffffff', fog_bottom=0.25, occl=None):
    if amount <= 0: return canvas
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    nx = (xx / W - 0.5) * 2; ny = (yy / H - 0.5) * 2
    edge = np.clip((np.maximum(np.abs(nx) ** 3, np.abs(ny) ** 3) - 0.35) / 0.65, 0, 1) ** 1.3
    fog = np.clip((yy / H - 0.78) / 0.22, 0, 1) ** 1.6 * fog_bottom
    m = np.clip(edge * amount + fog, 0, 0.9)
    if occl is not None: m = m * (1 - blur(occl, H * 0.01) * 0.65)
    return over(canvas, np.broadcast_to(hexc(color), canvas.shape), m)

def particles(canvas, cfg, W, H, layer='back', occl=None):
    if not cfg: return canvas
    kind = cfg.get('type', 'snow'); rng = np.random.default_rng(cfg.get('seed', 11) + (0 if layer == 'back' else 99))
    n = int(cfg.get('count', 140) * (1 if layer == 'back' else 0.18))
    col = hexc(cfg.get('color', '#ffffff' if kind in ('snow', 'dust') else '#ffd27a'))
    m = np.zeros((H, W), np.float32)
    for _ in range(n):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        if layer == 'back':
            r = rng.uniform(0.6, 2.4) * H / 1080; a = rng.uniform(0.35, 0.95)
        else:
            r = rng.uniform(4, 11) * H / 1080; a = rng.uniform(0.15, 0.45)
        if kind == 'embers':
            cv2.line(m, (int(x), int(y)), (int(x + r * 2.5), int(y - r * 4)), a, max(1, int(r * 0.8)), cv2.LINE_AA)
        else:
            cv2.circle(m, (int(x), int(y)), max(1, int(round(r))), a, -1, cv2.LINE_AA)
    m = blur(m, 0.6 if layer == 'back' else 3.5 * H / 1080)
    return screen(canvas, np.clip(m, 0, 1)[..., None] * col)

def bloom(canvas, amount, H):
    if amount <= 0: return canvas
    L = lin(canvas); b = np.clip(L - 0.6, 0, None)
    g = blur(b, 0.006 * H) * 0.5 + blur(b, 0.02 * H) * 0.8 + blur(b, 0.06 * H) * 1.0
    return srgb(np.clip(L + g * amount, 0, None) / (1 + g * amount * 0.25))

def film(canvas, grain, W, H, seed=3, texture=0.5):
    if grain <= 0: return canvas
    rng = np.random.default_rng(seed)
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    n = blur(n, 0.6) * grain
    # paper/canvas texture (ref #1): low-freq blotches + fine fibres
    tex = (fbm(H, W, 4, seed + 1, 24) - 0.5) * grain * texture * 1.2
    l = lum(canvas)[..., None]
    return np.clip(canvas + (n + tex)[..., None] * (0.4 + 0.6 * (1 - np.abs(l - 0.5) * 2)), 0, 1)

def frame(img: Image.Image, cfg):
    if cfg is False: return img
    cfg = cfg or {}
    W, H = img.size
    inset = int(cfg.get('inset', 0.016) * H); wdt = max(1, int(cfg.get('width', 0.0028) * H))
    rad = int(cfg.get('radius', 0.012) * H); op = int(255 * cfg.get('opacity', 0.75))
    ov = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    d.rounded_rectangle([inset, inset, W - inset, H - inset], rad, outline=(255, 255, 255, op), width=wdt)
    ov = Image.alpha_composite(ov.filter(ImageFilter.GaussianBlur(wdt * 1.5)), ov)
    out = img.convert('RGBA'); out.alpha_composite(ov); return out

# ----------------------------------------------------------------------------
# typography
# ----------------------------------------------------------------------------

def font(name, size):
    f = FONTS.get(name, name)
    p = f if os.path.isabs(f) else os.path.join(FONT_DIR, f)
    return ImageFont.truetype(p, size)

def text_mask(txt, fnt, stroke=0, spacing=0):
    bbox = fnt.getbbox(txt, stroke_width=stroke)
    w = bbox[2] - bbox[0] + 4 + spacing * len(txt); h = bbox[3] - bbox[1] + 4
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    if spacing:
        x = -bbox[0] + 2
        for ch in txt:
            d.text((x, -bbox[1] + 2), ch, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
            x += fnt.getlength(ch) + spacing
    else:
        d.text((-bbox[0] + 2, -bbox[1] + 2), txt, font=fnt, fill=255, stroke_width=stroke, stroke_fill=255)
    return im

def gradient_fill(size, colors, angle=90):
    w, h = size
    cols = [hexc(c) for c in colors]
    if len(cols) == 1: cols = cols * 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a = math.radians(angle); t = (xx * math.cos(a) + yy * math.sin(a))
    t = (t - t.min()) / max(t.max() - t.min(), 1)
    seg = t * (len(cols) - 1); i = np.clip(seg.astype(int), 0, len(cols) - 2); f = (seg - i)[..., None]
    c0 = np.array(cols)[i]; c1 = np.array(cols)[i + 1]
    return Image.fromarray(to_u8(c0 * (1 - f) + c1 * f))

def draw_title(img: Image.Image, t, W, H):
    size = int(t.get('size', 0.16) * H); fnt = font(t.get('font', 'lilita'), size)
    txt = t['text']; stroke = int(t.get('stroke', 0.075) * size)
    lines = txt.split('\n')
    layers = []
    for ln in lines:
        core = text_mask(ln, fnt, 0, t.get('spacing', 0))
        pad = stroke + int(size * 0.25)
        def padded(m):
            c = Image.new('L', (core.width + 2 * pad, core.height + 2 * pad), 0); c.paste(m, (pad, pad)); return c
        core = padded(core)
        outl = core.filter(ImageFilter.MaxFilter(stroke * 2 + 1)) if stroke > 0 else core
        layers.append((core, outl))
    Wt = max(c.width for c, _ in layers); lh = int(size * t.get('line_height', 1.0))
    canvas = Image.new('RGBA', (Wt, lh * (len(layers) - 1) + layers[-1][0].height), (0, 0, 0, 0))
    depth = int(t.get('extrude', 0.06) * size); ang = math.radians(t.get('extrude_angle', 90))
    ext_col = t.get('extrude_color', '#1a1a2e'); str_col = t.get('stroke_color', '#101018')
    for li, (core, outl) in enumerate(layers):
        x = (Wt - core.width) // 2 if t.get('align', 'center') == 'center' else 0; y = li * lh
        # drop shadow
        sh = Image.new('RGBA', core.size, (0, 0, 0, 0)); sh.putalpha(outl.point(lambda v: int(v * t.get('shadow', 0.55))))
        sh = sh.filter(ImageFilter.GaussianBlur(size * 0.06))
        canvas.alpha_composite(sh, (x + int(depth * math.cos(ang) * 1.5), y + int(depth * math.sin(ang) * 1.5) + int(size * 0.03)))
        # 3D extrusion (stack of outline copies)
        for k in range(depth, 0, -1):
            ex = Image.new('RGBA', core.size, hexc_to_rgba(ext_col)); ex.putalpha(outl)
            canvas.alpha_composite(ex, (x + int(round(k * math.cos(ang))), y + int(round(k * math.sin(ang)))))
        # stroke
        st = Image.new('RGBA', core.size, hexc_to_rgba(str_col)); st.putalpha(outl); canvas.alpha_composite(st, (x, y))
        # gradient fill + inner top highlight
        fill = gradient_fill(core.size, t.get('fill', ['#ffffff', '#ffffff']), t.get('fill_angle', 90)).convert('RGBA')
        fill.putalpha(core); canvas.alpha_composite(fill, (x, y))
        if t.get('gloss', 0.35) > 0:
            gl = Image.new('L', core.size, 0); gd = ImageDraw.Draw(gl)
            gd.rectangle([0, 0, core.width, int(core.height * 0.48)], fill=int(255 * t.get('gloss', 0.35)))
            gl = Image.fromarray(np.minimum(np.asarray(gl), np.asarray(core)))
            hl = Image.new('RGBA', core.size, (255, 255, 255, 0)); hl.putalpha(gl.filter(ImageFilter.GaussianBlur(1)))
            canvas.alpha_composite(hl, (x, y))
    if t.get('rotate'): canvas = canvas.rotate(t['rotate'], Image.BICUBIC, expand=True)
    tw = t.get('width');
    if tw: canvas = canvas.resize((int(tw * W), int(canvas.height * tw * W / canvas.width)), Image.LANCZOS)
    x = int(t.get('x', 0.5) * W - canvas.width / 2); y = int(t.get('y', 0.2) * H - canvas.height / 2)
    img.alpha_composite(canvas, (max(-canvas.width, x), max(-canvas.height, y)))

def hexc_to_rgba(h, a=255):
    c = hexc(h); return (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), a)

def draw_badge(img: Image.Image, b, W, H):
    """Rounded gradient panel + diagonal wave pattern + chunky 3D text (+ optional round icon)."""
    hgt = int(b.get('height', 0.25) * H); size = int(hgt * 0.62)
    fnt = font(b.get('font', 'bungee'), size)
    txt = b.get('text', 'LIVE')
    tm = text_mask(txt, fnt, 0, b.get('spacing', int(size * 0.04)))
    icon = b.get('icon'); icon_w = int(hgt * 0.8) if icon is not None else 0
    pad = int(hgt * 0.16)
    wdt = int(b.get('width', 0) * W) or (pad * 3 + icon_w + tm.width + (pad if icon_w else 0))
    rad = int(hgt * 0.14)
    panel = Image.new('RGBA', (wdt + 40, hgt + 40), (0, 0, 0, 0))
    # shadow
    sh = Image.new('L', panel.size, 0); ImageDraw.Draw(sh).rounded_rectangle([20, 26, 20 + wdt, 26 + hgt], rad, fill=150)
    panel.putalpha(sh.filter(ImageFilter.GaussianBlur(10)))
    # body gradient
    body = gradient_fill((wdt, hgt), b.get('colors', ['#18e0ff', '#0aa8d8', '#0b6f9e']), b.get('angle', 20)).convert('RGBA')
    # diagonal wave pattern (subtle darker bands)
    yy, xx = np.mgrid[0:hgt, 0:wdt].astype(np.float32)
    wave = 0.5 + 0.5 * np.sin((xx * 0.9 + yy * 1.4 + 18 * np.sin(yy / hgt * 3.1)) / (hgt * 0.11))
    arr = np.asarray(body, np.float32) / 255
    arr[..., :3] *= (1 - 0.12 * wave[..., None] * np.clip(xx / wdt + 0.2, 0, 1)[..., None])
    body = Image.fromarray(to_u8(arr))
    m = Image.new('L', (wdt, hgt), 0); ImageDraw.Draw(m).rounded_rectangle([0, 0, wdt - 1, hgt - 1], rad, fill=255)
    body.putalpha(m)
    panel.alpha_composite(body, (20, 20))
    # bevel: top light line + bottom dark line
    bev = Image.new('RGBA', (wdt, hgt), (0, 0, 0, 0)); bd = ImageDraw.Draw(bev)
    bd.rounded_rectangle([1, 1, wdt - 2, hgt - 2], rad, outline=(255, 255, 255, 90), width=max(2, hgt // 40))
    bd.rounded_rectangle([0, 0, wdt - 1, hgt - 1], rad, outline=hexc_to_rgba(b.get('border', '#06425e'), 255), width=max(2, hgt // 45))
    panel.alpha_composite(bev, (20, 20))
    x = 20 + pad
    if icon is not None:
        ic = Image.new('RGBA', (icon_w, icon_w), (0, 0, 0, 0)); idr = ImageDraw.Draw(ic)
        idr.ellipse([0, 0, icon_w - 1, icon_w - 1], fill=hexc_to_rgba('#101826'))
        if icon and os.path.exists(str(icon)):
            im = Image.open(icon).convert('RGBA').resize((int(icon_w * 0.84), int(icon_w * 0.84)), Image.LANCZOS)
            mm = Image.new('L', im.size, 0); ImageDraw.Draw(mm).ellipse([0, 0, im.width - 1, im.height - 1], fill=255)
            im.putalpha(Image.fromarray(np.minimum(np.asarray(im.split()[3]), np.asarray(mm))))
            ic.alpha_composite(im, ((icon_w - im.width) // 2, (icon_w - im.height) // 2))
        else:  # default: red recording dot
            r0 = int(icon_w * 0.22)
            idr.ellipse([r0, r0, icon_w - r0, icon_w - r0], fill=hexc_to_rgba(b.get('dot', '#ff3040')))
        panel.alpha_composite(ic, (x, 20 + (hgt - icon_w) // 2)); x += icon_w + pad
    # 3D text: grey extrusion down, white face
    depth = max(2, int(size * 0.09))
    ty = 20 + (hgt - tm.height) // 2 - depth // 2
    for k in range(depth, 0, -1):
        e = Image.new('RGBA', tm.size, hexc_to_rgba(b.get('text_extrude', '#b9c6d2'))); e.putalpha(tm)
        panel.alpha_composite(e, (x, ty + k))
    s2 = Image.new('RGBA', tm.size, (0, 0, 0, 0)); s2.putalpha(tm.point(lambda v: v // 3))
    panel.alpha_composite(s2.filter(ImageFilter.GaussianBlur(3)), (x + 2, ty + depth + 4))
    face = Image.new('RGBA', tm.size, hexc_to_rgba(b.get('text_color', '#ffffff'))); face.putalpha(tm)
    panel.alpha_composite(face, (x, ty))
    if b.get('rotate'): panel = panel.rotate(b['rotate'], Image.BICUBIC, expand=True)
    X = int(b.get('x', 0.3) * W - panel.width / 2); Y = int(b.get('y', 0.5) * H - panel.height / 2)
    img.alpha_composite(panel, (X, Y))

def draw_watermark(img, wm, W, H):
    size = int(wm.get('size', 0.075) * H); fnt = font(wm.get('font', 'kaushan'), size)
    tm = text_mask(wm['text'], fnt, 0)
    padd = int(size * 0.3)
    c = Image.new('L', (tm.width + 2 * padd, tm.height + 2 * padd), 0); c.paste(tm, (padd, padd))
    sh = Image.new('RGBA', c.size, (20, 30, 50, 0)); sh.putalpha(c.filter(ImageFilter.MaxFilter(3)).point(lambda v: int(v * 0.6)))
    sh = sh.filter(ImageFilter.GaussianBlur(size * 0.06))
    face = Image.new('RGBA', c.size, hexc_to_rgba(wm.get('color', '#ffffff'))); face.putalpha(c)
    X = int(wm.get('x', 0.5) * W - c.width / 2); Y = int(wm.get('y', 0.93) * H - c.height / 2)
    img.alpha_composite(sh, (X + int(size * 0.03), Y + int(size * 0.06))); img.alpha_composite(face, (X, Y))

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def build(spec):
    W, H = spec.get('size', (1920, 1080))
    preset = dict(PRESETS.get(spec.get('preset', 'airy'), PRESETS['airy'])); preset.update(spec.get('grade', {}))
    sun = spec.get('sun', {})
    out = spec.get('out', 'thumbnail.png')
    scratch = os.path.join(os.path.dirname(os.path.abspath(out)), '_mcthumb_cache'); os.makedirs(scratch, exist_ok=True)

    canvas = load_background(spec, W, H)
    env = sample_env(canvas, W, H)
    # sun glow sits BEHIND characters (so they silhouette against it)
    occl = np.zeros((H, W), np.float32)
    chars = sorted(spec.get('characters', []), key=lambda c: c.get('z', 0))
    canvas = sun_glow(canvas, sun, W, H)
    canvas = particles(canvas, spec.get('particles'), W, H, 'back')
    for pr in spec.get('props', []):
        canvas, A = place_prop(canvas, pr, sun, env, W, H)
        occl = np.maximum(occl, A * pr.get('occlude', 0.6))
    for ch in chars:
        canvas, A = place_character(canvas, ch, sun, env, W, H, scratch)
        occl = np.maximum(occl, A)
    canvas = god_rays(canvas, occl, sun, W, H)
    bright = float(lum(canvas).mean())
    hz = preset.get('haze', 0.3) * float(np.clip((bright - 0.2) / 0.4, 0.15, 1.0))
    hcol = spec.get('haze_color') or '#%02x%02x%02x' % tuple(int(v * 255) for v in np.clip(
        hexc(sun.get('color', '#fff1dc')) * 0.5 + np.array(env[0]) * 0.2 + 0.3, 0, 1))
    canvas = haze(canvas, hz, W, H, hcol, spec.get('fog', 0.1) * min(1, bright / 0.45), occl)
    canvas = particles(canvas, spec.get('particles'), W, H, 'front')
    canvas = bloom(canvas, preset.get('bloom', 0.35), H)
    canvas = grade(canvas, preset.get('exposure', 0), preset.get('sat', 1), preset.get('contrast', 1), preset.get('lift', 0),
                   preset.get('temp', 0), preset.get('tint_shadows'), preset.get('tint_highlights'), preset.get('split', 0))
    canvas = film(canvas, preset.get('grain', 0.03), W, H, texture=spec.get('paper', 0.5))

    img = Image.fromarray(to_u8(canvas)).convert('RGBA')
    for t in spec.get('text', []):
        kind = t.get('type', 'title')
        if kind == 'badge': draw_badge(img, t, W, H)
        elif kind == 'watermark': draw_watermark(img, t, W, H)
        else: draw_title(img, t, W, H)
    img = frame(img, spec.get('frame'))
    img = img.convert('RGB')
    img.save(out)
    base, _ = os.path.splitext(out)
    small = img.resize((1280, 720), Image.LANCZOS)
    small.save(base + '_720.jpg', quality=93, optimize=True, subsampling=0)
    # readability check at real YouTube sizes (home feed desktop / sidebar / mobile list)
    sizes = [(640, 360), (360, 202), (168, 94)]
    sheet = Image.new('RGB', (sum(w for w, _ in sizes) + 40, 380), (15, 15, 15)); x = 10
    for w, h in sizes:
        sheet.paste(img.resize((w, h), Image.LANCZOS), (x, 10)); x += w + 10
    sheet.save(base + '_sizes.png')
    return out

def resolve_paths(spec, base):
    def fix(p):
        return p if (not isinstance(p, str) or os.path.isabs(p) or not os.path.exists(os.path.join(base, p))) else os.path.join(base, p)
    bg = spec.get('background', {})
    if bg.get('image'): bg['image'] = fix(bg['image'])
    for c in spec.get('characters', []):
        for k in ('skin', 'image'):
            if c.get(k): c[k] = fix(c[k])
    for t in spec.get('text', []):
        if t.get('icon'): t['icon'] = fix(t['icon'])
    if spec.get('out') and not os.path.isabs(spec['out']): spec['out'] = os.path.join(base, spec['out'])
    return spec

if __name__ == '__main__':
    spec = resolve_paths(json.load(open(sys.argv[1])), os.path.dirname(os.path.abspath(sys.argv[1])))
    if len(sys.argv) > 2: spec['out'] = sys.argv[2]
    print(build(spec))
