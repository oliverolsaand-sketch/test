#!/usr/bin/env python3
"""
cutout.py - precise cut-out of a character image the USER supplies
(a Blender / Mine-imator / Cinema4D render, or a screenshot on a flat/green background).

Prefer mcrender.py when you only have a skin: a rendered alpha is always exact.
Use this when the user insists on their own render.

  python3 cutout.py in.png out.png [--key auto|#00ff00] [--tol 0.12] [--blocky] [--choke 0.5]

Steps
  1. alpha source: existing alpha  >  flat/chroma background key (flood-filled from
     the borders, so same-coloured pixels INSIDE the character survive)  >  rembg (if installed)
  2. clean-up: drop specks, fill enclosed holes that are not background-coloured
  3. --blocky : Minecraft characters are made of straight edges - snap the matte to
     a hard 0.5 threshold at 4x and downsample => crisp, perfectly straight anti-aliased edges
  4. colour decontamination: remove the background colour that bled into edge pixels
     (no green/white halo), optional choke
  5. writes out.png (RGBA) + out_check.png (on black / white / checker to inspect halos)
"""
import argparse, sys
import numpy as np
import cv2
from PIL import Image


def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255


def border_color(rgb):
    b = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    med = np.median(b, 0); spread = np.median(np.abs(b - med).sum(1))
    return med, spread


def extend_colors(rgb, solid):
    """Normalised convolution: push colours of `solid` pixels outward (small -> large radius)."""
    solid = solid.astype(np.float32)
    ext = rgb * solid[..., None]; have = solid > 0.5
    for sig in (1.0, 2.0, 4.0, 8.0, 16.0):
        k = int(sig * 3) * 2 + 1
        num = cv2.GaussianBlur(rgb * solid[..., None], (k, k), sig)
        den = cv2.GaussianBlur(solid, (k, k), sig)
        ok = (den > 0.02) & ~have
        ext[ok] = num[ok] / den[ok][:, None]; have |= ok
    return ext


def key_alpha(rgb, key, tol):
    """Matte against a known flat background colour.
    1) hard split: background = key-coloured pixels CONNECTED to the image border
       (same colour enclosed by the character stays foreground)
    2) in a 2px band around the boundary, alpha = projection of (C-B) onto (F-B) where F is the
       character colour extended from the solid core -> correct partial coverage, no halo."""
    d = np.sqrt(((rgb - key) ** 2).sum(-1))
    near = (d < max(tol, 0.05) * 2.2).astype(np.uint8)
    n, lab = cv2.connectedComponents(near, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border)) & (near > 0)
    fg = (~bg).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    core = cv2.erode(fg, k, iterations=2) > 0
    band = (cv2.dilate(fg, k, iterations=2) > 0) & ~core
    F = extend_colors(rgb, core)
    FB = F - key; CB = rgb - key
    proj = (CB * FB).sum(-1) / np.maximum((FB * FB).sum(-1), 1e-4)
    a = np.where(core, 1.0, np.where(band, np.clip(proj, 0, 1), 0.0)).astype(np.float32)
    return a, F


def try_rembg(im):
    try:
        from rembg import remove, new_session
    except Exception:
        return None
    out = remove(im, session=new_session('isnet-general-use'))
    return np.asarray(out, np.float32)[..., 3] / 255


def clean(a, min_area_frac=0.0005):
    m = (a > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 1:
        keep = np.zeros(n, bool); thr = m.size * min_area_frac
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= thr
        a = a * keep[lab]
    return a


def blocky_snap(a, ss=4):
    h, w = a.shape
    big = cv2.resize(a, (w * ss, h * ss), interpolation=cv2.INTER_LINEAR)
    big = (big > 0.5).astype(np.float32)
    big = cv2.morphologyEx(big, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)


def decontaminate(rgb, a, F):
    """Edge pixels take the character colour F (extended from the solid interior).
    For blocky Minecraft characters this is exact: an edge pixel is just a partially covered texel."""
    edge = (a > 0.0) & (a < 0.98)
    return np.where(edge[..., None], F, rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--key', default='auto', help="'auto' (flat border colour) or hex like #00ff00")
    ap.add_argument('--tol', type=float, default=0.14)
    ap.add_argument('--blocky', action='store_true', help='snap to straight Minecraft edges')
    ap.add_argument('--choke', type=float, default=0.0, help='shrink matte by N px (0.5 kills 1px halos)')
    a_ = ap.parse_args()

    im = Image.open(a_.inp).convert('RGBA'); arr = np.asarray(im, np.float32) / 255
    rgb = arr[..., :3]; alpha = arr[..., 3]
    bg = None
    if alpha.min() < 0.98:
        src = 'existing alpha'
        bg = np.array([0, 0, 0], np.float32)  # assume premultiplied-against-black fringe
    else:
        med, spread = border_color(rgb)
        if a_.key != 'auto':
            bg = hexc(a_.key); alpha, F = key_alpha(rgb, bg, a_.tol); src = f'key {a_.key}'
        elif spread < 0.08:
            bg = med; alpha, F = key_alpha(rgb, bg, a_.tol); src = f'flat border {np.round(med * 255).astype(int)}'
        else:
            r = try_rembg(im.convert('RGB'))
            if r is None:
                sys.exit('Background is not flat and rembg is not installed.\n'
                         'Ask the user for a render on a flat/green background or with transparency,\n'
                         'or (better) render the skin with mcrender.py for a mathematically exact cut-out.')
            alpha = r; src = 'rembg isnet'; bg = med; F = extend_colors(rgb, alpha > 0.97)
    alpha = clean(alpha)
    if a_.blocky: alpha = blocky_snap(alpha)
    if a_.choke > 0:
        k = max(1, int(round(a_.choke * 2)) * 2 + 1)
        er = cv2.erode(alpha, np.ones((k, k), np.uint8)); alpha = alpha * (1 - a_.choke) + er * a_.choke if a_.choke < 1 else er
    if src != 'existing alpha':
        rgb = decontaminate(rgb, alpha, F)
    out = np.dstack([rgb, alpha])
    Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save(a_.out)
    # inspection sheet: black | white | checker
    h, w = alpha.shape
    chk = ((np.indices((h, w)).sum(0) // 16) % 2)[..., None] * 0.25 + 0.6
    sheets = [np.zeros((h, w, 3)), np.ones((h, w, 3)), np.broadcast_to(chk, (h, w, 3))]
    comp = np.concatenate([s * (1 - alpha[..., None]) + rgb * alpha[..., None] for s in sheets], 1)
    Image.fromarray((np.clip(comp, 0, 1) * 255).astype(np.uint8)).save(a_.out.rsplit('.', 1)[0] + '_check.png')
    print(f'alpha from: {src}; saved {a_.out}')


if __name__ == '__main__':
    main()
