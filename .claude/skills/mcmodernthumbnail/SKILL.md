---
name: mcmodernthumbnail
description: Create pro "modern cinematic render" Minecraft YouTube thumbnails (bedwars, skywars, pvp, duels, SMP, DonutSMP/LIVE streams, hardcore) in the bright airy shader style - pixel-exact 3D skin renders with studio lighting (rim light, sun backlight, AO, self-shadows), depth-of-field background, god rays, haze, bloom, particles, 3D titles and LIVE badges. Use when the user asks for a Minecraft thumbnail, a bedwars/pvp/SMP/live thumbnail, a skin render for a thumbnail, or invokes /mcmodernthumbnail.
---

# /mcmodernthumbnail — modern Minecraft render thumbnails

Produces 1920×1080 PNG + 1280×720 JPG thumbnails in the style of the user's references
(Galris-FX-style bedwars island shot, DonutSMP LIVE badge, jumping hero shot): bright,
pastel, sun-drenched, defocused shader background, a crisply rendered and rim-lit
character, light shafts, haze, snow specks, thin white inner frame.

Read `references/style-guide.md` once per session before designing — it holds the
measured look (brightness, saturation, black point, inverse vignette) and the patterns
from 338 analysed YouTube thumbnails. `references/spec.md` documents every spec field.

## Tools (all in `scripts/`, Python 3 + numpy, Pillow, opencv-python-headless, scipy)

| script | does |
|---|---|
| `fetch_skin.py NAME -o skin.png` | Mojang username/UUID → skin PNG + slim flag (fallback mc-heads.net) |
| `mcrender.py` | own 3D rasteriser: posed player + overlay layer + held item, shadow-mapped studio lighting, supersampled exact alpha. Also `build_island()` voxel props |
| `compose.py spec.json` | full thumbnail pipeline from a JSON spec |
| `cutout.py in.png out.png [--blocky]` | precise matte for a user-supplied render on flat/green background |

Install deps if missing: `pip install numpy pillow opencv-python-headless scipy`.
Network hosts used: `api.mojang.com`, `sessionserver.mojang.com`, `textures.minecraft.net`,
`mc-heads.net` (skins). If they are blocked, ask the user for the skin PNG.

## Workflow

1. **Collect the brief** (ask only for what is missing; sensible defaults otherwise):
   - Minecraft name (Java) or the skin PNG; slim/classic is auto-detected.
   - Video type: bedwars / skywars / duels / SMP / LIVE / hardcore … and the title words
     (max 3 words, or a number like `#1`, `1v4`, `100 WINS`), or no text (refs 1 & 3 have none).
   - Background: best is a **shader screenshot** of their map/server (F1 to hide HUD, high
     render distance). Without one: procedural sky (+ optional voxel bedwars island prop).
   - Optional: logo for a LIVE badge, watermark/signature, mood (airy default / golden /
     frost / punchy / night).
2. **Design before rendering** — pick from the style guide:
   - Composition: subject on a third, text/badge/island on the other side, gaze into frame.
   - Sun position (a top corner) and `behind: true` for backlight + rim.
   - Pose & camera: action pose (`sprint_attack`, `swing`, `leap`, `wave`, `point`,
     `look_back`…), yaw −20…−45 for 3/4 front, ±140…170 for a back view, fov 34–45.
   - Character height 0.85–1.1 of the frame; `y` > 1 to crop at the legs for a hero shot.
3. **Write a spec** (start from `examples/*.json`, copy it next to the output) and run
   `python3 scripts/compose.py spec.json`. A render takes ~15–30 s.
4. **QA — always look at the result** (Read the `_720.jpg` and `_sizes.png`) and fix:
   - [ ] silhouette crisp, no halo, hat/jacket overlay pixels present, arms slim/classic right
   - [ ] rim light on the sun side, shadow side not muddy (raise `fill`/`sky`, lower `self_shadow`)
   - [ ] subject is the most contrasted thing; background clearly defocused
   - [ ] face not covered by text, nothing important bottom-right (timestamp)
   - [ ] readable at 168×94 in `_sizes.png`
   - [ ] overall bright & airy for the default style (mean lum ≈ 0.55–0.65); haze not washing out the face
   Iterate 1–3 times. Then deliver the PNG (master) and the JPG (upload, < 2 MB).
5. Offer variations (other pose/yaw, flipped layout, different preset) — cheap to make.

## Lighting knobs that matter most

`sun.pos`, `sun.behind`, `sun.rays`, character `rim` / `rim_width` / `rim_bloom`, `fill`,
`self_shadow`, `light_wrap`, `pop`, preset `haze` / `bloom`. Defaults are tuned to the
references; change one at a time.

## Skin cut-out quality (non-negotiable)

Render skins with `mcrender.py` — never cut a character out of a screenshot by hand. The
renderer samples texels nearest-neighbour, alpha-tests the second layer, renders at 3×
and box-filters premultiplied colour, so edges are exact and halo-free. Only use
`cutout.py` when the user supplies their own render; inspect its `_check.png`.

## Content rules

- Use only the user's own skin/name or skins they ask for; do not imitate another
  creator's branding/watermark — use the user's own signature text.
- Fonts in `assets/fonts` are SIL OFL; item sprites and block textures are original
  procedural art (no Mojang assets are bundled).
