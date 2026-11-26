# compose.py spec reference

All positions/sizes are **fractions of the canvas** (x: 0 left → 1 right, y: 0 top → 1 bottom;
sizes relative to canvas height). Relative file paths resolve against the spec file's folder.

```jsonc
{
  "out": "thumbnail.png",            // also writes thumbnail_720.jpg and thumbnail_sizes.png
  "size": [1920, 1080],
  "preset": "airy",                   // airy | golden | frost | punchy | night   (see table)
  "grade": { "sat": 0.95 },           // override any preset key: exposure sat contrast lift temp split
                                      //   tint_shadows tint_highlights haze bloom grain
  "background": {
    "image": "bg.png",                // shader screenshot (F1, no HUD). Omit => procedural sky
    "focus": [0.5, 0.5],              // which part of the image survives the 16:9 cover-crop
    "blur": 6,                        // DOF radius in px @1080p (0 = sharp). Refs: 4-10
    "dof": "uniform",                 // uniform | bottom (fg blurred, like ref 1) | radial
    "focus_y": 0.65, "focus_point": [0.5,0.5], "focus_radius": 0.25,
    "grade": {"exposure": 0.05, "saturation": 0.95, "contrast": 0.9, "lift": 0.04, "temperature": 0},
    "top": "#2f8fe8", "horizon": "#bfe8ff", "clouds": 0.55, "seed": 1,   // procedural sky only
    "flip": false
  },
  "sun": {
    "pos": [0.85, 0.05],              // may be outside 0..1 (off-frame sun)
    "behind": true,                   // sun behind subject => strong rim, shadow towards viewer
    "color": "#fff1dc", "rim_color": "#fff6e8",
    "key": 3.0, "sky": 0.8, "bounce": 0.3,
    "rays": 0.55,                     // volumetric + stylised shafts, 0 = off
    "beams": 9, "beam_strength": 1.0, "rays_on_subject": 0.75, "seed": 7,
    "glow": 0.8, "glow_size": 0.38
  },
  "props": [                          // procedural voxel island (bedwars style), drawn behind characters
    {"type": "island", "seed": 4, "theme": "grass|sand|end", "defense": "wool|obsidian|planks|endstone",
     "radius": 7, "depth": 6, "x": 0.7, "y": 0.47, "height": 0.9, "yaw": -35, "pitch": 24, "fov": 35,
     "blur": 1.2, "atmosphere": 0.12, "flip": false}
  ],
  "characters": [
    {
      "skin": "Username or skin.png", // username => fetched from Mojang (slim flag auto)
      "image": null,                  // OR a pre-cut RGBA render (see cutout.py) instead of skin
      "slim": null,                   // null = auto
      "pose": "sprint_attack",        // preset name, or {"base":"run","rarm":{"fwd":90}} overrides
      "yaw": -30,                     // camera orbit: 0 front, -/+ 3/4 views, ±150..180 back view
      "pitch": 6,                     // camera height (+ = from above)
      "fov": 34,                      // 25 = flat/tele, 45+ = dramatic wide-angle (big head/hands)
      "roll": 0,                      // dutch angle
      "height": 0.9, "x": 0.72, "y": 1.02,   // feet anchor; y>1 crops legs (hero crop)
      "flip": false,                  // mirror horizontally
      "z": 0,                         // draw order among characters
      "item": "sword:diamond",        // kind:palette  or {"kind":"sword","palette":"amethyst","enchant":true,
                                      //   "glint":"#b36bff","yaw":0,"pitch":0,"roll":0,"scale":0.68,"hand":"right"}
      "rim": 1.8, "rim_width": 1.1, "rim_bloom": 0.8,
      "fill": 1.0, "ao": 0.6, "self_shadow": 0.8, "spec": 0.2, "exposure": 1.0,
      "light_wrap": 0.35, "color_match": 0.12, "pop": 0.5,
      "ground_shadow": true, "contact_shadow": 0.55, "cast_shadow": 0.35, "cast_squash": 0.22, "cast_skew": 1.1,
      "glow": {"color": "#7ad7ff", "radius": 0.03, "strength": 1.0},   // coloured aura (pvp/tier look)
      "outline": {"color": "#ffffff", "width": 0.006},                 // sticker outline (punchy look)
      "light": {}                     // raw overrides for mcrender DEFAULT_LIGHT
    }
  ],
  "particles": {"type": "snow|dust|embers", "count": 140, "color": "#ffffff", "seed": 11},
  "haze_color": null, "fog": 0.1, "paper": 0.5,
  "text": [
    {"type": "title", "text": "SOLO\nBEDWARS", "font": "lilita", "size": 0.16, "x": 0.25, "y": 0.3,
     "fill": ["#ffffff", "#ffe066"], "fill_angle": 90, "stroke": 0.075, "stroke_color": "#101018",
     "extrude": 0.06, "extrude_angle": 90, "extrude_color": "#1a1a2e", "shadow": 0.55, "gloss": 0.35,
     "rotate": 4, "line_height": 1.0, "spacing": 0, "width": null},
    {"type": "badge", "text": "LIVE", "icon": "logo.png", "x": 0.3, "y": 0.5, "height": 0.25,
     "font": "bungee", "colors": ["#18e0ff", "#0aa8d8", "#0b6f9e"], "border": "#06425e",
     "text_extrude": "#b9c6d2", "dot": "#ff3040", "rotate": 0},   // icon "" => red rec dot, omit => no icon
    {"type": "watermark", "text": "YourName FX", "font": "kaushan", "size": 0.075, "x": 0.5, "y": 0.93}
  ],
  "frame": {"inset": 0.016, "width": 0.0028, "radius": 0.012, "opacity": 0.75}   // false = none
}
```

## Presets

| preset | use for | feel |
|---|---|---|
| `airy` | default; refs 1-3, bedwars/skywars daytime, LIVE | bright, pastel, white haze, lifted blacks |
| `golden` | sunset, SMP, emotional | warm highlights, purple shadows |
| `frost` | winter/snow maps, ice | cool, very bright, low sat |
| `punchy` | family B "flat punchy" thumbnails | saturated, contrasty, little haze |
| `night` | nether, end, night pvp | dark, neon, strong bloom |

## Pose presets (mcrender.POSES)

`idle, stand_sword, walk, run, sprint_attack, swing, jump, leap, wave, cheer, point,
look_back, crouch, flex, dance, t_pose`

Custom pose: `{"base": "run", "head": {"yaw": 20}, "rarm": {"fwd": 100, "side": 10, "twist": -20}}`
— limbs take `fwd` (raise forward), `side` (raise outward), `twist`; head takes `yaw`, `pitch`
(+ = look down), `roll`; `torso` takes `yaw`, `pitch` (+ = lean forward), `roll`; `root` adds
`lift` (px up) plus `yaw/pitch/roll` of the whole body.

## Fonts (assets/fonts, all SIL OFL)

`lilita` (Lilita One), `luckiest` (Luckiest Guy), `titan` (Titan One), `bungee`,
`pixel` (Press Start 2P), `silkscreen` (Silkscreen Bold), `kaushan` (Kaushan Script),
`pacifico`. Any absolute .ttf path also works.

## Items

`sword | pickaxe | axe` × palettes `diamond, netherite, gold, iron, amethyst, ruby, emerald, wood`
(original pixel art). `"enchant": true` adds the animated-looking glint + glow.
