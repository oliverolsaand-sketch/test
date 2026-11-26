# Modern Minecraft render thumbnail — style guide

Built from (a) a close read of the 3 reference thumbnails the user supplied and
(b) 338 YouTube thumbnails scraped on 2026-11-26 from 22 searches (bedwars, hypixel
bedwars, bedrock bedwars, skywars, hive, cubecraft eggwars, pika network, bridge duels,
1v1 pvp, pvp montage, donutsmp / donut smp live, lifesteal smp, smp live, hardcore,
speedrun, render thumbnail…). About 225 of them were reviewed visually on contact sheets;
all 338 went through the numeric pass (`stats` below).

## 1. The three references

| | Bedrock Bedwars (Galris FX) | DonutSMP LIVE | Jumping hero (mountains) |
|---|---|---|---|
| Subject | 1 char, **from behind**, 3/4, left third, looking at the island | 1 char, 3/4 front, right half, waving/dynamic | 1 char, centre, **mid-jump**, huge |
| Story | gaze line char → floating bedwars island with defense pyramid | badge on the left, char on the right | pure hero shot |
| Sun | top-left, diagonal shafts across the whole frame | top-right **behind** the char → bright rim on right edges | top-right, god rays + rainbow |
| Background | sky + sharp island, **foreground sand blurred** (DOF) | village heavily blurred (DOF ~1.5% of width) | mountains moderately blurred, birds, foreground tree defocused |
| Atmosphere | snow specks, lens glow, haze | snow specks, warm haze, white fog at the bottom | haze, petals/snow, light leak |
| Text | none (script watermark bottom centre) | **LIVE badge**: cyan gradient, wave pattern, round logo, chunky white 3D letters | none (logo watermark bottom left) |
| Finish | thin white rounded **inner frame**, paper/canvas texture | inner frame | faint inner frame |
| mean lum / sat | 0.56 / 0.44 | 0.60 / 0.28 | 0.63 / 0.23 |

Numbers (`lum` = mean luminance 0..1, `sat` = mean HSV saturation):

| metric | refs (3) | corpus p10 | corpus median | corpus p90 |
|---|---|---|---|---|
| mean luminance | **0.56–0.63** | 0.27 | 0.44 | 0.59 |
| saturation | **0.23–0.44** | 0.30 | 0.49 | 0.70 |
| black point (p2) | 0.016–0.063 (lifted) | 0.00 | 0.036 | 0.11 |
| white point (p98) | 0.89–1.00 | 0.75 | 0.96 | 1.00 |
| centre − edge lum | **−0.10** (edges brighter!) | −0.16 | +0.01 | +0.22 |
| clipped highlights | 0.5–10 % | 0 % | 3.6 % | 13 % |

**Takeaways:** the reference style is *brighter than 90 % of the corpus* and *calmer in
saturation* — airy, pastel, sun-drenched. Edges are hazed toward white (inverse
vignette) instead of darkened. Blacks are lifted (never pure black). The character is the
most contrasted, most saturated thing in the frame.

## 2. What the 338-thumbnail corpus shows

Two big families:

* **A. Cinematic render** (this skill's default; refs belong here) — shader screenshot
  background, strong DOF, god rays, haze, pastel grade, rim-lit 3D characters, light
  particles, minimal or badge text.
* **B. Punchy flat** — saturated sky-blue or solid backgrounds, huge stroked text,
  arrows/red circles, keyboard + mouse props. Use `"preset": "punchy"` when the user wants
  this look; keep the same render quality.

Recurring patterns (counted roughly over the reviewed sheets):

* **Character render in ~85 %** of thumbnails. 1 character is most common; 2 for "vs" /
  duels (split diagonal, red vs blue); 3–6 for team / leaderboard shots.
* **Size:** character spans 60–100 % of the height; often cropped at the knees/hips so the
  head is huge. Heads are the focal point — eyes ~ upper third.
* **Placement:** rule of thirds, character on one side, text/badge/island on the other.
  Characters look or point *into* the frame, never out of it.
* **Poses:** sprinting with sword raised (most common in bedwars/pvp), swinging, jumping,
  pointing, waving (LIVE thumbnails), flexing, looking over the shoulder (back view).
  Static "T-pose / arms down" reads as amateur.
* **Items:** diamond / netherite / purple-enchanted swords, often glowing; ender pearls,
  fireballs, beds. Enchant glint = purple/pink glow.
* **Text:** 1–3 words or one number, 15–30 % of frame height. Heavy rounded display font
  (Lilita / Luckiest style) or blocky pixel font for ranks (`[CHAMPION]`, `[MVP++]`).
  White or yellow fill, dark stroke, 3D extrusion, soft drop shadow, slight rotation.
  Common words: SOLO BEDWARS, LIVE, #1, 1v4, 100 WINS, TIER 1, ROOKIE vs MASTER.
* **LIVE** thumbnails: badge in a rounded panel (red pill with dot, or the cyan panel
  from ref 2) + waving/greeting character + server logo.
* **Light:** sun behind or beside the subject; bright rim on the silhouette edge facing
  the sun; sky-coloured fill on the shadow side; soft contact shadow under feet.
* **Background:** blurred Minecraft builds (villages, castles, bedwars maps), floating
  islands, sky; nearly always defocused so the character pops.
* **Finish:** thin white inner frame (refs + several corpus items), grain/texture,
  particles (snow, dust, embers for nether/pvp, confetti for wins).

## 3. Lighting recipe (what makes it look "extremely pro")

1. **One motivated sun.** Decide where the sun is in the frame (usually a top corner).
   Everything follows it: god-ray origin, rim side, cast-shadow direction, haze bias.
2. **Backlight + rim.** Sun slightly behind the character (`"behind": true`). The edges
   facing the sun get a 1–2 px hot rim + a soft bloom of the rim. This single detail
   separates pro renders from flat ones.
3. **Fill from the sky.** The shadow side is never black: cool sky fill + camera-side
   fill light. Ground bounce warm (sand/wood) from below.
4. **Self-shadow + AO.** Head shadows the neck, arms shadow the torso, crevices darker.
5. **Light wrap.** Blurred background light bleeds 1–2 % over the silhouette edge — glues
   the character into the plate.
6. **Contact shadow** under the feet when they stand on something visible.
7. **Atmosphere.** Volumetric shafts (occluded by the character), sun glow *behind* the
   character, haze on the frame edges, bottom fog band, bloom on highlights.
8. **Grade last.** Lift blacks ~3–5 %, gentle S-curve, split tone (cool shadows / warm
   highlights), saturation moderate, fine grain + paper texture.

## 4. Skin cut-out rules

* **Never** cut a character from a busy screenshot by hand. Render it (mcrender.py):
  the alpha is exact, overlay pixels are alpha-tested, edges are supersampled (3×) and
  box-filtered, so there are no halos, no missing hat pixels, no jaggies.
* Respect the skin model: slim (3 px arms) vs classic (4 px). The Mojang API tells you
  which; `fetch_skin.py` reports it and compose passes it on.
* Keep texels square and sharp — nearest-neighbour sampling only, no blur on the
  character (DOF applies to the background, not the subject).
* If the user insists on their own render: `cutout.py` (flat/green bg key with
  connected-to-border logic, projection matte on a 2 px band, colour decontamination,
  `--blocky` straight-edge snap). Check `_check.png` on black/white/checker for halos.

## 5. Composition checklist

* Character fills ≥ 70 % height (hero) or ≥ 85 % (with heavy crop) — readable at 168×94.
* Face/head in the upper third, never cut by the frame edge.
* Max 3 words of text; text never overlaps the face.
* Keep the bottom-right ~18 % × 12 % free of important stuff (YouTube timestamp).
* Strong value separation: bright subject edge vs darker/defocused background (or the
  inverse for night scenes).
* Check `_sizes.png` (640/360/168 px wide). If the subject is not instantly readable at
  168 px, go bigger / simpler.
