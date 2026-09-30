# Blender Virtual Community

This repository contains tools for generating 3D city environments in Blender using OpenStreetMap data and height field information.

## Workflow Overview

### 1. City Objects and Buildings Processing

**Step 1: Height Field Processing**

- `process_height_field.py` - Run outside Blender
- Processes height data from `height_field.npz` files for each city
- Assigns elevation values to objects

**Step 2: Blender Scene Generation**

- `blender_process_cities.py` - Run inside Blender (paste code into Scripting workspace)
- Generates complete USD scenes for each city including:
  - Terrain mesh (`terrain.glb`)
  - Building models
  - Objects positioned with elevation data

- `pipeline/blender_process_cities_with_roads.py` Run inside Blender (paste code into Scripting workspace)
- Generates complete USD scenes for each city including:
  - Terrain mesh (`terrain.glb`)
  - Building models
  - Objects positioned with elevation data
  - Add road objects based on road data - traffic lights, signs

### 2. Road Network Visualization

**Step 1: Road Height Processing**

- `process_road_for_height.py` - Run outside Blender
- Extracts road elevation data from `height_field.npz` files and road width data from road_tada/roads.pkl
- Prepares road geometry with accurate height and width information

**Step 2: Road Scene Creation**

- `blender_add_road_with_height.py` - Run inside Blender (paste code into Scripting workspace)
- Generates road network and terrain for visualization
- Creates viewable road infrastructure with proper elevation

## Usage Notes

- Run external Python scripts first to prepare height data
- Copy and paste Blender scripts into Blender's Scripting workspace for execution

## TODO

- Segment terrain with different heights based on road infomation

<!-- VC_PLUS_PIPELINE_NOTES -->

## VC Plus USD Layer Pipeline Notes

This directory is now the collection point for the Virtual Community plus VC+ scripts that were previously split across `navverse_tools/blender_virtual_community`, `navverse_tools/3dmodel`, and `a separate local tools directory`.

### Directory layout

- General VC+ helpers live directly in this directory; orchestration and composition scripts live under `pipeline/`; visual QA and acceptance renderers live under `render/`.
- `pipeline/merge_usd.py`: copied from `navverse_tools/3dmodel/merge_usd.py`. It reorganizes an existing city USD into `/World/city` and `/World/store`, then adds road objects and Objaverse assets. It does not generate storefronts from OSM and does not perform boolean cutting.
- `pipeline/merge_usd_innout.py`: copied from `navverse_tools/3dmodel/merge_usd_innout.py`. It consumes `{CITY}_gr_merged.usd` style inputs and writes `{CITY}_innout.usd`, preserving `/World/start_result_navigation` and adding nearby outdoor objects for indoor/outdoor scenes. It assumes rooms/holes already exist in the source USD.
- `pipeline/generate_vc_full.py`: current orchestration entry point for
  `vc_full` cities, with `vc_full_raw` still usable through
  `--source-dataset vc_full_raw`. It runs raw USD generation, point/object planning,
  outdoor composition, facade scoring, final innout composition,
  texture/material optimization, validation, QA rendering, scene-photo rendering,
  and manifests.
- `pipeline/compose_vc_full_innout.py`: final raw-work-root innout composer invoked by
  `pipeline/generate_vc_full.py`.

### Current vc_plus generation flow

The current recommended full-city flow starts from
`navverse_data/vc_full/<CITY>` and writes generated layers to
`navverse_data/vc_plus/<CITY>`.

Use `pipeline/generate_vc_full.py` as the only batch entry point. In flat-ground
mode it first preprocesses the city source into the work root before raw USD
export:

- compresses terrain, building roots, object roots, OSM node heights, and
  `height_field.npz` vertical relief around a ground reference;
- uses `--flat-ground-height-scale 0.3` for the current benchmark flat terrain;
- lowers OSM road corridors during flat-ground preprocessing with
  `--road-lower-depth 0.2`, so the later raw-road lowering pass is skipped;
- cuts both sides of every road boundary before lowering: the road core is
  lowered by 0.2 m and a 0.2 m curb strip continuously joins it to the original
  sidewalk height;
- copies the processed `height_field.npz` back to the source city when
  `--flat-ground-copy-height-field-to-source` is used, preserving a
  `.bak_flat_ground_*` backup;
- removes the old city `roof.glb` layer from raw USD export. The roof layer is
  derived from the original terrain relief and does not stay aligned after
  flat-ground preprocessing.

After preprocessing, the pipeline generates:

- `{CITY}.usd` and `{CITY}_with_road_objects.usd` from the processed flat city
  directory;
- `{CITY}_outdoor.usd` with road objects, storefronts, and Objaverse assets;
- `{CITY}_innout.usd` with the selected GR commercial indoor scene aligned to
  the scored doorway;
- visible semantic `/World/TerrainRegions/{road,curb,sidewalk,crosswalk,green,water}`
  meshes clipped from the final lowered terrain, with the identical source mesh
  hidden at `/World/TerrainRegions/Terrain`;
- per-city planning, material, height-adjustment, opening, and manifest JSON
  files.

Storefronts are scaled to a 4 m target height with uniform scale. The facade
planner reads the scaled storefront bbox and writes matching `opening_width` and
`opening_height`, so changing the storefront target height must be done before
planning/opening cuts. Do not resize storefronts after hole cutting without
rerunning the facade planner and opening application.

Objaverse generation targets 600 sidewalk props and 200 vehicles per outdoor
city. Props are sampled only from OSM ways explicitly tagged with
`footway=sidewalk` or road-side `sidewalk=left/right/both/yes` semantics;
generic footways, unclassified terrain assigned to the visual sidewalk region,
and synthetic road-edge fallback points are not eligible. Vehicle and prop
selection is balanced by asset name and semantic group
to avoid dense repetition of the same trash can, umbrella, bicycle, or vehicle.
When semantic groups do not fill the target, a balanced fallback samples the
remaining explicit sidewalk points while preserving same-asset spacing. The
planner reads each asset's actual USD bounds and reserves its complete XY
footprint plus a safety gap. Prop pairs keep at least 0.75 m between footprints;
any pair containing a vehicle keeps at least 2.5 m. Fallback passes never relax
these footprint clearances.
The vehicle catalog is explicit, so names such as `bus_stop`, `cargo_pallet`,
or `vans_shoes` are not misclassified as vehicles. Cities
without explicit OSM sidewalks generate no sidewalk props. Asset-specific
size corrections live in `pipeline/objaverse_asset_rules.py` and are applied
while writing `objaverse_objects.json`; the USD composer then uses those stored
scales.

Innout content is planned independently after the pipeline selects the connected
building and exact doorway from the normalized city scene. Candidate buildings
are scored together with their usable storefront capacity inside the 50 m local
area; the best candidate with enough local facade capacity is preferred. The
resulting anchor plan is passed unchanged to final composition, so a later
planning pass cannot silently choose another building, wall, or door. The innout
planner clips lane and explicit sidewalk samples to 50 m around that doorway,
then targets 12 facade-aligned storefronts with a capacity-aware hard floor of
up to 7, plus up to 120 props and 30 vehicles in the local area. The selected
indoor building is excluded from normal
storefront placement so storefront cuts cannot obstruct the room entrance.
Innout counts are upper bounds: usable facade capacity and footprint safety may
reduce them when the neighborhood cannot hold the targets without overlap. The
building boolean opening spans the transformed indoor room's full entrance-side
frontage and reaches above its transformed top, rather than being sized only
from the detected door. Generated tunnel faces use a dedicated rough gray-white
material while the exterior facade keeps the canonical city texture. The
boolean cutter extends primarily inward, with only 0.25 m outside the facade,
and cut-interior faces are clipped to the source building XY bounds, so open or
concave building meshes cannot create a cutter-sized cap beyond the building.
These plans are written as `innout_storefront_plan.json`,
`innout_objaverse_objects.json`, and `innout_road_objects.json`; the outdoor
plans and counts remain unchanged. The doorway is reserved during object and
vehicle planning with a footprint-expanded corridor: 1.5 m beyond each side
of the door, 1 m inward, and 10 m outward. This keeps both small props and large
vehicles out of the entrance path. The connecting threshold depth includes the
transformed indoor door thickness. Once a work city contains
`flat_ground_summary.json`, `height_field.npz`, and processed `terrain.glb`,
later plan/compose-only runs automatically use that processed city as their
geometry and height source instead of falling back to the original terrain.
The defaults can be changed with
`--innout-radius-m`, `--innout-storefronts`, `--innout-props`, and
`--innout-vehicles`.

Example full-flat regeneration command:

```bash
data_root="${NAVVERSE_DATA_ROOT:-$PWD/navverse_data}"
python navverse_tools/vc_plus/pipeline/generate_vc_full.py \
  --repo-root "$PWD" \
  --data-root "$data_root" \
  --source-dataset vc_full \
  --work-root "$data_root/vc_plus" \
  --store-root "$data_root/storefront" \
  --grscene-root "$data_root/grscenes_commercial" \
  --objaverse-root "$data_root/objaverse" \
  --vc-plus-root "$data_root/vc_apr14" \
  --log-root "$data_root/vc_plus/logs" \
  --qa-root "$data_root/vc_qa" \
  --scene-data-root "$data_root/../scene_data" \
  --variant both --stage all \
  --flat-ground \
  --flat-ground-height-scale 0.3 \
  --flat-ground-reference min \
  --flat-ground-copy-height-field-to-source \
  --storefront-target-height 4.0 \
  --road-lower-depth 0.2 \
  --road-lower-subdivide-cuts 1 \
  --usd-python "${CONDA_PREFIX}/bin/python" \
  --blender blender
```

### Portable flat dataset

After every city worker has finished, run the asset-localization step once from
the repository root:

```bash
python \
  navverse_tools/vc_plus/pipeline/localize_vc_plus_assets.py \
  --flat-root "$data_root/vc_plus" \
  --data-root "$data_root" \
  --grscene-root "$data_root/grscenes_commercial" \
  --strict-missing
```

This copies the runtime Objaverse USD library to `vc_plus/objaverse`, the
storefront library to `vc_plus/storefront`, and the CloudySky assets to
`vc_plus/nvidia`. It then rewrites authored USD paths relative to each
layer and writes `vc_plus/portable_assets_manifest.json`. Run this as a
single final packaging step rather than from parallel city workers, because all
workers share the localized asset directories. The audit recursively follows
all final VC+ and GRScenes layers and separately checks the effective composed
shader assets; a nonzero exit status is a packaging blocker.

The scene assets remain self-contained in two sibling directories. A runnable
benchmark release also includes navmeshes and exposes the bundled CloudySky at
the runtime path expected by evaluation:

```text
<bundle>/vc_plus/
<bundle>/grscenes_commercial/
<bundle>/navmesh/
<bundle>/nvidia -> vc_plus/nvidia
```

Transfer the layout with a tool that preserves relative symlinks, such as
`rsync -a` or `tar`. Scene USDs no longer depend on the original sibling
`objaverse`, `storefront`, or `nvidia` directories. Re-run localization after a
later scene regeneration, because the composer intentionally continues to read
the canonical source asset libraries while generating scenes.

### Storefront and indoor composition

- `pipeline/compose_vc_full_innout.py` is the forward raw-generation path used by
  `pipeline/generate_vc_full.py`. It consumes `building_geometry_score.json`, matches
  the generated outdoor building prim, cuts the raw building with
  `../3dmodel/cut_vc_building_blender.py`, composes the assigned GR commercial
  room into `{CITY}_innout.usd`, and renders QA images under
  `navverse_data/vc_qa/<CITY>/`. It does not add a storefront at the innout
  doorway unless `--add-exterior-storefront` is explicitly passed.

The composer removes the selected building footprint from the visible region
meshes and hidden source terrain, then adds a ground-textured collision ramp
between the indoor floor and sampled exterior terrain. Storefront openings from
the local innout plan are applied before the doorway cut; only a storefront that
actually overlaps the reserved doorway is removed. A narrow four-meter indoor
doorway corridor is cleared of intersecting furniture/animation instances, and
the first 2.5 m outdoors is cleared of intersecting RoadObjects/Objaverse
instances. The green exit sign is placed above the interior side of the door.

For batch generation, the pipeline first scores a road-facing facade without
generated content and fixes the connected building and doorway. It then plans
the outdoor scene globally and the innout scene locally around that doorway,
applies their storefront openings to separate building layers, composes the GR
indoor room, and renders doorway QA before accepting the city. Outdoor OSM
storefronts use `storefront_plan.json`; local innout storefronts use
`innout_storefront_plan.json`. Both are instantiated by `pipeline/merge_usd.py`
and cut by `pipeline/apply_storefront_openings.py`.

Height alignment is terrain/road driven. `terrain_utils.py` loads the final
`terrain.glb`, maps its Y-up coordinates into map Z-up coordinates, and uses
spatially indexed barycentric interpolation on the top terrain surface.
`height_field.npz` is only a fallback when mesh coverage is unavailable. Innout
openings sample the door center, door
endpoints, and outside points, then place both the building hole bottom and the
GR indoor floor at the local terrain height. Storefront openings and storefront
asset z placement prefer the closest facade-side `road_data_with_height.osm`
segment height, falling back to nearby terrain only when no road segment is
available. This avoids placing stores under elevated road/sidewalk slabs when
the height field has multiple nearby levels. Outdoor Objaverse props, road
objects, and default cameras are also grounded through this helper instead of
trusting stale OSM node z values.

Storefront facade placement also checks the building edge endpoints. For each
asset, `storefront_facade_planner.py` reads the referenced storefront USD bbox
when USD Python is available and uses the larger of that facade width and the
minimum opening width. The written building opening is the asset facade width
plus a small width margin and the asset height plus a small height margin, so
the cut does not leave a large empty hole above the storefront. Storefront
placement also uses the asset bbox's inward local-Y extent to compute a per-asset
offset from the facade, keeping the storefront close to the exterior wall while
avoiding wall penetration. The final point on the selected building edge is
clamped so `edge_s +/- width / 2` stays inside the edge endpoints with a 2 m
safety margin. Candidate facades also sample a short strip in front of the
storefront; if that strip falls inside a building footprint, the edge is
rejected. Generic combined building meshes named exactly `buildings_element` are
excluded from storefront candidates because they can cover whole blocks and
produce interior-looking street views. If a short corner edge cannot contain the
asset or does not have this front clearance, another road-facing facade is
chosen.

When a city's OSM commercial POIs are sparse, `generate_vc_full.py` supplements
roadside synthetic storefront candidates before facade alignment. The default
minimum is 50 storefronts per city (`--min-storefronts 50`), with
`--max-storefronts` still available as a hard cap. Synthetic storefronts use
store-like asset categories only, use globally unique stable `index` values,
then follow the same facade alignment, road/terrain-height, and
building-opening path as OSM-derived storefronts.
The minimum is enforced on valid `facade_aligned` storefronts, not on fallback
records. If OSM/roadside candidates do not provide enough road-facing building
matches, `storefront_facade_planner.py` adds `synthetic_facade_fill` placements
directly on eligible building edges until the target is met. Fallback records
remain in `storefront_plan.json` for diagnostics, but `pipeline/merge_usd.py`,
`pipeline/apply_storefront_openings.py`, and the Isaac storefront renderer only consume
`facade_aligned` placements, so fallback candidates do not become floating
storefronts or unmatched holes.

`cut_vc_building_blender.py` supports hollow-shell and boolean-tunnel modes.
Innout composition uses the boolean-tunnel mode on the storefront-cut building,
preserving the original building geometry, materials, and non-overlapping store
openings. Storefront-cut GLBs are Z-up intermediate assets, so the door-cut
helper restores their coordinate system after Blender's standard glTF Y-up
import before applying the final doorway boolean. The opening bottom clears the
closed solid threshold profile while the opening width and top expose the full
transformed indoor frontage; the connector overlaps both the real facade and
the transformed indoor door. Storefront opening application preserves the
original building mesh and uses shallow
per-opening boolean cutters: the cutter extends slightly outside the facade and
about one storefront depth inside the building instead of cutting a tunnel
through the whole structure. Storefront assets are placed with their local
`Y=0` facade plane nearly coplanar with the building facade, leaving only a
small anti-z-fighting offset; do not derive this offset from sign/canopy
geometry. The original GR scene is not modified; innout layers deactivate the
referenced GR door prim and
`/World/start_result_navigation/__default_setting/HDR_Sphere` locally.

Indoor scenes should come from `grscenes_commercial/scenes/scene*`.
`pipeline/generate_vc_full.py` assigns them by shuffling the catalog with `--seed` and
cycling through the city list. This keeps the assignment random while ensuring
each GR commercial scene is used at least once in the full 52-city run.

Before using GR commercial scenes for innout composition, scan their semantic
door meshes and write one `door_info.json` per scene:

```bash
python navverse_tools/vc_plus/detect_grscene_doors.py \
  --grscene-root navverse_data/grscenes_commercial
```

The detector reads each
`grscenes_commercial/scenes/scene*/start_result_navigation.usd`, finds direct
children under `/Root/Meshes/BaseAnimation/door`, computes mesh-point
world-space bounding boxes for each door, and writes the door center, bottom,
size, orientation-axis guess, and a non-environment scene bbox to the matching
scene folder. The output is geometric metadata only: it identifies modeled door
objects and positions, but it does not prove that a door is open or walkable.

The forward innout placement now expects this metadata. `geometry_score_blender.py`
only catalogs GR commercial scenes with valid likely entry doors, and
`pipeline/compose_vc_full_innout.py` aligns the chosen GR door center and bottom edge to
the VC building doorway. It tries both possible door normal directions and
accepts only a transform whose transformed indoor scene bbox stays inside the
selected building footprint.

### Outdoor terrain and region visualization

`pipeline/blender_process_cities_with_roads.py` applies terrain normal smoothing by
default after importing `terrain.glb` without moving terrain vertices. The VC
terrain mesh contains layered textured faces and `Ground_base_mat` fill faces;
Blender's geometric Smooth modifier can separate those layers during USD export
and expose a regular triangle/grid artifact in Isaac Sim. Use
`--no-smooth-terrain` for exact raw mesh debugging. Only use
`--terrain-geometric-smooth` with `--terrain-smooth-iterations` and
`--terrain-smooth-factor` for controlled Blender-side experiments, then validate
the exported USD terrain render before batch generation.

`pipeline/merge_usd.py` adds `/World/TerrainRegions` to the outdoor layer unless
`--skip_terrain_regions` is passed. Semantic meshes are visible by default and
preserve the final terrain Z, UVs, normals, and material bindings. The original
terrain is moved to `/World/TerrainRegions/Terrain` and hidden. Categories with
zero OSM area are omitted; unclassified terrain is assigned to `sidewalk`.

The copied `/World/city` geometry remains at scale `1.0`; only storefront assets
use the centimeter-to-meter parent scale. `audit_vc_plus.py` validates both
the city scale and its composed world bounds so a standalone merge cannot bury
the buildings below the terrain through an accidental `0.01` scale.

### Texture/material optimization status

Texture downsampling is part of the final scene build, not a separate analysis
step. The initial optimization runs after the outdoor/innout USD layers are
composed. A second effective-binding pass runs after storefront adjustment and
any render-driven storefront repair, before dependency validation, navmesh
packaging, and benchmark use.

The raw city `/_materials/...` shader paths are normalized to their composed
`/World/city/_materials/...` keys and matched against
`<data-root>/vc_full_generated/<CITY>/<CITY>_outdoor.usd`. The pipeline copies
the canonical upscaled city DDS files plus the 64 Nano Banana ground PNGs into
the work city and rewrites every raw city texture input to DDS. After
composition it repeats the canonical rewrite on outdoor, innout base, and final
innout layers, so rebuilding a scene cannot silently restore PNG bindings from
an older base layer. `audit_vc_plus.py` treats any remaining non-DDS
`/World/city` texture or missing city DDS as a hard failure. Shader-path matching
is required because filenames containing dots or Unicode may be truncated or
renamed during GLB/USD conversion. Override the canonical source with
`--city-texture-source-root` when needed.

`pipeline/generate_vc_full.py --stage all` now runs
`../texture_downsampling/process_material.py` on the composed
output layers unless `--skip-texture-optimization` is passed. The default pass:

- rewrites duplicate material bindings and deletes redundant material prims with
  `--texture-material-op merge_materials_prune`;
- converts texture references to DDS with `--texture-variant a_bc7_nomip`;
- uses BC7 for color/other maps, BC5 for normal maps, and BC4 for scalar maps;
- uses `--texture-scene-root` to decide which asset tree is eligible for DDS
  conversion. It defaults to `--data-root` so shared storefront, GR scene, and
  Objaverse textures are included;
- writes per-city JSON manifests next to the generated USDs.

After texture optimization, the pipeline runs `pipeline/adjust_storefront_materials.py`
on the generated outdoor and innout layers. This is scene-local: it does not
modify the source `storefront/` assets. The tool writes adjusted albedo copies
under `<CITY>/storefront_albedo_adjusted/`, overrides `/World/store/*` texture
inputs to those copies, and sets storefront `UsdPreviewSurface` shaders to
non-metallic, rougher, lower-specular values. Bright source storefront PNGs are
scaled toward `target_mean=125` and `target_p90=175`, clamped by
`min_factor=0.58`, which reduces white storefront facades without darkening
already dark assets. Per-layer summaries are written as
`<CITY>_outdoor_storefront_material_adjust.json` and
`<CITY>_innout_storefront_material_adjust.json`.

Because storefront adjustment creates new PNGs and later innout composition can
restore a stronger shared PNG opinion, `generate_vc_full.py` finishes with
`pipeline/finalize_texture_bindings.py`. This pure-PXR pass scans only effective
composed image attributes, generates BC7/BC5/BC4 DDS files with 4x4 block
padding, and updates the layer that authored each effective value. It avoids
processing weaker historical PNG opinions and avoids writing one redundant
override per object instance. The per-city result is
`<CITY>_final_texture_bindings.json`.

To repair an already generated batch, run the same finalizer over all final
entrypoints:

```bash
python navverse_tools/vc_plus/pipeline/finalize_texture_bindings.py \
  --usd navverse_data/vc_plus/*/*_outdoor.usd \
        navverse_data/vc_plus/*/*_innout.usd \
        navverse_data/grscenes_commercial/scenes/*/start_result_navigation.usd \
  --output-json navverse_data/final_texture_bindings.json \
  --workers 4
```

Review `status_counts.missing_source` in the output manifest. Use `--strict`
when every source image is expected to be present.

Install AMD Compressonator CLI before running final batches:

```bash
curl -fL -o /tmp/compressonatorcli_4.5.52_amd64.deb \
  https://github.com/GPUOpen-Tools/compressonator/releases/download/V4.5.52/compressonatorcli_4.5.52_amd64.deb
sudo apt-get install -y /tmp/compressonatorcli_4.5.52_amd64.deb
sudo ln -sf /usr/compressonatorcli /usr/local/bin/compressonatorcli
```

The Debian package installs a wrapper at `/usr/compressonatorcli`; the pipeline
invokes it through `bash` when needed because that wrapper begins with a leading
blank line before the shebang. Verify with a small PNG-to-DDS compression test,
not just `which compressonatorcli`.

For reversible debugging, use
`--texture-material-op merge_materials_deactivate` instead of pruning. For final
batch output, treat missing DDS files, stale PNG/JPEG references, unresolved
material targets, or undocumented source omissions as hard validation failures.
Because DDS outputs are written next to the source textures, point
`--texture-scene-root` at a copied data root when the original dataset must
remain unchanged.

### QA and smoke policy

Before launching all 52 cities, run three representative cities and inspect the
outputs manually. QA images for every attempted city should be saved under:

```text
navverse_data/vc_qa/<CITY>/
```

The QA set must include at least ten cameras for every outdoor and innout USD.
Use road overviews, vehicle views, sidewalk-prop views, doorway front/oblique
views, and an indoor-to-outdoor view. `render/render_isaac_vc_street_grid.py`
supports both variants with `--variant outdoor|innout`, defaults to ten views,
aims object cameras directly at selected assets, avoids known object footprints
and OSM building footprints, records camera and sightline clearances, and
retries transient black/flat RTX frames. It also records black-frame,
white-frame, and low-contrast metrics. Automatic image metrics are only a first
gate: inspect the contact sheets for scale, spacing, occlusion, facade cuts,
terrain continuity, and indoor/outdoor connectivity. Classify isolated asset or
camera defects as local follow-up work. Treat repeated geometry, placement,
material, or connectivity defects as systemic; fix the pipeline and rebuild all
affected scenes before acceptance. The acceptance checks are:

- road core is 0.2 m below the sidewalk and the 0.2 m curb is continuous;
- semantic TerrainRegions reconstruct the visible terrain without gaps or
  broken triangular islands;
- the cut-building facade meets the exterior doorway plane, the opening bottom
  clears the complete threshold profile, and a closed textured collision
  threshold joins the indoor floor to the exterior surface;
- GR indoor room is behind the opening, has matching floor height, and has a
  robot-width clear corridor to the outside;
- vehicle and object footprints do not overlap, their clearances are respected,
  and their scale looks consistent with the Apr14 VC+ reference scenes;
- texture/material optimization produced valid DDS references and no unresolved
  material targets.

Run the structural audit after generation:

```bash
python navverse_tools/vc_plus/pipeline/audit_vc_plus.py \
  --root "$data_root/vc_plus" \
  --output "$data_root/vc_plus/audits/full_audit.json"
```

For innout connectivity, run `render/check_innout_connectivity_blender.py` with
five width samples, four height samples, a 4 m indoor extent, and a 7 m outdoor
extent. Every city must have at least one complete low-height column; blocked
top-only rays may represent the modeled ceiling and should be reviewed in the
Isaac doorway renders.

Each QA directory also contains `README.txt` and `qa_legend.json`. The rendered
colors are: dark gray for the cut building mesh, yellow for the optional
exterior storefront/facade overlay when explicitly enabled, green translucent
for the door opening/cut-volume marker, and blue translucent for the indoor-room
placement proxy.

Outdoor door width and height use a fixed GR-scene doorway size by default. The
cut depth remains fixed so the doorway reliably reaches the indoor room.
`--fit-door-to-storefront` is available only for storefront-specific QA.

`pipeline/generate_vc_full.py --stage all` also runs the Isaac Sim photo renderers unless
`--skip-scene-photo-render` is passed. Storefront photos are rendered with
`render/render_isaac_storefront_photos.py` at 960x540 for every facade-aligned
storefront to:

```text
navverse_data/scene_data/<CITY>/store/
```

The storefront renderer uses RTX `RayTracedLighting`, the original VC building
materials, and multi-GPU Isaac startup. It disables lights referenced under
`/World/store`, adds only weak neutral fill lights by default
(`--ambient-intensity 1.5`, `--dome-intensity 900`, `--sun-intensity 2500`,
`--headlight-intensity 4500`), and keeps the temporary target-mask material
non-emissive so magenta QA masks do not contaminate final lighting. If a
storefront still looks too white after this, inspect the storefront texture
albedo first; some source storefront PNGs contain large baked-white facade
regions. It tries several street-side camera distances and lateral offsets
(`6,8,12,18,24` meters and `-8,-4,0,4,8` meters) so views include more exterior
building context without reversing the facade normal.
`--preview-building-material` can still be enabled for geometry-only debugging,
but should not be used for acceptance screenshots. During camera selection, the
renderer temporarily masks the target storefront and scores only cameras where
that target is visible, while penalizing mostly blank wall or dark
under-overhang views. Storefront prims are named and tagged with the stable
`index` field from `storefront_plan.json`, so image names, JSON records, and USD
prims can be matched directly. `--allow-flip` exists only for diagnostics; any
`flipped=true` entry or `qa_status=failed` entry in
`isaac_storefront_views.json` should be inspected before acceptance. The
renderer QA is an image-quality and street-context gate, not a pure geometry
gate: a `failed` record may still have a correctly sized, facade-aligned
storefront if the best camera lacks enough street context or is partly occluded.
For geometry acceptance, also check `opening_width`, `opening_height`,
`storefront_offset`, `storefront_asset_height`, and
`storefront_asset_inward_y` in `storefront_plan.json`, plus the corresponding
opening count in `storefront_openings_manifest.json`.

For innout scenes it renders exterior-to-interior and interior-to-exterior
doorway views with `render/render_isaac_exit_views.py`:

```text
navverse_data/scene_data/<CITY>/innout/outside_to_inside.png
navverse_data/scene_data/<CITY>/innout/inside_to_outside.png
```

`compose_vc_full_innout.py` also mounts `green_exit_sign.usd` on the transformed
interior doorway wall, rather than at a fixed offset from the exterior facade.
The placement code projects the source door bbox to find the room-facing wall
surface and probes the composed `/Meshes/Base/ceiling` triangles to find the
actual ceiling underside. The sign is kept between the door and ceiling when
space permits; low-clearance entrances scale it down and may overlap the top of
the doorway slightly. Only the room-facing textured surface is retained, with
a neutral exterior backing so `EXIT` is not visible or mirrored from the street.
Accept the result only after both `exit_sign_close` and
`exit_sign_outer_side` confirm a complete interior label and no exterior label.

Manual storefront rerender example:

```bash
python navverse_tools/vc_plus/render/render_isaac_storefront_photos.py \
  --usd "$data_root/vc_full_generated/MADRID/MADRID_outdoor.usd" \
  --storefront-plan "$data_root/vc_full_generated/MADRID/storefront_plan.json" \
  --output-dir "$data_root/../scene_data/MADRID/store" \
  --max-stores 0 \
  --resolution 960 540 \
  --warmup-frames 4 \
  --camera-distances 6,8,12,18,24 \
  --camera-lateral-offsets=-8,-4,0,4,8
```

For selected storefront QA, pass `--indices 0,1,3` using the stable
`index` fields in `storefront_plan.json`, not the array positions. Use these
photos as quick acceptance images; use the separate QA renders under
`vc_qa/<CITY>/` for detailed cut-volume debugging.

Example full regeneration command for the previously validated 10-city smoke
set:

```bash
data_root="${NAVVERSE_DATA_ROOT:-$PWD/navverse_data}"
python navverse_tools/vc_plus/pipeline/generate_vc_full.py \
  --repo-root "$PWD" \
  --data-root "$data_root" \
  --source-dataset vc_full \
  --work-root "$data_root/vc_full_generated" \
  --store-root "$data_root/storefront" \
  --grscene-root "$data_root/grscenes_commercial" \
  --objaverse-root "$data_root/objaverse" \
  --vc-plus-root "$data_root/vc_apr14" \
  --log-root "$data_root/vc_full_generated/logs" \
  --qa-root "$data_root/vc_qa" \
  --scene-data-root "$data_root/../scene_data" \
  --variant both --stage all --no-skip-existing --min-storefronts 50 \
  --cities COLUMBUS BERLIN AMSTERDAM ZURICH YALE SANFRANCISCO AUSTIN MADRID FLORENCE SANFRANCISCO2 \
  --usd-python "${CONDA_PREFIX}/bin/python" \
  --blender blender \
  --compressonator compressonatorcli
```

Use an official Blender build with USD export support for this smoke run. Some
distribution builds expose `bpy.ops.wm.usd_export` as an attribute but cannot
execute it. `generate_vc_full.py` sets a child-process umask of `002` so USD's
safe-output writer can create valid `.usd`/`.usda` files even when the parent
shell uses a restrictive umask.
