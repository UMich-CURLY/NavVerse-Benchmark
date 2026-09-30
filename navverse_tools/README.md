# NavVerse tools

The release source includes scene preparation, task generation, terrain
validation, reusable metric analysis, and the experimental real-world client.
Large assets, generated outputs and research notebooks are distributed
separately from the source tree.

- `vc_plus/`: outdoor and indoor-to-outdoor composition, navmesh preparation,
  inspection and rendering. See its README for the pipeline stages.
- `blender_virtual_community/` and `3dmodel/`: Blender and USD scene utilities.
- `storefront/`: storefront discovery and placement tools.
- `model_decimation/`, `texture_downsampling/`, `texture_upscaling/`: mesh and
  texture processing. External binaries and model weights are installed
  separately; read each tool's setup instructions.
- `terrain_eval/`: controlled terrain generation and evaluation, with its
  own configurations and tests.
- `statistics/`: metric correction and analysis utilities; supply your own
  result directories.
- `vln_tool/`: instruction-generation tools.
- `real_world_exp_server/`: Python service, mobile web relay and iOS client.
  See its README for supported observation protocols and local operation.

Scene processing tools can modify their input assets. Use an explicit copied
working directory and verify output dependencies before processing a full
collection. Some legacy standalone scripts retain example data paths and must
be configured for the user's dataset layout.

Third-party scenes, textures, objects and models keep their upstream licenses;
see `THIRD_PARTY_NOTICES.md` in the repository root. This candidate does not
grant blanket redistribution rights over processed asset collections.
