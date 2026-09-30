# Texture Upscaling

`navverse_upscaler.py` provides local and batch utilities for texture
upscaling/conversion. Generated outputs, batch jobs, checkpoints, and external
model workspaces are local artifacts and should not be committed.

## HYPIR Setup

The local HYPIR model checkout is expected at:

```text
navverse_tools/texture_upscaling/HYPIR/
```

Clone it manually when local HYPIR upscaling is needed:

```bash
git clone https://github.com/XPixelGroup/HYPIR.git navverse_tools/texture_upscaling/HYPIR
```

Then install the dependencies and download weights according to the upstream
HYPIR instructions. The default weight path used by the tool is:

```text
navverse_tools/texture_upscaling/HYPIR/weights/HYPIR_sd2.pth
```

The HYPIR checkout and weights are ignored by git.

## Batch Modes

Use the UI for interactive review:

```bash
python navverse_tools/texture_upscaling/navverse_upscaler.py --mode ui
```

Use batch mode for automated processing. Nano Banana modes require a
`GEMINI_API_KEY` in `navverse_tools/texture_upscaling/.env`; DDS conversion uses
the local texture downsampling helper and Compressonator.
