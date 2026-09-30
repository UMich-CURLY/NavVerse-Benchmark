# Baselines 
## UniNaVid agent

### 1. Download model
Download Encoder and Finetuned model from:

	https://github.com/CharlesChang012/Uni-NaVid?tab=readme-ov-file#model

Place the model files with the following structure under `baselines/submodules/Uni-NaVid/model_zoo/`:

```
model_zoo/
├── eva_vit_g.pth
└── uninavid-7b-full-224-video-fps-1-grid-2/
```

### 2. Setup conda environment

#### 2.1 Create and activate the conda environment (uses the provided environment_uninavid.yml):

```bash
bash baselines/uninavid/uninavid_setup.sh uninavid
conda activate uninavid
```

#### 2.2 From the repository root, run the UniNaVid baseline agent:

```bash
navverse-run benchmark --preset benchmark_uninavid
```

### Notes
- The setup script installs this repository and `baselines/submodules/Uni-NaVid` as editable packages.
- Ensure the model directory `uninavid-7b-full-224-video-fps-1-grid-2` is accessible at `baselines/submodules/Uni-NaVid/model_zoo/`.
- Setup uses isolated package builds and installs the NavVerse runtime dependencies.
  PyTorch, torchvision and torchaudio use a matching CUDA 12.8 stack.
- The upstream package still declares older Torch versions and bitsandbytes, which
  this full-precision inference setup omits. Those declarations and decord's wheel
  platform metadata currently prevent a clean `pip check`; see the release status.
