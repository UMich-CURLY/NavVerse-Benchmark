ARG FINAL_BASE_IMAGE=customized-isaaclab
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS customized-isaaclab

# Base System Environment
WORKDIR /workspace
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    HOME=/workspace \
    ZDOTDIR=/workspace

# Grouped APT Installations
# Fix: Force libbrotli1 downgrade to resolve sury.org repo conflict.
RUN apt-get update && \
    apt-get install -y --no-install-recommends locales ca-certificates curl wget gnupg gnupg2 software-properties-common git cmake build-essential ninja-build unzip pigz vim tmux zsh python3 python3-dev python3-pip net-tools xz-utils dbus-x11 xvfb nginx supervisor novnc websockify lxde gtk2-engines-murrine gtk2-engines-pixbuf arc-theme libfreetype6-dev libfontconfig1-dev mesa-utils libgl1-mesa-dri x11-xserver-utils libvulkan1 vulkan-tools pv squashfs-tools x11-utils && \
    sed -i 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# GUI Tools: TurboVNC 3.3 Installation
RUN wget -q https://github.com/TurboVNC/turbovnc/releases/download/3.3/turbovnc_3.3_amd64.deb -O /tmp/turbovnc.deb && \
    apt-get update && \
    apt-get install -y /tmp/turbovnc.deb && \
    rm -f /tmp/turbovnc.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
ENV PATH=/opt/TurboVNC/bin:$PATH

# ROS 2 & Zenoh-Bridge Installation
# ROS 2 Humble
RUN add-apt-repository universe && \
    export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F'"' '{print $4}') && \
    curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb" && \
    dpkg -i /tmp/ros2-apt-source.deb
RUN apt-get update && \
    apt-get install -y ros-humble-desktop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Zenoh-Bridge-ROS2DDS (Linux Binary) - Updated to 1.7.2
RUN wget -q https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/1.7.2/zenoh-plugin-ros2dds-1.7.2-x86_64-unknown-linux-gnu-standalone.zip -O /tmp/zenoh.zip && \
    unzip /tmp/zenoh.zip -d /usr/local/bin && \
    rm -f /tmp/zenoh.zip

# Miniconda & GPU Tools
ENV CONDA_DIR=/workspace/conda
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    ${CONDA_DIR}/bin/conda init bash && \
    ${CONDA_DIR}/bin/conda init zsh && \
    ${CONDA_DIR}/bin/conda config --set auto_activate_base false && \
    ${CONDA_DIR}/bin/pip install --no-cache-dir nvitop

# Modular Conda Environments
SHELL ["/bin/zsh", "-lc"]

# navverse env (Main Application)
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n navverse python=3.11 && \
    conda run -n navverse pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision torchaudio && \
    conda run -n navverse pip install "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com && \
    conda run -n navverse pip install git+https://github.com/isaac-sim/rl_games.git@python3.11

# Set default python to navverse env
ENV CONDA_DEFAULT_ENV=navverse \
    PATH=/workspace/conda/envs/navverse/bin:$PATH

# CUDA Environment Paths (inherited from Base Image)
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH=${CUDA_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"

# HPC Compatibility (SINGULARITY_BUILD support)
# ARG SINGULARITY_BUILD=0
# RUN if [ "$SINGULARITY_BUILD" = "1" ]; then chmod -R 777 /workspace /workspace/conda /root; fi

FROM ${FINAL_BASE_IMAGE} AS navverse
SHELL ["/bin/zsh", "-lc"]

# Baseline environments build on top of the warmed Isaac Lab image.
# Uninavid setup
COPY docker/environment_uninavid.yml /tmp/environment_uninavid.yml
RUN sed -i '/^prefix:/d' /tmp/environment_uninavid.yml && \
    PIP_USE_PEP517=1 conda env create -y -f /tmp/environment_uninavid.yml -n uninavid && \
    conda clean -afy && \
    rm -f /tmp/environment_uninavid.yml

# Poliformer setup
RUN conda create -y -n poliformer python=3.10 && \
    conda run -n poliformer pip install --no-cache-dir \
        torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    conda run -n poliformer pip install --no-cache-dir \
        --extra-index-url https://miropsota.github.io/torch_packages_builder \
        "detectron2==0.6+pt2.3.1cu121" && \
    conda run -n poliformer pip install --no-cache-dir \
        --extra-index-url https://ai2thor-pypi.allenai.org \
        "ai2thor==0+966bd7758586e05d18f6181f459c0e90ba318bec" && \
    conda run -n poliformer pip install --no-cache-dir \
        "allenact==0.5.4" "allenact_plugins==0.5.4" && \
    conda run -n poliformer pip install --no-cache-dir \
        "git+https://github.com/openai/CLIP.git" \
        numpy==1.23.5 opencv-python pillow omegaconf \
        timm transformers lightning nltk open3d shapely h5py \
        objathor prior \
        "moviepy==1.0.3" && \
    conda clean -afy

# Poliformer additional packages (fire, wandb, etc.)
RUN conda run -n poliformer pip install --no-cache-dir \
        fire wandb phonemizer stringcase canonicaljson \
        petname scikit-video av shortuuid timeout-decorator \
        open-clip-torch "transformers==4.44.2" && \
    conda run -n poliformer pip install --no-cache-dir \
        torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    conda clean -afy

# Repository code is shared by native Docker runs and the direct SageMaker
# ContainerEntrypoint. Large scenes, checkpoints, and model weights are inputs.
COPY . /opt/navverse
WORKDIR /opt/navverse

RUN conda run -n navverse pip install --no-cache-dir -e . && \
    conda run -n navverse pip install --no-cache-dir boto3 questionary rich omegaconf && \
    conda run -n uninavid pip install --no-cache-dir -e . --no-deps && \
    conda run -n poliformer pip install --no-cache-dir -e . --no-deps && \
    conda run -n navverse bash navverse_tools/install_pyrecastdetour.sh && \
    conda run -n navverse python baselines/eval_navverse.py --help

RUN bash baselines/longnav_r1/setup.sh

# Preserve Isaac Sim's runtime pins after installing baseline dependencies.
RUN conda run -n navverse pip install --no-cache-dir -r scripts/requirements_isaac_compat.txt && \
    conda run -n navverse pip check

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=all \
    NAVVERSE_REPO_ROOT=/opt/navverse
CMD ["/bin/zsh"]
