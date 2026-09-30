from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

from .backends import run_native
from .tasks import build_task


INPUT_ROOT = Path("/opt/ml/input/data")


def _extract_channel(name: str) -> Path | None:
    channel = INPUT_ROOT / name
    if not channel.is_dir():
        return None
    files = [path for path in channel.iterdir() if path.is_file()]
    archives = [path for path in files if tarfile.is_tarfile(path) or zipfile.is_zipfile(path)]
    if len(files) == 1 and len(archives) == 1:
        target = channel / "release"
        target.mkdir(exist_ok=True)
        archive = archives[0]
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as handle:
                handle.extractall(target, filter="data")
        else:
            with zipfile.ZipFile(archive) as handle:
                target_root = target.resolve()
                for member in handle.infolist():
                    if target_root not in (target / member.filename).resolve().parents:
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                handle.extractall(target)
        return target
    return channel


def _find_scene_root(root: Path) -> Path:
    if all((root / name).is_dir() for name in ("vc_plus", "grscenes_commercial", "navmesh")):
        return root
    markers = list(root.rglob("NAVVERSE_DATA_RELEASE_*.json"))
    return markers[0].parent if markers else root


def _prepare_uninavid(root: Path) -> Path:
    configs = list(root.rglob("config.json"))
    model_config = None
    payload = None
    for candidate in configs:
        candidate_payload = json.loads(candidate.read_text(encoding="utf-8"))
        if "mm_vision_tower" in candidate_payload:
            model_config, payload = candidate, candidate_payload
            break
    eva_weights = next(iter(root.rglob("eva_vit_g.pth")), None)
    if model_config is None or payload is None or eva_weights is None:
        raise ValueError("UniNaVid input must contain its model config and eva_vit_g.pth")
    payload["mm_vision_tower"] = str(eva_weights)
    payload["image_processor"] = str(
        Path(__file__).resolve().parents[2]
        / "baselines/submodules/Uni-NaVid/uninavid/processor/clip-patch14-224"
    )
    model_config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return model_config.parent


def prepare_config(config: dict) -> dict:
    config["execution"]["backend"] = "native"
    config["execution"]["output_dir"] = "/opt/ml/model"
    channel_paths = {name: _extract_channel(name) for name in ("scenes", "uninavid")}
    if channel_paths["scenes"]:
        scene_root = _find_scene_root(channel_paths["scenes"])
        config["inputs"]["scene_folder"] = str(scene_root)
        config["inputs"]["episode_folder"] = str(scene_root / "episodes")
        config["benchmark"]["scene_folder"] = str(scene_root)
        config["benchmark"]["episode_folder"] = str(scene_root / "episodes")
    if channel_paths["uninavid"]:
        os.environ["UNINAVID_MODEL_PATH"] = str(_prepare_uninavid(channel_paths["uninavid"]))
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    task = build_task(prepare_config(config))
    run_native(task, dry_run=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
