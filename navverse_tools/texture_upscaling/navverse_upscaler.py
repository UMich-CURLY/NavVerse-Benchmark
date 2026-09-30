"""
NavVerse Texture Processor — viewer + batch upscaling in one script.

Models:
  HYPIR        — local SD2-based super-resolution (GPU required)
  nano-banana-pro / nano-banana-2 — Google Gemini image editing via official API
                                    (requires GEMINI_API_KEY in .env)
  dds          — convert final VC scene texture references to DDS

CLI:
  python navverse_upscaler.py                           # launch UI (default)
  python navverse_upscaler.py --mode ui [--port N]
  python navverse_upscaler.py --mode batch
      [--model hypir|nano-banana-pro|nano-banana-2|dds]
      [--city AMSTERDAM] [--cities AMSTERDAM,BOSTON]
      [--filter all|ground|building] [--resume]
      [--target_size 2048] [--patch_size 512] [--stride 256]
      [--seed 231] [--device cuda]
      [--nb_resolution 1K|2K|4K]
  python navverse_upscaler.py --mode single
      --city AMSTERDAM --file Material_element.001.png
      [--model hypir|nano-banana-pro|nano-banana-2]
      [--prompt "..."] [--target_size 2048] [--patch_size 512]
      [--stride 256] [--device cuda] [--nb_resolution 1K]
  python navverse_upscaler.py --mode clear [--city AMSTERDAM]
"""

import argparse
import base64
import email.utils
import io
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time as _time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from time import time

import gradio as gr
import requests
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from PIL import Image

# ── paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
HYPIR_DIR   = SCRIPT_DIR / "HYPIR"
sys.path.insert(0, str(HYPIR_DIR))

from HYPIR.enhancer.sd2 import SD2Enhancer

DATA_DIR         = SCRIPT_DIR.parent.parent / "navverse_data" / "vc_apr14"
WEIGHT_PATH      = str(HYPIR_DIR / "weights" / "HYPIR_sd2.pth")
BASE_MODEL_PATH  = "Manojb/stable-diffusion-2-1-base"
PROMPTS_DIR      = SCRIPT_DIR / "prompts"
DDS_HELPER_PATH  = SCRIPT_DIR.parent / "texture_downsampling" / "process_material.py"
DDS_VARIANT      = "a_bc7_nomip"
DDS_LAYERS       = ("outdoor", "innout")
BATCH_STATE_FILE = Path("/tmp/navverse_batch.json")
ENV_FILE         = SCRIPT_DIR / ".env"
UI_STATE_FILE    = SCRIPT_DIR / ".navverse_ui_state.json"
JOBS_DIR         = SCRIPT_DIR / "batch_jobs"

LORA_MODULES = [
    "to_k", "to_q", "to_v", "to_out.0",
    "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
    "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"
]

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}


def _retry_after_seconds(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        if parsed is None:
            return None
        return max(parsed.timestamp() - _time.time(), 0.0)


def _response_snippet(resp: requests.Response, limit: int = 300) -> str:
    text = (resp.text or "").replace("\n", " ").strip()
    return text[:limit]


def _gemini_request_with_retry(
    method: str,
    url: str,
    *,
    label: str,
    max_attempts: int = 8,
    base_delay: float = 15.0,
    max_delay: float = 300.0,
    **kwargs,
) -> requests.Response:
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.request(method, url, **kwargs)
        except requests.RequestException as exc:
            if attempt >= max_attempts:
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= random.uniform(0.85, 1.15)
            print(f"  [{label}] request error ({exc}); retry {attempt}/{max_attempts} in {delay:.1f}s")
            _time.sleep(delay)
            continue

        if resp.status_code not in GEMINI_RETRY_STATUS_CODES or attempt >= max_attempts:
            return resp

        delay = _retry_after_seconds(resp)
        if delay is None:
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay *= random.uniform(0.85, 1.15)
        print(
            f"  [{label}] HTTP {resp.status_code}; retry {attempt}/{max_attempts} "
            f"in {delay:.1f}s: {_response_snippet(resp)}"
        )
        _time.sleep(delay)

    raise RuntimeError(f"{label}: exhausted retries")


# UI display name → (CLI id, API model id)
NB_MODELS: dict[str, tuple[str, str]] = {
    "Nano Banana Pro": ("nano-banana-pro", "gemini-3-pro-image-preview"),
    "Nano Banana 2":   ("nano-banana-2",   "gemini-3.1-flash-image-preview"),
}
# CLI model id → API model id
NB_CLI_TO_API: dict[str, str] = {cli: api for cli, api in NB_MODELS.values()}
DDS_MODEL_LABEL = "DDS"
DDS_MODEL_ID = "dds"

# ── .env / API key ─────────────────────────────────────────────────────────────
_gemini_api_key: str | None = None
_gemini_key_status = "⬜ Gemini API key not loaded"


def load_env():
    global _gemini_api_key, _gemini_key_status
    try:
        from dotenv import load_dotenv
        if ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=False)
    except ImportError:
        if ENV_FILE.exists():
            for line in ENV_FILE.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
    _gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if _gemini_api_key:
        _gemini_key_status = "🟢 Gemini API key loaded"
    else:
        _gemini_key_status = "🔴 GEMINI_API_KEY not found in .env or environment"


# ── UI state persistence ───────────────────────────────────────────────────────
def load_ui_state() -> dict:
    try:
        if UI_STATE_FILE.exists():
            return json.loads(UI_STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def save_ui_state(city: str, fname: str):
    try:
        UI_STATE_FILE.write_text(json.dumps({"city": city, "fname": fname}))
    except Exception:
        pass


# ── default prompts ────────────────────────────────────────────────────────────
_DEFAULT_HYPIR_BUILDING = (
    "8k building facade, sharp brick texture, clear windows, concrete panels, "
    "orthographic UV, even lighting, photorealistic, architectural photography"
)
_DEFAULT_HYPIR_GROUND = (
    "8k satellite orthophoto, asphalt road, clear lane markings, crosswalk, "
    "nadir view, no shadows, sharp, urban street texture"
)
_DEFAULT_NB_BUILDING = (
    "High-resolution building facade texture, sharp architectural details, "
    "remove shadows, clean uniform surface, photorealistic, orthographic view, "
    "no occlusion, consistent lighting"
)
_DEFAULT_NB_GROUND = (
    "High-resolution satellite top-down road texture, remove all cast shadows, "
    "clear white lane markings, clean asphalt, nadir view, no distortion, "
    "even lighting, sharp"
)


# ── prompt file I/O ────────────────────────────────────────────────────────────
def _prompt_path(name: str) -> Path:
    return PROMPTS_DIR / f"{name}_prompt.txt"


def load_prompt(name: str, default: str) -> str:
    p = _prompt_path(name)
    if p.exists():
        return p.read_text().strip()
    PROMPTS_DIR.mkdir(exist_ok=True)
    p.write_text(default)
    return default


def load_hypir_prompts() -> tuple[str, str]:
    return (
        load_prompt("building",    _DEFAULT_HYPIR_BUILDING),
        load_prompt("ground",      _DEFAULT_HYPIR_GROUND),
    )


def load_nb_prompts() -> tuple[str, str]:
    return (
        load_prompt("nb_building", _DEFAULT_NB_BUILDING),
        load_prompt("nb_ground",   _DEFAULT_NB_GROUND),
    )


def save_prompt(name: str, text: str):
    PROMPTS_DIR.mkdir(exist_ok=True)
    _prompt_path(name).write_text(text.strip())


def select_prompt(fname: str, prompt_building: str, prompt_ground: str) -> str:
    return prompt_ground if fname.startswith("Material_Ground") else prompt_building


# ── HYPIR model (UI mode) ──────────────────────────────────────────────────────
_model        = None
_model_lock   = threading.Lock()
_model_status = "⬜ HYPIR model not loaded — will load on first HYPIR run"


def _get_hypir_model(device: str = "cuda"):
    global _model, _model_status
    with _model_lock:
        if _model is not None:
            return _model
        try:
            _model_status = "🟡 Loading HYPIR model..."
            print("Loading HYPIR model on first use...")
            m = SD2Enhancer(
                base_model_path=BASE_MODEL_PATH,
                weight_path=WEIGHT_PATH,
                lora_modules=LORA_MODULES,
                lora_rank=256, model_t=200, coeff_t=200, device=device,
            )
            m.init_models()
            _model = m
            _model_status = "🟢 HYPIR model ready"
            print("HYPIR model ready.")
            return _model
        except Exception as e:
            _model_status = f"🔴 HYPIR load failed: {e}"
            print(f"HYPIR model load error: {e}")
            raise


# ── Google Gemini image generation client ──────────────────────────────────────
def _img_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _gemini_generate(model_id: str, prompt: str, resolution: str,
                     image: Image.Image | None = None) -> Image.Image:
    """
    Call the official Gemini generateContent API.
    Endpoint: POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    Auth:      x-goog-api-key header
    Image I/O: inlineData (camelCase, base64 PNG)
    """
    if not _gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to .env")

    parts: list[dict] = [{"text": prompt}]
    if image is not None:
        parts.append({
            "inlineData": {
                "mimeType": "image/png",
                "data": _img_to_b64(image),
            }
        })

    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {
                "imageSize": resolution,   # "1K" | "2K" | "4K"
                "aspectRatio": "1:1",
            },
        },
    }

    resp = _gemini_request_with_retry(
        "POST",
        f"{GEMINI_API_BASE}/{model_id}:generateContent",
        label="NB generate",
        json=body,
        headers={
            "x-goog-api-key": _gemini_api_key,
            "Content-Type": "application/json",
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    # Extract image from response parts (API returns camelCase inlineData)
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates in response: {data}")
    text_parts = []
    for part in candidates[0]["content"]["parts"]:
        img_block = part.get("inlineData") or part.get("inline_data")
        if img_block:
            img_bytes = base64.b64decode(img_block["data"])
            return Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if "text" in part:
            text_parts.append(part["text"])
    model_text = " | ".join(text_parts) if text_parts else "(no text)"
    raise RuntimeError(f"Model returned text instead of image: {model_text}")


def nb_enhance_single(model_id: str, image: Image.Image, prompt: str,
                      resolution: str) -> Image.Image:
    """Single image: one synchronous Gemini generateContent call."""
    print(f"  [NB] calling {model_id} ({resolution})...")
    return _gemini_generate(model_id, prompt, resolution, image)


# ── Gemini Batch API (50% cost vs standard) ────────────────────────────────────
GEMINI_BASE        = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_UPLOAD_BASE = "https://generativelanguage.googleapis.com/upload/v1beta/files"


def _upload_batch_file(content: bytes, display_name: str) -> str:
    """Upload JSONL bytes via Files API resumable upload; return file name."""
    headers_start = {
        "x-goog-api-key": _gemini_api_key,
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(len(content)),
        "X-Goog-Upload-Header-Content-Type": "application/jsonl",
        "Content-Type": "application/json",
    }
    r = _gemini_request_with_retry(
        "POST",
        GEMINI_UPLOAD_BASE,
        label="NB upload-start",
        headers=headers_start,
        json={"file": {"display_name": display_name}}, timeout=30,
    )
    r.raise_for_status()
    upload_url = r.headers["X-Goog-Upload-URL"]

    r2 = _gemini_request_with_retry(
        "POST",
        upload_url,
        label="NB upload-finalize",
        headers={
            "x-goog-api-key": _gemini_api_key,
            "X-Goog-Upload-Command": "upload, finalize",
            "X-Goog-Upload-Offset": "0",
            "Content-Type": "application/jsonl",
        },
        data=content, timeout=300,
    )
    r2.raise_for_status()
    return r2.json()["file"]["name"]   # e.g. "files/abc123"


def _gemini_batch_submit(model_id: str, file_name: str, display_name: str) -> str:
    """Submit batch job referencing uploaded JSONL file; return job name."""
    r = _gemini_request_with_retry(
        "POST",
        f"{GEMINI_BASE}/models/{model_id}:batchGenerateContent",
        label="NB batch-submit",
        json={
            "batch": {
                "displayName": display_name,
                "inputConfig": {"fileName": file_name},
            }
        },
        headers={"x-goog-api-key": _gemini_api_key, "Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["name"]   # e.g. "batches/abc123"


def _gemini_batch_poll(job_name: str, timeout: int = 86400) -> dict:
    """Poll batch job until terminal state; return job data.
    The Gemini Batch API nests state inside metadata:
      data["metadata"]["state"] == "BATCH_STATE_RUNNING" | "BATCH_STATE_SUCCEEDED" | ...
    """
    terminal = {"BATCH_STATE_SUCCEEDED", "BATCH_STATE_FAILED",
                "BATCH_STATE_CANCELLED", "BATCH_STATE_EXPIRED"}
    deadline = _time.time() + timeout
    interval = 15.0
    while _time.time() < deadline:
        r = _gemini_request_with_retry(
            "GET",
            f"{GEMINI_BASE}/{job_name}",
            label="NB batch-poll",
            headers={"x-goog-api-key": _gemini_api_key}, timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        metadata = data.get("metadata", {})
        state = metadata.get("state", "")
        stats = metadata.get("batchStats", {})
        print(f"  [NB batch] state={state} | stats={stats}")
        if state == "BATCH_STATE_SUCCEEDED":
            print(f"  [NB batch] succeeded — full response: {json.dumps(data)}")
            return data
        if state in terminal:
            raise RuntimeError(f"Batch job ended with {state}: {data}")
        _time.sleep(interval)
        interval = min(interval * 1.5, 60)   # back off up to 60s
    raise TimeoutError(f"Batch job timed out after {timeout}s")


def _gemini_batch_results(job_data: dict) -> dict[str, Image.Image | Exception]:
    """Download result JSONL and parse images keyed by request key.
    The result file name is in metadata.outputConfig.fileName (Gemini Batch API).
    """
    metadata = job_data.get("metadata", {})
    result_file = (
        metadata.get("output", {}).get("responsesFile")
        or job_data.get("response", {}).get("responsesFile")
    )
    if not result_file:
        raise RuntimeError(f"No result file in job metadata: {metadata}")
    r = _gemini_request_with_retry(
        "GET",
        f"https://generativelanguage.googleapis.com/download/v1beta/{result_file}:download?alt=media",
        label="NB batch-download",
        headers={"x-goog-api-key": _gemini_api_key}, timeout=300,
    )
    r.raise_for_status()
    out: dict[str, Image.Image | Exception] = {}
    for line in r.text.strip().splitlines():
        if not line.strip():
            continue
        entry = {}
        try:
            entry = json.loads(line)
            key = entry.get("key", "")
            if "error" in entry:
                out[key] = RuntimeError(f"API error: {entry['error']}")
                continue
            response = entry.get("response", {})
            candidates = response.get("candidates", [])
            if not candidates:
                out[key] = RuntimeError(f"No candidates in batch response: {response}")
                continue
            img = None
            text_parts = []
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                finish_reason = candidates[0].get("finishReason") or candidates[0].get("finish_reason")
                out[key] = RuntimeError(
                    f"No parts in batch response candidate"
                    f"{f' (finishReason={finish_reason})' if finish_reason else ''}: {candidates[0]}"
                )
                continue
            for part in parts:
                img_block = part.get("inlineData") or part.get("inline_data")
                if img_block:
                    img = Image.open(
                        io.BytesIO(base64.b64decode(img_block["data"]))
                    ).convert("RGB")
                    break
                if "text" in part:
                    text_parts.append(part["text"])
            if img is None:
                model_text = " | ".join(text_parts) if text_parts else "(no text)"
                out[key] = RuntimeError(f"Model returned text instead of image: {model_text}")
            else:
                out[key] = img
        except Exception as e:
            out[entry.get("key", "?")] = e
    return out


def nb_enhance_batch(model_id: str, items: list[tuple[Image.Image, str]],
                     resolution: str) -> list[Image.Image | Exception]:
    """
    Batch path using official Gemini Batch API (50% cost).
    Flow: build JSONL → upload via Files API → submit batch job →
          poll until done → download & parse result JSONL.
    """
    import datetime
    display_name = f"navverse-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:8]}"

    # Build JSONL in memory
    lines = []
    for i, (img, prompt) in enumerate(items):
        parts: list[dict] = [{"text": prompt}]
        if img is not None:
            parts.append({"inlineData": {"mimeType": "image/png",
                                         "data": _img_to_b64(img)}})
        lines.append(json.dumps({
            "key": str(i),
            "request": {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["IMAGE"],
                    "imageConfig": {"imageSize": resolution, "aspectRatio": "1:1"},
                },
            },
        }))
    jsonl_bytes = "\n".join(lines).encode("utf-8")
    print(f"  [NB batch] uploading {len(jsonl_bytes)/1024:.0f} KB JSONL ({len(items)} requests)...")

    file_name = _upload_batch_file(jsonl_bytes, display_name)
    print(f"  [NB batch] file uploaded: {file_name}")

    job_name = _gemini_batch_submit(model_id, file_name, display_name)
    print(f"  [NB batch] job submitted: {job_name}")

    job_data = _gemini_batch_poll(job_name)
    key_results = _gemini_batch_results(job_data)

    return [key_results.get(str(i), RuntimeError(f"missing key {i}"))
            for i in range(len(items))]


# ── data helpers ───────────────────────────────────────────────────────────────
def get_city_dirs(city: str | None = None, cities: str | None = None) -> list[Path]:
    allowed = None
    if cities:
        allowed = {c.strip().upper() for c in cities.split(",")}
    elif city:
        allowed = {city.strip().upper()}
    return sorted(
        [d for d in DATA_DIR.iterdir()
         if d.is_dir() and d.name.isupper() and (allowed is None or d.name in allowed)],
        key=lambda d: d.name,
    )


def get_city_names() -> list[str]:
    return [d.name for d in get_city_dirs()]


def should_process_source_image(path: Path, max_longest_side: int) -> bool:
    if max_longest_side <= 0:
        return True
    try:
        with Image.open(path) as img:
            return max(img.size) < max_longest_side
    except Exception:
        return True


def apply_file_shard(files: list[Path], args) -> list[Path]:
    shard_count = max(int(getattr(args, "file_shard_count", 1) or 1), 1)
    shard_index = int(getattr(args, "file_shard_index", 0) or 0)
    if shard_count <= 1:
        return files
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"file_shard_index must be in [0, {shard_count}); got {shard_index}")
    return [path for index, path in enumerate(files) if index % shard_count == shard_index]


def build_file_rows(city: str, filter_mode: str) -> list[list[str]]:
    src_dir = DATA_DIR / city / "textures_source"
    out_dir = DATA_DIR / city / "textures"
    gen_dir = DATA_DIR / city / "textures_generated"
    if not src_dir.exists():
        return []
    pngs = sorted(f.name for f in src_dir.iterdir() if f.suffix.lower() == ".png")
    if filter_mode == "ground":
        pngs = [f for f in pngs if f.startswith("Material_Ground")]
    elif filter_mode == "building":
        pngs = [f for f in pngs if f.startswith("Material_") and not f.startswith("Material_Ground")]
    elif filter_mode == "copy-only":
        pngs = [f for f in pngs if not f.startswith("Material_")]

    # Count textures_generated copies per original file
    orig_stems = {Path(f).stem: f for f in pngs}
    gen_counts: dict[str, int] = {}
    if gen_dir.exists():
        for p in gen_dir.iterdir():
            if p.suffix.lower() != ".png":
                continue
            for orig_stem, orig_fname in orig_stems.items():
                if p.stem.startswith(orig_stem + "."):
                    gen_counts[orig_fname] = gen_counts.get(orig_fname, 0) + 1
                    break

    rows = []
    for fname in pngs:
        done    = (out_dir / fname).exists()
        n_gen   = gen_counts.get(fname, 0)
        suffix  = f" ({n_gen})" if n_gen else ""
        rows.append([f"{'✅' if done else '⬜'}  {fname}{suffix}"])
    return rows


def _row_to_fname(row: list) -> str:
    name = row[0].split("  ", 1)[1]
    # Strip trailing " (N)" generated-count annotation
    if name.endswith(")") and " (" in name:
        name = name[:name.rfind(" (")]
    return name


# ── image helpers ──────────────────────────────────────────────────────────────
def backup_and_clear_textures(city_dir: Path):
    src = city_dir / "textures"
    dst = city_dir / "textures_source"
    if not src.exists():
        print(f"  [warn] No textures/ in {city_dir.name}, skipping"); return
    if dst.exists():
        print(f"  [skip] textures_source already exists in {city_dir.name}"); return
    print(f"  Backing up textures → textures_source in {city_dir.name}")
    shutil.copytree(src, dst)
    shutil.rmtree(src); src.mkdir()


def save_png(img: Image.Image, path: Path):
    img.save(path, format="PNG", optimize=True, compress_level=9)


def hypir_enhance(model, lq_pil: Image.Image, prompt: str,
                  target_size: int, patch_size: int, stride: int) -> Image.Image:
    lq_tensor = transforms.ToTensor()(lq_pil).unsqueeze(0)
    w, h = lq_pil.size
    kw = dict(prompt=prompt, patch_size=patch_size, stride=stride, return_type="pil")
    if max(w, h) <= target_size:
        return model.enhance(lq=lq_tensor, scale_by="longest_side",
                             target_longest_side=target_size, **kw)[0]
    else:
        return model.enhance(lq=lq_tensor, scale_by="factor", upscale=1, **kw)[0]


# ── batch subprocess management ────────────────────────────────────────────────
def get_batch_state() -> dict:
    empty = {"running": False, "pid": None, "log": None, "cmd": "", "started": "", "tail": ""}
    if not BATCH_STATE_FILE.exists():
        return empty
    try:
        state = json.loads(BATCH_STATE_FILE.read_text())
        pid = state["pid"]
        log_path = Path(state.get("log", ""))

        # Primary: PID liveness check (also catches zombie processes)
        try:
            os.kill(pid, 0)
            running = True
        except (ProcessLookupError, PermissionError):
            running = False

        # Check for zombie — /proc/{pid}/stat field 3 == 'Z'
        if running:
            try:
                stat = Path(f"/proc/{pid}/stat").read_text()
                proc_state = stat.split("(", 1)[-1].rsplit(")", 1)[-1].strip().split()[0]
                if proc_state == "Z":
                    running = False
                    # Reap the zombie so the PID is released
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
            except FileNotFoundError:
                running = False

        # Secondary: cross-check job record — PID may have been recycled after job ended
        if running:
            job_id = state.get("started")   # ts string doubles as job_id
            if job_id:
                job = _read_job(job_id)
                if job and job.get("status") in ("completed", "failed", "stopped"):
                    running = False

        tail = ""
        if log_path.exists():
            lines = log_path.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-1000:])
        return {**state, "running": running, "tail": tail}
    except Exception as e:
        return {**empty, "tail": f"State read error: {e}"}


def _batch_status_text(state: dict) -> str:
    if state["running"]:
        return f"🟢 Running — PID {state['pid']}  (started {state['started']})"
    if state["pid"] is not None:
        return f"⬜ Finished/stopped — PID {state['pid']}  (started {state['started']})"
    return "⬜ No batch job"


def launch_batch(
    model_choice: str,
    city_mode: str, city_single: str, cities_multi: list,
    filter_mode: str, resume: bool, dry_run: bool,
    city_workers: int,
    target_size: int, patch_size: int, stride: int, seed: int, device: str,
    nb_resolution: str,
) -> tuple[str, str, str, str]:
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"/tmp/navverse_batch_{ts}.log")
    try:
        city_workers_i = max(int(city_workers or 1), 1)
    except (TypeError, ValueError):
        city_workers_i = 1

    if model_choice == DDS_MODEL_LABEL:
        model_id = DDS_MODEL_ID
    elif model_choice in NB_MODELS:
        model_id = NB_MODELS[model_choice][0]   # CLI id, e.g. "nano-banana-2"
    else:
        model_id = "hypir"

    # Determine city list for the record
    if city_mode == "single city" and city_single:
        cities_list = [city_single]
    elif city_mode == "multiple cities" and cities_multi:
        cities_list = list(cities_multi)
    else:
        cities_list = get_city_names()

    # Snapshot prompts at submission time
    if model_id == DDS_MODEL_ID:
        _pb, _pg = "", ""
    else:
        _pb, _pg = (load_hypir_prompts() if model_id == "hypir" else load_nb_prompts())

    # Create job record before spawning subprocess
    job_record: dict = {
        "job_id":       ts,
        "submitted_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "pid":          None,
        "log":          str(log_path),
        "cmd":          None,   # filled in after building cmd
        "status":       "pending",
        "exit_code":    None,
        "started_at":   None,
        "finished_at":  None,
        "error":        None,
        "params": {
            "data_dir":    str(DATA_DIR.resolve()),
            "model":       model_id,
            "api_model":   NB_CLI_TO_API.get(model_id),
            "dds_variant": DDS_VARIANT if model_id == DDS_MODEL_ID else None,
            "cities":      cities_list,
            "filter":      filter_mode,
            "resume":      resume,
            "dry_run":     dry_run,
            "city_workers": city_workers_i,
            "target_size": int(target_size),
            "patch_size":  int(patch_size),
            "stride":      int(stride),
            "seed":        int(seed),
            "device":      device,
            "nb_resolution": nb_resolution,
        },
        "prompts": {"building": _pb, "ground": _pg},
    }
    _write_job(ts, job_record)

    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--mode", "batch",
           "--data-dir", str(DATA_DIR.resolve()),
           "--model", model_id, "--job_id", ts]
    if city_mode == "single city" and city_single:
        cmd += ["--city", city_single]
    elif city_mode == "multiple cities" and cities_multi:
        cmd += ["--cities", ",".join(cities_multi)]
    cmd += [
        "--filter", filter_mode,
        "--target_size", str(int(target_size)),
        "--patch_size",  str(int(patch_size)),
        "--stride",      str(int(stride)),
        "--seed",        str(int(seed)),
        "--device",      device,
        "--nb_resolution", nb_resolution,
        "--city-workers", str(city_workers_i),
    ]
    if resume:
        cmd.append("--resume")
    if dry_run:
        cmd.append("--dry_run")

    job_record["cmd"] = " ".join(cmd)
    _write_job(ts, job_record)

    with open(log_path, "w") as log_fd:
        proc = subprocess.Popen(
            cmd, stdout=log_fd, stderr=subprocess.STDOUT,
            start_new_session=True, cwd=str(SCRIPT_DIR),
        )

    new_state = {"pid": proc.pid, "log": str(log_path), "cmd": " ".join(cmd), "started": ts}
    BATCH_STATE_FILE.write_text(json.dumps(new_state))

    action = f"✅ Batch started (PID {proc.pid}) → {log_path}"
    _time.sleep(0.5)
    summary = get_batch_summary()
    log     = get_job_log(ts)
    return action, summary, log, ts   # ts is the new job_id → select it


def do_stop_batch(selected_job_id: str | None = None) -> str:
    """Stop the selected job, or the most recently started running job."""
    # Try selected job first
    if selected_job_id:
        job = _read_job(selected_job_id)
        if job and job.get("status") == "running" and job.get("pid"):
            pid = int(job["pid"])
            try:
                os.kill(pid, signal.SIGTERM)
                _update_job(selected_job_id, status="stopped")
                return f"⏹ Sent SIGTERM to PID {pid} (job {selected_job_id})"
            except Exception as e:
                return f"Error stopping job {selected_job_id}: {e}"
    # Fallback: stop most recent running job
    for job in get_recent_jobs():
        if job.get("status") == "running" and job.get("pid"):
            pid = int(job["pid"])
            try:
                os.kill(pid, signal.SIGTERM)
                _update_job(job["job_id"], status="stopped")
                return f"⏹ Sent SIGTERM to PID {pid}"
            except Exception as e:
                return f"Error: {e}"
    return "No running batch found"


def poll_batch() -> tuple[str, str]:
    """Legacy compat shim."""
    state = get_batch_state()
    return _batch_status_text(state), state["tail"]


# ── Job record helpers ─────────────────────────────────────────────────────────
_STATUS_ICON = {
    "pending":   "⏳",
    "running":   "🟢",
    "completed": "✅",
    "failed":    "❌",
    "stopped":   "⏹",
}


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _write_job(job_id: str, data: dict):
    JOBS_DIR.mkdir(exist_ok=True)
    _job_path(job_id).write_text(json.dumps(data, indent=2, default=str))


def _read_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _update_job(job_id: str, **kwargs):
    data = _read_job(job_id) or {}
    data.update(kwargs)
    _write_job(job_id, data)


def _pid_is_alive(pid: int) -> bool:
    """True if PID exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, TypeError):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        proc_state = stat.split("(", 1)[-1].rsplit(")", 1)[-1].strip().split()[0]
        if proc_state == "Z":
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
            return False
    except FileNotFoundError:
        return False
    return True


def get_recent_jobs(n: int = 20) -> list[dict]:
    """Return last n job records newest-first, with live PID check to fix stale 'running' status."""
    JOBS_DIR.mkdir(exist_ok=True)
    jobs = []
    for p in sorted(JOBS_DIR.glob("*.json"), reverse=True)[:n]:
        try:
            job = json.loads(p.read_text())
            # If still marked running but PID is gone/zombie, mark stopped
            if job.get("status") == "running" and job.get("pid"):
                if not _pid_is_alive(int(job["pid"])):
                    import datetime as _dt
                    job["status"] = "stopped"
                    job.setdefault("finished_at", _dt.datetime.now().isoformat(timespec="seconds"))
                    p.write_text(json.dumps(job, indent=2, default=str))
            jobs.append(job)
        except Exception:
            pass
    return jobs


def get_batch_summary() -> str:
    """One-line summary of recent job statuses."""
    jobs = get_recent_jobs(20)
    running   = sum(1 for j in jobs if j.get("status") == "running")
    completed = sum(1 for j in jobs if j.get("status") == "completed")
    failed    = sum(1 for j in jobs if j.get("status") == "failed")
    parts = []
    if running:
        parts.append(f"🟢 {running} running")
    if completed:
        parts.append(f"✅ {completed} completed")
    if failed:
        parts.append(f"❌ {failed} failed")
    return " | ".join(parts) if parts else "⬜ No batch jobs"


def get_job_log(job_id: str | None) -> str:
    """Return last 50 lines of the log file for a given job_id."""
    if not job_id:
        return "← Click a row in Job History to view its log"
    job = _read_job(job_id)
    if not job:
        return f"Job record not found: {job_id}"
    log_path = Path(job.get("log", ""))
    if not log_path.exists():
        return f"Log file not found: {log_path}"
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-1000:])


def poll_batch_multi(selected_job_id: str | None) -> tuple[str, str]:
    """Periodic timer tick: return (summary_md, log_for_selected_job)."""
    return get_batch_summary(), get_job_log(selected_job_id)


def jobs_table_data() -> tuple[list[list[str]], list[str]]:
    """Return (rows, job_ids) for the job history Dataframe."""
    rows = []
    ids  = []
    for job in get_recent_jobs():
        icon   = _STATUS_ICON.get(job.get("status", ""), "❓")
        status = f"{icon} {job.get('status', '?')}"
        p      = job.get("params", {})
        cities = ", ".join(p.get("cities", [])) or "all"
        model  = p.get("model", "?")
        if p.get("api_model"):
            model += f" ({p['api_model']})"
        flags  = []
        if p.get("dry_run"):
            flags.append("dry-run")
        if p.get("resume"):
            flags.append("resume")
        if int(p.get("city_workers") or 1) > 1:
            flags.append(f"workers={p['city_workers']}")
        pid    = str(job.get("pid") or "—")
        log    = Path(job.get("log", "")).name or "—"
        err    = (job.get("error") or "")[:80]
        rows.append([
            job.get("submitted_at", "?")[:16],
            status,
            model,
            cities,
            p.get("filter", "all"),
            " ".join(flags) or "—",
            pid,
            log,
            err,
        ])
        ids.append(job.get("job_id", ""))
    return rows, ids


def jobs_table_rows() -> list[list[str]]:
    """Compat shim — returns only rows (for places that don't need ids)."""
    rows, _ = jobs_table_data()
    return rows


# ── HTML / JS (slider) ─────────────────────────────────────────────────────────
def img_to_b64(img: Image.Image, max_px: int = 1200) -> tuple[str, tuple[int, int]]:
    w, h = img.size
    if max(w, h) > max_px:
        scale = max_px / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), img.size


def make_slider_html(src_img: Image.Image | None, out_img: Image.Image | None) -> str:
    if src_img is None or out_img is None:
        return "<div style='padding:40px;text-align:center;color:#888'>Select a processed file to compare</div>"
    out_b64, (dw, dh) = img_to_b64(out_img)
    src_display = src_img.convert("RGB").resize((dw, dh), Image.NEAREST)
    buf = BytesIO(); src_display.save(buf, format="JPEG", quality=88)
    src_b64 = base64.b64encode(buf.getvalue()).decode()
    uid = uuid.uuid4().hex[:8]
    return f"""<div data-sid="{uid}" style="position:relative;width:100%;line-height:0;touch-action:none;user-select:none;cursor:col-resize">
  <img src="data:image/jpeg;base64,{out_b64}" draggable="false" style="display:block;width:100%;height:auto;pointer-events:none">
  <div class="s-before" style="position:absolute;inset:0;width:50%;overflow:hidden;pointer-events:none">
    <img class="s-before-img" src="data:image/jpeg;base64,{src_b64}" draggable="false" style="display:block;width:{dw}px;max-width:none;height:auto">
  </div>
  <div class="s-line" style="position:absolute;top:0;left:50%;width:3px;height:100%;background:#fff;transform:translateX(-50%);pointer-events:none;box-shadow:0 0 8px rgba(0,0,0,.6)">
    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:44px;height:44px;background:#fff;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:20px;color:#333;box-shadow:0 2px 10px rgba(0,0,0,.4)">⟺</div>
  </div>
  <span style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,.6);color:#fff;padding:3px 10px;border-radius:4px;font-size:13px;pointer-events:none">Before</span>
  <span style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,.6);color:#fff;padding:3px 10px;border-radius:4px;font-size:13px;pointer-events:none">After</span>
</div>"""


SLIDER_JS = """() => {
  function initSlider(w) {
    if (w._sidInited) return;
    w._sidInited = true;
    var b  = w.querySelector('.s-before');
    var bi = w.querySelector('.s-before-img');
    var l  = w.querySelector('.s-line');
    if (!b || !bi || !l) return;
    function syncW() { bi.style.width = w.offsetWidth + 'px'; }
    new ResizeObserver(syncW).observe(w); syncW();
    function move(cx) {
      var r = w.getBoundingClientRect();
      var p = Math.max(0, Math.min(1, (cx - r.left) / r.width));
      b.style.width = (p * 100) + '%'; l.style.left = (p * 100) + '%';
    }
    w.addEventListener('pointerdown', function(e) {
      w.setPointerCapture(e.pointerId); move(e.clientX); e.preventDefault();
    });
    w.addEventListener('pointermove', function(e) { if (e.buttons > 0) move(e.clientX); });
  }
  function scan(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('[data-sid]').forEach(initSlider);
  }
  new MutationObserver(function(muts) {
    muts.forEach(function(m) {
      m.addedNodes.forEach(function(n) {
        if (n.nodeType !== 1) return;
        if (n.getAttribute && n.getAttribute('data-sid')) initSlider(n);
        scan(n);
      });
    });
  }).observe(document.body, {childList: true, subtree: true});
  scan(document.body);
}"""

CSS = """
.left-col { min-width: 280px; max-width: 320px; }
.left-col > div { gap: 6px !important; }
.file-list { height: 600px; overflow-y: auto; }
.log-box textarea { font-family: monospace !important; font-size: 12px !important; }
"""

MODEL_CHOICES = ["HYPIR", "Nano Banana Pro", "Nano Banana 2"]
BATCH_MODEL_CHOICES = MODEL_CHOICES + [DDS_MODEL_LABEL]


# ── model copy helper ─────────────────────────────────────────────────────────
def save_model_copy(result: Image.Image, city_dir: Path, fname: str, model_tag: str):
    """Save a model-output copy to textures_generated/ with a tagged filename.

    E.g. fname='Material_building.001.png', model_tag='nano-banana-2.2K'
    → saves city_dir/textures_generated/Material_building.001.nano-banana-2.2K.png
    """
    gen_dir = city_dir / "textures_generated"
    gen_dir.mkdir(exist_ok=True)
    stem = Path(fname).stem
    copy_name = f"{stem}.{model_tag}.png"
    try:
        save_png(result, gen_dir / copy_name)
    except Exception as e:
        print(f"  [warn] Could not save model copy {copy_name}: {e}")


# ── version helpers ────────────────────────────────────────────────────────────
def get_file_versions(city: str, fname: str) -> list[str]:
    """Return ordered list of version labels available for fname in city.

    Labels:
      "Original"       → textures_source/{fname}              (always first)
      "Current output" → textures/{fname}                     (if exists)
      "method.res"     → textures_generated/{stem}.method.res.png (each generated copy)
    """
    versions: list[str] = ["Original"]
    out_dir = DATA_DIR / city / "textures"
    gen_dir = DATA_DIR / city / "textures_generated"
    stem = Path(fname).stem
    if (out_dir / fname).exists():
        versions.append("Current output")
    if gen_dir.exists():
        for p in sorted(gen_dir.glob(f"{stem}.*.png")):
            tag = p.stem[len(stem) + 1:]   # everything after "stem."
            if tag:
                versions.append(tag)
    return versions


def load_version_image(city: str, fname: str, version: str | None) -> Image.Image | None:
    if not version:
        return None
    if version == "Original":
        p = DATA_DIR / city / "textures_source" / fname
    elif version == "Current output":
        p = DATA_DIR / city / "textures" / fname
    else:
        stem = Path(fname).stem
        p = DATA_DIR / city / "textures_generated" / f"{stem}.{version}.png"
    try:
        return Image.open(p).convert("RGB") if p.exists() else None
    except Exception:
        return None


# ── UI event handlers ──────────────────────────────────────────────────────────
def on_file_select(evt: gr.SelectData, city: str, rows_state: list) -> tuple:
    if not rows_state or evt.index[0] >= len(rows_state):
        return None, None, make_slider_html(None, None), "No file selected"
    fname = _row_to_fname(rows_state[evt.index[0]])
    src_path = DATA_DIR / city / "textures_source" / fname
    out_path = DATA_DIR / city / "textures" / fname
    src_img = Image.open(src_path).convert("RGB") if src_path.exists() else None
    out_img = Image.open(out_path).convert("RGB") if out_path.exists() else None
    src_size = f"{src_img.size[0]}×{src_img.size[1]}" if src_img else "missing"
    out_size = f"{out_img.size[0]}×{out_img.size[1]}" if out_img else "not processed"
    info = f"{fname}  |  Before: {src_size}  →  After: {out_size}"
    return src_img, out_img, make_slider_html(src_img, out_img), info


def on_model_change(model_choice: str):
    """Toggle visibility of HYPIR vs NB controls and reload prompt."""
    is_hypir = (model_choice == "HYPIR")
    pb_hypir, pg_hypir = load_hypir_prompts()
    pb_nb,    pg_nb    = load_nb_prompts()
    return (
        gr.update(visible=is_hypir),      # hypir_settings_row
        gr.update(visible=not is_hypir),  # nb_settings_row
        gr.update(value=pb_hypir if is_hypir else pb_nb),  # prompt_box default
    )


def on_template_change(template: str, model_choice: str):
    pb_hypir, pg_hypir = load_hypir_prompts()
    pb_nb,    pg_nb    = load_nb_prompts()
    is_hypir = (model_choice == "HYPIR")
    if template == "building (facade)":
        return gr.update(value=pb_hypir if is_hypir else pb_nb)
    if template == "ground (road/satellite)":
        return gr.update(value=pg_hypir if is_hypir else pg_nb)
    return gr.update()


def run_enhance_ui(
    city, rows_state, sel_idx,
    model_choice, prompt,
    target_size, patch_size, stride, seed,
    nb_resolution,
) -> tuple:
    _no_file = (None, None, make_slider_html(None, None),
                "No file selected", gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False))
    if sel_idx is None or not rows_state:
        return _no_file
    fname = _row_to_fname(rows_state[sel_idx])
    src_path = DATA_DIR / city / "textures_source" / fname
    if not src_path.exists():
        return (None, None, make_slider_html(None, None), "Source not found",
                gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False))

    lq_pil = Image.open(src_path).convert("RGB")
    out_path = DATA_DIR / city / "textures" / fname
    out_path.parent.mkdir(exist_ok=True)

    try:
        if model_choice == "HYPIR":
            m = _get_hypir_model()
            if int(seed) >= 0:
                set_seed(int(seed))
            result = hypir_enhance(m, lq_pil, prompt, int(target_size), int(patch_size), int(stride))
            tag = f"HYPIR {lq_pil.size[0]}×{lq_pil.size[1]} → {result.size[0]}×{result.size[1]}"
            model_tag = f"hypir.{int(target_size)}"
        else:
            cli_id, api_model_id = NB_MODELS[model_choice]
            result = nb_enhance_single(api_model_id, lq_pil, prompt, nb_resolution)
            tag = f"{model_choice} {nb_resolution}"
            model_tag = f"{cli_id}.{nb_resolution}"
    except Exception as e:
        return (None, None, make_slider_html(None, None), f"Error: {e}",
                gr.update(), gr.update(), gr.update(visible=False), gr.update(visible=False))

    save_png(result, out_path)
    save_model_copy(result, DATA_DIR / city, fname, model_tag)

    # Refresh version dropdowns — right auto-switches to "Current output"
    versions = get_file_versions(city, fname)
    info = f"{fname}  |  {tag}  ✅"
    return (lq_pil, result, make_slider_html(lq_pil, result), info,
            gr.update(choices=versions, value="Original"),
            gr.update(choices=versions, value="Current output"),
            gr.update(visible=False),
            gr.update(visible=False, value=""))


# ── build UI ───────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    city_names   = get_city_names()
    pb_hypir, pg_hypir = load_hypir_prompts()
    pb_nb,    pg_nb    = load_nb_prompts()

    # ── Restore last-used city + file ─────────────────────────────────────────
    _saved      = load_ui_state()
    default_city = (
        _saved["city"] if _saved.get("city") in city_names
        else (city_names[0] if city_names else "")
    )
    default_rows = build_file_rows(default_city, "all") if default_city else []

    # Compute initial Browse-tab state for saved file (if still valid)
    _saved_fname   = _saved.get("fname", "")
    _init_sel_idx: int | None = None
    _init_fname    = ""
    _init_versions = ["Original"]
    _init_right_ver: str | None = None
    _init_left_img = None
    _init_right_img = None
    _init_slider   = make_slider_html(None, None)
    _init_file_info = ""
    _init_prompt   = pb_hypir
    _init_template = "building (facade)"

    for _i, _row in enumerate(default_rows):
        if _row_to_fname(_row) == _saved_fname:
            _init_sel_idx   = _i
            _init_fname     = _saved_fname
            _init_versions  = get_file_versions(default_city, _saved_fname)
            _init_right_ver = (
                "Current output" if "Current output" in _init_versions
                else _init_versions[-1] if len(_init_versions) > 1 else _init_versions[0]
            )
            _init_left_img  = load_version_image(default_city, _saved_fname, "Original")
            _init_right_img = load_version_image(default_city, _saved_fname, _init_right_ver)
            _out_img        = load_version_image(default_city, _saved_fname, "Current output")
            _src_sz = (f"{_init_left_img.size[0]}×{_init_left_img.size[1]}"
                       if _init_left_img else "missing")
            _out_sz = (f"{_out_img.size[0]}×{_out_img.size[1]}"
                       if _out_img else "not processed")
            _init_file_info = (f"{_saved_fname}  |  Original: {_src_sz}  →  Current: {_out_sz}")
            _init_slider    = make_slider_html(_init_left_img, _init_right_img)
            _is_ground      = _saved_fname.startswith("Material_Ground")
            _init_prompt    = pg_hypir if _is_ground else pb_hypir
            _init_template  = "ground (road/satellite)" if _is_ground else "building (facade)"
            break

    with gr.Blocks(title="NavVerse Texture Processor", js=SLIDER_JS, css=CSS) as demo:
        gr.Markdown("## NavVerse Texture Processor")

        with gr.Row():
            model_status_md  = gr.Markdown(_model_status)
            gemini_status_md = gr.Markdown(_gemini_key_status)
        _model_timer = gr.Timer(value=2, active=True)
        _model_timer.tick(lambda: (_model_status, _gemini_key_status),
                          outputs=[model_status_md, gemini_status_md])

        with gr.Tabs():

            # ── Tab 1: Browse ────────────────────────────────────────────────
            with gr.Tab("🖼 Browse"):
                rows_state      = gr.State(default_rows)
                sel_idx_state   = gr.State(_init_sel_idx)
                sel_fname_state = gr.State(_init_fname)

                with gr.Row():
                    # Left: file list
                    with gr.Column(scale=1, min_width=280, elem_classes=["left-col"]):
                        city_dd   = gr.Dropdown(choices=city_names, value=default_city, label="City")
                        filter_dd = gr.Dropdown(
                            choices=["all", "ground", "building", "copy-only"],
                            value="all", label="Filter",
                        )
                        refresh_btn = gr.Button("🔄 Refresh", size="sm")
                        done_count  = sum(1 for r in default_rows if r[0].startswith("✅"))
                        file_df = gr.Dataframe(
                            headers=["Filename"], datatype=["str"],
                            value=default_rows,
                            label=f"Files — {done_count}/{len(default_rows)} processed",
                            interactive=False, wrap=False,
                            elem_classes=["file-list"],
                        )

                    # Right: image display
                    with gr.Column(scale=3):
                        file_info = gr.Textbox(
                            value=_init_file_info, label="Selected file",
                            interactive=False, lines=1,
                        )

                        # Version selectors
                        _init_is_copy = _init_right_ver not in ("Original", "Current output", None)
                        with gr.Row():
                            left_ver_dd = gr.Dropdown(
                                choices=_init_versions, value="Original",
                                label="Left image", scale=2, interactive=True,
                            )
                            right_ver_dd = gr.Dropdown(
                                choices=_init_versions, value=_init_right_ver,
                                label="Right image", scale=2, interactive=True,
                            )
                            with gr.Column(scale=1, min_width=160):
                                set_current_btn = gr.Button(
                                    "🔄 Set as Current",
                                    visible=_init_is_copy, variant="primary",
                                )
                                set_current_st = gr.Textbox(
                                    interactive=False, lines=1, label="",
                                    visible=_init_is_copy,
                                )

                        with gr.Tabs():
                            with gr.Tab("⟺ Slider"):
                                slider_html = gr.HTML(_init_slider)
                            with gr.Tab("◫ Side by Side"):
                                with gr.Row():
                                    before_img = gr.Image(
                                        value=_init_left_img, label="Left", type="pil",
                                        interactive=False, show_download_button=True,
                                    )
                                    after_img  = gr.Image(
                                        value=_init_right_img, label="Right", type="pil",
                                        interactive=False, show_download_button=True,
                                    )

                        with gr.Accordion("▶ Re-process", open=False):
                            model_rd = gr.Radio(
                                choices=MODEL_CHOICES, value="HYPIR", label="Model",
                            )
                            template_dd = gr.Dropdown(
                                choices=["building (facade)", "ground (road/satellite)", "custom"],
                                value=_init_template, label="Prompt template",
                            )
                            prompt_box = gr.Textbox(value=_init_prompt, label="Prompt", lines=5)

                            # HYPIR-specific controls
                            with gr.Row(visible=True) as hypir_settings_row:
                                target_sl = gr.Slider(512, 4096, value=2048, step=256,
                                                      label="Target longest side")
                                patch_sl  = gr.Slider(256, 1024, value=512,  step=128,
                                                      label="Patch size")
                                stride_sl = gr.Slider(128, 512,  value=256,  step=128,
                                                      label="Stride")
                                seed_num  = gr.Number(value=-1, label="Seed (−1=random)")

                            # NB-specific controls
                            with gr.Row(visible=False) as nb_settings_row:
                                nb_res_dd = gr.Dropdown(
                                    choices=["1K", "2K", "4K"], value="2K",
                                    label="Output resolution (1:1)",
                                )

                            process_btn = gr.Button("▶ Run", variant="primary")

                # Browse wiring
                def _refresh(city, fmode):
                    rows = build_file_rows(city, fmode)
                    done = sum(1 for r in rows if r[0].startswith("✅"))
                    save_ui_state(city, "")
                    return gr.update(value=rows, label=f"Files — {done}/{len(rows)} processed"), rows, None

                for trigger in [city_dd.change, filter_dd.change, refresh_btn.click]:
                    trigger(_refresh, [city_dd, filter_dd], [file_df, rows_state, sel_idx_state])

                def _select(evt: gr.SelectData, city, rows, model_choice):
                    _empty = (None, None, make_slider_html(None, None), "No file selected",
                              None, "",
                              gr.update(choices=["Original"], value="Original"),
                              gr.update(choices=[], value=None),
                              gr.update(visible=False), gr.update(visible=False),
                              gr.update(), gr.update())
                    if not rows or evt.index[0] >= len(rows):
                        return _empty
                    fname = _row_to_fname(rows[evt.index[0]])
                    save_ui_state(city, fname)

                    versions = get_file_versions(city, fname)
                    default_right = (
                        "Current output" if "Current output" in versions
                        else versions[-1] if len(versions) > 1 else versions[0]
                    )
                    left_img  = load_version_image(city, fname, "Original")
                    right_img = load_version_image(city, fname, default_right)
                    out_img   = load_version_image(city, fname, "Current output")
                    src_size  = f"{left_img.size[0]}×{left_img.size[1]}" if left_img else "missing"
                    out_size  = f"{out_img.size[0]}×{out_img.size[1]}" if out_img else "not processed"
                    info = f"{fname}  |  Original: {src_size}  →  Current: {out_size}"
                    is_copy = default_right not in ("Original", "Current output", None)

                    # Auto-select prompt and template based on filename
                    _pb_h, _pg_h = load_hypir_prompts()
                    _pb_n, _pg_n = load_nb_prompts()
                    is_hypir  = (model_choice == "HYPIR")
                    is_ground = fname.startswith("Material_Ground")
                    auto_prompt   = (_pg_h if is_ground else _pb_h) if is_hypir else (_pg_n if is_ground else _pb_n)
                    auto_template = "ground (road/satellite)" if is_ground else "building (facade)"

                    return (left_img, right_img, make_slider_html(left_img, right_img), info,
                            evt.index[0], fname,
                            gr.update(choices=versions, value="Original"),
                            gr.update(choices=versions, value=default_right),
                            gr.update(visible=is_copy),
                            gr.update(visible=is_copy),
                            gr.update(value=auto_prompt),
                            gr.update(value=auto_template))

                def _update_prompt_template(fname, model_choice):
                    """Force prompt_box and template_dd update after _select completes.
                    Runs via .then() so it fires AFTER any template_dd.change chain,
                    ensuring the correct prompt is always applied on file selection.
                    """
                    if not fname:
                        return gr.update(), gr.update()
                    _pb_h, _pg_h = load_hypir_prompts()
                    _pb_n, _pg_n = load_nb_prompts()
                    is_hypir  = (model_choice == "HYPIR")
                    is_ground = fname.startswith("Material_Ground")
                    prompt   = (_pg_h if is_ground else _pb_h) if is_hypir else (_pg_n if is_ground else _pb_n)
                    template = "ground (road/satellite)" if is_ground else "building (facade)"
                    return gr.update(value=template), gr.update(value=prompt)

                file_df.select(_select, [city_dd, rows_state, model_rd],
                               [before_img, after_img, slider_html, file_info,
                                sel_idx_state, sel_fname_state,
                                left_ver_dd, right_ver_dd,
                                set_current_btn, set_current_st,
                                prompt_box, template_dd]
                ).then(_update_prompt_template, [sel_fname_state, model_rd],
                       [template_dd, prompt_box])

                def _ver_change(city, fname, left_ver, right_ver):
                    if not fname:
                        return (None, None, make_slider_html(None, None),
                                gr.update(visible=False), gr.update(visible=False))
                    l = load_version_image(city, fname, left_ver)
                    r = load_version_image(city, fname, right_ver)
                    is_copy = right_ver not in ("Original", "Current output", None, "")
                    return (l, r, make_slider_html(l, r),
                            gr.update(visible=is_copy), gr.update(visible=is_copy))

                for _vd in [left_ver_dd, right_ver_dd]:
                    _vd.change(_ver_change,
                               [city_dd, sel_fname_state, left_ver_dd, right_ver_dd],
                               [before_img, after_img, slider_html, set_current_btn, set_current_st])

                def _set_current(city, fname, right_ver, left_ver):
                    if not fname or right_ver in ("Original", "Current output", None, ""):
                        return "Nothing to set.", gr.update(), None, None, make_slider_html(None, None)
                    stem = Path(fname).stem
                    src = DATA_DIR / city / "textures_generated" / f"{stem}.{right_ver}.png"
                    dst = DATA_DIR / city / "textures" / fname
                    if not src.exists():
                        return f"❌ Not found: {src.name}", gr.update(), None, None, make_slider_html(None, None)
                    dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(src, dst)
                    l = load_version_image(city, fname, left_ver)
                    r = load_version_image(city, fname, "Current output")
                    return (f"✅ Set '{right_ver}' as current output",
                            gr.update(value="Current output"),
                            l, r, make_slider_html(l, r))

                set_current_btn.click(
                    _set_current,
                    [city_dd, sel_fname_state, right_ver_dd, left_ver_dd],
                    [set_current_st, right_ver_dd, before_img, after_img, slider_html],
                )

                model_rd.change(on_model_change, [model_rd],
                                [hypir_settings_row, nb_settings_row, prompt_box])

                template_dd.change(on_template_change, [template_dd, model_rd], [prompt_box])

                process_btn.click(
                    run_enhance_ui,
                    [city_dd, rows_state, sel_idx_state,
                     model_rd, prompt_box,
                     target_sl, patch_sl, stride_sl, seed_num,
                     nb_res_dd],
                    [before_img, after_img, slider_html, file_info,
                     left_ver_dd, right_ver_dd, set_current_btn, set_current_st],
                )

            # ── Tab 2: Batch ─────────────────────────────────────────────────
            with gr.Tab("⚙ Batch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Configuration")
                        batch_model_rd = gr.Radio(
                            choices=BATCH_MODEL_CHOICES, value="HYPIR", label="Model",
                        )
                        city_mode_rd = gr.Radio(
                            choices=["all cities", "single city", "multiple cities"],
                            value="all cities", label="City selection",
                        )
                        city_single_dd = gr.Dropdown(
                            choices=city_names, value=None, label="City", visible=False,
                        )
                        cities_multi_cb = gr.CheckboxGroup(
                            choices=city_names, value=[], label="Cities", visible=False,
                        )
                        batch_filter_dd = gr.Dropdown(
                            choices=["all", "ground", "building"],
                            value="all", label="Filter (texture type)",
                        )
                        resume_cb  = gr.Checkbox(value=True,  label="Resume — skip already processed")
                        city_workers_num = gr.Number(
                            value=1, minimum=1, precision=0,
                            label="City workers",
                        )

                        with gr.Group(visible=True) as batch_hypir_group:
                            b_target  = gr.Slider(512, 4096, value=2048, step=256,
                                                  label="Target longest side")
                            with gr.Row():
                                b_patch  = gr.Slider(256, 1024, value=512, step=128, label="Patch size")
                                b_stride = gr.Slider(128, 512,  value=256, step=128, label="Stride")
                            with gr.Row():
                                b_seed   = gr.Number(value=231, label="Seed")
                                b_device = gr.Textbox(value="cuda", label="Device")

                        with gr.Group(visible=False) as batch_nb_group:
                            b_nb_res = gr.Dropdown(
                                choices=["1K", "2K", "4K"], value="2K",
                                label="Output resolution (1:1)",
                            )

                        with gr.Row():
                            start_btn   = gr.Button("▶ Start Batch", variant="primary")
                            dry_run_btn = gr.Button("🔍 Dry Run",    variant="secondary")
                            stop_btn    = gr.Button("⏹ Stop",        variant="stop")
                        action_out = gr.Textbox(label="", interactive=False, lines=1)

                    with gr.Column(scale=2):
                        gr.Markdown("### Status")
                        _init_summary = get_batch_summary()
                        batch_status_md = gr.Markdown(_init_summary)
                        batch_log_box = gr.Textbox(
                            value="← Click a row in Job History to view its log",
                            label="Log (last 50 lines)", lines=18,
                            interactive=False, elem_classes=["log-box"],
                        )
                        selected_job_state = gr.State(None)
                        _batch_timer = gr.Timer(value=1, active=True)
                        _batch_timer.tick(poll_batch_multi, [selected_job_state],
                                          [batch_status_md, batch_log_box])

                        gr.Markdown("### Job History  *(click a row to view its log)*")
                        _JOB_COLS = ["Submitted", "Status", "Model", "Cities",
                                     "Filter", "Flags", "PID", "Log file", "Error"]
                        _init_rows, _init_ids = jobs_table_data()
                        job_ids_state = gr.State(_init_ids)
                        jobs_df = gr.Dataframe(
                            value=_init_rows,
                            headers=_JOB_COLS,
                            datatype=["str"] * len(_JOB_COLS),
                            interactive=False, wrap=True,
                            label="Recent batch jobs (auto-refreshes every 1 s)",
                        )
                        _jobs_timer = gr.Timer(value=1, active=True)
                        _jobs_timer.tick(jobs_table_data, outputs=[jobs_df, job_ids_state])

                # Batch wiring
                def _batch_model_change(m):
                    is_h = (m == "HYPIR")
                    is_nb = m in NB_MODELS
                    return gr.update(visible=is_h), gr.update(visible=is_nb)

                batch_model_rd.change(_batch_model_change, [batch_model_rd],
                                      [batch_hypir_group, batch_nb_group])

                def _city_mode_change(mode):
                    return (gr.update(visible="single" in mode),
                            gr.update(visible="multiple" in mode))

                city_mode_rd.change(_city_mode_change, [city_mode_rd],
                                    [city_single_dd, cities_multi_cb])

                _batch_inputs = [batch_model_rd,
                                 city_mode_rd, city_single_dd, cities_multi_cb,
                                 batch_filter_dd, resume_cb, city_workers_num,
                                 b_target, b_patch, b_stride, b_seed, b_device,
                                 b_nb_res]
                _batch_outputs = [action_out, batch_status_md, batch_log_box, selected_job_state]

                def _start(bm, mode, cs, cm, filt, res, workers, tgt, patch, stride, seed, dev, nbr):
                    action, summary, log, job_id = launch_batch(
                        bm, mode, cs, cm, filt, res, dry_run=False,
                        city_workers=workers,
                        target_size=tgt, patch_size=patch, stride=stride,
                        seed=seed, device=dev, nb_resolution=nbr,
                    )
                    return action, summary, log, job_id

                def _dry_run(bm, mode, cs, cm, filt, res, workers, tgt, patch, stride, seed, dev, nbr):
                    action, summary, log, job_id = launch_batch(
                        bm, mode, cs, cm, filt, res, dry_run=True,
                        city_workers=workers,
                        target_size=tgt, patch_size=patch, stride=stride,
                        seed=seed, device=dev, nb_resolution=nbr,
                    )
                    return action, summary, log, job_id

                start_btn.click(_start, _batch_inputs, _batch_outputs)
                dry_run_btn.click(_dry_run, _batch_inputs, _batch_outputs)
                stop_btn.click(do_stop_batch, [selected_job_state], [action_out])

                def _on_job_select(evt: gr.SelectData, job_ids: list):
                    if not job_ids or evt.index[0] >= len(job_ids):
                        return None, ""
                    job_id = job_ids[evt.index[0]]
                    return job_id, get_job_log(job_id)

                jobs_df.select(_on_job_select, [job_ids_state],
                               [selected_job_state, batch_log_box])

            # ── Tab 3: Prompts ───────────────────────────────────────────────
            with gr.Tab("📝 Prompts"):
                with gr.Tabs():
                    with gr.Tab("HYPIR"):
                        gr.Markdown(
                            "> CLIP 75-token limit — keep prompts under ~60 words, "
                            "comma-separated nouns/adjectives, no instructions."
                        )
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Building\n*`Material_*` excluding Ground*")
                                h_building_box = gr.Textbox(value=pb_hypir, lines=8, label="HYPIR building prompt")
                                h_save_b_btn   = gr.Button("💾 Save", variant="primary")
                                h_save_b_st    = gr.Textbox(interactive=False, lines=1, label="")
                            with gr.Column():
                                gr.Markdown("### Ground\n*`Material_Ground*`*")
                                h_ground_box   = gr.Textbox(value=pg_hypir, lines=8, label="HYPIR ground prompt")
                                h_save_g_btn   = gr.Button("💾 Save", variant="primary")
                                h_save_g_st    = gr.Textbox(interactive=False, lines=1, label="")

                        h_save_b_btn.click(
                            lambda t: (save_prompt("building", t), "✅ Saved")[1],
                            [h_building_box], [h_save_b_st],
                        )
                        h_save_g_btn.click(
                            lambda t: (save_prompt("ground", t), "✅ Saved")[1],
                            [h_ground_box], [h_save_g_st],
                        )

                    with gr.Tab("Nano Banana"):
                        gr.Markdown(
                            "> Nano Banana uses a full language model — prompts can be descriptive "
                            "and instruction-like (e.g. 'remove shadows, enhance texture'). "
                            "No token limit concerns."
                        )
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Building\n*`Material_*` excluding Ground*")
                                nb_building_box = gr.Textbox(value=pb_nb, lines=8, label="NB building prompt")
                                nb_save_b_btn   = gr.Button("💾 Save", variant="primary")
                                nb_save_b_st    = gr.Textbox(interactive=False, lines=1, label="")
                            with gr.Column():
                                gr.Markdown("### Ground\n*`Material_Ground*`*")
                                nb_ground_box   = gr.Textbox(value=pg_nb, lines=8, label="NB ground prompt")
                                nb_save_g_btn   = gr.Button("💾 Save", variant="primary")
                                nb_save_g_st    = gr.Textbox(interactive=False, lines=1, label="")

                        nb_save_b_btn.click(
                            lambda t: (save_prompt("nb_building", t), "✅ Saved")[1],
                            [nb_building_box], [nb_save_b_st],
                        )
                        nb_save_g_btn.click(
                            lambda t: (save_prompt("nb_ground", t), "✅ Saved")[1],
                            [nb_ground_box], [nb_save_g_st],
                        )

    return demo


# ── CLI processing modes ───────────────────────────────────────────────────────
def _effective_city_workers(args, model_id: str, city_count: int) -> int:
    try:
        requested = int(getattr(args, "city_workers", 1) or 1)
    except (TypeError, ValueError):
        requested = 1
    requested = max(requested, 1)
    if model_id == "hypir" and requested > 1:
        print("  [workers] HYPIR uses one shared GPU model; forcing city_workers=1")
        requested = 1
    return max(1, min(requested, max(city_count, 1)))


def _run_city_tasks(city_dirs: list[Path], args, model_id: str, run_one):
    workers = _effective_city_workers(args, model_id, len(city_dirs))
    if workers <= 1 or len(city_dirs) <= 1:
        for city_dir in city_dirs:
            run_one(city_dir)
        return

    print(f"  [workers] Processing {len(city_dirs)} cities with city_workers={workers}")
    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_one, city_dir): city_dir for city_dir in city_dirs}
        for future in as_completed(futures):
            city_dir = futures[future]
            try:
                future.result()
                print(f"  [workers] {city_dir.name}: completed")
            except BaseException as exc:
                print(f"  [workers] {city_dir.name}: failed — {exc}")
                failures.append((city_dir.name, exc))

    if failures:
        detail = "; ".join(f"{city}: {exc}" for city, exc in failures)
        raise RuntimeError(f"{len(failures)} city job(s) failed: {detail}")


def cli_batch(args):
    import datetime as _dt
    job_id = getattr(args, "job_id", None)

    def _job_update(**kw):
        if job_id:
            _update_job(job_id, **kw)

    _job_update(
        pid=os.getpid(),
        status="running",
        started_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )

    try:
        set_seed(args.seed)
        model_id = args.model

        if model_id == DDS_MODEL_ID:
            pb, pg = "", ""
        elif model_id == "hypir":
            pb, pg = load_hypir_prompts()
        else:
            pb, pg = load_nb_prompts()

        city_dirs = get_city_dirs(args.city, args.cities)
        if not city_dirs:
            raise RuntimeError("No cities found")
        workers = _effective_city_workers(args, model_id, len(city_dirs))
        print(f"Cities: {[c.name for c in city_dirs]},  model: {model_id},  city_workers={workers}")

        if getattr(args, "dry_run", False):
            print("━━━ DRY RUN — no files will be written ━━━")
            for city_dir in city_dirs:
                if model_id == DDS_MODEL_ID:
                    _run_city_dds(city_dir, args)
                else:
                    _dry_run_city(city_dir, model_id, args, pb, pg)
            print("\n━━━ DRY RUN done ━━━")
            _job_update(status="completed",
                        finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
                        exit_code=0)
            return

        if model_id != DDS_MODEL_ID:
            for city_dir in city_dirs:
                backup_and_clear_textures(city_dir)

        if model_id == DDS_MODEL_ID:
            _run_city_tasks(city_dirs, args, model_id, lambda city_dir: _run_city_dds(city_dir, args))
        elif model_id == "hypir":
            print("\nLoading HYPIR model...")
            t0 = time()
            model = SD2Enhancer(
                base_model_path=BASE_MODEL_PATH, weight_path=WEIGHT_PATH,
                lora_modules=LORA_MODULES, lora_rank=256, model_t=200, coeff_t=200,
                device=args.device,
            )
            model.init_models()
            print(f"Model loaded in {time() - t0:.1f}s\n")
            for city_dir in city_dirs:
                _run_city_hypir(city_dir, model, args, pb, pg)
        else:
            load_env()
            _run_city_tasks(
                city_dirs,
                args,
                model_id,
                lambda city_dir: _run_city_nb(city_dir, model_id, args, pb, pg),
            )

        print("\nAll done.")
        _job_update(status="completed",
                    finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
                    exit_code=0)

    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(f"\n[FATAL] {e}\n{err_detail}", flush=True)
        _job_update(status="failed",
                    finished_at=_dt.datetime.now().isoformat(timespec="seconds"),
                    exit_code=1,
                    error=f"{e}\n\n{err_detail}")
        sys.exit(1)


def _dry_run_city(city_dir: Path, model_id: str, args, prompt_building: str, prompt_ground: str):
    src_dir = city_dir / "textures_source"
    out_dir = city_dir / "textures"
    if not src_dir.exists():
        print(f"[warn] No textures_source in {city_dir.name}"); return

    all_pngs      = sorted(f for f in src_dir.iterdir() if f.suffix.lower() == ".png")
    ground_pngs   = [f for f in all_pngs if f.name.startswith("Material_Ground")]
    building_pngs = [f for f in all_pngs if f.name.startswith("Material_")
                     and not f.name.startswith("Material_Ground")]

    filt = args.filter
    process_pngs = (ground_pngs if filt == "ground"
                    else building_pngs if filt == "building"
                    else ground_pngs + building_pngs)

    to_process = [f for f in process_pngs
                  if should_process_source_image(f, args.max_source_longest_side)
                  and not (args.resume and (out_dir / f.name).exists())]
    to_process = apply_file_shard(to_process, args)
    high_res_skips = [f for f in process_pngs
                      if not should_process_source_image(f, args.max_source_longest_side)]

    # Model params header
    if model_id == "hypir":
        params = (f"target_size={args.target_size}  patch={args.patch_size}  "
                  f"stride={args.stride}  seed={args.seed}  device={args.device}  "
                  f"max_source_longest_side={args.max_source_longest_side}")
    else:
        api_id = NB_CLI_TO_API.get(model_id, model_id)
        params = f"api_model={api_id}  resolution={args.nb_resolution}"

    print(f"\n=== {city_dir.name} [{model_id}/{filt}]  {len(to_process)} files "
          f"({'--resume: ' + str(len(process_pngs) - len(to_process)) + ' skip' if args.resume else 'no resume'}) ===")
    print(f"  params : {params}")
    print(f"  prompts:")
    print(f"    building : {prompt_building}")
    print(f"    ground   : {prompt_ground}")
    print(f"  high-res source skips: {len(high_res_skips)}")
    print()

    for img_path in to_process:
        try:
            img = Image.open(img_path)
            in_str = f"{img.size[0]}×{img.size[1]}"
        except Exception:
            in_str = "?"
        if model_id == "hypir":
            out_str = f"→ ~{args.target_size}px"
        else:
            out_str = f"→ {args.nb_resolution}"
        already = " [exists]" if (out_dir / img_path.name).exists() else ""
        ptype = 'ground' if img_path.name.startswith('Material_Ground') else 'building'
        print(f"  {img_path.name:<50s}  {in_str} {out_str:<10s}  prompt={ptype}{already}")
    for img_path in high_res_skips[:20]:
        try:
            img = Image.open(img_path)
            in_str = f"{img.size[0]}×{img.size[1]}"
        except Exception:
            in_str = "?"
        print(f"  [copy-high-res] {img_path.name:<38s} {in_str}")
    if len(high_res_skips) > 20:
        print(f"  ... {len(high_res_skips) - 20} more high-res source file(s)")


def _run_city_hypir(city_dir: Path, model, args, prompt_building: str, prompt_ground: str):
    src_dir = city_dir / "textures_source"
    out_dir = city_dir / "textures"
    out_dir.mkdir(exist_ok=True)
    if not src_dir.exists():
        print(f"[warn] No textures_source in {city_dir.name}"); return

    all_pngs      = sorted(f for f in src_dir.iterdir() if f.suffix.lower() == ".png")
    ground_pngs   = [f for f in all_pngs if f.name.startswith("Material_Ground")]
    building_pngs = [f for f in all_pngs if f.name.startswith("Material_")
                     and not f.name.startswith("Material_Ground")]
    copy_pngs     = [f for f in all_pngs if not f.name.startswith("Material_")]
    others        = [f for f in src_dir.iterdir() if f.suffix.lower() != ".png"]

    filt = args.filter
    process_pngs = (ground_pngs if filt == "ground"
                    else building_pngs if filt == "building"
                    else ground_pngs + building_pngs)
    high_res_pngs = [f for f in process_pngs
                     if not should_process_source_image(f, args.max_source_longest_side)]
    process_pngs = [f for f in process_pngs
                    if should_process_source_image(f, args.max_source_longest_side)]
    process_pngs = apply_file_shard(process_pngs, args)

    for f in copy_pngs + high_res_pngs + others:
        dst = out_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst); print(f"  [copy] {f.name}")

    print(
        f"\n=== {city_dir.name} [HYPIR/{filt}]: {len(process_pngs)} to process; "
        f"{len(high_res_pngs)} high-res source PNG(s) copied unchanged ==="
    )
    processed = skipped = 0
    for img_path in process_pngs:
        out_path = out_dir / img_path.name
        if args.resume and out_path.exists():
            skipped += 1; continue
        try:
            lq_pil = Image.open(img_path).convert("RGB"); lq_pil.load()
        except Exception as e:
            print(f"  [skip] {img_path.name} — {e}"); continue
        prompt = select_prompt(img_path.name, prompt_building, prompt_ground)
        t_img = time()
        try:
            result = hypir_enhance(model, lq_pil, prompt,
                                   args.target_size, args.patch_size, args.stride)
        except Exception as e:
            print(f"  [error] {img_path.name} — {e}"); continue
        save_png(result, out_path)
        save_model_copy(result, city_dir, img_path.name, f"hypir.{args.target_size}")
        processed += 1
        print(f"  [done] {img_path.name}  {lq_pil.size} → {result.size}  ({time()-t_img:.1f}s)")
    print(f"  {city_dir.name}: processed={processed}, skipped={skipped}")


def _run_city_nb(city_dir: Path, cli_model_id: str, args, prompt_building: str, prompt_ground: str):
    api_model_id = NB_CLI_TO_API[cli_model_id]
    model_tag = f"{cli_model_id}.{args.nb_resolution}"
    src_dir = city_dir / "textures_source"
    out_dir = city_dir / "textures"
    out_dir.mkdir(exist_ok=True)
    if not src_dir.exists():
        print(f"[warn] No textures_source in {city_dir.name}"); return

    all_pngs      = sorted(f for f in src_dir.iterdir() if f.suffix.lower() == ".png")
    ground_pngs   = [f for f in all_pngs if f.name.startswith("Material_Ground")]
    building_pngs = [f for f in all_pngs if f.name.startswith("Material_")
                     and not f.name.startswith("Material_Ground")]
    copy_pngs     = [f for f in all_pngs if not f.name.startswith("Material_")]
    others        = [f for f in src_dir.iterdir() if f.suffix.lower() != ".png"]

    filt = args.filter
    process_pngs = (ground_pngs if filt == "ground"
                    else building_pngs if filt == "building"
                    else ground_pngs + building_pngs)

    for f in copy_pngs + others:
        dst = out_dir / f.name
        if not dst.exists():
            shutil.copy2(f, dst); print(f"  [copy] {f.name}")

    # Filter already-done if resuming
    to_process = []
    for img_path in process_pngs:
        out_path = out_dir / img_path.name
        if args.resume and out_path.exists():
            continue
        to_process.append(img_path)
    to_process = apply_file_shard(to_process, args)

    print(f"\n=== {city_dir.name} [{cli_model_id}/{filt}]: {len(to_process)} to submit ===")
    if not to_process:
        print("  Nothing to do."); return

    # Load images + prompts
    items: list[tuple[Image.Image, str]] = []
    img_paths_valid: list[Path] = []
    for img_path in to_process:
        try:
            lq_pil = Image.open(img_path).convert("RGB"); lq_pil.load()
            prompt = select_prompt(img_path.name, prompt_building, prompt_ground)
            items.append((lq_pil, prompt))
            img_paths_valid.append(img_path)
        except Exception as e:
            print(f"  [skip] {img_path.name} — {e}")

    results = nb_enhance_batch(api_model_id, items, args.nb_resolution)
    processed = 0
    for img_path, result in zip(img_paths_valid, results):
        out_path = out_dir / img_path.name
        if isinstance(result, Exception):
            print(f"  [error] {img_path.name} — {result}")
        else:
            save_png(result, out_path)
            save_model_copy(result, city_dir, img_path.name, model_tag)
            processed += 1
            print(f"  [done] {img_path.name}")
    print(f"  {city_dir.name}: processed={processed}/{len(items)}")


def cli_single(args):
    if not args.city or not args.file:
        print("--city and --file are required for --mode single"); sys.exit(1)
    model_id = args.model
    if model_id == DDS_MODEL_ID:
        print("--model dds is only supported with --mode batch"); sys.exit(1)
    if model_id == "hypir":
        pb, pg = load_hypir_prompts()
    else:
        pb, pg = load_nb_prompts()

    city_dir = DATA_DIR / args.city.upper()
    src_path = city_dir / "textures_source" / args.file
    out_path = city_dir / "textures" / args.file
    if not src_path.exists():
        print(f"Source not found: {src_path}"); sys.exit(1)
    prompt = args.prompt or select_prompt(args.file, pb, pg)

    lq_pil = Image.open(src_path).convert("RGB")

    if model_id == "hypir":
        print("Loading HYPIR model...")
        t0 = time()
        model = SD2Enhancer(
            base_model_path=BASE_MODEL_PATH, weight_path=WEIGHT_PATH,
            lora_modules=LORA_MODULES, lora_rank=256, model_t=200, coeff_t=200,
            device=args.device,
        )
        model.init_models()
        print(f"Model loaded in {time() - t0:.1f}s")
        if args.seed >= 0:
            set_seed(args.seed)
        result = hypir_enhance(model, lq_pil, prompt,
                               args.target_size, args.patch_size, args.stride)
    else:
        load_env()
        api_model_id = NB_CLI_TO_API[model_id]
        print(f"Submitting to {api_model_id} ({args.nb_resolution})...")
        result = nb_enhance_single(api_model_id, lq_pil, prompt, args.nb_resolution)

    out_path.parent.mkdir(exist_ok=True)
    save_png(result, out_path)
    if model_id != "hypir":
        model_tag = f"{model_id}.{args.nb_resolution}"
    else:
        model_tag = f"hypir.{args.target_size}"
    save_model_copy(result, city_dir, args.file, model_tag)
    print(f"Saved: {out_path}  ({result.size[0]}×{result.size[1]})")


def cli_clear(args):
    city_dirs = get_city_dirs(args.city)
    if not city_dirs:
        print("No cities found."); sys.exit(1)
    for city_dir in city_dirs:
        textures = city_dir / "textures"
        if not textures.exists():
            print(f"  [skip] {city_dir.name}: no textures/ dir"); continue
        files = list(textures.iterdir())
        for f in files:
            f.unlink() if f.is_file() else shutil.rmtree(f)
        print(f"  [clear] {city_dir.name}: removed {len(files)} items")


def _delete_city_dds_outputs(city_dir: Path) -> tuple[int, int]:
    dds_files = list(city_dir.rglob("*.dds"))
    lock_files = list(city_dir.rglob("*.dds.lock"))
    for path in dds_files + lock_files:
        path.unlink(missing_ok=True)
    return len(dds_files), len(lock_files)


def _run_city_dds(city_dir: Path, args):
    if not DDS_HELPER_PATH.exists():
        raise FileNotFoundError(DDS_HELPER_PATH)

    if getattr(args, "dry_run", False):
        print(f"\n=== {city_dir.name} [DDS/{DDS_VARIANT}] dry run ===")
    else:
        print(f"\n=== {city_dir.name} [DDS/{DDS_VARIANT}] ===")
        if not args.resume:
            dds_count, lock_count = _delete_city_dds_outputs(city_dir)
            print(f"  [force] removed {dds_count} .dds and {lock_count} lock file(s)")

    for label in DDS_LAYERS:
        scene_usd = city_dir / f"{city_dir.name}_{label}.usd"
        output_json = city_dir / f"{city_dir.name}_{label}_{DDS_VARIANT}.json"
        if not scene_usd.exists():
            print(f"  [warn] missing {scene_usd.name}; skipping {label}")
            continue

        cmd = [
            sys.executable, "-u", str(DDS_HELPER_PATH),
            "--op", "convert",
            "--variant", DDS_VARIANT,
            "--scene_usd", str(scene_usd),
            "--scene_root", str(DATA_DIR.resolve()),
            "--output_json", str(output_json),
            "--compressonator", "compressonatorcli",
            "--headless",
        ]
        if getattr(args, "dry_run", False):
            print("  [dry-run] " + " ".join(cmd))
            continue

        print(f"  [dds] {label}: {scene_usd.name} -> {output_json.name}")
        subprocess.run(cmd, check=True, cwd=str(SCRIPT_DIR.parent.parent))
    print(f"  {city_dir.name}: DDS conversion done")


# ── entry point ────────────────────────────────────────────────────────────────
def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(description="NavVerse Texture Processor")
    parser.add_argument("--mode", choices=["ui", "batch", "single", "clear"], default="ui")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    # UI
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    # model
    parser.add_argument("--model",  type=str, default="hypir",
                        choices=["hypir", "nano-banana-pro", "nano-banana-2", DDS_MODEL_ID])
    # batch / single
    parser.add_argument("--city",        type=str, default=None)
    parser.add_argument("--cities",      type=str, default=None)
    parser.add_argument("--filter",      type=str, default="all",
                        choices=["all", "ground", "building"])
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Print files and prompts without processing")
    parser.add_argument("--job_id",      type=str, default=None,
                        help="Internal: job record ID written by launch_batch")
    parser.add_argument(
        "--city-workers",
        type=int,
        default=1,
        help="Number of cities to process concurrently in batch mode. HYPIR remains single-city.",
    )
    parser.add_argument("--target_size", type=int, default=2048)
    parser.add_argument("--patch_size",  type=int, default=512)
    parser.add_argument("--stride",      type=int, default=256)
    parser.add_argument("--seed",        type=int, default=231)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument(
        "--max-source-longest-side",
        type=int,
        default=1500,
        help="Only upscale source PNGs whose longest side is below this value; use 0 to process all.",
    )
    parser.add_argument(
        "--file-shard-index",
        type=int,
        default=0,
        help="Process only files whose sorted index modulo --file-shard-count matches this value.",
    )
    parser.add_argument(
        "--file-shard-count",
        type=int,
        default=1,
        help="Split each selected city's files into this many deterministic shards.",
    )
    # single only
    parser.add_argument("--file",        type=str, default=None)
    parser.add_argument("--prompt",      type=str, default=None)
    # nano banana
    parser.add_argument("--nb_resolution", type=str, default="2K",
                        choices=["1K", "2K", "4K"],
                        help="Output resolution for Nano Banana models")
    args = parser.parse_args()
    DATA_DIR = args.data_dir

    # Force line-buffered stdout so every print() flushes immediately to the log file
    if args.mode in ("batch", "single", "clear"):
        sys.stdout.reconfigure(line_buffering=True)

    if args.mode == "batch":
        cli_batch(args)
    elif args.mode == "single":
        cli_single(args)
    elif args.mode == "clear":
        cli_clear(args)
    else:
        load_env()
        demo = build_ui()
        demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
