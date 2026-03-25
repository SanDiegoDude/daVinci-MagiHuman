"""
Gradio frontend for daVinci-MagiHuman.

Keeps models resident in GPU memory across generations.
Supports I2V, T2V, and optional lipsync with user-provided audio.
Includes OAI-compatible prompt enhancement with vision support.

Launch:
    python app.py [--port 7860] [--models-dir ~/shared/models]
"""

import argparse
import base64
import gc
import io
import os
import random
import re
import resource
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import gradio as gr
import torch

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompt_system.txt")


def _load_dotenv():
    """Minimal .env loader — no external dependency needed."""
    if not os.path.exists(_ENV_PATH):
        return
    with open(_ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODELS_DIR = os.path.expanduser("~/shared/models")

MODEL_MANIFEST = {
    # --- BF16 originals (used with --fp16) ---
    "distill": {
        "repo": "GAIR/daVinci-MagiHuman",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["distill/*"],
        "check_file": "distill/model.safetensors.index.json",
    },
    "base": {
        "repo": "GAIR/daVinci-MagiHuman",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["base/*"],
        "check_file": "base/model.safetensors.index.json",
    },
    "540p_sr": {
        "repo": "GAIR/daVinci-MagiHuman",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["540p_sr/*"],
        "check_file": "540p_sr/model.safetensors.index.json",
    },
    "1080p_sr": {
        "repo": "GAIR/daVinci-MagiHuman",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["1080p_sr/*"],
        "check_file": "1080p_sr/model.safetensors.index.json",
    },
    # --- FP8 quantized (default) ---
    "distill_fp8": {
        "repo": "SanDiegoDude/daVinci-MagiHuman-FP8",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["distill_fp8/*"],
        "check_file": "distill_fp8/model.safetensors.index.json",
    },
    "base_fp8": {
        "repo": "SanDiegoDude/daVinci-MagiHuman-FP8",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["base_fp8/*"],
        "check_file": "base_fp8/model.safetensors.index.json",
    },
    "540p_sr_fp8": {
        "repo": "SanDiegoDude/daVinci-MagiHuman-FP8",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["540p_sr_fp8/*"],
        "check_file": "540p_sr_fp8/model.safetensors.index.json",
    },
    "1080p_sr_fp8": {
        "repo": "SanDiegoDude/daVinci-MagiHuman-FP8",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["1080p_sr_fp8/*"],
        "check_file": "1080p_sr_fp8/model.safetensors.index.json",
    },
    # --- Shared models (used by both modes) ---
    "turbo_vae": {
        "repo": "GAIR/daVinci-MagiHuman",
        "subdir": "daVinci-MagiHuman",
        "allow_patterns": ["turbo_vae/*"],
        "check_file": "turbo_vae/checkpoint-340000.ckpt",
    },
    "stable_audio": {
        "repo": "stabilityai/stable-audio-open-1.0",
        "subdir": "stable-audio-open-1.0",
        "allow_patterns": None,
        "check_file": "model.safetensors",
    },
    "t5gemma": {
        "repo": "google/t5gemma-9b-9b-ul2",
        "subdir": "t5gemma-9b-9b-ul2",
        "allow_patterns": None,
        "check_file": "model.safetensors.index.json",
    },
    "wan_vae": {
        "repo": "Wan-AI/Wan2.2-TI2V-5B",
        "subdir": "Wan2.2-TI2V-5B",
        "allow_patterns": ["Wan2.2_VAE.pth"],
        "check_file": "Wan2.2_VAE.pth",
    },
}

SR_SHORT_SIDE = {
    "540p": 512,
    "1080p": 1088,
}

ALIGNMENT = 32  # vae_stride(16) * patch_size(2)


def _snap(val: int, alignment: int = ALIGNMENT) -> int:
    """Round to nearest multiple of alignment."""
    return max(alignment, round(val / alignment) * alignment)


def _sr_dims_for(base_w: int, base_h: int, sr_tier: str):
    """Compute SR output dimensions that preserve the base aspect ratio.

    Scales so the short side matches the tier target, snapped to ALIGNMENT.
    """
    target_short = SR_SHORT_SIDE[sr_tier]
    short = min(base_w, base_h)
    if short <= 0:
        short = 1
    scale = target_short / short
    sr_w = _snap(int(base_w * scale))
    sr_h = _snap(int(base_h * scale))
    return sr_w, sr_h

BASE_RESOLUTION_PRESETS = {
    "Auto (match image)": None,
    "448 x 256  (16:9 landscape)": (448, 256),
    "256 x 448  (9:16 portrait)": (256, 448),
    "352 x 352  (1:1 square)": (352, 352),
    "480 x 288  (5:3 landscape)": (480, 288),
    "288 x 480  (3:5 portrait)": (288, 480),
    "480 x 272  (16:9 landscape)": (480, 272),
    "640 x 384  (5:3 landscape)": (640, 384),
    "384 x 640  (3:5 portrait)": (384, 640),
}

TARGET_PIXELS_BASE = 448 * 256  # ~115k — keeps VRAM reasonable at base res




def _auto_resolution_from_image(image_path: str) -> tuple[int, int]:
    """Compute base resolution matching the image's aspect ratio.

    Keeps total pixel count near TARGET_PIXELS_BASE, snapped to multiples
    of ALIGNMENT.
    """
    from PIL import Image as PILImage
    with PILImage.open(image_path) as img:
        iw, ih = img.size

    aspect = iw / ih
    # w * h ≈ TARGET_PIXELS_BASE and w/h = aspect
    # => w = sqrt(TARGET_PIXELS_BASE * aspect), h = w / aspect
    w = (TARGET_PIXELS_BASE * aspect) ** 0.5
    h = w / aspect
    w, h = _snap(int(w)), _snap(int(h))
    return w, h

# ---------------------------------------------------------------------------
# Auto-download
# ---------------------------------------------------------------------------


def ensure_models(models_dir: str, sr_model: str):
    """Check for missing models and download them."""
    from huggingface_hub import snapshot_download

    suffix = "_fp8" if _use_fp8 else ""
    dit_name = "distill" if _use_distill else "base"
    required = [f"{dit_name}{suffix}", "turbo_vae", "stable_audio", "t5gemma", "wan_vae"]
    if sr_model == "540p":
        required.append(f"540p_sr{suffix}")
    elif sr_model == "1080p":
        required.append(f"1080p_sr{suffix}")
    elif sr_model == "both":
        required.extend([f"540p_sr{suffix}", f"1080p_sr{suffix}"])

    for key in required:
        entry = MODEL_MANIFEST[key]
        local_dir = os.path.join(models_dir, entry["subdir"])
        check_path = os.path.join(local_dir, entry["check_file"])

        if os.path.exists(check_path):
            print(f"  [ok] {key}: {check_path}")
            continue

        print(f"  [downloading] {key} from {entry['repo']} -> {local_dir}")
        kwargs = dict(
            repo_id=entry["repo"],
            local_dir=local_dir,
            repo_type="model",
        )
        if entry["allow_patterns"]:
            kwargs["allow_patterns"] = entry["allow_patterns"]

        snapshot_download(**kwargs)
        print(f"  [done] {key}")


# ---------------------------------------------------------------------------
# Prompt enhancement (OAI-compatible)
# ---------------------------------------------------------------------------

_system_prompt_cache: str | None = None


def _get_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        if os.path.exists(_SYSTEM_PROMPT_PATH):
            with open(_SYSTEM_PROMPT_PATH) as f:
                _system_prompt_cache = f.read().strip()
        else:
            _system_prompt_cache = "You are a helpful prompt enhancement assistant for AI video generation."
    return _system_prompt_cache


def reload_system_prompt() -> str:
    """Force-reload the system prompt from disk (e.g. after user edits it)."""
    global _system_prompt_cache
    _system_prompt_cache = None
    return _get_system_prompt()


def _image_to_data_uri(image_path: str) -> str:
    """Convert an image file to a base64 data URI for the vision API."""
    import mimetypes
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"


def enhance_prompt(
    user_prompt: str,
    image_path: str | None = None,
    progress_callback=None,
    api_base: str = "",
    api_key: str = "",
    model: str = "",
    vision_model: str = "",
) -> str:
    """Call an OAI-compatible endpoint to enhance the user's prompt.

    If image_path is provided and a vision model is configured, the image
    is sent as part of the message for I2V-aware enhancement.
    """
    import httpx

    api_base = (api_base or os.environ.get("LLM_API_BASE_URL", "")).strip()
    api_key = (api_key or os.environ.get("LLM_API_KEY", "")).strip()
    model = (model or os.environ.get("LLM_MODEL", "default")).strip()
    vision_model = (vision_model or os.environ.get("LLM_VISION_MODEL", "")).strip()

    if not api_base:
        raise gr.Error(
            "Prompt enhancement not configured. Set the API Base URL in LLM Settings."
        )

    use_vision = bool(image_path and os.path.exists(image_path or ""))
    active_model = vision_model if (use_vision and vision_model) else model

    system_prompt = _get_system_prompt()

    if progress_callback:
        progress_callback(0.1, "Building enhancement request...")

    user_content: list | str
    if use_vision:
        data_uri = _image_to_data_uri(image_path)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": data_uri},
            },
            {
                "type": "text",
                "text": f"First-frame image is attached above.\n\nUser Prompt: {user_prompt}",
            },
        ]
    else:
        user_content = f"User Prompt: {user_prompt}"

    payload = {
        "model": active_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
    }

    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    if progress_callback:
        progress_callback(0.3, "Sending to LLM...")

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
    except httpx.ConnectError:
        raise gr.Error(f"Cannot connect to LLM at {api_base}. Is LM Studio running?")
    except httpx.HTTPStatusError as e:
        raise gr.Error(f"LLM API error {e.response.status_code}: {e.response.text[:300]}")
    except Exception as e:
        raise gr.Error(f"LLM request failed: {e}")

    if progress_callback:
        progress_callback(0.9, "Processing response...")

    data = resp.json()
    enhanced = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    if not enhanced.strip():
        raise gr.Error("LLM returned an empty response.")

    return enhanced.strip()


# ---------------------------------------------------------------------------
# Pipeline singleton
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_loading = False
_pipeline_error: str | None = None
_models_dir: str = DEFAULT_MODELS_DIR
_loaded_sr: str | None = None  # "540p", "1080p", or None
_parsed_config = None  # cached MagiPipelineConfig for SR loading


_use_fp8 = True
_use_distill = True
_highvram = False


# ---------------------------------------------------------------------------
# VRAM offload manager — swaps models between CPU and GPU on demand.
# In --highvram mode everything stays resident; otherwise only the active
# model lives on GPU, keeping peak usage 4090-friendly (~20 GB).
# ---------------------------------------------------------------------------

class VRAMManager:
    """Tracks which heavy DiT model is on GPU and swaps as needed.

    Components managed:
        dit      – base DiT (~14 GB FP8)
        sr       – SR DiT   (~14 GB FP8, optional)

    T5Gemma and VAEs use CPUOffloadWrapper internally and are not
    managed here — their wrappers handle GPU hops transparently.
    """

    def __init__(self):
        self._on_gpu: set[str] = set()
        self._models: dict[str, torch.nn.Module | object] = {}
        self._device = torch.device("cuda")
        self._cpu = torch.device("cpu")
        self.enabled = False

    def register(self, name: str, model):
        self._models[name] = model
        self._on_gpu.add(name)

    def _move(self, name: str, device: torch.device):
        model = self._models.get(name)
        if model is None:
            return
        if isinstance(model, torch.nn.Module):
            model.to(device, non_blocking=True)
        elif hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
            model.model.to(device, non_blocking=True)
        if device.type == "cuda":
            self._on_gpu.add(name)
        else:
            self._on_gpu.discard(name)

    def ensure_on_gpu(self, *names: str):
        """Move *names* to GPU; if offload is enabled, move everything else to CPU."""
        if not self.enabled:
            return
        needed = set(names)
        to_offload = self._on_gpu - needed
        for n in to_offload:
            print(f"  [vram] offloading {n} → CPU")
            self._move(n, self._cpu)
        torch.cuda.empty_cache()
        for n in needed:
            if n not in self._on_gpu and n in self._models:
                print(f"  [vram] loading {n} → GPU")
                self._move(n, self._device)
        torch.cuda.synchronize()

    def mark_offloaded(self, name: str):
        """Update tracking after external code (e.g. evaluate()) moved a model to CPU."""
        self._on_gpu.discard(name)

    def offload_all(self):
        if not self.enabled:
            return
        for n in list(self._on_gpu):
            self._move(n, self._cpu)
        torch.cuda.empty_cache()


_vram = VRAMManager()


def get_model_paths(models_dir: str):
    suffix = "_fp8" if _use_fp8 else ""
    dit_name = "distill" if _use_distill else "base"
    return {
        "dit": os.path.join(models_dir, "daVinci-MagiHuman", f"{dit_name}{suffix}"),
        "540p_sr": os.path.join(models_dir, "daVinci-MagiHuman", f"540p_sr{suffix}"),
        "1080p_sr": os.path.join(models_dir, "daVinci-MagiHuman", f"1080p_sr{suffix}"),
        "turbo_vae_config": os.path.join(
            models_dir, "daVinci-MagiHuman", "turbo_vae", "TurboV3-Wan22-TinyShallow_7_7.json"
        ),
        "turbo_vae_ckpt": os.path.join(models_dir, "daVinci-MagiHuman", "turbo_vae", "checkpoint-340000.ckpt"),
        "stable_audio": os.path.join(models_dir, "stable-audio-open-1.0"),
        "t5gemma": os.path.join(models_dir, "t5gemma-9b-9b-ul2"),
        "wan_vae": os.path.join(models_dir, "Wan2.2-TI2V-5B"),
    }


def init_pipeline(models_dir: str):
    """Initialize torch.distributed and build MagiPipeline (no SR model)."""
    global _pipeline, _models_dir, _parsed_config

    _models_dir = models_dir
    paths = get_model_paths(models_dir)

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if _vram.enabled:
        os.environ["CPU_OFFLOAD"] = "1"
        os.environ["SR2_1080"] = "1"

        def _fake_arch_memory(unit="GB"):
            """Report 24 GB so CPUOffloadWrapper activates for all sub-models."""
            scales = {"B": 24 * 1024**3, "KB": 24 * 1024**2, "MB": 24 * 1024, "GB": 24.0}
            return scales.get(unit, 24.0)

        import inference.common.arch as _arch_mod
        import inference.common as _common_mod
        _arch_mod._original_get_arch_memory = _arch_mod.get_arch_memory
        _arch_mod.get_arch_memory = _fake_arch_memory
        _common_mod.get_arch_memory = _fake_arch_memory

    import json, tempfile as _tf

    config_dict = {
        "engine_config": {
            "load": paths["dit"],
            "cp_size": 1,
        },
        "evaluation_config": {
            "cfg_number": 1 if _use_distill else 2,
            "num_inference_steps": 8 if _use_distill else 32,
            "audio_model_path": paths["stable_audio"],
            "txt_model_path": paths["t5gemma"],
            "vae_model_path": paths["wan_vae"],
            "use_turbo_vae": True,
            "student_config_path": paths["turbo_vae_config"],
            "student_ckpt_path": paths["turbo_vae_ckpt"],
            "use_sr_model": False,
            "sr_model_path": "",
            "sr_cfg_number": 1,
            "sr_num_inference_steps": 5,
        },
    }

    config_file = _tf.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(config_dict, config_file)
    config_file.close()

    sys.argv = ["app.py", "--config-load-path", config_file.name]

    from inference.infra import initialize_infra
    from inference.common import parse_config
    from inference.model.dit import get_dit
    from inference.pipeline import MagiPipeline

    initialize_infra()

    _parsed_config = parse_config()
    model = get_dit(_parsed_config.arch_config, _parsed_config.engine_config)
    _pipeline = MagiPipeline(model, _parsed_config.evaluation_config)

    os.unlink(config_file.name)
    _install_tiled_decode()

    _vram.register("dit", _pipeline.model)

    if _vram.enabled:
        _pipeline.model.to(torch.device("cpu"))
        _vram._on_gpu.discard("dit")
        gc.collect()
        torch.cuda.empty_cache()
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n*** Pipeline ready (VRAM offload active). ***")
        print(f"    GPU: {vram_used:.1f} / {vram_total:.1f} GB used (heavy models on CPU)\n")
    else:
        print("\n*** Pipeline ready (--highvram — all models resident). ***\n")


def _ensure_sr_model(sr_tier: str):
    """Lazy-load an SR model into the evaluator. Downloads if needed.

    sr_tier should be "540p" or "1080p".
    If the requested tier is already loaded, this is a no-op.
    """
    global _loaded_sr

    if _loaded_sr == sr_tier:
        return

    paths = get_model_paths(_models_dir)
    sr_key = f"{sr_tier}_sr"
    sr_path = paths[sr_key]

    manifest_key = f"{sr_key}_fp8" if _use_fp8 else sr_key
    check_path = os.path.join(sr_path, "model.safetensors.index.json")
    if not os.path.exists(check_path):
        print(f"  [sr] Downloading {manifest_key}...")
        from huggingface_hub import snapshot_download
        entry = MODEL_MANIFEST[manifest_key]
        local_dir = os.path.join(_models_dir, entry["subdir"])
        kwargs = dict(repo_id=entry["repo"], local_dir=local_dir, repo_type="model")
        if entry["allow_patterns"]:
            kwargs["allow_patterns"] = entry["allow_patterns"]
        snapshot_download(**kwargs)
        print(f"  [sr] Download complete: {manifest_key}")

    print(f"  [sr] Loading SR model: {sr_tier} from {sr_path}")
    from inference.model.dit import get_dit
    import copy

    engine_cfg = copy.deepcopy(_parsed_config.engine_config)
    engine_cfg.load = sr_path
    sr_model = get_dit(_parsed_config.sr_arch_config, engine_cfg)

    _pipeline.evaluator.sr_model = sr_model
    _pipeline.evaluator.sr_model.eval()
    _vram.register("sr", sr_model)
    if _vram.enabled:
        sr_model.to(torch.device("cpu"))
        _vram._on_gpu.discard("sr")
        gc.collect()
        torch.cuda.empty_cache()
    _loaded_sr = sr_tier
    print(f"  [sr] SR model loaded and resident: {sr_tier}")


# ---------------------------------------------------------------------------
# Tiled VAE decode (prevents RAM OOM at high resolutions like 1080p)
# ---------------------------------------------------------------------------

# Tile = 512px (32 latent @16x), stride = 384px (24 latent) → 128px overlap.
# For 1080p (1920×1088) this yields 3×5 = 15 tiles — manageable.
TILE_SAMPLE_MIN_SIZE = 512
TILE_SAMPLE_STRIDE = 384
TILE_PIXEL_THRESHOLD = 720  # tile when ANY pixel dim exceeds this


def _blend_v(a, b, blend_extent):
    """Vertical (height-dim) linear blend of two overlapping pixel tiles."""
    blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
    if blend_extent <= 0:
        return b
    ramp = torch.linspace(0.0, 1.0, blend_extent, device=b.device, dtype=b.dtype)
    ramp = ramp[None, None, None, :, None]
    b[:, :, :, :blend_extent, :] = (
        a[:, :, :, -blend_extent:, :] * (1 - ramp) + b[:, :, :, :blend_extent, :] * ramp
    )
    return b


def _blend_h(a, b, blend_extent):
    """Horizontal (width-dim) linear blend of two overlapping pixel tiles."""
    blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
    if blend_extent <= 0:
        return b
    ramp = torch.linspace(0.0, 1.0, blend_extent, device=b.device, dtype=b.dtype)
    ramp = ramp[None, None, None, None, :]
    b[:, :, :, :, :blend_extent] = (
        a[:, :, :, :, -blend_extent:] * (1 - ramp) + b[:, :, :, :, :blend_extent] * ramp
    )
    return b


def _tiled_turbo_vae_decode(turbo_vae, latent, dtype):
    """Spatially-tiled decode for TurboVAED.  Prevents RAM OOM at 1080p+.

    Splits latent into overlapping spatial tiles, decodes each through
    TurboVAED's sliding-window temporal decoder (with output_offload=True
    so decoded pixels live on CPU), blends overlapping pixel regions, and
    stitches the result.  Follows the same tile/stride/blend pattern as
    diffusers' AutoencoderKLWan.tiled_decode().
    """
    _, _, _, lat_h, lat_w = latent.shape
    spatial_r = turbo_vae.spatial_compression_ratio  # 16

    tile_lat = TILE_SAMPLE_MIN_SIZE // spatial_r    # 32
    stride_lat = TILE_SAMPLE_STRIDE // spatial_r    # 24
    blend_px = TILE_SAMPLE_MIN_SIZE - TILE_SAMPLE_STRIDE  # 128 px overlap

    target_h = lat_h * spatial_r
    target_w = lat_w * spatial_r

    # ----- phase 1: decode all spatial tiles ---------------------
    rows = []
    tile_idx = 0
    for i in range(0, lat_h, stride_lat):
        row = []
        for j in range(0, lat_w, stride_lat):
            tile_z = latent[:, :, :, i : i + tile_lat, j : j + tile_lat].to(dtype)
            tile_idx += 1
            print(f"  [tiled-vae] tile {tile_idx}  "
                  f"lat y={i}:{i + tile_z.shape[3]}  x={j}:{j + tile_z.shape[4]}")
            tile_out = turbo_vae.decode(tile_z, output_offload=True).cpu().float()
            row.append(tile_out)
            del tile_z, tile_out
            torch.cuda.empty_cache()
            gc.collect()
        rows.append(row)
    print(f"  [tiled-vae] Decoded {tile_idx} tiles, blending…")

    # ----- phase 2: blend overlaps & stitch ---------------------
    stride_px = TILE_SAMPLE_STRIDE

    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = _blend_v(rows[i - 1][j], tile, blend_px)
            if j > 0:
                tile = _blend_h(row[j - 1], tile, blend_px)
            result_row.append(tile[:, :, :, :stride_px, :stride_px])
        result_rows.append(torch.cat(result_row, dim=-1))
        gc.collect()

    dec = torch.cat(result_rows, dim=3)[:, :, :, :target_h, :target_w]
    del rows, result_rows
    gc.collect()
    return dec


def _patched_decode_video(evaluator, latent, group=None):
    """Drop-in for MagiEvaluator.decode_video — adds spatial tiling."""
    import numpy as np

    _, _, _, lat_h, lat_w = latent.shape
    spatial_ratio = evaluator.config.vae_stride[1]  # 16
    pixel_h = lat_h * spatial_ratio
    pixel_w = lat_w * spatial_ratio
    need_tile = pixel_h > TILE_PIXEL_THRESHOLD or pixel_w > TILE_PIXEL_THRESHOLD

    if evaluator.config.use_turbo_vae and need_tile:
        print(f"  [tiled-vae] Decoding {pixel_w}×{pixel_h} with spatial tiling")
        videos = _tiled_turbo_vae_decode(evaluator.turbo_vae, latent, evaluator.dtype)
    elif evaluator.config.use_turbo_vae:
        videos = evaluator.turbo_vae.decode(
            latent.to(evaluator.dtype), output_offload=False
        ).float()
    else:
        videos = evaluator.vae.decode(
            latent.squeeze(0).to(evaluator.dtype), group=group
        )

    if videos is None:
        return None
    videos.mul_(0.5).add_(0.5).clamp_(0, 1)
    videos = [video.cpu() for video in videos]
    videos = [video.permute(1, 2, 3, 0) * 255 for video in videos]
    videos = [video.numpy().astype(np.uint8) for video in videos]
    return videos


def _install_tiled_decode():
    """Monkey-patch the evaluator's decode_video with our tiled version."""
    if _pipeline is None:
        return
    import types
    _pipeline.evaluator.decode_video = types.MethodType(
        _patched_decode_video, _pipeline.evaluator
    )
    print("  [tiled-vae] Tiled VAE decode installed")


# ---------------------------------------------------------------------------
# Generation wrapper
# ---------------------------------------------------------------------------


@contextmanager
def _capture_stdout():
    """Capture stdout while still letting it through to the real stdout."""
    buf = io.StringIO()
    real_stdout = sys.stdout

    class Tee:
        def write(self, s):
            buf.write(s)
            real_stdout.write(s)
        def flush(self):
            real_stdout.flush()

    sys.stdout = Tee()
    try:
        yield buf
    finally:
        sys.stdout = real_stdout


def _parse_timing_lines(captured: str) -> list[tuple[str, str]]:
    """Extract step timing info from captured pipeline stdout."""
    timings = []
    for m in re.finditer(
        r"Time Elapsed: \[([^\]]+)\] From \[([^\(]+?)\s*\(", captured
    ):
        duration, from_step = m.group(1), m.group(2).strip()
        timings.append((from_step, duration))
    return timings


def _do_enhance(
    mode: str,
    raw_prompt: str,
    image_path: str | None,
    llm_api_base: str,
    llm_api_key: str,
    llm_model: str,
    llm_vision_model: str,
    progress=gr.Progress(),
):
    """Gradio callback for the Enhance Prompt button."""
    if not raw_prompt or not raw_prompt.strip():
        raise gr.Error("Enter a prompt first before enhancing.")

    img = image_path if mode == "Image to Video" else None

    def _progress_cb(frac, msg):
        progress(frac, desc=msg)

    enhanced = enhance_prompt(
        raw_prompt,
        image_path=img,
        progress_callback=_progress_cb,
        api_base=llm_api_base,
        api_key=llm_api_key,
        model=llm_model,
        vision_model=llm_vision_model,
    )
    progress(1.0, "Enhancement complete")
    return enhanced


def generate(
    mode: str,
    raw_prompt: str,
    enhanced_prompt: str,
    use_enhanced: bool,
    image,
    audio,
    seed: int,
    randomize_seed: bool,
    seconds: int,
    base_res: str,
    sr_choice: str,
    sr_steps: int,
    sr_guidance: float,
    base_steps: int,
    vid_guidance: float,
    aud_guidance: float,
    preview_raw: bool = False,
    progress=gr.Progress(),
):
    # --- Wait for pipeline if still loading ---
    if _pipeline_loading or _pipeline is None:
        wait_start = time.time()
        timeout = 600  # 10 min max
        while _pipeline_loading and (time.time() - wait_start) < timeout:
            elapsed_wait = int(time.time() - wait_start)
            progress(0.0, desc=f"Waiting for models to load… ({elapsed_wait}s)")
            time.sleep(2)
        if _pipeline is None:
            msg = f"Pipeline not initialized: {_pipeline_error}" if _pipeline_error else "Pipeline not initialized. Check server logs."
            raise gr.Error(msg)

    used_enhanced = use_enhanced and bool(enhanced_prompt.strip())
    prompt = enhanced_prompt.strip() if used_enhanced else raw_prompt.strip()

    if not prompt:
        raise gr.Error("Prompt is required.")

    if mode == "Image to Video" and image is None:
        raise gr.Error("Image is required for Image-to-Video mode.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)

    image_input = image if mode == "Image to Video" else None
    audio_input = audio if audio else None

    preset = BASE_RESOLUTION_PRESETS.get(base_res)
    if preset is None:
        if image_input is not None:
            br_width, br_height = _auto_resolution_from_image(image_input)
        else:
            br_width, br_height = 448, 256
    else:
        br_width, br_height = preset

    # --- Lazy SR model management ---
    sr_tier_map = {"540p": "540p", "1080p": "1080p"}
    sr_tier = sr_tier_map.get(sr_choice)
    sr_width, sr_height = None, None
    use_sr = sr_tier is not None

    if use_sr:
        progress(0.0, desc=f"Checking SR model ({sr_tier})...")
        _ensure_sr_model(sr_tier)
        sr_width, sr_height = _sr_dims_for(br_width, br_height, sr_tier)
        print(f"  [sr] SR output: {sr_width}×{sr_height} (preserving {br_width}×{br_height} aspect ratio)")

    _pipeline.evaluation_config.num_inference_steps = base_steps
    _pipeline.evaluation_config.video_txt_guidance_scale = vid_guidance
    _pipeline.evaluation_config.audio_txt_guidance_scale = aud_guidance
    _pipeline.evaluation_config.sr_num_inference_steps = sr_steps
    _pipeline.evaluation_config.sr_video_txt_guidance_scale = sr_guidance
    evaluator = _pipeline.evaluator
    evaluator.video_txt_guidance_scale = vid_guidance
    evaluator.audio_txt_guidance_scale = aud_guidance
    evaluator.sr_video_txt_guidance_scale = sr_guidance

    output_dir = os.path.join(Path(__file__).resolve().parent, "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    save_prefix = os.path.join(output_dir, f"{timestamp}_{seed}")

    progress(0.02, desc="Starting generation...")

    # Tell the evaluator to stash base-resolution latent if we want raw preview
    want_raw = preview_raw and use_sr
    evaluator._capture_br_latent = want_raw

    # Step callback → Gradio progress bar
    def _on_step(step, total, is_sr):
        phase = "SR denoising" if is_sr else "Base denoising"
        base_frac = 0.05
        end_frac = 0.90
        frac = base_frac + (step / total) * (end_frac - base_frac)
        if is_sr:
            frac = 0.55 + (step / total) * 0.35
        progress(frac, desc=f"{phase} step {step}/{total}")

    evaluator._step_callback = _on_step

    t0 = time.time()
    with _pipeline_lock:
        progress(0.05, desc="Generating…")
        with _capture_stdout() as captured_buf:
            save_path = _pipeline.run_offline(
                prompt=prompt,
                image=image_input,
                audio=audio_input,
                save_path_prefix=save_prefix,
                seed=seed,
                seconds=seconds,
                br_width=br_width,
                br_height=br_height,
                sr_width=sr_width,
                sr_height=sr_height,
            )
        _vram.mark_offloaded("dit")
        _vram.mark_offloaded("sr")

        # Decode raw (pre-SR) preview if requested
        raw_video_path = None
        if want_raw and hasattr(evaluator, '_br_latent_video'):
            progress(0.92, desc="Decoding raw (pre-SR) preview…")
            import numpy as np
            raw_np = evaluator.decode_video(evaluator._br_latent_video)
            if raw_np is not None:
                import imageio
                raw_video_path = os.path.join(output_dir, f"{timestamp}_{seed}_raw.mp4")
                imageio.mimwrite(
                    raw_video_path, raw_np[0],
                    fps=_pipeline.evaluation_config.fps, quality=8,
                    output_params=["-loglevel", "error"],
                )
            del evaluator._br_latent_video, evaluator._br_latent_audio
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    evaluator._capture_br_latent = False
    evaluator._step_callback = None

    progress(1.0, desc="Complete")

    if not os.path.exists(save_path):
        raise gr.Error(f"Generation failed — output not found at {save_path}")

    # --- Build detailed report ---
    lines = []
    lines.append(f"Total: {elapsed:.1f}s  |  Seed: {seed}")
    lines.append(f"Mode: {mode}  |  Prompt source: {'enhanced' if used_enhanced else 'raw'}")
    lines.append(f"Base: {br_width}x{br_height}  |  Duration: {seconds}s  |  Steps: {base_steps}")
    lines.append(f"Guidance — Video: {vid_guidance}  Audio: {aud_guidance}")
    if sr_width:
        lines.append(f"SR: {sr_tier} ({sr_width}x{sr_height})  |  SR Steps: {sr_steps}  |  SR Guidance: {sr_guidance}")
    else:
        lines.append("SR: off")
    if raw_video_path:
        lines.append(f"Raw preview: {os.path.basename(raw_video_path)}")
    lines.append(f"Output: {os.path.basename(save_path)}")
    lines.append("")

    timings = _parse_timing_lines(captured_buf.getvalue())
    if timings:
        lines.append("── Timing Breakdown ──")
        for step_name, duration in timings:
            lines.append(f"  {step_name}: {duration}")

    lines.append("")
    lines.append("── Prompt Used ──")
    lines.append(prompt[:500] + ("..." if len(prompt) > 500 else ""))

    report = "\n".join(lines)

    raw_out = gr.update(value=raw_video_path, visible=True) if raw_video_path else gr.update(value=None, visible=False)
    return save_path, raw_out, report, seed


# ---------------------------------------------------------------------------
# Standalone SR upscale
# ---------------------------------------------------------------------------


def upscale_video(
    input_video: str,
    prompt: str,
    sr_tier_choice: str,
    sr_steps: int,
    sr_guidance: float,
    seed: int,
    randomize_seed: bool,
    progress=gr.Progress(),
):
    """Encode an external video through the VAE, run SR, decode back."""
    import numpy as np

    if _pipeline_loading or _pipeline is None:
        wait_start = time.time()
        while _pipeline_loading and (time.time() - wait_start) < 600:
            elapsed_wait = int(time.time() - wait_start)
            progress(0.0, desc=f"Waiting for models to load… ({elapsed_wait}s)")
            time.sleep(2)
        if _pipeline is None:
            raise gr.Error("Pipeline not initialized. Check server logs.")
    if not input_video:
        raise gr.Error("Upload a video to upscale.")

    sr_tier_map = {"540p": "540p", "1080p": "1080p"}
    sr_tier = sr_tier_map.get(sr_tier_choice)
    if sr_tier is None:
        raise gr.Error("Select an SR resolution (540p or 1080p).")

    if not prompt or not prompt.strip():
        raise gr.Error("A text prompt is required — the SR model is text-guided.")

    if randomize_seed:
        seed = random.randint(0, 2**31 - 1)

    progress(0.0, desc="Loading input video to determine dimensions...")
    import imageio.v3 as iio

    frames = iio.imread(input_video, plugin="pyav")  # (T, H, W, C) uint8
    num_frames = frames.shape[0]
    input_h, input_w = frames.shape[1], frames.shape[2]

    progress(0.02, desc=f"Ensuring SR model ({sr_tier})...")
    _ensure_sr_model(sr_tier)

    sr_width, sr_height = _sr_dims_for(input_w, input_h, sr_tier)
    print(f"  [sr] Upscale SR output: {sr_width}×{sr_height} "
          f"(preserving {input_w}×{input_h} aspect ratio)")

    evaluator = _pipeline.evaluator
    vae_stride = evaluator.vae_stride
    patch_size = evaluator.patch_size
    device = evaluator.device
    dtype = evaluator.dtype

    sr_latent_height = sr_height // vae_stride[1] // patch_size[1] * patch_size[1]
    sr_latent_width = sr_width // vae_stride[2] // patch_size[2] * patch_size[2]
    sr_height = sr_latent_height * vae_stride[1]
    sr_width = sr_latent_width * vae_stride[2]

    progress(0.05, desc="Preparing video tensor...")
    video_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T,C,H,W)
    video_tensor = video_tensor * 2.0 - 1.0  # normalize to [-1, 1]
    video_tensor = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)

    from torch.nn.functional import interpolate as F_interpolate

    br_latent_h = input_h // vae_stride[1] // patch_size[1] * patch_size[1]
    br_latent_w = input_w // vae_stride[2] // patch_size[2] * patch_size[2]
    enc_h = br_latent_h * vae_stride[1]
    enc_w = br_latent_w * vae_stride[2]

    if enc_h != input_h or enc_w != input_w:
        video_tensor = F_interpolate(
            video_tensor.reshape(1, 3, num_frames, input_h, input_w),
            size=(num_frames, enc_h, enc_w),
            mode="trilinear",
            align_corners=True,
        )

    progress(0.10, desc="Encoding video through VAE...")
    video_tensor = video_tensor.to(device=device, dtype=dtype)

    t0 = time.time()
    with _pipeline_lock:
        with torch.inference_mode():
            br_latent_video = evaluator.vae.encode(video_tensor).to(torch.float32)

            latent_length = br_latent_video.shape[2]

            progress(0.25, desc="Interpolating latent to SR resolution...")
            latent_video = F_interpolate(
                br_latent_video,
                size=(latent_length, sr_latent_height, sr_latent_width),
                mode="trilinear",
                align_corners=True,
            )

            if evaluator.noise_value != 0:
                noise = torch.randn_like(latent_video, device=latent_video.device)
                sigmas = evaluator.sigmas.to(latent_video.device)
                sigma = sigmas[evaluator.noise_value]
                latent_video = latent_video * sigma + noise * (1 - sigma**2) ** 0.5

            latent_audio = torch.randn(
                1, num_frames, 64, dtype=torch.float32, device=device
            )

            progress(0.30, desc="Encoding text prompt...")
            from inference.pipeline.prompt_process import get_padded_t5_gemma_embedding
            context, original_context_len = get_padded_t5_gemma_embedding(
                prompt,
                evaluator.txt_model_path,
                device,
                dtype,
                evaluator.config.t5_gemma_target_length,
            )

            progress(0.40, desc=f"Running SR denoising ({sr_steps} steps)...")
            _pipeline.evaluation_config.sr_num_inference_steps = sr_steps
            _pipeline.evaluation_config.sr_video_txt_guidance_scale = sr_guidance
            evaluator.sr_video_txt_guidance_scale = sr_guidance

            _vram.ensure_on_gpu("sr")

            with _capture_stdout() as captured_buf:
                torch.random.manual_seed(seed)
                sr_latent_video, _ = evaluator.evaluate_with_latent(
                    context,
                    original_context_len,
                    None,
                    latent_video.clone(),
                    latent_audio.clone(),
                    sr_steps,
                    is_a2v=False,
                    use_sr_model=True,
                )

            progress(0.85, desc="Decoding SR video...")
            videos_np = evaluator.decode_video(sr_latent_video)
            video_np = videos_np[0]  # (T, H, W, C) uint8

            if _vram.enabled:
                _pipeline.evaluator.sr_model.to(torch.device("cpu"))
                _vram.mark_offloaded("sr")
                gc.collect()
                torch.cuda.empty_cache()

    elapsed = time.time() - t0
    progress(0.95, desc="Saving output...")

    output_dir = os.path.join(Path(__file__).resolve().parent, "output")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    out_path = os.path.join(output_dir, f"{timestamp}_{seed}_sr_{sr_tier}.mp4")

    import imageio, subprocess, tempfile

    tmp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    input_fps = _pipeline.evaluation_config.fps
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", input_video],
            capture_output=True, text=True, timeout=10,
        )
        num, den = probe.stdout.strip().split("/")
        input_fps = round(int(num) / int(den))
    except Exception:
        pass
    imageio.mimwrite(tmp_video, video_np, fps=input_fps, quality=8, output_params=["-loglevel", "error"])

    has_audio = False
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", input_video],
            capture_output=True, text=True, timeout=10,
        )
        has_audio = "audio" in probe.stdout
    except Exception:
        pass

    if has_audio:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_video, "-i", input_video,
             "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
             "-shortest", "-loglevel", "error", out_path],
            timeout=120,
        )
        os.unlink(tmp_video)
    else:
        os.rename(tmp_video, out_path)

    progress(1.0, desc="Complete")

    lines = [
        f"Total: {elapsed:.1f}s  |  Seed: {seed}",
        f"Input: {input_w}x{input_h} ({num_frames} frames)  →  SR: {sr_width}x{sr_height}",
        f"SR: {sr_tier}  |  Steps: {sr_steps}  |  Guidance: {sr_guidance}",
        f"Audio: {'carried forward' if has_audio else 'none in source'}",
        f"Output: {os.path.basename(out_path)}",
    ]

    timings = _parse_timing_lines(captured_buf.getvalue())
    if timings:
        lines.append("")
        lines.append("── Timing Breakdown ──")
        for step_name, duration in timings:
            lines.append(f"  {step_name}: {duration}")

    report = "\n".join(lines)
    return out_path, report, seed


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_ui():
    with gr.Blocks(title="daVinci-MagiHuman") as demo:
        gr.Markdown("# daVinci-MagiHuman\nAudio-video generation from text and images.")
        with gr.Row():
            pipeline_status = gr.Textbox(
                label="Pipeline Status",
                value="Loading models in background... (you can set up prompts & settings while waiting)",
                interactive=False,
                max_lines=1,
            )
            check_status_btn = gr.Button("Check Status", size="sm", scale=0)

        def _check_pipeline_status():
            if _pipeline is not None:
                sr_status = f"SR: {_loaded_sr}" if _loaded_sr else "SR: not loaded (select in dropdown)"
                return f"Ready  |  {sr_status}"
            if _pipeline_loading:
                return "Loading base models... (check server console for progress)"
            if _pipeline_error:
                return f"FAILED: {_pipeline_error}"
            return "Not started"

        check_status_btn.click(fn=_check_pipeline_status, outputs=[pipeline_status])

        with gr.Tabs():
            # ==================================================================
            # TAB 1 — Generate
            # ==================================================================
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mode = gr.Radio(
                            choices=["Image to Video", "Text to Video"],
                            value="Image to Video",
                            label="Mode",
                        )
                        raw_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe the scene, character, dialogue, and background sound...",
                            lines=5,
                        )
                        image = gr.Image(label="Reference Image", type="filepath")
                        audio = gr.Audio(label="Audio (optional — lipsync)", type="filepath")

                        with gr.Group():
                            gr.Markdown("### Prompt Enhancement")
                            with gr.Row():
                                enhance_btn = gr.Button("Enhance Prompt", variant="secondary")
                                use_enhanced = gr.Checkbox(
                                    label="Use enhanced prompt for generation",
                                    value=True,
                                )
                            enhanced_prompt = gr.Textbox(
                                label="Enhanced Prompt (editable)",
                                placeholder="Click 'Enhance Prompt' to generate, or write your own enhanced prompt here...",
                                lines=8,
                                interactive=True,
                            )
                            with gr.Accordion("LLM & System Prompt Settings", open=False):
                                gr.Markdown("LLM connection settings (changes take effect immediately, no restart needed):")
                                llm_api_base = gr.Textbox(
                                    label="API Base URL",
                                    value=os.environ.get("LLM_API_BASE_URL", "http://localhost:1234/v1"),
                                    placeholder="http://192.168.1.100:1234/v1",
                                )
                                with gr.Row():
                                    llm_model = gr.Textbox(
                                        label="Model Name",
                                        value=os.environ.get("LLM_MODEL", "default"),
                                    )
                                    llm_vision_model = gr.Textbox(
                                        label="Vision Model (blank = use above)",
                                        value=os.environ.get("LLM_VISION_MODEL", ""),
                                        placeholder="(optional)",
                                    )
                                llm_api_key = gr.Textbox(
                                    label="API Key (only for commercial APIs)",
                                    value=os.environ.get("LLM_API_KEY", ""),
                                    placeholder="sk-... or lm-studio",
                                    type="password",
                                )
                                gr.Markdown("---")
                                gr.Markdown(
                                    f"System prompt loaded from:\n`{_SYSTEM_PROMPT_PATH}`\n\n"
                                    "Edit the file and click reload to pick up changes."
                                )
                                reload_sp_btn = gr.Button("Reload System Prompt", size="sm")
                                system_prompt_preview = gr.Textbox(
                                    label="Current System Prompt (preview)",
                                    value=_get_system_prompt()[:500] + ("..." if len(_get_system_prompt()) > 500 else ""),
                                    lines=4,
                                    interactive=False,
                                )

                        with gr.Group():
                            gr.Markdown("### Generation Settings")
                            with gr.Row():
                                seed = gr.Number(label="Seed", value=42, precision=0)
                                randomize_seed = gr.Checkbox(label="Randomize", value=True)
                            seconds = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1, label="Duration (seconds)"
                            )
                            base_res = gr.Dropdown(
                                choices=list(BASE_RESOLUTION_PRESETS.keys()),
                                value="Auto (match image)",
                                label="Base Resolution",
                            )
                            auto_res_info = gr.Textbox(
                                label="Auto Resolution",
                                value="Will match uploaded image aspect ratio (default 448x256 for T2V)",
                                interactive=False,
                                max_lines=1,
                            )

                        with gr.Group():
                            gr.Markdown("### Super-Resolution")
                            sr_choice = gr.Dropdown(
                                choices=["None", "540p", "1080p"],
                                value="None",
                                label="Super-Resolution Upscale (preserves aspect ratio)",
                            )
                            sr_steps = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="SR Inference Steps",
                            )
                            sr_guidance = gr.Slider(
                                minimum=1.0, maximum=10.0, value=3.5, step=0.5,
                                label="SR Guidance Scale",
                            )
                            preview_raw = gr.Checkbox(
                                label="Also decode raw (pre-SR) output for comparison",
                                value=False,
                            )

                        with gr.Accordion("Advanced", open=False):
                            base_steps = gr.Slider(
                                minimum=1, maximum=50, value=8 if _use_distill else 32, step=1,
                                label="Base Inference Steps",
                            )
                            vid_guidance = gr.Slider(
                                minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                                label="Video Guidance Scale",
                            )
                            aud_guidance = gr.Slider(
                                minimum=1.0, maximum=15.0, value=5.0, step=0.5,
                                label="Audio Guidance Scale",
                            )

                        generate_btn = gr.Button("Generate", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="Generated Video")
                        raw_video_output = gr.Video(
                            label="Raw (pre-SR) Preview",
                            visible=False,
                        )
                        status_output = gr.Textbox(
                            label="Generation Details",
                            interactive=False,
                            lines=12,
                            max_lines=25,
                        )
                        seed_output = gr.Number(label="Seed Used", interactive=False)

            # ==================================================================
            # TAB 2 — Upscale Video
            # ==================================================================
            with gr.TabItem("Upscale Video"):
                gr.Markdown(
                    "Upload an existing video and run the SR model to upscale it. "
                    "The SR model is text-guided, so a prompt describing the content helps quality."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        up_video_in = gr.Video(label="Input Video")
                        up_prompt = gr.Textbox(
                            label="Prompt (describes video content for SR guidance)",
                            placeholder="A woman with dark hair speaking into a microphone...",
                            lines=3,
                        )
                        up_sr_choice = gr.Dropdown(
                            choices=["540p", "1080p"],
                            value="540p",
                            label="Target Resolution (preserves aspect ratio)",
                        )
                        with gr.Row():
                            up_sr_steps = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="SR Inference Steps",
                            )
                            up_sr_guidance = gr.Slider(
                                minimum=1.0, maximum=10.0, value=3.5, step=0.5,
                                label="SR Guidance Scale",
                            )
                        with gr.Row():
                            up_seed = gr.Number(label="Seed", value=42, precision=0)
                            up_randomize = gr.Checkbox(label="Randomize", value=True)
                        up_btn = gr.Button("Upscale", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        up_video_out = gr.Video(label="Upscaled Video")
                        up_status = gr.Textbox(
                            label="Upscale Details",
                            interactive=False,
                            lines=8,
                            max_lines=20,
                        )
                        up_seed_out = gr.Number(label="Seed Used", interactive=False)

        # --- Event wiring: Generate tab ---

        def toggle_image_visibility(mode_value):
            return gr.update(visible=mode_value == "Image to Video")

        mode.change(
            fn=toggle_image_visibility,
            inputs=[mode],
            outputs=[image],
        )

        def _update_auto_res(image_path, res_choice):
            if res_choice != "Auto (match image)":
                return gr.update(visible=False)
            if image_path is None:
                return gr.update(
                    value="No image uploaded — will use 448x256 default",
                    visible=True,
                )
            w, h = _auto_resolution_from_image(image_path)
            from PIL import Image as PILImage
            with PILImage.open(image_path) as img:
                iw, ih = img.size
            return gr.update(
                value=f"Input: {iw}x{ih} → Base: {w}x{h} (aspect-matched, {w*h//1000}k pixels)",
                visible=True,
            )

        image.change(
            fn=_update_auto_res,
            inputs=[image, base_res],
            outputs=[auto_res_info],
        )
        base_res.change(
            fn=_update_auto_res,
            inputs=[image, base_res],
            outputs=[auto_res_info],
        )

        enhance_btn.click(
            fn=_do_enhance,
            inputs=[mode, raw_prompt, image, llm_api_base, llm_api_key, llm_model, llm_vision_model],
            outputs=[enhanced_prompt],
        )

        def _reload_sp():
            sp = reload_system_prompt()
            preview = sp[:500] + ("..." if len(sp) > 500 else "")
            return preview

        reload_sp_btn.click(fn=_reload_sp, outputs=[system_prompt_preview])

        generate_btn.click(
            fn=generate,
            inputs=[
                mode, raw_prompt, enhanced_prompt, use_enhanced,
                image, audio,
                seed, randomize_seed, seconds, base_res,
                sr_choice, sr_steps, sr_guidance,
                base_steps, vid_guidance, aud_guidance,
                preview_raw,
            ],
            outputs=[video_output, raw_video_output, status_output, seed_output],
            concurrency_limit=1,
        )

        # --- Event wiring: Upscale tab ---

        up_btn.click(
            fn=upscale_video,
            inputs=[
                up_video_in, up_prompt, up_sr_choice,
                up_sr_steps, up_sr_guidance,
                up_seed, up_randomize,
            ],
            outputs=[up_video_out, up_status, up_seed_out],
            concurrency_limit=1,
        )

    return demo


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="daVinci-MagiHuman Gradio UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--models-dir",
        type=str,
        default=DEFAULT_MODELS_DIR,
        help=f"Root directory for model checkpoints (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--fp16", action="store_true", help="Use original BF16 models instead of FP8-quantized (default: FP8)")
    parser.add_argument("--base", action="store_true", help="Load the full (non-distill) base model (32 steps, CFG=2). Default is distill (8 steps, no CFG).")
    parser.add_argument("--highvram", action="store_true", help="Keep all models resident in GPU (no CPU offloading). Default is to swap models in/out for lower VRAM usage.")
    parser.add_argument(
        "--ram-limit",
        type=float,
        default=0,
        help="Fraction of system RAM to use as virtual-memory ceiling, applied after model loading (0 to disable, default: disabled)",
    )
    return parser.parse_args()


def _background_load(models_dir: str, ram_limit: float = 0):
    """Download base models and initialize the pipeline in a background thread."""
    global _pipeline_loading, _pipeline_error
    _pipeline_loading = True
    try:
        print(f"\n[bg] Checking base models (dir: {models_dir})")
        ensure_models(models_dir, "none")
        print(f"\n[bg] Initializing pipeline (SR loads on demand)...")
        init_pipeline(models_dir)
        if ram_limit > 0:
            _apply_ram_ceiling(ram_limit)
    except Exception as e:
        _pipeline_error = str(e)
        print(f"\n[bg] *** Pipeline initialization FAILED: {e} ***\n")
        import traceback
        traceback.print_exc()
    finally:
        _pipeline_loading = False


def _apply_ram_ceiling(fraction: float = 0.90):
    """Limit this process's virtual-memory to `fraction` of total system RAM.

    If the process tries to allocate beyond this, it gets a MemoryError
    instead of destabilising the whole server.  Only effective on Linux.
    """
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        limit = int(mem_bytes * fraction)
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        gb = limit / (1024 ** 3)
        print(f"  [ram-limit] Virtual-memory ceiling set to {gb:.1f} GB "
              f"({fraction:.0%} of {mem_bytes / (1024**3):.1f} GB total)")
    except Exception as exc:
        print(f"  [ram-limit] Could not set RAM ceiling: {exc}")


def main():
    global _use_fp8, _use_distill, _highvram
    args = parse_args()
    _use_fp8 = not args.fp16
    _use_distill = not args.base
    _highvram = args.highvram
    _vram.enabled = not _highvram

    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))
    os.chdir(project_root)

    print("=" * 60)
    mode_tags = []
    if _use_fp8:
        mode_tags.append("FP8")
    mode_tags.append("distill" if _use_distill else "base")
    mode_tags.append("high-VRAM" if _highvram else "offload")
    print(f"daVinci-MagiHuman — Gradio Frontend [{' | '.join(mode_tags)}]")
    print("=" * 60)

    loader = threading.Thread(
        target=_background_load,
        args=(args.models_dir, args.ram_limit),
        daemon=True,
    )
    loader.start()

    print(f"\n[1/2] Launching Gradio on {args.host}:{args.port}")
    print("       (base models loading in background — SR loads on demand)")
    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
