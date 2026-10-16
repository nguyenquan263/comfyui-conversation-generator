"""
Workflow builders for the video API. Each pipeline turns (prompt, negative_prompt, options)
into one or more ComfyUI API-format graphs.

Pipelines
  wan          Wan 2.1 1.3B + Flat Color LoRA, text -> 5 s cartoon (832x480 @ 16 fps). Light, proven.
  ltx25        LTX 2.5 22B distilled (Q3_K_S GGUF), text -> video with audio (768x512 @ 24 fps).
  flux2_ltx25  FLUX.2 klein 4B renders a keyframe, LTX 2.5 animates it (image -> video).
  flux2_image  FLUX.2 klein 4B text -> image only.

Every builder returns a dict with the graph and the node id of its output node so the worker
can find the saved file in ComfyUI's history.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path

COMFY_DIR = Path(__file__).resolve().parent / "ComfyUI"
MODELS = COMFY_DIR / "models"

# --------------------------------------------------------------------------- model files
# Everything here is downloadable without a HuggingFace login.
FILES = {
    # Wan 2.1 1.3B cartoon stack
    "wan_unet": ("diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors",
                 "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_fp16.safetensors"),
    "wan_clip": ("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                 "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "wan_vae": ("vae/wan_2.1_vae.safetensors",
                "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"),
    "wan_lora": ("loras/wan_flat_color_1.3b_v2.safetensors",
                 "https://huggingface.co/motimalu/wan-flat-color-1.3b-v2/resolve/main/wan_flat_color_1.3b_v2.safetensors"),
    # FLUX.2 klein 4B (distilled, Apache-2.0)
    "flux2_unet": ("diffusion_models/flux-2-klein-4b.safetensors",
                   "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/diffusion_models/flux-2-klein-4b.safetensors"),
    "flux2_clip": ("text_encoders/qwen_3_4b.safetensors",
                   "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"),
    "flux2_vae": ("vae/flux2-vae.safetensors",
                  "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-4b/resolve/main/split_files/vae/flux2-vae.safetensors"),
    # LTX 2.5 (22B distilled transformer as Q3_K_S GGUF; needs custom_nodes/ComfyUI-GGUF)
    "ltx_unet": ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16-Q3_K_S.gguf",
                 "https://huggingface.co/agosh/LTX-2.5-Comfy-GGUF/resolve/main/ltx-2.5-22b-distilled-transformer-bf16-Q3_K_S.gguf"),
    "ltx_clip": ("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
                 "https://huggingface.co/comfyicu/LTX-2.5/resolve/main/text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
    "ltx_vae": ("vae/ltx-2.5-video-vae-bf16.safetensors",
                "https://huggingface.co/comfyicu/LTX-2.5/resolve/main/vae/ltx-2.5-video-vae-bf16.safetensors"),
    "ltx_audio_vae": ("vae/ltx-2.5-audio-vae-bf16.safetensors",
                      "https://huggingface.co/comfyicu/LTX-2.5/resolve/main/vae/ltx-2.5-audio-vae-bf16.safetensors"),
    "ltx_upscaler": ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
                     "https://huggingface.co/comfyicu/LTX-2.5/resolve/main/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
}

# Exact byte sizes, so a half-downloaded file does not count as present.
SIZES = {
    "wan_unet": 2838303560, "wan_clip": 6735906897, "wan_vae": 253815318, "wan_lora": 87564160,
    "flux2_unet": 7751105712, "flux2_clip": 8044982048, "flux2_vae": 336211292,
    "ltx_unet": 9446940768, "ltx_clip": 15372971786, "ltx_vae": 1472223346,
    "ltx_audio_vae": 364866540, "ltx_upscaler": 995778752,
}

# SHA-256 from HuggingFace, verified by setup_models.py (a resumed download once came back corrupt).
SHA256 = {
    "wan_unet": "be531024cd9018cb5b48c40cfbb6a6191645b1c792eb8bf4f8c1c6e10f924dc5",
    "wan_clip": "c3355d30191f1f066b26d93fba017ae9809dce6c627dda5f6a66eaa651204f68",
    "wan_vae": "2fc39d31359a4b0a64f55876d8ff7fa8d780956ae2cb13463b0223e15148976b",
    "wan_lora": "a95bbbfb764802492702abe98d4841415517708477520cdfacf6a2d81a87357d",
    "flux2_unet": "ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343",
    "flux2_clip": "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
    "flux2_vae": "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
    "ltx_unet": "01fd15e08e33935e7895776e22945b284cdd09660a4df69d343823695015bb9f",
    "ltx_clip": "09a89e084de1a149c3de60cfe9dfd3e5161967eb09eea39e806fcdeffdd568de",
    "ltx_vae": "847e14ca7f3355debca0cea4eaa24ac0fbcdf0061da054ac89ca638a869ddba3",
    "ltx_audio_vae": "c52733d37f6a7fb7949c3dc0fb468c6cb2169e4d836983a73babb9f0d54837a5",
    "ltx_upscaler": "eb5a71fe4068ee87ccdb1c3aa635e547ca76bd2d30ae20ae889f2c325c0677e8",
}

PIPELINE_FILES = {
    "wan": ["wan_unet", "wan_clip", "wan_vae", "wan_lora"],
    "flux2_image": ["flux2_unet", "flux2_clip", "flux2_vae"],
    "ltx25": ["ltx_unet", "ltx_clip", "ltx_vae", "ltx_audio_vae"],
    "flux2_ltx25": ["flux2_unet", "flux2_clip", "flux2_vae", "ltx_unet", "ltx_clip", "ltx_vae", "ltx_audio_vae"],
    "dialogue": ["ltx_unet", "ltx_clip", "ltx_vae", "ltx_audio_vae"],
}

PIPELINE_INFO = {
    "wan": "Wan 2.1 1.3B + Flat Color LoRA: 5 s flat-color cartoon, 832x480 @ 16 fps, ~5 min",
    "ltx25": "LTX 2.5 22B distilled (GGUF): text to video with audio, 768x512 @ 24 fps",
    "flux2_ltx25": "FLUX.2 klein 4B keyframe, then LTX 2.5 image-to-video with audio",
    "flux2_image": "FLUX.2 klein 4B text-to-image (png only)",
    "dialogue": "Two characters talk: one LTX 2.5 shot per line with spoken audio, joined into one video",
}


def file_name(key: str) -> str:
    return Path(FILES[key][0]).name


def file_ok(key: str) -> bool:
    path = MODELS / FILES[key][0]
    return path.exists() and path.stat().st_size >= SIZES.get(key, 1)


def missing_files(pipeline: str) -> list[str]:
    return [FILES[k][0] for k in PIPELINE_FILES[pipeline] if not file_ok(k)]


def pipeline_status() -> dict:
    out = {}
    for p, desc in PIPELINE_INFO.items():
        miss = missing_files(p)
        out[p] = {"description": desc, "available": not miss, "missing_files": miss}
    return out


# --------------------------------------------------------------------------- options

@dataclass
class Options:
    prompt: str
    negative_prompt: str = ""
    pipeline: str = "wan"
    seed: int | None = None
    seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    steps: int | None = None          # wan only (LTX distilled uses fixed sigmas, flux2 klein uses 4)
    cfg: float | None = None          # wan only
    lora_strength: float = 1.0        # wan only
    two_stage: bool = False           # ltx: add the x2 latent upscale pass (needs more VRAM)
    name: str = "video"
    dialogue: dict | None = None      # {"character_a", "character_b", "line_a", "line_b", "scene", "style"}
    extra: dict = field(default_factory=dict)

    def seed_value(self) -> int:
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        return self.seed


def _round(v: int, m: int) -> int:
    return max(m, int(round(v / m)) * m)


WAN_NEGATIVE = (
    "overly bright colors, overexposed, static, blurry details, subtitles, style, artwork, painting, "
    "still image, overall grayish, worst quality, low quality, JPEG artifacts, ugly, incomplete, "
    "extra fingers, badly drawn hands, badly drawn face, deformed, disfigured, malformed limbs, "
    "fused fingers, motionless frame, cluttered background, three legs, many people in background, "
    "walking backwards"
)
LTX_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"


def save_video_node(video_ref: list, prefix: str) -> dict:
    # SaveVideo uses nested "dynamic combo" inputs; the API form is dotted keys.
    return {"class_type": "SaveVideo", "inputs": {
        "video": video_ref, "filename_prefix": f"video/{prefix}",
        "format": "mp4", "format.codec": "h264",
        "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 18.0}}


# --------------------------------------------------------------------------- wan

def build_wan(o: Options) -> dict:
    fps = o.fps or 16
    seconds = o.seconds or 5.0
    frames = int(round(seconds * fps / 4)) * 4 + 1              # 4k+1
    width, height = _round(o.width or 832, 16), _round(o.height or 480, 16)
    neg = o.negative_prompt or WAN_NEGATIVE
    prompt = "flat color, no lineart, cartoon style. " + o.prompt
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": file_name("wan_unet"), "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": file_name("wan_lora"), "strength_model": o.lora_strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": file_name("wan_clip"), "type": "wan", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": file_name("wan_vae")}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["3", 0], "text": neg}},
        "7": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["2", 0], "shift": 8.0}},
        "9": {"class_type": "KSampler", "inputs": {
            "model": ["8", 0], "seed": o.seed_value(), "steps": o.steps or 30, "cfg": o.cfg or 6.0,
            "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0,
            "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["4", 0]}},
        "11": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": float(fps)}},
        "12": save_video_node(["11", 0], o.name),
    }
    return {"graph": g, "output_node": "12", "kind": "video",
            "meta": {"width": width, "height": height, "frames": frames, "fps": fps}}


# --------------------------------------------------------------------------- flux2 klein

def build_flux2_image(o: Options, width: int | None = None, height: int | None = None) -> dict:
    width, height = _round(width or o.width or 1024, 16), _round(height or o.height or 1024, 16)
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": file_name("flux2_unet"), "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": file_name("flux2_clip"), "type": "flux2", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": file_name("flux2_vae")}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": o.prompt}},
        # klein 4B is guidance-distilled: cfg 1 and a zeroed negative, exactly like the official template.
        "5": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["4", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": width, "height": height}},
        "8": {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "cfg": 1.0}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": o.seed_value()}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["9", 0], "guider": ["8", 0], "sampler": ["10", 0], "sigmas": ["7", 0], "latent_image": ["6", 0]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"image/{o.name}"}},
    }
    return {"graph": g, "output_node": "13", "kind": "image", "meta": {"width": width, "height": height}}


# --------------------------------------------------------------------------- ltx 2.5

LTX_SIGMAS_STAGE1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"


def build_ltx25(o: Options, image_file: str | None = None) -> dict:
    """LTX 2.5 distilled text-to-video, or image-to-video when image_file (in ComfyUI/input) is given."""
    fps = o.fps or 24
    seconds = o.seconds or 5.0
    frames = int(round(seconds * fps / 8)) * 8 + 1              # 8k+1
    width, height = _round(o.width or 768, 32), _round(o.height or 512, 32)
    neg = o.negative_prompt or LTX_NEGATIVE
    if o.two_stage:                                             # sample at half size, upscale x2 in latent space
        w1, h1 = _round(width // 2, 32), _round(height // 2, 32)
    else:
        w1, h1 = width, height

    g = {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": file_name("ltx_unet")}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": file_name("ltx_clip"), "type": "ltxv", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": file_name("ltx_vae")}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": file_name("ltx_audio_vae")}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": o.prompt}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": neg}},
        "7": {"class_type": "LTXVConditioning", "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": float(fps)}},
        "8": {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": w1, "height": h1, "length": frames, "batch_size": 1}},
        "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": frames, "frame_rate": float(fps), "batch_size": 1, "audio_vae": ["4", 0]}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "11": {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_STAGE1}},
        "12": {"class_type": "LTXVDualCFGGuider", "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1], "video_cfg": 1.0, "audio_cfg": 1.0}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": o.seed_value()}},
    }
    video_latent = ["8", 0]
    if image_file:
        g["20"] = {"class_type": "LoadImage", "inputs": {"image": image_file}}
        g["21"] = {"class_type": "LTXVPreprocess", "inputs": {"image": ["20", 0], "img_compression": 18}}
        g["22"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["3", 0], "image": ["21", 0], "latent": ["8", 0],
            "strength": 0.7 if o.two_stage else 1.0, "bypass": False}}
        video_latent = ["22", 0]
    g["14"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": video_latent, "audio_latent": ["9", 0]}}
    g["15"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
        "noise": ["13", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["14", 0]}}
    g["16"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["15", 0]}}
    final_video, final_audio = ["16", 0], ["16", 1]

    if o.two_stage:
        g["30"] = {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": file_name("ltx_upscaler")}}
        g["31"] = {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["16", 0], "upscale_model": ["30", 0], "vae": ["3", 0]}}
        up_latent = ["31", 0]
        if image_file:
            g["32"] = {"class_type": "LTXVImgToVideoInplace", "inputs": {
                "vae": ["3", 0], "image": ["21", 0], "latent": ["31", 0], "strength": 1.0, "bypass": False}}
            up_latent = ["32", 0]
        g["33"] = {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": up_latent, "audio_latent": ["16", 1]}}
        g["34"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": LTX_SIGMAS_STAGE2}}
        g["35"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}
        g["36"] = {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["35", 0], "guider": ["12", 0], "sampler": ["10", 0], "sigmas": ["34", 0], "latent_image": ["33", 0]}}
        g["37"] = {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["36", 0]}}
        final_video, final_audio = ["37", 0], ["37", 1]

    g["40"] = {"class_type": "VAEDecodeTiled", "inputs": {
        "samples": final_video, "vae": ["3", 0], "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 16}}
    g["41"] = {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": final_audio, "audio_vae": ["4", 0]}}
    g["42"] = {"class_type": "CreateVideo", "inputs": {"images": ["40", 0], "fps": float(fps), "audio": ["41", 0]}}
    g["43"] = save_video_node(["42", 0], o.name)
    return {"graph": g, "output_node": "43", "kind": "video",
            "meta": {"width": width, "height": height, "frames": frames, "fps": fps, "two_stage": o.two_stage}}


# --------------------------------------------------------------------------- dialogue

def _speech_seconds(line: str) -> float:
    """Shot length from the spoken text: ~2.3 words/s plus room to start and finish."""
    return min(8.0, max(3.0, len(line.split()) / 2.3 + 1.5))


def dialogue_shots(d: dict) -> list[dict]:
    """Two prompts, one per line, written the way LTX wants them: one flowing present-tense
    paragraph covering shot, scene, character, action, camera and audio, with the spoken line in
    quotes so the model voices it with matching lip movement."""
    a, b = d["character_a"].strip(), d["character_b"].strip()
    scene = (d.get("scene") or "a quiet, softly lit room").strip().rstrip(".")
    style = (d.get("style") or "cinematic, realistic lighting, shallow depth of field").strip().rstrip(".")
    shots = []
    for speaker, listener, line in ((a, b, d["line_a"]), (b, a, d["line_b"])):
        line = line.strip().strip('"')
        prompt = (
            f"Medium close-up of {speaker} in {scene}, {style}. {speaker} looks at {listener}, who is partly "
            f"visible at the edge of the frame listening, and speaks clearly with natural lip movement and "
            f"small expressive gestures. {speaker} says: \"{line}\" The camera holds steady with a slow, "
            f"subtle push-in. Audio: {speaker}'s clear voice in sync with the lips, quiet room ambience, no music."
        )
        shots.append({"speaker": speaker, "line": line, "prompt": prompt, "seconds": _speech_seconds(line)})
    return shots


# --------------------------------------------------------------------------- registry

def stages(o: Options) -> list[dict]:
    """The ComfyUI graphs to run for a job, in order. A stage with kind 'image' feeds the next stage."""
    if o.pipeline == "wan":
        return [build_wan(o)]
    if o.pipeline == "flux2_image":
        return [build_flux2_image(o)]
    if o.pipeline == "ltx25":
        return [build_ltx25(o)]
    if o.pipeline == "flux2_ltx25":
        # keyframe at the video's size; the worker fills in image_file after stage 1 finishes.
        w, h = _round(o.width or 768, 32), _round(o.height or 512, 32)
        return [build_flux2_image(o, w, h), {"deferred": "ltx25_i2v"}]
    if o.pipeline == "dialogue":
        if not o.dialogue:
            raise ValueError("the dialogue pipeline needs character_a, character_b, line_a and line_b")
        out = []
        base_seed = o.seed_value()
        for i, shot in enumerate(dialogue_shots(o.dialogue)):
            so = replace(o, prompt=shot["prompt"], seconds=o.seconds or shot["seconds"],
                         seed=base_seed + i, name=f"{o.name}_shot{i + 1}")
            st = build_ltx25(so)
            st["meta"] = {**st["meta"], "shot": i + 1, "speaker": shot["speaker"], "line": shot["line"]}
            out.append(st)
        out.append({"deferred": "concat"})
        return out
    raise ValueError(f"unknown pipeline {o.pipeline!r}; choose one of {list(PIPELINE_INFO)}")
