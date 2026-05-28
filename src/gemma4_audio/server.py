"""Gemma 4 Audio Analysis MCP Server.

Splits audio into 30-second clips via FFmpeg, sends each clip to Gemma 4
for analysis using local inference (gemma4-cli) or cloud APIs.

Backends:
  - "swiftlm": Local Gemma 4 E2B/E4B via gemma4-cli (VincentGourbin/gemma-4-swift-mlx).
    Native Conformer audio encoder on Apple Silicon. Real audio understanding.
  - "google": Uses google-genai SDK with AI Studio (Gemma 4 E2B/E4B).
  - "omlx-stt": Uses oMLX /v1/audio/transcriptions (Whisper/Qwen3-ASR).
  - "openai": OpenAI-compatible text chat (cannot hear audio — fallback only).
"""

from __future__ import annotations

import asyncio
import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

import httpx
from fastmcp import FastMCP

# --- Config -------------------------------------------------------------------

GEMMA_MODEL = os.environ.get("GEMMA4_AUDIO_MODEL", "gemma-4-e4b-it")
CLIP_SECONDS = int(os.environ.get("GEMMA4_CLIP_SECONDS", "30"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
BACKEND = os.environ.get("GEMMA4_AUDIO_BACKEND", "swiftlm")

# SwiftLM / gemma4-cli backend (native audio tower on Apple Silicon)
GEMMA4_CLI_PATH = os.environ.get(
    "GEMMA4_CLI_PATH",
    "/Users/gwyn/projects/gemma-4-swift-mlx/.build/xcode/Build/Products/Release/gemma4-cli",
)
SWIFTLM_MODEL_PATH = os.environ.get(
    "GEMMA4_AUDIO_SWIFTLM_MODEL_PATH",
    "/Users/gwyn/.omlx/models/google/gemma-4-E4B-it",
)

# oMLX STT backend
OMLX_BASE_URL = os.environ.get(
    "GEMMA4_AUDIO_OMLX_BASE_URL", "http://100.75.192.109:27473"
)

# OpenAI-compatible chat backend (limited)
OPENAI_BASE_URL = os.environ.get(
    "GEMMA4_AUDIO_OPENAI_BASE_URL", "http://100.75.192.109:27473/v1"
)

# Shared API key getters
def _get_omlx_key():
    return os.environ.get("GEMMA4_AUDIO_OMLX_API_KEY", "donthackme")

def _get_google_key():
    # Try GEMMA4_AUDIO_GOOGLE_API_KEY first (avoids masking), then GOOGLE_API_KEY
    val = os.environ.get("GEMMA4_AUDIO_GOOGLE_API_KEY", "")
    if val:
        return val
    return os.environ.get("GOOGLE_API_KEY", "")

# ------------------------------------------------------------------------------

mcp = FastMCP(
    name="gemma4-audio",
    instructions=(
        "Audio analysis powered by Gemma 4 and Gemini. "
        "Use analyze_audio to split a file into 30s clips and analyze each. "
        "Use analyze_clip to analyze a single audio segment. "
        "Use split_audio to just split without analysis. "
        "Backends: 'swiftlm' (native audio tower on Apple Silicon), "
        "'gemini-flash' (Gemini 3.5 Flash with thinking, cloud), "
        "'gemma4-cloud' (Gemini 2.5 Flash native audio on AI Studio), "
        "'google' (any AI Studio model, specify model name), "
        "'omlx-stt' (transcription only), "
        "'openai' (text-only fallback)."
    ),
)


# --- FFmpeg helpers -----------------------------------------------------------

def _probe_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def _split_audio(path: str, clip_secs: int = CLIP_SECONDS) -> list[str]:
    """Split audio file into clips of clip_secs seconds using FFmpeg.

    Outputs 16kHz mono WAV (required by Gemma 4 Conformer encoder).
    """
    duration = _probe_duration(path)
    n_clips = int(duration // clip_secs) + (1 if duration % clip_secs > 0 else 0)
    if n_clips == 0:
        n_clips = 1
    tmpdir = tempfile.mkdtemp(prefix="gemma4_audio_")
    out_paths: list[str] = []
    for i in range(n_clips):
        out_path = os.path.join(tmpdir, f"clip_{i:04d}.wav")
        cmd = [
            FFMPEG_BIN, "-y", "-i", path,
            "-ss", str(i * clip_secs), "-t", str(clip_secs),
            "-vn", "-ac", "1", "-ar", "16000",
            "-acodec", "pcm_s16le", out_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        out_paths.append(out_path)
    return out_paths


def _cleanup_clips(paths: list[str]) -> None:
    """Remove temporary clip files and their directory."""
    if not paths:
        return
    tmpdir = os.path.dirname(paths[0])
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


# --- Backend: SwiftLM (native Gemma 4 audio tower via gemma4-cli) -----------

async def _analyze_swiftlm(clip_path: str, prompt: str, model: str) -> str:
    """Send audio to gemma4-cli for local Gemma 4 audio analysis.

    Uses VincentGourbin/gemma-4-swift-mlx which implements:
      - Conformer audio encoder (12-layer)
      - Mel spectrogram (128 bins, 16kHz, USM features)
      - masked_scatter embedding merge (audio tokens at audio_token_id positions)
      - WeightSanitizer for Google BF16 conv weight transpose

    Max audio: 30 seconds (750 audio tokens).
    Tested on Apple M5 Pro 48 GB with Gemma 4 E4B BF16 (~20 GB GPU peak).
    """
    cli_path = os.environ.get("GEMMA4_CLI_PATH", GEMMA4_CLI_PATH)
    model_path = os.environ.get("GEMMA4_AUDIO_SWIFTLM_MODEL_PATH", SWIFTLM_MODEL_PATH)
    max_tokens = int(os.environ.get("GEMMA4_AUDIO_SWIFTLM_MAX_TOKENS", "512"))

    def _run() -> str:
        cmd = [
            cli_path, "describe",
            "--model-path", model_path,
            "--audio", clip_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"gemma4-cli failed (exit {result.returncode}): {result.stderr[:500]}")
        return _parse_cli_output(result.stdout)

    return await asyncio.to_thread(_run)


def _parse_cli_output(stdout: str) -> str:
    """Extract the analysis text from gemma4-cli describe output.

    Output format:
        --- Generation multimodale ---
          Mode: audio
          Prompt: ...
        ---
        <model output here>

        --- Stats ---
    """
    lines = stdout.split("\n")
    in_generation = False
    past_header = False  # True after the second '---' separator
    output_lines = []
    for line in lines:
        if "Generation multimodale" in line:
            in_generation = True
            continue
        if in_generation:
            if line.strip() == "---" or line.strip().startswith("--- "):
                if not past_header:
                    past_header = True
                    continue
                break
            if past_header:
                output_lines.append(line)
    text = "\n".join(output_lines).strip()
    # Strip trailing <turn|> marker that Gemma 4 models emit
    text = re.sub(r"<turn\|?>?\s*$", "", text).strip()
    return text if text else stdout.strip()


# --- Backend: Google AI Studio (native audio) --------------------------------

async def _analyze_google(clip_path: str, prompt: str, model: str) -> str:
    """Analyze audio via Google AI Studio (Gemma 4, Gemini, any multimodal model).

    Supports all models with audio capability on AI Studio:
      - gemma-4-31b-it, gemma-4-26b-a4b-it (native Gemma 4 audio)
      - gemini-3.5-flash (with thinking)
      - gemini-2.5-flash-native-audio-latest (native audio output)
    """
    from google import genai
    from google.genai import types

    key = _get_google_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY or GEMMA4_AUDIO_GOOGLE_API_KEY env var required for google/gemini/gemma4-cloud backends")

    audio_bytes = Path(clip_path).read_bytes()
    ext = Path(clip_path).suffix.lower()
    mime_map = {
        ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }
    mime = mime_map.get(ext, "audio/wav")

    def _call() -> str:
        client = genai.Client(api_key=key)

        # Build config — enable thinking for Gemini models
        config = types.GenerateContentConfig()
        if "gemini" in model.lower():
            config.thinking_config = types.ThinkingConfig(thinking_budget=10000)

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(mime_type=mime, data=audio_bytes)),
                    types.Part(text=prompt),
                ]),
            ],
            config=config,
        )
        return response.text or ""

    return await asyncio.to_thread(_call)


# --- Backend: oMLX STT (transcription only) ----------------------------------

async def _analyze_omlx_stt(clip_path: str, prompt: str) -> str:
    """Transcribe via oMLX /v1/audio/transcriptions (STT models only)."""
    base = os.environ.get("GEMMA4_AUDIO_OMLX_BASE_URL", OMLX_BASE_URL)
    akey = _get_omlx_key()
    audio_bytes = Path(clip_path).read_bytes()
    filename = Path(clip_path).name
    stt_model = os.environ.get("GEMMA4_AUDIO_OMLX_STT_MODEL", "")

    async with httpx.AsyncClient(timeout=120.0) as http:
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"response_format": "verbose_json"}
        if stt_model:
            data["model"] = stt_model

        resp = await http.post(
            f"{base}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {akey}"},
            files=files,
            data=data,
        )
        resp.raise_for_status()
        result = resp.json()

    text = result.get("text", "")
    duration = result.get("duration", "")
    language = result.get("language", "")
    parts = []
    if language:
        parts.append(f"Language: {language}")
    if duration:
        parts.append(f"Duration: {duration}s")
    parts.append(f"Transcription: {text}")
    return "\n".join(parts)


# --- Backend: OpenAI-compatible (text-only, limited) -------------------------

async def _analyze_openai(clip_path: str, prompt: str, model: str) -> str:
    """Text-only metadata analysis. Cannot hear audio."""
    path = Path(clip_path)
    size_kb = path.stat().st_size / 1024
    duration = _probe_duration(clip_path)
    user_msg = (
        f"[Audio clip: {path.name}]\n"
        f"Format: WAV 16kHz mono\n"
        f"Size: {size_kb:.1f} KB, Duration: {duration:.1f}s\n\n"
        f"{prompt}"
    )
    base = os.environ.get("GEMMA4_AUDIO_OPENAI_BASE_URL", OPENAI_BASE_URL)
    akey = _get_omlx_key()

    async with httpx.AsyncClient(timeout=120.0) as http:
        resp = await http.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {akey}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Audio analysis assistant. You can only see file metadata, not hear audio. Be honest about this."},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 256,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# --- Unified ------------------------------------------------------------------

async def _analyze_single_clip(
    clip_path: str, prompt: str, model: str = GEMMA_MODEL, backend: str = BACKEND,
) -> str:
    """Route audio analysis to the selected backend."""
    if backend == "swiftlm":
        return await _analyze_swiftlm(clip_path, prompt, model)
    if backend == "omlx-stt":
        return await _analyze_omlx_stt(clip_path, prompt)
    if backend == "openai":
        chat_model = os.environ.get("GEMMA4_AUDIO_OMLX_CHAT_MODEL", "gemma-4-E4B-it")
        return await _analyze_openai(clip_path, prompt, chat_model)
    if backend == "gemini-flash":
        return await _analyze_google(clip_path, prompt, "gemini-3.5-flash")
    if backend == "gemma4-cloud":
        return await _analyze_google(clip_path, prompt, "gemini-3.5-flash")
    # Default: google backend with specified model
    return await _analyze_google(clip_path, prompt, model)


# --- MCP Tools ---------------------------------------------------------------

@mcp.tool
def split_audio(audio_path: str, clip_seconds: int = CLIP_SECONDS) -> dict:
    """Split an audio file into clips of N seconds using FFmpeg.

    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, flac, etc.)
        clip_seconds: Duration of each clip in seconds (default: 30, max for Gemma 4 audio)

    Returns:
        Dict with clip_count, clip_paths, duration_seconds, clip_seconds
    """
    duration = _probe_duration(audio_path)
    clip_paths = _split_audio(audio_path, clip_seconds)
    return {
        "clip_count": len(clip_paths),
        "clip_paths": clip_paths,
        "duration_seconds": round(duration, 2),
        "clip_seconds": clip_seconds,
    }


@mcp.tool
async def analyze_clip(
    audio_path: str,
    prompt: str = "Describe what you hear in this audio clip.",
    model: str = GEMMA_MODEL,
    backend: str = BACKEND,
) -> str:
    """Analyze a single audio file. Backend determines analysis depth.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, flac, etc.)
        prompt: Question or instruction about the audio
        model: Model name (default: gemma-4-e4b-it)
        backend: Analysis backend — 'swiftlm' (native audio, local),
                 'gemini-flash' (Gemini 3.5 Flash with thinking, cloud),
                 'gemma4-cloud' (Gemini 2.5 Flash native audio on AI Studio),
                 'google' (any AI Studio model via model param),
                 'omlx-stt' (transcription only),
                 'openai' (metadata-only fallback)

    Returns:
        Analysis text from the model
    """
    return await _analyze_single_clip(audio_path, prompt, model, backend)


@mcp.tool
async def analyze_audio(
    audio_path: str,
    prompt: str = "Describe what you hear in this audio clip.",
    model: str = GEMMA_MODEL,
    clip_seconds: int = CLIP_SECONDS,
    backend: str = BACKEND,
) -> dict:
    """Split audio into clips, analyze each, return combined results.

    Best for files longer than 30s — splits into clip_seconds chunks,
    sends each to the selected backend for analysis, returns per-clip
    results and a combined summary.

    Args:
        audio_path: Path to audio file
        prompt: Question about the audio (applied to each clip)
        model: Model name
        clip_seconds: Seconds per clip (default 30, max for Gemma 4)
        backend: 'swiftlm', 'google', 'omlx-stt', or 'openai'

    Returns:
        Dict with total_clips, per-clip analyses, and combined text
    """
    clip_paths = _split_audio(audio_path, clip_seconds)
    sem = asyncio.Semaphore(1 if backend == "swiftlm" else 3)

    async def _guarded(i: int, path: str) -> dict:
        async with sem:
            try:
                analysis = await _analyze_single_clip(path, prompt, model, backend)
            except Exception as exc:
                analysis = f"[ERROR clip {i}: {exc}]"
            return {"clip_index": i, "offset_seconds": i * clip_seconds, "analysis": analysis}

    results = list(await asyncio.gather(*[_guarded(i, p) for i, p in enumerate(clip_paths)]))
    _cleanup_clips(clip_paths)

    parts = []
    for r in results:
        idx = r["clip_index"]
        off = r["offset_seconds"]
        txt = r["analysis"]
        parts.append(f"--- Clip {idx} (offset {off}s) ---\n{txt}")
    combined = "\n\n".join(parts)

    return {
        "total_clips": len(results),
        "clip_seconds": clip_seconds,
        "model": model,
        "backend": backend,
        "analyses": results,
        "combined": combined,
    }


# --- Entry -------------------------------------------------------------------

def main():
    mcp.run()

if __name__ == "__main__":
    main()
