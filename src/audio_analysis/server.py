"""Audio Analysis MCP Server.

Splits audio into clips via FFmpeg, sends each to an audio-capable model
for analysis — genre classification, instrument identification, production
analysis, vocal transcription, and more.

Backends:
  - "swiftlm": Local inference via gemma4-cli (VincentGourbin/gemma-4-swift-mlx).
    Gemma 4 with native Conformer audio encoder on Apple Silicon.
  - "gemini-flash": Gemini 3.5 Flash with thinking (cloud). Best accuracy.
  - "google": Any Google AI Studio model with audio capability.
  - "omlx-stt": oMLX /v1/audio/transcriptions (Whisper/Qwen3-ASR, transcription only).
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

# --- Dotenv -------------------------------------------------------------------
# Load .env from project dir, ~/.env, or CWD — checked in order, first wins.
# This lets you keep API keys out of claude_desktop_config.json.
try:
    from dotenv import load_dotenv
    for _p in [
        Path(__file__).resolve().parents[2] / ".env",  # project root
        Path.home() / ".env",
        Path.cwd() / ".env",
    ]:
        if _p.exists():
            load_dotenv(_p)
            break
except ImportError:
    pass

# --- Config -------------------------------------------------------------------

DEFAULT_MODEL = os.environ.get("AUDIO_ANALYSIS_MODEL", "gemma-4-e4b-it")
CLIP_SECONDS = int(os.environ.get("AUDIO_ANALYSIS_CLIP_SECONDS", "30"))
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
BACKEND = os.environ.get("AUDIO_ANALYSIS_BACKEND", "swiftlm")

# SwiftLM / gemma4-cli backend (Gemma 4 native audio tower on Apple Silicon)
GEMMA4_CLI_PATH = os.environ.get(
    "GEMMA4_CLI_PATH",
    "/Users/gwyn/projects/gemma-4-swift-mlx/.build/xcode/Build/Products/Release/gemma4-cli",
)
SWIFTLM_MODEL_PATH = os.environ.get(
    "AUDIO_ANALYSIS_SWIFTLM_MODEL_PATH",
    "/Users/gwyn/.omlx/models/google/gemma-4-E4B-it",
)

# oMLX STT backend
OMLX_BASE_URL = os.environ.get(
    "AUDIO_ANALYSIS_OMLX_BASE_URL", "http://100.75.192.109:27473"
)

# OpenAI-compatible chat backend (limited)
OPENAI_BASE_URL = os.environ.get(
    "AUDIO_ANALYSIS_OPENAI_BASE_URL", "http://100.75.192.109:27473/v1"
)

# API key getters
def _get_omlx_key():
    return os.environ.get("AUDIO_ANALYSIS_OMLX_API_KEY", "donthackme")

def _get_google_key():
    # Try prefixed var first (avoids security masking), then standard
    val = os.environ.get("AUDIO_ANALYSIS_GOOGLE_API_KEY", "")
    if val:
        return val
    return os.environ.get("GOOGLE_API_KEY", "")

# ------------------------------------------------------------------------------

mcp = FastMCP(
    name="audio-analysis",
    instructions=(
        "Audio analysis — split files into clips and analyze with AI. "
        "Use analyze_audio to split a file into 30s clips and analyze each. "
        "Use analyze_clip to analyze a single audio segment. "
        "Use split_audio to just split without analysis. "
        "Backends: 'swiftlm' (Gemma 4 local on Apple Silicon), "
        "'gemini-flash' (Gemini 3.5 Flash with thinking, cloud), "
        "'google' (any AI Studio model with audio), "
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

    Outputs 16kHz mono WAV (compatible with most audio models).
    """
    duration = _probe_duration(path)
    n_clips = int(duration // clip_secs) + (1 if duration % clip_secs > 0 else 0)
    if n_clips == 0:
        n_clips = 1
    tmpdir = tempfile.mkdtemp(prefix="audio_analysis_")
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


# --- Backend: SwiftLM (Gemma 4 native audio tower via gemma4-cli) -----------

async def _analyze_swiftlm(clip_path: str, prompt: str, model: str) -> str:
    """Analyze audio locally via gemma4-cli (Gemma 4 with Conformer audio encoder).

    Uses VincentGourbin/gemma-4-swift-mlx: 12-layer Conformer encoder,
    mel spectrogram (128 bins, 16kHz), masked_scatter embedding merge.
    Max audio: 30 seconds (750 audio tokens).
    Tested on Apple M5 Pro 48 GB with Gemma 4 E4B BF16 (~20 GB GPU peak).
    """
    cli_path = os.environ.get("GEMMA4_CLI_PATH", GEMMA4_CLI_PATH)
    model_path = os.environ.get("AUDIO_ANALYSIS_SWIFTLM_MODEL_PATH", SWIFTLM_MODEL_PATH)
    max_tokens = int(os.environ.get("AUDIO_ANALYSIS_SWIFTLM_MAX_TOKENS", "512"))

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
    """Extract analysis text from gemma4-cli describe output."""
    lines = stdout.split("\n")
    in_generation = False
    past_header = False
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
    text = re.sub(r"<turn\|?>?\s*$", "", text).strip()
    return text if text else stdout.strip()


# --- Backend: Google AI Studio (Gemini, Gemma 4, any multimodal model) -------

async def _analyze_google(clip_path: str, prompt: str, model: str) -> str:
    """Analyze audio via Google AI Studio.

    Supports all models with audio capability:
      - gemini-3.5-flash (with thinking — best accuracy)
      - gemma-4-31b-it, gemma-4-26b-a4b-it (text+vision only, no audio on API)
      - gemini-2.5-flash-native-audio-latest (native audio output)
    """
    from google import genai
    from google.genai import types

    key = _get_google_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY or AUDIO_ANALYSIS_GOOGLE_API_KEY env var required for google/gemini backends")

    audio_bytes = Path(clip_path).read_bytes()
    ext = Path(clip_path).suffix.lower()
    mime_map = {
        ".wav": "audio/wav", ".mp3": "audio/mp3", ".m4a": "audio/mp4",
        ".flac": "audio/flac", ".ogg": "audio/ogg",
    }
    mime = mime_map.get(ext, "audio/wav")

    def _call() -> str:
        client = genai.Client(api_key=key)

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
    base = os.environ.get("AUDIO_ANALYSIS_OMLX_BASE_URL", OMLX_BASE_URL)
    akey = _get_omlx_key()
    audio_bytes = Path(clip_path).read_bytes()
    filename = Path(clip_path).name
    stt_model = os.environ.get("AUDIO_ANALYSIS_OMLX_STT_MODEL", "")

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
    base = os.environ.get("AUDIO_ANALYSIS_OPENAI_BASE_URL", OPENAI_BASE_URL)
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
    clip_path: str, prompt: str, model: str = DEFAULT_MODEL, backend: str = BACKEND,
) -> str:
    """Route audio analysis to the selected backend."""
    if backend == "swiftlm":
        return await _analyze_swiftlm(clip_path, prompt, model)
    if backend == "omlx-stt":
        return await _analyze_omlx_stt(clip_path, prompt)
    if backend == "openai":
        chat_model = os.environ.get("AUDIO_ANALYSIS_CHAT_MODEL", "gemma-4-e4b-it")
        return await _analyze_openai(clip_path, prompt, chat_model)
    if backend == "gemini-flash":
        return await _analyze_google(clip_path, prompt, "gemini-3.5-flash")
    # Default: google backend with specified model
    return await _analyze_google(clip_path, prompt, model)


# --- MCP Tools ---------------------------------------------------------------

@mcp.tool
def split_audio(audio_path: str, clip_seconds: int = CLIP_SECONDS) -> dict:
    """Split an audio file into clips of N seconds using FFmpeg.

    Args:
        audio_path: Path to the audio file (wav, mp3, m4a, flac, etc.)
        clip_seconds: Duration of each clip in seconds (default: 30)

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
    model: str = DEFAULT_MODEL,
    backend: str = BACKEND,
) -> str:
    """Analyze a single audio file. Backend determines analysis depth.

    Args:
        audio_path: Path to audio file (wav, mp3, m4a, flac, etc.)
        prompt: Question or instruction about the audio
        model: Model name (default: gemma-4-e4b-it)
        backend: 'swiftlm' (Gemma 4 local, Apple Silicon),
                 'gemini-flash' (Gemini 3.5 Flash with thinking, cloud),
                 'google' (any AI Studio audio model),
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
    model: str = DEFAULT_MODEL,
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
        clip_seconds: Seconds per clip (default 30)
        backend: 'swiftlm', 'gemini-flash', 'google', 'omlx-stt', or 'openai'

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


# --- MCP Resources: Suggested Prompts ----------------------------------------

PROMPT_CATEGORIES = {
    "genre-classification": {
        "name": "Genre & Style Classification",
        "prompts": [
            "What genre of music is this? Be specific about subgenres.",
            "Classify this track's genre and list the musical traditions it draws from.",
            "What era or decade does this music sound like it's from? What gives it away?",
            "Is this more [genre A] or [genre B]? Explain the hybrid elements.",
            "What scene or movement does this track belong to?",
        ],
    },
    "instrumentation": {
        "name": "Instruments & Sound Sources",
        "prompts": [
            "List every instrument and sound source you can identify in this clip.",
            "Is that a real [instrument] or a synthesizer imitating one? How can you tell?",
            "What's creating that [description] sound around [timestamp]?",
            "Describe the bass tone — is it synthetic, sampled, or played live?",
            "How many layers of sound can you count? Describe each.",
        ],
    },
    "production": {
        "name": "Production & Engineering",
        "prompts": [
            "Describe the production style — compression, reverb, stereo width, distortion.",
            "How would you recreate this sound? What effects chains are on the vocals?",
            "Is this mixed for headphones, club PA, or radio? What tells you?",
            "Rate the loudness and dynamic range. Is it crushed or breathing?",
            "What production techniques define this track's sound?",
        ],
    },
    "vocals": {
        "name": "Vocal Analysis",
        "prompts": [
            "Transcribe any lyrics you can make out.",
            "Describe the vocal processing — autotune, reverb, delay, compression, distortion.",
            "What vocal technique is being used? Belting, falsetto, talk-singing, screaming?",
            "Is this one vocalist or multiple? How are they layered?",
            "Describe the emotional delivery of the vocals.",
        ],
    },
    "mood-atmosphere": {
        "name": "Mood & Atmosphere",
        "prompts": [
            "Describe the emotional landscape of this track in three words, then explain.",
            "What would you be doing if this was playing? Set the scene.",
            "Does this track feel nostalgic, futuristic, or outside of time? Why?",
            "Map the energy arc — does it build, plateau, drop, or stay flat?",
            "What color does this music make you think of? Why?",
        ],
    },
    "rhythm-tempo": {
        "name": "Rhythm & Tempo",
        "prompts": [
            "Estimate the BPM and time signature.",
            "Describe the drum pattern — four-on-the-floor, breakbeat, half-time, swung?",
            "Is there a groove or is the rhythm deliberately unstable?",
            "Where does the rhythmic tension come from?",
            "Compare this rhythm to [reference track/artist].",
        ],
    },
    "comparison": {
        "name": "Comparative & Reference",
        "prompts": [
            "What does this remind you of? Name specific artists and tracks.",
            "How does this compare to [artist]'s typical sound?",
            "If you had to file this next to one album in a record store, which one?",
            "What's the closest mainstream equivalent to this underground track?",
            "Who are the 3 artists that most influenced this sound?",
        ],
    },
    "musicology": {
        "name": "Musicology & Theory",
        "prompts": [
            "What key is this in? Major, minor, or modal?",
            "Describe the harmonic movement — is it static, cyclical, or progressive?",
            "What's the song structure? Intro-verse-chorus or something less conventional?",
            "Identify any samples, interpolations, or references you recognize.",
            "What's the most interesting musical choice in this clip?",
        ],
    },
    "a-b-diagnostic": {
        "name": "A/B Testing & Diagnostic",
        "prompts": [
            "Analyze this 30-second clip for genre, instruments, production, and mood. Be as specific as possible about subgenres and production techniques.",
            "Listen carefully and describe what you hear — focus on details that would distinguish this from similar-sounding tracks.",
            "What would a music critic write about this track? Be honest about both strengths and derivativeness.",
            "What's the single most distinctive element in this clip?",
            "If this were playing at a party, who would ask 'what is this' and who would leave the room?",
        ],
    },
}


@mcp.resource("audio-analysis://prompts")
def get_prompt_catalog() -> str:
    """Full catalog of suggested prompts organized by category."""
    lines = ["# Suggested Prompts for Audio Analysis", ""]
    for key, cat in PROMPT_CATEGORIES.items():
        lines.append(f"## {cat['name']} (`{key}`)")
        lines.append("")
        for p in cat["prompts"]:
            lines.append(f"- {p}")
        lines.append("")
    return "\n".join(lines)


@mcp.resource("audio-analysis://prompts/{category}")
def get_prompts_by_category(category: str) -> str:
    """Get suggested prompts for a specific analysis category."""
    cat = PROMPT_CATEGORIES.get(category)
    if not cat:
        available = ", ".join(sorted(PROMPT_CATEGORIES.keys()))
        return f"Unknown category '{category}'. Available: {available}"
    lines = [f"# {cat['name']}", ""]
    for i, p in enumerate(cat["prompts"], 1):
        lines.append(f"{i}. {p}")
    return "\n".join(lines)


@mcp.resource("audio-analysis://backends")
def get_backend_info() -> str:
    """Info about available backends and when to use each."""
    return """# Audio Analysis Backends

## swiftlm (default)
- **Model**: Gemma 4 E4B BF16 (local, ~20 GB GPU on Apple Silicon)
- **Speed**: ~5.5 tok/s, ~55s for 30s clip + 300 tokens
- **Audio**: Real — Conformer encoder + mel spectrogram (750 tokens/30s)
- **Best for**: Local inference, privacy, no API dependency
- **Weakness**: Dense layered tracks (wall of sound) may need genre hints in prompt
- **Tip**: Add genre hints for dense material (e.g., "This is hyperpop territory")

## gemini-flash
- **Model**: Gemini 3.5 Flash (cloud, with 10k-token thinking budget)
- **Speed**: ~2-5s response
- **Audio**: Real — full native audio pipeline
- **Best for**: Dense/complex tracks, BPM estimation, lyric transcription, subgenre precision
- **Weakness**: Requires API key, cloud dependency
- **Requires**: `AUDIO_ANALYSIS_GOOGLE_API_KEY` or `GOOGLE_API_KEY`

## google
- **Model**: Any AI Studio model (specify via `model` param)
- **Speed**: Varies
- **Audio**: Gemini models yes, Gemma 4 models NO (text+vision only on API)
- **Best for**: Custom model selection
- **Requires**: `GOOGLE_API_KEY`

## omlx-stt
- **Model**: Whisper/Qwen3-ASR (requires STT model loaded on oMLX)
- **Speed**: Fast
- **Audio**: Transcription only — no analysis
- **Best for**: Speech-to-text when you don't need understanding

## openai
- **Model**: Any OpenAI-compatible endpoint
- **Speed**: Fast
- **Audio**: NONE — can only see file metadata
- **Best for**: Fallback only
"""


# --- Entry -------------------------------------------------------------------

def main():
    mcp.run()

if __name__ == "__main__":
    main()
