# audio-analysis-mcp

MCP server for audio analysis. Split audio into clips via FFmpeg, analyze with local or cloud AI models.

Supports **genre classification, instrument identification, vocal transcription, production analysis, and mood mapping** across multiple backends — from local Gemma 4 inference on Apple Silicon to cloud Gemini with thinking.

## What It Does

| Tool | Description |
|------|-------------|
| `split_audio` | Split an audio file into N-second clips using FFmpeg |
| `analyze_clip` | Analyze a single audio clip |
| `analyze_audio` | Split into clips, analyze each, return combined results |

## Backends

| Backend | Audio Understanding | Speed | Requirements |
|---------|-------------------|-------|--------------|
| **`swiftlm`** | Full — Gemma 4 Conformer encoder + mel spectrogram | ~5.5 tok/s | gemma4-cli binary + Gemma 4 model |
| **`gemini-flash`** ✨ | Full — Gemini 3.5 Flash with thinking | ~2-5s (cloud) | `GOOGLE_API_KEY` |
| `google` | Full — any AI Studio audio model | Varies | `GOOGLE_API_KEY` |
| `omlx-stt` | Transcription only | Fast | oMLX with STT model |
| `openai` | None — metadata only | Fast | Any OpenAI-compatible endpoint |

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **FFmpeg** (`brew install ffmpeg`)
3. For local inference: **[gemma4-cli](https://github.com/VincentGourbin/gemma-4-swift-mlx)** + Gemma 4 model
4. For cloud: **Google AI Studio API key**

### Install

```bash
pip install -e .
```

### Local Inference (Gemma 4 on Apple Silicon)

Build [gemma4-cli](https://github.com/VincentGourbin/gemma-4-swift-mlx) (requires Xcode):

```bash
git clone https://github.com/VincentGourbin/gemma-4-swift-mlx.git
cd gemma-4-swift-mlx
xcodebuild -scheme gemma4-cli -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build
```

Download a Gemma 4 model with audio tower weights:

```bash
huggingface-cli download google/gemma-4-E4B-it --local-dir ~/.omlx/models/google/gemma-4-E4B-it
```

### Cloud (Gemini 3.5 Flash)

Set your API key:

```bash
export GOOGLE_API_KEY=your-key-here
```

Gemini 3.5 Flash with thinking provides the best accuracy — especially for dense, layered tracks where local models may misclassify.

### Run

```bash
# Default: local swiftlm backend
audio-analysis-mcp

# Cloud: Gemini 3.5 Flash
AUDIO_ANALYSIS_BACKEND=gemini-flash audio-analysis-mcp
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIO_ANALYSIS_BACKEND` | `swiftlm` | Default backend |
| `AUDIO_ANALYSIS_MODEL` | `gemma-4-e4b-it` | Model name |
| `AUDIO_ANALYSIS_CLIP_SECONDS` | `30` | Clip duration (seconds) |
| `GEMMA4_CLI_PATH` | (built-in default) | Path to gemma4-cli binary |
| `AUDIO_ANALYSIS_SWIFTLM_MODEL_PATH` | `~/.omlx/models/google/gemma-4-E4B-it` | Local model path |
| `AUDIO_ANALYSIS_GOOGLE_API_KEY` | (empty) | Google AI Studio API key (preferred) |
| `GOOGLE_API_KEY` | (empty) | Fallback key for google/gemini backends |
| `AUDIO_ANALYSIS_OMLX_BASE_URL` | (default oMLX endpoint) | oMLX endpoint |

## Resources

The server exposes MCP resources for guided analysis:

| Resource URI | Contents |
|---|---|
| `audio-analysis://prompts` | 9 categories, 45 suggested prompts |
| `audio-analysis://prompts/{category}` | Prompts for one category |
| `audio-analysis://backends` | Backend comparison guide |

Prompt categories: `genre-classification`, `instrumentation`, `production`, `vocals`, `mood-atmosphere`, `rhythm-tempo`, `comparison`, `musicology`, `a-b-diagnostic`

## Hermes / Claude Desktop Integration

```yaml
mcp_servers:
  audio-analysis:
    command: audio-analysis-mcp
    env:
      AUDIO_ANALYSIS_BACKEND: swiftlm
      AUDIO_ANALYSIS_GOOGLE_API_KEY: your-key
      GEMMA4_CLI_PATH: /path/to/gemma4-cli
      AUDIO_ANALYSIS_SWIFTLM_MODEL_PATH: /path/to/gemma-4-E4B-it
```

Then: *"Analyze the audio at /path/to/song.mp3 — what genre is it?"*

## Architectural Notes

### Audio Pipeline (swiftlm backend)

```
Audio file → FFmpeg: 16kHz mono WAV, 30s clips
  → gemma4-cli: Conformer encoder (mel spec → audio tokens)
  → masked_scatter: splice embeddings at audio_token_id positions
  → autoregressive text generation → analysis
```

### WeightSanitizer Fix (for Google BF16 models)

Google's BF16 safetensors store conv weights in PyTorch convention `[O, I, kH, kW]`, but MLX Swift expects `[O, kH, kW, I]`. The WeightSanitizer in gemma4-swift-mlx transposes these, but originally only matched keys with `.conv` (leading dot). Audio tower keys like `depthwise_conv1d` and `lconv1d` were missed.

**Fix**: Added `conv1d` to the pattern match in `WeightSanitizer.swift`.

### A/B Testing Results (JACKIE EXTREME - DRUGGABLE)

| Backend | Genre | Accuracy |
|---|---|---|
| swiftlm (local E4B) | Hyperpop / Digital Hardcore / Glitchcore | ✅ Correct with hint prompt |
| gemini-flash (3.5 thinking) | Hyperpop / Digital Hardcore / Electroclash / Glitchcore / Industrial Pop | 🔥 Best — transcribed actual lyrics |

Gemini 3.5 Flash with thinking provides superior accuracy on dense, saturated tracks. Local E4B is solid for cleaner material and when cloud isn't available.

## Performance

M5 Pro 48 GB, Gemma 4 E4B BF16 (~15 GB model):

| Metric | Value |
|--------|-------|
| Generation | ~5.5 tok/s |
| GPU peak | ~20 GB |
| 30s clip + 300 tokens | ~55s |

## Related Projects

- [VincentGourbin/gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx) — Gemma 4 multimodal inference for Apple Silicon
- [SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM) — SwiftLM server (audio crashes at embedding merge)
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — MLX Swift LM library

## License

MIT
