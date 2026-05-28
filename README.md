# gemma4-audio-mcp

MCP server for audio analysis using **Gemma 4** with local inference on Apple Silicon.

Splits audio into 30-second clips via FFmpeg and sends each to Gemma 4 for real audio understanding — genre classification, instrument identification, transcription, mood analysis, and more. Runs **entirely locally** on Mac with M-series chips via [gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx).

## What It Does

| Tool | Description |
|------|-------------|
| `split_audio` | Split an audio file into N-second clips using FFmpeg |
| `analyze_clip` | Analyze a single audio clip (any duration up to 30s) |
| `analyze_audio` | Split into clips, analyze each, return combined results |

## Backends

| Backend | Audio Understanding | Speed | Requirements |
|---------|-------------------|-------|--------------|
| **`swiftlm`** ✨ | **Full** — Conformer encoder + mel spectrogram | ~5.5 tok/s (E4B BF16) | gemma4-cli binary + Gemma 4 model |
| `google` | Full — AI Studio native audio | Cloud | `GOOGLE_API_KEY` |
| `omlx-stt` | Transcription only | Cloud | oMLX with STT model loaded |
| `openai` | None — metadata only | Cloud | Any OpenAI-compatible endpoint |

**`swiftlm` is the recommended backend** — it provides real audio understanding locally on Apple Silicon with no API keys needed.

## Quick Start

### Prerequisites

1. **Python 3.11+** with `pip`
2. **FFmpeg** installed (`brew install ffmpeg`)
3. **gemma4-cli** built from [VincentGourbin/gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx)
4. **Gemma 4 model** with audio tower weights (E2B or E4B)

### Install

```bash
pip install -e .
```

### Build gemma4-cli

> **Requires Xcode** (not just Command Line Tools). Must use `xcodebuild`, not `swift build`.

```bash
git clone https://github.com/VincentGourbin/gemma-4-swift-mlx.git
cd gemma-4-swift-mlx
xcodebuild -scheme gemma4-cli -configuration Release \
  -destination "platform=macOS" -derivedDataPath .build/xcode \
  -skipMacroValidation build
```

Binary: `.build/xcode/Build/Products/Release/gemma4-cli`

### Download Model

Gemma 4 E2B and E4B models include audio tower weights. Only the Google BF16 format is confirmed working:

```bash
# Option A: huggingface-cli (large download, ~15 GB)
huggingface-cli download google/gemma-4-E4B-it --local-dir ~/.omlx/models/google/gemma-4-E4B-it

# Option B: mlx-community 4-bit quant (~5 GB, may need WeightSanitizer fix)
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit --local-dir ~/.omlx/models/gemma-4-e4b-it-4bit
```

### Test

```bash
gemma4-cli describe \
  --model-path ~/.omlx/models/google/gemma-4-E4B-it \
  --audio /path/to/song.mp3 \
  --prompt "What genre is this? Describe instruments and mood."
```

### Run the MCP Server

```bash
# Default: swiftlm backend
gemma4-audio-mcp

# Or with specific backend
GEMMA4_AUDIO_BACKEND=google gemma4-audio-mcp
```

## Configuration

All configuration via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMMA4_AUDIO_BACKEND` | `swiftlm` | Default backend: `swiftlm`, `google`, `omlx-stt`, `openai` |
| `GEMMA4_AUDIO_MODEL` | `gemma-4-e4b-it` | Model name (for API calls) |
| `GEMMA4_CLIP_SECONDS` | `30` | Clip duration in seconds (30 = max for Gemma 4 audio) |
| `GEMMA4_CLI_PATH` | (built-in default) | Path to gemma4-cli binary |
| `GEMMA4_AUDIO_SWIFTLM_MODEL_PATH` | `~/.omlx/models/google/gemma-4-E4B-it` | Path to local Gemma 4 model |
| `GEMMA4_AUDIO_SWIFTLM_MAX_TOKENS` | `512` | Max generation tokens |
| `GOOGLE_API_KEY` | (empty) | Required for `google` backend |
| `GEMMA4_AUDIO_OMLX_BASE_URL` | `http://100.75.192.109:27473` | oMLX endpoint |
| `GEMMA4_AUDIO_OMLX_API_KEY` | (empty) | oMLX API key |

## Hermes Agent Integration

Add to `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  gemma4-audio:
    command: gemma4-audio-mcp
    env:
      GEMMA4_AUDIO_BACKEND: swiftlm
      GEMMA4_CLI_PATH: /path/to/gemma4-cli
      GEMMA4_AUDIO_SWIFTLM_MODEL_PATH: /path/to/gemma-4-E4B-it
```

Then use via any Hermes tool:

```
"Analyze the audio at /Users/gwyn/Downloads/song.mp3 — what genre is it?"
```

## How It Works

### Audio Pipeline (swiftlm backend)

```
Audio file (any format)
  → FFmpeg: convert to 16kHz mono WAV, split into 30s clips
  → gemma4-cli describe --audio clip.wav
    → AVFoundation: decode WAV to float PCM
    → Gemma4AudioFeatureExtractor: mel spectrogram (128 bins, 20ms frames, 10ms hop)
    → AudioEncoder: 12-layer Conformer (depthwise conv, self-attention, FFN)
    → MultimodalEmbedder: project audio features → text embedding space
    → masked_scatter: splice audio embeddings at audio_token_id (258881) positions
    → Gemma4TextModel: autoregressive generation
  → Parse output, return analysis text
```

### Key Technical Details

- **Audio token budget**: 750 tokens max per 30-second clip
- **Audio token ID**: 258881 (gemma4 specific)
- **Conformer encoder**: 12 layers, 1024 hidden dim, depthwise conv1d
- **Mel spectrogram**: 128 mel bins, 16kHz sample rate, USM features
- **Embedding merge**: `masked_scatter` replaces audio token positions with projected audio features
- **GPU memory**: ~20 GB peak for E4B BF16 on M-series Mac

### WeightSanitizer Fix

The Google BF16 model stores conv weights in PyTorch convention `[O, I, kH, kW]`, but MLX Swift expects `[O, kH, kW, I]`. The `WeightSanitizer` in gemma4-swift-mlx transposes these — but had a bug where it only matched keys containing `.conv` (with leading dot), missing audio tower keys like `depthwise_conv1d` and `lconv1d`.

**Fix applied**: Added `conv1d` to the pattern match in `WeightSanitizer.swift`:

```swift
// Before (broken for audio tower):
if isGoogle && newKey.contains(".conv") && newKey.hasSuffix(".weight") && newValue.ndim == 3 {

// After (works):
if isGoogle && (newKey.contains(".conv") || newKey.contains("conv1d")) && newKey.hasSuffix(".weight") && newValue.ndim == 3 {
```

This transposes all 12 depthwise Conv1d weights and 2 Conv2d weights in the audio tower.

## Performance

Tested on Apple M5 Pro 48 GB with Gemma 4 E4B BF16 (~15 GB model):

| Metric | Value |
|--------|-------|
| Generation speed | ~5.5 tok/s |
| GPU peak | ~20 GB |
| Audio preprocessing | <2s for 30s clip |
| Total time (30s clip, 300 tokens) | ~55s |
| Model load time | ~10s cold start |

## Troubleshooting

### `Mismatched parameter ... depthwise_conv1d.weight shape`

The WeightSanitizer fix isn't applied. Ensure you're using the patched `WeightSanitizer.swift` from this project. See [WeightSanitizer Fix](#weightsanitizer-fix) above.

### `SmallVector out of range` crash

This happens with SharpAI/SwiftLM's audio code. Use `gemma4-cli` from VincentGourbin/gemma-4-swift-mlx instead, which uses `masked_scatter` for embedding merge instead of the broken `audioPadding` token insertion approach.

### `audio_config` stripped by oMLX

oMLX's `_strip_audio_config_if_orphaned` removes `audio_config` from Gemma 4 models at load time. This means oMLX chat completions cannot process audio even though the weights exist. Use `swiftlm` backend or `google` backend instead.

### Model not found

Check `GEMMA4_AUDIO_SWIFTLM_MODEL_PATH` points to a directory containing `config.json` and `.safetensors` files. The Google BF16 model at `~/.omlx/models/google/gemma-4-E4B-it/` is confirmed working.

## Project Structure

```
gemma4-audio-mcp/
├── src/gemma4_audio/
│   ├── __init__.py
│   └── server.py          # MCP server with 3 tools, 4 backends
├── pyproject.toml          # Package config (FastMCP, httpx, google-genai)
├── README.md               # This file
├── LICENSE                 # MIT
└── .gitignore
```

## Related Projects

- [VincentGourbin/gemma-4-swift-mlx](https://github.com/VincentGourbin/gemma-4-swift-mlx) — Native Gemma 4 multimodal inference for Apple Silicon via MLX Swift
- [SharpAI/SwiftLM](https://github.com/SharpAI/SwiftLM) — SwiftLM inference server (audio tower crashes at embedding merge, not recommended for audio)
- [ml-explore/mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm) — MLX Swift LM library (no Gemma 4 audio tower in main branch)
- [google/gemma_pytorch](https://github.com/google/gemma_pytorch) — Reference PyTorch implementation

## License

MIT
