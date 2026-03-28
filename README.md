# Audio LLM Demo

Real-time audio transcription demo powered by Amphion (vLLM) with TEN VAD speech segmentation.
Supports dual-ASR parallel inference (Amphion + Qwen) with normalized, risk-aware fusion.

## Prerequisites

- Python 3.10+
- A running vLLM server with Amphion (OpenAI-compatible API)
- OpenSSL (for self-signed certificate generation)

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set vLLM endpoint (default: http://localhost:8000)
export VLLM_BASE_URL="http://localhost:8000"
export VLLM_MODEL_NAME="Amphion/Amphion-3B"
export SECONDARY_VLLM_BASE_URL="http://localhost:8001"
export SECONDARY_VLLM_MODEL_NAME="Qwen/Qwen3-ASR-1.7B"
export ENABLE_SECONDARY_ASR="1"
export FUSION_SIMILARITY_THRESHOLD="0.85"
export FUSION_MIN_PRIMARY_SCORE="0.55"
export FUSION_MAX_REPETITION_RATIO="0.35"
export FUSION_DISAGREEMENT_THRESHOLD="0.55"
export FUSION_HOTWORD_BOOST="0.12"
export FUSION_PRIMARY_SCORE_MARGIN="0.08"

# Start the server
bash start.sh
```

Open `https://<your-server-ip>:8443` in your browser.

> On first visit, the browser will warn about the self-signed certificate.
> Click **Advanced** -> **Proceed** to continue.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM server address |
| `VLLM_MODEL_NAME` | `Amphion/Amphion-3B` | Model name |
| `SECONDARY_VLLM_BASE_URL` | `http://localhost:8001` | Secondary vLLM server address (Qwen) |
| `SECONDARY_VLLM_MODEL_NAME` | `Qwen/Qwen3-ASR-1.7B` | Secondary model name |
| `ENABLE_SECONDARY_ASR` | `1` | Enable/disable dual-model parallel ASR |
| `FUSION_SIMILARITY_THRESHOLD` | `0.85` | Similarity threshold used in fusion logic |
| `FUSION_MIN_PRIMARY_SCORE` | `0.55` | Minimum quality score required before trusting Amphion |
| `FUSION_MAX_REPETITION_RATIO` | `0.35` | Repetition risk threshold for hallucination fallback |
| `FUSION_DISAGREEMENT_THRESHOLD` | `0.55` | Max disagreement (1-similarity) before fallback checks |
| `FUSION_HOTWORD_BOOST` | `0.12` | Per-hotword boost applied to Amphion quality score |
| `FUSION_PRIMARY_SCORE_MARGIN` | `0.08` | Required Amphion score margin to beat Qwen in conflicts |
| `ASR_REQUEST_TIMEOUT` | `120` | Timeout (seconds) for each ASR model request |
| `VAD_THRESHOLD` | `0.6` | VAD speech probability threshold |
| `VAD_SMOOTHING_ALPHA` | `0.35` | EMA smoothing for VAD probability (larger = smoother) |
| `VAD_START_FRAMES` | `3` | Consecutive speech frames required to start a segment |
| `VAD_END_FRAMES` | `SILENCE_DURATION_MS/10` | Consecutive non-speech frames required to end a segment |
| `SILENCE_DURATION_MS` | `600` | Silence duration (ms) to end a speech segment |
| `PORT` | `8443` | HTTPS server port |

## Run Two vLLM Servers

Start Amphion first:

```bash
bash backend/server.sh
```

Then start Qwen on port 8001 in another terminal:

```bash
bash backend/server_qwen.sh
```

## Architecture

```
Browser (Mic) --WSS--> FastAPI --HTTP--> vLLM#1 (Amphion)
                          |       \
                          |        --> vLLM#2 (Qwen)
                       TEN VAD
                    (speech detection)
                          |
                     Similarity Fusion
```

- **Frontend**: Web Audio API AudioWorklet captures 16kHz PCM, sends via WebSocket
- **Backend**: FastAPI with two concurrent async tasks per connection:
  - VAD Task: processes audio frames, detects speech segments (non-blocking)
  - LLM Task: consumes segments from asyncio.Queue, calls vLLM API (independent)
- **Hotwords**: Managed in the browser UI, synced to backend via WebSocket in real-time
