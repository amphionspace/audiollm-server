# Contributing to Audio LLM Demo

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/open-mmlab/Amphion.git
cd audiollm-demo

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies (with dev extras)
pip install -e ".[dev]"

# Copy the example environment file
cp .env.example .env
```

## Code Style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
ruff check .

# Auto-fix lint errors
ruff check --fix .

# Format code
ruff format .
```

## Project Structure

```
backend/
  main.py                  # FastAPI entry point
  config.py                # Environment variable configuration
  session.py               # WebSocket session orchestration
  asr_streaming_session.py # Streaming ASR session
  audio/                   # Audio signal processing
    utils.py               # Resampler, PCM/WAV conversion
    vad.py                 # Voice Activity Detection
  asr/                     # ASR model interaction
    client.py              # vLLM API calls
    fusion.py              # Dual-model fusion logic
    hotword.py             # Hotword extraction service
    prompt.py              # LLM prompt templates
frontend/                  # Static web frontend
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Make your changes and ensure `ruff check .` passes.
3. Update documentation if your changes affect the public API or configuration.
4. Submit a pull request with a clear description of the changes.

## Reporting Issues

Please use GitHub Issues to report bugs or request features. Include:

- Steps to reproduce (for bugs)
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
