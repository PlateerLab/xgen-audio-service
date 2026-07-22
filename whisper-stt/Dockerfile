# ──────────────────────────────────────────────────────────────────────
#  Whisper STT — vLLM 기반 음성 전사 서비스 (Blackwell-compatible)
#
#  Hosts openai/whisper-large-v3 behind vLLM's OpenAI-compatible
#  /v1/audio/transcriptions endpoint. Profile-gated (audio-local)
#  alongside omnivoice on the same GPU.
#
#  Why the official vllm/vllm-openai base?
#  ──────────────────────────────────────
#  The original revision built on top of `nvidia/cuda:12.4.1-runtime`
#  and pip-installed `vllm==0.7.3`, which transitively pulls PyTorch
#  2.5.1 compiled for sm_50..sm_90 only. The prod host is an
#  RTX 5070 (Blackwell, sm_120) → engine init crashes with
#  `RuntimeError: CUDA error: no kernel image is available for
#  execution on the device`. PyTorch 2.5.1 simply has no Blackwell
#  kernels.
#
#  vLLM publishes an official docker image with the right
#  PyTorch / CUDA / cuDNN versions for current GPUs (their
#  Blackwell-capable build pins PyTorch 2.7+ and CUDA 12.8). Pinning
#  ourselves to that image avoids the GPU-version maze entirely —
#  vLLM maintainers do the matching, we just add our entrypoint.
#
#  GPU split (RTX 5070 / 12 GB) — unchanged:
#    omnivoice              0.50  (≈ 6 GB)
#    whisper-stt (this)     0.35  (≈ 4 GB)
#    ──────────────────────────────────────
#    total                  0.85  (≈ 10 GB) — 15% headroom
#
#  Pinning policy: use a specific tag (latest stable) rather than
#  `:latest`. When upgrading, re-verify the API server flags
#  (`--task transcription`, `--gpu-memory-utilization`,
#  `--max-model-len`) still parse — they're stable across the
#  0.10/0.11 line.
# ──────────────────────────────────────────────────────────────────────

FROM vllm/vllm-openai:latest

# HF model cache — bind to the docker volume so we don't re-download
# ~3 GB on every container rebuild. The upstream image doesn't set
# these by default.
ENV HUGGINGFACE_HUB_CACHE=/cache/huggingface \
    HF_HOME=/cache/huggingface
RUN mkdir -p /cache/huggingface

# curl for the docker healthcheck (compose-level). The base image
# may or may not include it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    || true

# Audio decoder dependency for vLLM's OpenAI-compatible transcription
# endpoint. vLLM tries ``soundfile`` (libsndfile) first — fine for
# WAV/FLAC/OGG but **not WebM/Opus**, which is exactly what browser
# MediaRecorder emits in the W3 manual record modal and the V2 VTuber
# STT mode. Without pyav installed, the fallback path raises
# ``ImportError: Please install vllm[audio]`` which the API surface
# re-raises as ``ValueError: Invalid or unsupported audio file``
# (HTTP 400 to the caller, blank transcript on the inbox note).
#
# We force ``--only-binary :all:`` so pip pulls the manylinux wheel
# (bundled ffmpeg libraries) instead of trying a source build — the
# vLLM base image doesn't ship pkg-config / ffmpeg dev headers, so a
# source build trips with "No package 'libavformat' found". The
# pre-built wheel works out of the box on the upstream image's
# Python 3.12 x86_64 layer.
#
# Pinning to ``av`` alone (not the ``vllm[audio]`` extra) keeps the
# image lean — librosa / audioread aren't used by the transcription
# endpoint. Diagnosed against prod webm samples that ``ffprobe`` and
# ``ffmpeg`` decode fine — the bytes are valid, the decoder was
# missing.
RUN pip install --no-cache-dir --only-binary :all: "av>=12.0"

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8001

# Override the upstream ENTRYPOINT (which would run vllm directly
# with its own arg parsing) so we can apply our env-driven flag
# composition in one place.
ENTRYPOINT ["/app/entrypoint.sh"]
