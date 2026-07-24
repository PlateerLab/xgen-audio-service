#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
#  Whisper STT entrypoint
#
#  Boots vLLM's OpenAI-compatible API server with the transcription
#  task for openai/whisper-large-v3. The model loads lazily on the
#  first request, so the healthcheck `start_period` in docker-compose
#  is generous (600s) to absorb the first-time HF download.
#
#  Uses the `vllm serve` CLI shipped by the upstream image (in
#  /usr/local/bin/vllm). The earlier `python -m
#  vllm.entrypoints.openai.api_server` invocation broke on the
#  upstream image because it only ships `python3`, not a `python`
#  shim, and `exec python` aborts with `not found`.
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${WHISPER_MODEL:-openai/whisper-large-v3}"
PORT="${WHISPER_PORT:-8001}"
GPU_MEM_FRACTION="${WHISPER_GPU_MEMORY_FRACTION:-0.35}"
DTYPE="${WHISPER_DTYPE:-auto}"
# CLI flag in vllm 0.20.x (the env var of the same name was removed
# — it now warns "Unknown vLLM environment variable detected").
# TRITON_ATTN compiles its kernel JIT, so it doesn't depend on a
# precompiled FA2 binary that would need an sm_120 build for the
# RTX 5070. FLASH_ATTN / FLASHINFER can be tried via this env once
# vllm publishes Blackwell-ready binaries.
# 'auto'(또는 빈 값) → --attention-backend 를 넘기지 않고 vLLM 이 자동 선택(최신 vllm +
# sm_120 Blackwell 권장 — 구 vllm 워크어라운드 TRITON_ATTN 강제가 오히려 최신 스택에서
# 엔진 코어 초기화를 깨는 경우가 있음). TRITON_ATTN/FLASH_ATTN/FLASHINFER 명시도 가능.
ATTENTION_BACKEND="${WHISPER_ATTENTION_BACKEND:-TRITON_ATTN}"
# --enforce-eager 토글. 1/true(기본) → eager(컴파일/cudagraph 스톨 회피, 구 sm_120 안전).
# 0/false → torch.compile 허용(최신 vllm+새 GPU 에서 성능 이득 가능).
ENFORCE_EAGER="${WHISPER_ENFORCE_EAGER:-1}"

echo "[whisper-stt] starting vLLM"
echo "[whisper-stt]   model    = ${MODEL}"
echo "[whisper-stt]   port     = ${PORT}"
echo "[whisper-stt]   gpu_frac = ${GPU_MEM_FRACTION}"
echo "[whisper-stt]   dtype    = ${DTYPE}"
# Multi-GPU: tensor-parallel-size shards the model across N GPUs. Whisper-
# large-v3 fits comfortably on one GPU, so the default is 1; raise it only for
# a bigger STT model or to pool VRAM. The container must SEE that many GPUs
# (docker-compose device_ids / NVIDIA_VISIBLE_DEVICES).
TENSOR_PARALLEL="${WHISPER_TENSOR_PARALLEL:-1}"
MAX_MODEL_LEN="${WHISPER_MAX_MODEL_LEN:-448}"
echo "[whisper-stt]   attn     = ${ATTENTION_BACKEND}"
echo "[whisper-stt]   tp_size  = ${TENSOR_PARALLEL}"

# `--task transcription` was the explicit selector under vllm 0.7/0.8.
# vllm ≥ 0.10 introspects the model architecture and routes Whisper
# automatically to the audio/transcription path — the CLI rejects
# the flag now (`unrecognized arguments: --task transcription`).
# Loading the model alone is enough; the OpenAI-compatible
# /v1/audio/transcriptions endpoint comes up for free.
#
# `--enforce-eager` disables vLLM's inductor (torch.compile) +
# cudagraph capture pipeline. On the prod RTX 5070 (sm_120) the
# default VLLM_COMPILE path stalled at the "Using FLASH_ATTN
# attention backend" step and never made it to listening — the
# encoder-decoder + sm_120 + inductor combo hangs (or takes 30 min+
# even on a hot cache). Whisper-large-v3 at ~1.5 B params is small
# enough that eager mode gives us plenty of headroom on this GPU
# (RTF still well below 1 for 30 s chunks). Drop the flag if a
# future vllm release publishes precompiled kernels for Blackwell.
ARGS=(
    --host 0.0.0.0
    --port "${PORT}"
    --gpu-memory-utilization "${GPU_MEM_FRACTION}"
    --dtype "${DTYPE}"
    --max-model-len "${MAX_MODEL_LEN}"
)
case "${ENFORCE_EAGER}" in
    0|false|no|off) echo "[whisper-stt]   eager    = off (torch.compile enabled)" ;;
    *) ARGS+=(--enforce-eager); echo "[whisper-stt]   eager    = on" ;;
esac
# 'auto'/빈 값이면 --attention-backend 를 생략 → vLLM 자동 선택.
if [ -n "${ATTENTION_BACKEND}" ] && [ "${ATTENTION_BACKEND}" != "auto" ]; then
    ARGS+=(--attention-backend "${ATTENTION_BACKEND}")
    echo "[whisper-stt]   attn_arg = --attention-backend ${ATTENTION_BACKEND}"
else
    echo "[whisper-stt]   attn_arg = (auto — vLLM selects)"
fi
# Only pass --tensor-parallel-size when sharding (>1); passing 1 is a no-op but
# some vllm versions still validate GPU count against it.
if [ "${TENSOR_PARALLEL}" -gt 1 ] 2>/dev/null; then
    ARGS+=(--tensor-parallel-size "${TENSOR_PARALLEL}")
fi

exec vllm serve "${MODEL}" "${ARGS[@]}"
