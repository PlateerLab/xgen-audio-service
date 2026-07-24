# xgen-audio-service

Standalone, GPU-hosted **voice services** for XGEN. Deploy them on a GPU box;
XGEN (a separate server) connects over the network by **endpoint only**. These
containers host **no application logic** — they just serve TTS and STT.

```
   ┌────────────────────┐        HTTP         ┌──────────────────────────────┐
   │  XGEN  (server 2)  │  ───────────────▶   │  xgen-audio-service (server 1) │
   │                    │   :9881  /tts       │   omnivoice   (GPU)            │
   │  points its TTS/   │   :8001  /v1/audio  │   whisper-stt (GPU)           │
   │  STT config here   │ ◀───────────────    │                              │
   └────────────────────┘     audio/text      └──────────────────────────────┘
```

| Service | Port | Endpoints | What |
|---|---|---|---|
| **omnivoice** (TTS) | `9881` | `POST /tts` · `GET /voices` · `GET /languages` · `GET /health` | [OmniVoice](https://github.com/k2-fsa/OmniVoice) — 600+ languages, voice **cloning** + voice **design**, streaming (wav/mp3/ogg/pcm) |
| **whisper-stt** (STT) | `8001` | `POST /v1/audio/transcriptions` (OpenAI-compatible) · `GET /health` | `openai/whisper-large-v3` on vLLM |

Self-contained and consumer-independent — add engines, endpoints, or XGEN-specific
tuning here without touching any consumer.

## Requirements
- NVIDIA GPU(s) + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Docker + Docker Compose

## Deploy
```bash
cp .env.example .env       # set GPU ids / ports / models
docker compose up --build -d
docker compose ps          # wait for (healthy) — first boot downloads models
curl -fsS http://localhost:9881/health   # omnivoice
curl -fsS http://localhost:8001/health   # whisper
```
First boot downloads model weights (~3 GB) into named volumes; later boots are
fast (cache hit).

## Multi-GPU

Each service is **pinned to a physical GPU** via `*_GPU_ID` in `.env`
(implemented as compose `device_ids`). `cuda:0` inside a container always means
its *first visible* GPU, so you only change the id.

| Box | Config |
|---|---|
| **1 GPU** | `OMNIVOICE_GPU_ID=0` `WHISPER_GPU_ID=0` — they share GPU 0, split by `*_GPU_MEMORY_FRACTION` (defaults 0.65 / 0.35). |
| **2 GPUs** | `OMNIVOICE_GPU_ID=0` `WHISPER_GPU_ID=1` — each owns a GPU; raise both fractions to ~0.85. |
| **3+ GPUs (pool)** | Use `deploy/gpu-pool.example.yml`: N omnivoice replicas (one per GPU) behind an nginx LB on a single `:9881`, whisper on its own GPU. Clients still see one endpoint. |
| **Big STT model** | Give whisper several GPUs (multiple `device_ids`) and set `WHISPER_TENSOR_PARALLEL` to that count to shard the model. |

GPU-pool deploy:
```bash
# edit deploy/gpu-pool.example.yml device_ids / replica count for your box
docker compose -f deploy/gpu-pool.example.yml up --build -d
```

## Connecting XGEN
Point XGEN's TTS/STT endpoint config at this host's published ports — nothing
else changes; the request/response contracts are stable.
- TTS → `http://<gpu-host>:9881`  (`POST /tts`)
- STT → `http://<gpu-host>:8001`  (`POST /v1/audio/transcriptions`)

### Example — synthesize
Just pass a `voice_profile` id (from `GET /voices`) — the server resolves the clone
reference (and its transcript) from `voices/<profile>/profile.json`, with
`emotion` → `neutral` → any-available fallback. No need to know on-disk paths.
```bash
curl -X POST http://<gpu-host>:9881/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요","voice_profile":"paimon_ko","emotion":"joy","format":"wav"}' \
  --output out.wav
```
`emotion` is optional (defaults to `neutral`). Advanced callers can still pass an
explicit `ref_audio_path` + `mode:"clone"` instead of `voice_profile`.
### Example — transcribe
```bash
curl -X POST http://<gpu-host>:8001/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 -F file=@clip.wav
```

## Voice profiles
TTS clone references live in [`voices/`](./voices/) — one dir per `voice_profile`
(`<profile>/profile.json` + `ref_*.wav`). See [`voices/README.md`](./voices/README.md).
A few defaults ship so it works immediately.

## Layout
```
xgen-audio-service/
├── docker-compose.yml            # the two services (GPU-pinned, published ports)
├── .env.example                  # ports · per-service GPU ids · fractions · models
├── deploy/
│   ├── gpu-pool.example.yml       # scale: omnivoice replicas per GPU + nginx LB
│   └── nginx-omnivoice-lb.conf    # LB config for the pool
├── omnivoice/                     # OmniVoice TTS service (FastAPI + vendored model)
├── whisper-stt/                   # Whisper STT service (vLLM image)
└── voices/                        # TTS clone reference profiles
```

## Customizing
- **Add a TTS engine / endpoint** → extend `omnivoice/server/`.
- **Swap the STT model** → set `WHISPER_MODEL` (any vLLM-supported ASR model).
- **Tune GPU** → `.env` (ids, fractions, tensor-parallel, Blackwell attention backend).
- Consumers only depend on the HTTP contract, so internal changes are safe.

License: Apache-2.0. OmniVoice inference code is a vendored snapshot of
[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) under its upstream license.
