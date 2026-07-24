# xgen-omnivoice

TTS microservice for XGEN — an HTTP wrapper around [OmniVoice](https://github.com/k2-fsa/OmniVoice)
(600+ languages, voice **cloning** + voice **design**, streaming). Runs on a GPU box;
XGEN connects over the network by endpoint only.

> This directory is the build context for the `omnivoice` image (see the repo-root
> `docker-compose.yml`). The repo root has the top-level deploy guide.

## Endpoints (`:9881`)

| Method / path | Purpose |
|---|---|
| `POST /tts` | Single-shot synthesis → audio bytes (`wav`/`mp3`/`ogg`/`pcm`). |
| `POST /tts/stream` | Sentence-streaming synthesis → NDJSON frames (base64 audio per sentence). |
| `GET /voices` | List available voice profiles (id, name, per-emotion reference audio). |
| `GET /languages` | Supported language names. |
| `GET /health` | Model load phase / readiness. |

## Synthesize — the easy path (`voice_profile`)

Pass a `voice_profile` id from `GET /voices`. The server resolves the clone reference
(and its transcript) from `voices/<profile>/profile.json`, with
`emotion` → `neutral` → any-available **fallback** — you don't need to know on-disk paths.

```bash
curl -X POST http://<gpu-host>:9881/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"안녕하세요","voice_profile":"paimon_ko","emotion":"joy","format":"wav"}' \
  --output out.wav
```

- `voice_profile` (aliases: `voice`, `profile`) — profile id. Implies `mode=clone`.
- `emotion` — optional (default `neutral`); falls back to `neutral`, then any available ref.
- `format` (alias of `audio_format`) — `wav` | `mp3` | `ogg` | `pcm`.
- `speed` (0<x≤4, default 1.0), `language` (auto-detect if omitted), `seed`, `num_step`,
  `guidance_scale`, `sample_rate` — see `server/schemas.py`.

Advanced callers can instead pass an explicit `ref_audio_path` + `mode:"clone"`, or
`instruct` + `mode:"design"` (attribute string, no reference audio), or `mode:"auto"`
(random voice).

## Voice profiles

One directory per profile under [`../voices/`](../voices/):

```
<voices_dir>/<profile_id>/
    profile.json        # name, language, emotion_refs{emotion → {file, prompt_text, prompt_lang}}
    ref_neutral.wav     # per-emotion reference clips (3–10s recommended)
    ref_joy.wav
```

`voices_dir` defaults to `/voices` (mounted read-only from the repo `voices/` in
`docker-compose.yml`). A few presets (`paimon_ko`, `ellen_joe`, `ruan_mei`) ship so it
works immediately — each has at least a `neutral` reference.

## Config

Environment (see the repo-root `.env.example`): `OMNIVOICE_MODEL`, `OMNIVOICE_DEVICE`,
`OMNIVOICE_DTYPE`, `OMNIVOICE_MAX_CONCURRENCY`, `OMNIVOICE_DEFAULT_NUM_STEP`,
`OMNIVOICE_GPU_MEMORY_FRACTION`, `OMNIVOICE_AUTO_ASR`, `OMNIVOICE_LOG_LEVEL`.
