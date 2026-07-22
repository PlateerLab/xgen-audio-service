# Voice profiles (OmniVoice clone references)

One directory per `voice_profile` id used in `POST /tts`. Layout:

```
voices/
└── <profile>/
    ├── profile.json         # metadata (display name, language, default emotion…)
    ├── ref_neutral.wav      # reference clip for the neutral emotion
    ├── ref_<emotion>.wav    # optional per-emotion reference clips
    └── …                    # source clips
```

Mounted read-only into the container at `/voices`. Drop new profiles here; the
service lists them via `GET /voices`. A few defaults ship so it works out of the
box (`paimon_ko`, `ellen_joe`, `mao_pro`, `ruan_mei`).
