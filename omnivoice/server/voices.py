"""Voice profile discovery — scans ``settings.voices_dir`` and surfaces
each subdirectory as a ``VoiceProfile``.

Compatible with the existing GPT-SoVITS profile layout used by XGEN:

    <voices_dir>/<profile_id>/
        profile.json
        ref_neutral.wav
        ref_joy.wav
        ...

Only ``profile.json`` keys we actually consume are touched; unknown keys
are passed through unchanged so the file can stay shared with GPT-SoVITS.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from server.schemas import VoiceProfile, VoiceRefAudio

logger = logging.getLogger(__name__)


def _safe_load_profile_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except FileNotFoundError:
        return {}
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to parse profile.json at %s", path)
        return {}


def _scan_one(profile_dir: str, profile_id: str) -> Optional[VoiceProfile]:
    if not os.path.isdir(profile_dir):
        return None

    data = _safe_load_profile_json(os.path.join(profile_dir, "profile.json"))
    refs: list[VoiceRefAudio] = []

    emotion_refs = data.get("emotion_refs") if isinstance(data, dict) else None
    if isinstance(emotion_refs, dict):
        for emotion, meta in emotion_refs.items():
            if not isinstance(meta, dict):
                continue
            file_name = meta.get("file") or f"ref_{emotion}.wav"
            full = os.path.join(profile_dir, file_name)
            if not os.path.isfile(full):
                continue
            refs.append(
                VoiceRefAudio(
                    emotion=emotion,
                    file=full,
                    prompt_text=meta.get("prompt_text"),
                    prompt_lang=meta.get("prompt_lang"),
                )
            )

    if not refs:
        # Fall back to filesystem scan for any ref_<emotion>.wav files.
        for entry in sorted(os.listdir(profile_dir)):
            if entry.startswith("ref_") and entry.endswith(".wav"):
                emotion = entry[len("ref_"):-len(".wav")]
                refs.append(
                    VoiceRefAudio(
                        emotion=emotion,
                        file=os.path.join(profile_dir, entry),
                    )
                )

    return VoiceProfile(
        id=profile_id,
        name=str(data.get("display_name") or data.get("name") or profile_id),
        language=data.get("language") if isinstance(data, dict) else None,
        is_template=bool(data.get("is_template", False)) if isinstance(data, dict) else False,
        ref_audios=refs,
    )


def list_profiles(voices_dir: str) -> list[VoiceProfile]:
    if not os.path.isdir(voices_dir):
        logger.warning("voices_dir does not exist: %s", voices_dir)
        return []
    out: list[VoiceProfile] = []
    for entry in sorted(os.listdir(voices_dir)):
        full = os.path.join(voices_dir, entry)
        profile = _scan_one(full, entry)
        if profile is not None:
            out.append(profile)
    return out


def get_profile(voices_dir: str, profile_id: str) -> Optional[VoiceProfile]:
    return _scan_one(os.path.join(voices_dir, profile_id), profile_id)


def resolve_ref_audio(voices_dir: str, profile_id: str, emotion: str) -> Optional[VoiceRefAudio]:
    """Pick a reference audio for ``emotion`` with neutral fallback."""
    profile = get_profile(voices_dir, profile_id)
    if profile is None or not profile.ref_audios:
        return None
    by_emotion = {r.emotion: r for r in profile.ref_audios}
    if emotion in by_emotion:
        return by_emotion[emotion]
    if "neutral" in by_emotion:
        return by_emotion["neutral"]
    return profile.ref_audios[0]


# ── Voice profile management (create / update / delete + emotion refs) ────────
#
# XGEN 관리자 Voice Studio 가 사용하는 쓰기 표면. 프로필 = 디렉터리 + profile.json,
# 레퍼런스 = ref_<emotion>.<ext> 파일 + profile.json 의 emotion_refs 항목.
# 템플릿(is_template=true) 프로필은 삭제/레퍼런스 변경을 거부해 기본 보이스를 보호한다.

import re
import shutil

PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
ALLOWED_REF_EXTS = (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac")


class VoiceManagementError(Exception):
    """Management failure with an HTTP-ish status code."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def _profile_dir(voices_dir: str, profile_id: str) -> str:
    if not PROFILE_ID_RE.match(profile_id or ""):
        raise VoiceManagementError(400, f"invalid profile id: {profile_id!r} (lowercase a-z0-9_-)")
    return os.path.join(voices_dir, profile_id)


def _write_profile_json(profile_dir: str, data: dict) -> None:
    with open(os.path.join(profile_dir, "profile.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def _load_for_update(voices_dir: str, profile_id: str) -> tuple[str, dict]:
    pdir = _profile_dir(voices_dir, profile_id)
    if not os.path.isdir(pdir):
        raise VoiceManagementError(404, f"voice profile not found: {profile_id}")
    return pdir, _safe_load_profile_json(os.path.join(pdir, "profile.json"))


def _reject_template(data: dict, action: str) -> None:
    if bool(data.get("is_template", False)):
        raise VoiceManagementError(403, f"template profile cannot be {action}")


def create_profile(voices_dir: str, profile_id: str, display_name: str, language: str | None) -> VoiceProfile:
    pdir = _profile_dir(voices_dir, profile_id)
    if os.path.isdir(pdir):
        raise VoiceManagementError(409, f"voice profile already exists: {profile_id}")
    os.makedirs(pdir, exist_ok=False)
    _write_profile_json(pdir, {
        "display_name": display_name or profile_id,
        "language": language or None,
        "is_template": False,
        "emotion_refs": {},
    })
    profile = get_profile(voices_dir, profile_id)
    assert profile is not None
    return profile


def update_profile(voices_dir: str, profile_id: str, display_name: str | None, language: str | None) -> VoiceProfile:
    pdir, data = _load_for_update(voices_dir, profile_id)
    _reject_template(data, "updated")
    if display_name is not None:
        data["display_name"] = display_name
    if language is not None:
        data["language"] = language or None
    _write_profile_json(pdir, data)
    profile = get_profile(voices_dir, profile_id)
    assert profile is not None
    return profile


def delete_profile(voices_dir: str, profile_id: str) -> None:
    pdir, data = _load_for_update(voices_dir, profile_id)
    _reject_template(data, "deleted")
    shutil.rmtree(pdir)


def save_ref(
    voices_dir: str,
    profile_id: str,
    emotion: str,
    filename: str,
    content: bytes,
    prompt_text: str | None,
    prompt_lang: str | None,
) -> VoiceProfile:
    """감정 레퍼런스 저장/교체 — 파일 + profile.json emotion_refs 갱신."""
    pdir, data = _load_for_update(voices_dir, profile_id)
    _reject_template(data, "modified")
    emotion = (emotion or "").strip().lower()
    if not re.match(r"^[a-z]{2,16}$", emotion):
        raise VoiceManagementError(400, f"invalid emotion: {emotion!r}")
    if not content:
        raise VoiceManagementError(400, "empty audio file")
    ext = os.path.splitext(filename or "")[1].lower() or ".wav"
    if ext not in ALLOWED_REF_EXTS:
        raise VoiceManagementError(400, f"unsupported audio extension: {ext}")

    # 기존 같은 감정의 레퍼런스 파일 제거 (확장자가 달라졌을 수 있음)
    refs = data.get("emotion_refs")
    if not isinstance(refs, dict):
        refs = {}
    old = refs.get(emotion)
    if isinstance(old, dict) and old.get("file"):
        old_path = os.path.join(pdir, str(old["file"]))
        if os.path.isfile(old_path):
            try:
                os.unlink(old_path)
            except OSError:
                logger.warning("failed to remove old ref %s", old_path)

    ref_name = f"ref_{emotion}{ext}"
    with open(os.path.join(pdir, ref_name), "wb") as fh:
        fh.write(content)
    entry: dict = {"file": ref_name}
    if prompt_text:
        entry["prompt_text"] = prompt_text
    if prompt_lang:
        entry["prompt_lang"] = prompt_lang
    refs[emotion] = entry
    data["emotion_refs"] = refs
    _write_profile_json(pdir, data)
    profile = get_profile(voices_dir, profile_id)
    assert profile is not None
    return profile


def delete_ref(voices_dir: str, profile_id: str, emotion: str) -> VoiceProfile:
    pdir, data = _load_for_update(voices_dir, profile_id)
    _reject_template(data, "modified")
    refs = data.get("emotion_refs")
    refs = refs if isinstance(refs, dict) else {}
    entry = refs.pop((emotion or "").strip().lower(), None)
    if isinstance(entry, dict) and entry.get("file"):
        path = os.path.join(pdir, str(entry["file"]))
        if os.path.isfile(path):
            try:
                os.unlink(path)
            except OSError:
                logger.warning("failed to remove ref %s", path)
    # 파일시스템 스캔 fallback 대상(ref_<emotion>.wav)도 제거
    scan_path = os.path.join(pdir, f"ref_{(emotion or '').strip().lower()}.wav")
    if os.path.isfile(scan_path):
        try:
            os.unlink(scan_path)
        except OSError:
            pass
    data["emotion_refs"] = refs
    _write_profile_json(pdir, data)
    profile = get_profile(voices_dir, profile_id)
    assert profile is not None
    return profile


def ref_audio_path(voices_dir: str, profile_id: str, emotion: str) -> str:
    """재생용 레퍼런스 파일 경로 (관리 UI 미리듣기)."""
    pdir, _data = _load_for_update(voices_dir, profile_id)
    profile = get_profile(voices_dir, profile_id)
    if profile is None:
        raise VoiceManagementError(404, f"voice profile not found: {profile_id}")
    emotion = (emotion or "").strip().lower()
    for r in profile.ref_audios:
        if r.emotion == emotion and os.path.isfile(r.file):
            return r.file
    raise VoiceManagementError(404, f"ref audio not found: {profile_id}/{emotion}")
