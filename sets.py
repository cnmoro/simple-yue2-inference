"""Sets (playlists) for YouTube-length releases.

Storage is one JSON file per set under ``sets/``. The optional LLM drafting goes
through OpenRouter with the caller's own key; the key is never written here.
Concatenation shells out to ffmpeg, using the binary the caller passes in.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS_DIR = Path(os.environ.get("YUE2_SETS", ROOT / "sets"))
OUTPUTS = Path(os.environ.get("YUE2_OUTPUTS", ROOT / "outputs"))
OPENROUTER_API = os.environ.get("YUE2_OPENROUTER_API", "https://openrouter.ai/api/v1")

# Under pythonw there is no console, so ffmpeg/ffprobe would each open a window.
NO_CONSOLE = getattr(subprocess, "CREATE_NO_WINDOW", 0)

MIN_TRACK_SECONDS = 15
MAX_TRACK_SECONDS = 900  # matches the webUI's max_duration ceiling

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

SONG_FIELDS = (
    "title", "title_translation", "style", "lyrics", "target_seconds",
    "notes", "job_id", "seed", "status", "audio", "audio_seconds",
)


def display_title(song: dict) -> str:
    """What the exports and the chapter list show: Original (Translation)."""
    title = str(song.get("title") or "").strip() or "Untitled"
    translation = str(song.get("title_translation") or "").strip()
    return f"{title} ({translation})" if translation else title


def bake_translation(song: dict) -> bool:
    """Write the translation into the title itself, as Original (Translation)."""
    translation = str(song.get("title_translation") or "").strip()
    if not translation:
        return False
    song["title"] = display_title(song)
    song["title_translation"] = ""
    return True


# --- storage ---------------------------------------------------------------

def _dir() -> Path:
    SETS_DIR.mkdir(parents=True, exist_ok=True)
    return SETS_DIR


def _path(set_id: str) -> Path:
    safe = "".join(c for c in str(set_id) if c.isalnum() or c in "-_")
    if not safe or safe != str(set_id):
        raise ValueError("bad set id")
    return _dir() / f"{safe}.json"


def clamp_seconds(value) -> int:
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        seconds = 300
    return max(MIN_TRACK_SECONDS, min(MAX_TRACK_SECONDS, seconds))


def new_song(order: int, **fields) -> dict:
    song = {
        "id": uuid.uuid4().hex[:8],
        "order": order,
        "title": "",
        "title_translation": "",
        "style": "",
        "lyrics": "",
        "target_seconds": 300,
        "notes": "",
        "job_id": None,
        "seed": None,
        "status": None,
        "audio": None,
        "audio_seconds": None,
    }
    song.update({k: v for k, v in fields.items() if k in song})
    song["order"] = int(order)
    song["target_seconds"] = clamp_seconds(song["target_seconds"])
    return song


def normalize(data: dict) -> dict:
    data = dict(data)
    data.setdefault("id", uuid.uuid4().hex[:12])
    data.setdefault("name", "Untitled set")
    data.setdefault("brief", "")
    data.setdefault("model", "")
    data.setdefault("background", "")
    data.setdefault("translation_language", "")
    data.setdefault("lyrics_language", "English")
    data.setdefault("track_count", 12)
    data.setdefault("target_minutes", 60)
    data.setdefault("created", time.time())
    if data.get("duration_mode") not in ("auto", "fixed"):
        data["duration_mode"] = "auto"
    # Sets used to carry one background prompt per track; promote the first one.
    if not data.get("background"):
        for raw in data.get("songs") or []:
            if isinstance(raw, dict) and raw.get("background"):
                data["background"] = str(raw["background"])
                break

    songs = []
    for index, raw in enumerate(data.get("songs") or [], start=1):
        if not isinstance(raw, dict):
            continue
        base = new_song(index)
        if raw.get("id") is not None:
            base["id"] = str(raw["id"])
        for key in SONG_FIELDS:
            if raw.get(key) is not None:
                base[key] = raw[key]
        base["id"] = str(base.get("id") or uuid.uuid4().hex[:8])
        songs.append(base)
    songs.sort(key=lambda s: int(s.get("order") or 0))
    for index, song in enumerate(songs, start=1):
        song["order"] = index
    data["songs"] = songs
    return data


def summary(data: dict) -> dict:
    songs = data.get("songs") or []
    rendered = [s for s in songs if s.get("audio_seconds")]
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "brief": data.get("brief", ""),
        "target_minutes": data.get("target_minutes"),
        "duration_mode": data.get("duration_mode", "auto"),
        "created": data.get("created"),
        "model": data.get("model", ""),
        "songs": len(songs),
        "rendered": len(rendered),
        "rendered_seconds": round(sum(float(s["audio_seconds"]) for s in rendered), 1),
        "planned_seconds": sum(int(s.get("target_seconds") or 0) for s in songs),
    }


def list_sets() -> list[dict]:
    rows = []
    for path in _dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # A job can finish a moment before its result is written back here, so
            # reconcile from meta.json instead of trusting the file alone.
            rows.append(summary(reconcile(normalize(data))))
        except Exception:  # noqa: BLE001
            continue
    rows.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return rows


def load_set(set_id: str) -> dict:
    path = _path(set_id)
    if not path.is_file():
        raise FileNotFoundError(set_id)
    return normalize(json.loads(path.read_text(encoding="utf-8")))


def save_set(data: dict) -> dict:
    data = normalize(data)
    _path(data["id"]).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return data


def create_set(name: str, target_minutes: float = 60, brief: str = "") -> dict:
    return save_set(
        {
            "id": uuid.uuid4().hex[:12],
            "name": (name or "").strip() or "Untitled set",
            "target_minutes": target_minutes,
            "brief": brief,
            "created": time.time(),
        }
    )


def delete_set(set_id: str) -> None:
    _path(set_id).unlink(missing_ok=True)
    shutil.rmtree(SETS_DIR / set_id, ignore_errors=True)


def find_song(data: dict, song_id: str) -> dict | None:
    for song in data.get("songs") or []:
        if song.get("id") == song_id:
            return song
    return None


def record_job_result(set_id: str, song_id: str, job) -> None:
    """Called by the worker when a job belonging to a set finishes."""
    try:
        data = load_set(set_id)
    except Exception:  # noqa: BLE001
        return
    song = find_song(data, song_id)
    if song is None:
        return
    song["job_id"] = job.id
    song["status"] = job.status
    song["seed"] = job.seed
    song["audio"] = job.audio
    song["audio_seconds"] = job.audio_seconds
    if job.error:
        song["notes"] = str(job.error)[:400]
    try:
        save_set(data)
    except Exception:  # noqa: BLE001
        pass


def resolve_audio_file(job_id) -> Path | None:
    """The rendered file for a job, whether or not the job is still in memory.

    meta.json records the *URL* under "audio", so never trust that field as a
    filename: the file on disk is the source of truth.
    """
    if not job_id:
        return None
    folder = OUTPUTS / str(job_id)
    if not folder.is_dir():
        return None
    found = sorted(path for path in folder.glob("song.*") if path.is_file())
    return found[0] if found else None


def audio_path(song: dict) -> Path:
    resolved = resolve_audio_file(song.get("job_id"))
    if resolved is not None:
        return resolved
    return OUTPUTS / str(song.get("job_id")) / str(song.get("audio") or "song.wav")


def reconcile(data: dict) -> dict:
    """Pick up results written by jobs that finished before an app restart."""
    for song in data.get("songs") or []:
        job_id = song.get("job_id")
        if not job_id or song.get("audio_seconds"):
            continue
        meta = OUTPUTS / str(job_id) / "meta.json"
        if not meta.is_file():
            continue
        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        song["status"] = info.get("status")
        song["audio_seconds"] = info.get("audio_seconds")
        song["seed"] = info.get("seed") or song.get("seed")
        resolved = resolve_audio_file(job_id)
        song["audio"] = resolved.name if resolved is not None else None
    return data


# --- OpenRouter ------------------------------------------------------------

_MODELS_CACHE: dict = {"at": 0.0, "rows": None}


def _request(url: str, payload: dict | None = None, headers: dict | None = None, timeout: int = 180):
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:600]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenRouter unreachable: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenRouter returned invalid JSON: {exc}") from exc


def openrouter_models(force: bool = False) -> list[dict]:
    if not force and _MODELS_CACHE["rows"] and time.time() - _MODELS_CACHE["at"] < 900:
        return _MODELS_CACHE["rows"]
    payload = _request(f"{OPENROUTER_API}/models", timeout=60)
    rows = []
    for item in payload.get("data") or []:
        pricing = item.get("pricing") or {}
        rows.append(
            {
                "id": item.get("id"),
                "name": item.get("name") or item.get("id"),
                "context_length": item.get("context_length"),
                "prompt_price": pricing.get("prompt"),
                "completion_price": pricing.get("completion"),
            }
        )
    rows.sort(key=lambda row: row["id"] or "")
    _MODELS_CACHE.update(at=time.time(), rows=rows)
    return rows


def openrouter_chat(
    api_key: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.85,
    max_tokens: int = 9000,
    json_object: bool = True,
    timeout: int = 300,
) -> str:
    if not api_key or not model:
        raise ValueError("api_key and model are required")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_object:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:7860",
        "X-Title": "YuE2 Studio sets",
    }
    try:
        result = _request(f"{OPENROUTER_API}/chat/completions", payload, headers, timeout=timeout)
    except RuntimeError as exc:
        # Plenty of models reject response_format; retry once without it.
        if not json_object or "response_format" not in str(exc):
            raise
        payload.pop("response_format", None)
        result = _request(f"{OPENROUTER_API}/chat/completions", payload, headers, timeout=timeout)
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {str(result)[:400]}")
    return choices[0].get("message", {}).get("content") or ""


# --- drafting --------------------------------------------------------------

def build_set_messages(
    brief: str,
    target_minutes: float,
    track_count: int,
    language: str = "English",
    existing: list[dict] | None = None,
):
    total = int(round(float(target_minutes) * 60))
    adding = bool(existing)
    system = (
        "You are a music curator and lyricist assembling one continuous set for a YouTube upload.\n"
        "Reply with strict JSON only: no prose, no markdown, no code fences.\n"
        'Schema: {"set_name": string, "background": string, "songs": [{"title": string, '
        '"style": string, "lyrics": string, "target_seconds": integer}]}\n'
        f"Produce exactly {track_count} songs, ordered as they should play.\n"
        f"The target_seconds values must sum to about {total} seconds (within 5% of it), and each "
        f"value must be between {MIN_TRACK_SECONDS} and {MAX_TRACK_SECONDS}.\n"
        "Rules:\n"
        f"- Write titles, styles and lyrics in {language}.\n"
        "- style: one single line covering genre, era, mood, instruments, vocal type and BPM. "
        "Keep the set coherent (neighbouring tracks share tempo range and instrumentation) while "
        "making each track clearly distinguishable.\n"
        "- lyrics: use [Verse], [Chorus], [Bridge] and [Instrumental] section markers. Scale the "
        "length to target_seconds, about one short line per three seconds. For a purely "
        "instrumental track use an empty string and describe the instruments in style.\n"
        "- background: ONE text-to-image prompt for the whole set, describing a single still image "
        "that will sit behind every track of the video. Describe the shared aesthetic, framing and "
        "palette in detail, 16:9, with no text and no watermark. Do not describe individual tracks.\n"
        "- Never reuse a title inside the set."
    )
    if adding:
        system += (
            f"\n- This is an addition to a set that already has {len(existing)} tracks, listed "
            "below. Continue that set: the new songs must flow on from the last existing one, "
            "share its palette and tempo range, and must not repeat any existing title."
        )
        listing = "\n".join(
            f"  {index}. {song.get('title')} — {str(song.get('style') or '')[:90]}"
            for index, song in enumerate(existing, start=1)
        )
        user = (
            f"Set brief: {brief.strip() or 'a continuous one hour set'}\n"
            f"Existing tracks:\n{listing}\n"
            f"Add {track_count} more songs, about {round(float(target_minutes))} minutes in total."
        )
    else:
        user = (
            f"Set brief: {brief.strip() or 'a continuous one hour set'}\n"
            f"Total length: about {round(float(target_minutes))} minutes across {track_count} songs."
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def merge_tracks(existing: list[dict], added: list[dict]) -> list[dict]:
    """Append drafted tracks to the current list, renumbering and de-duplicating titles."""
    merged = list(existing) + list(added)
    seen: dict[str, int] = {}
    for song in merged:
        title = str(song.get("title") or "").strip() or "Untitled"
        key = title.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            title = f"{title} ({seen[key]})"
        song["title"] = title
    for index, song in enumerate(merged, start=1):
        song["order"] = index
    return merged


def parse_json_object(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the model did not return a JSON object")
    return json.loads(text[start : end + 1])


def rescale_seconds(songs: list[dict], target_minutes: float) -> None:
    total = int(round(float(target_minutes) * 60))
    current = sum(int(song["target_seconds"]) for song in songs)
    if not total or not current:
        return
    ratio = total / current
    if 0.95 <= ratio <= 1.05:
        return
    for song in songs:
        song["target_seconds"] = clamp_seconds(round(song["target_seconds"] * ratio))


def songs_from_llm(payload: dict, target_minutes: float) -> tuple[str, str, list[dict]]:
    raw_songs = payload.get("songs")
    if not isinstance(raw_songs, list) or not raw_songs:
        raise ValueError("the model returned no songs")
    songs = []
    for index, raw in enumerate(raw_songs, start=1):
        if not isinstance(raw, dict):
            continue
        songs.append(
            new_song(
                index,
                title=str(raw.get("title") or f"Track {index}").strip(),
                style=str(raw.get("style") or "").strip(),
                lyrics=str(raw.get("lyrics") or "").strip("\n"),
                target_seconds=clamp_seconds(raw.get("target_seconds")),
            )
        )
    if not songs:
        raise ValueError("the model returned no usable songs")
    rescale_seconds(songs, target_minutes)
    seen: dict[str, int] = {}
    for song in songs:
        title = song["title"]
        seen[title] = seen.get(title, 0) + 1
        if seen[title] > 1:
            song["title"] = f"{title} ({seen[title]})"
    return (
        str(payload.get("set_name") or "").strip(),
        str(payload.get("background") or "").strip(),
        songs,
    )


# --- export ----------------------------------------------------------------

def _stamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def timeline(data: dict) -> tuple[list[dict], float]:
    """Timestamps for chapters. Unrendered tracks fall back to their target length."""
    rows, at = [], 0.0
    for song in sorted(data.get("songs") or [], key=lambda s: int(s["order"])):
        rendered = song.get("audio_seconds")
        seconds = float(rendered or song.get("target_seconds") or 0)
        rows.append(
            {
                "at": at,
                "stamp": _stamp(at),
                "title": display_title(song),
                "seconds": seconds,
                "rendered": bool(rendered),
            }
        )
        at += seconds
    return rows, at


def build_translation_messages(titles: list[str], language: str):
    system = (
        "You translate song titles for a music release.\n"
        "Reply with strict JSON only: no prose, no code fences.\n"
        'Schema: {"translations": [string, ...]} in the same order and with the same length '
        "as the input titles.\n"
        f"Translate every title into {language}. Keep each one short and natural as a song title, "
        "with no quotes and no explanation. If a title is already in the target language, "
        "return it unchanged."
    )
    user = json.dumps({"titles": titles}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def apply_translations(songs: list[dict], translations) -> None:
    if not isinstance(translations, list) or len(translations) != len(songs):
        raise ValueError("expected one translation per title")
    for song, text in zip(songs, translations):
        song["title_translation"] = str(text or "").strip()


def export_json(data: dict) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def export_markdown(data: dict) -> str:
    rows, total = timeline(data)
    lines = [f"# {data.get('name')}", ""]
    if data.get("brief"):
        lines += [str(data["brief"]).strip(), ""]
    lines += [
        f"- Target length: {data.get('target_minutes')} min",
        f"- Tracks: {len(rows)}",
        f"- Rendered: {sum(1 for r in rows if r['rendered'])}/{len(rows)}"
        + (f" ({_stamp(total)})" if total else ""),
        "",
    ]
    if data.get("background"):
        lines += ["## Background image prompt", "", str(data["background"]).strip(), ""]
    lines += ["## Tracklist", ""]
    for index, (row, song) in enumerate(
        zip(rows, sorted(data.get("songs") or [], key=lambda s: int(s["order"]))), start=1
    ):
        lines += [
            f"### {index}. {row['title']}",
            "",
            f"- Order: {song.get('order')}",
            f"- Target: {song.get('target_seconds')}s"
            + (f" · rendered: {song.get('audio_seconds')}s" if song.get("audio_seconds") else ""),
            f"- Style: {song.get('style') or '(none)'}",
        ]
        if song.get("seed"):
            lines.append(f"- Seed: {song['seed']}")
        if song.get("lyrics"):
            lines += ["", "```text", str(song["lyrics"]).strip(), "```"]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_youtube(data: dict) -> str:
    rows, total = timeline(data)
    title = data.get("name") or "Untitled set"
    lines = [f"**Title**", title, "", "**Description**", ""]
    if data.get("brief"):
        lines += [str(data["brief"]).strip(), ""]
    lines.append(f"A {round(total / 60)} minute continuous set.")
    lines += ["", "Chapters:", ""]
    for index, row in enumerate(rows, start=1):
        lines.append(f"{row['stamp']} {index}. {row['title']}")
    missing = [row for row in rows if not row["rendered"]]
    if missing:
        lines += ["", f"(!) {len(missing)} track(s) not rendered yet: "
                      + ", ".join(str(r["title"]) for r in missing)]
    lines += ["", "Background image prompt (one image for the whole set):", ""]
    lines.append(str(data.get("background") or "(none)").strip())
    lines += ["", "Generated with YuE2-3B (m-a-p) locally."]
    return "\n".join(lines).rstrip() + "\n"


# --- concatenation ---------------------------------------------------------

def missing_tracks(data: dict) -> list[int]:
    return [
        int(song["order"])
        for song in sorted(data.get("songs") or [], key=lambda s: int(s["order"]))
        if not song.get("audio_seconds") or not song.get("audio")
    ]


def concat_set(data: dict, ffmpeg: str, out_format: str = "mp3", bitrate: str = "320k", on_progress=None) -> Path:
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: install it, or set YUE2_FFMPEG to its full path")
    if not data.get("songs"):
        raise RuntimeError("this set has no tracks yet")
    pending = missing_tracks(data)
    if pending:
        raise RuntimeError(f"tracks not rendered yet: {pending}")
    songs = sorted(data["songs"], key=lambda s: int(s["order"]))
    files = []
    for song in songs:
        path = audio_path(song)
        if not path.is_file():
            raise RuntimeError(f"missing audio file for track {song['order']}: {path}")
        files.append(path)

    folder = SETS_DIR / str(data["id"])
    folder.mkdir(parents=True, exist_ok=True)
    listing = folder / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in files), encoding="utf-8"
    )
    suffix = "mp3" if out_format == "mp3" else "wav"
    output = folder / f"set.{suffix}"
    # write beside the final name and swap it in, so the download never sees half a file
    partial = folder / f"set.part.{suffix}"
    partial.unlink(missing_ok=True)
    total = sum(float(song.get("audio_seconds") or 0) for song in songs)
    codec = ["-c:a", "libmp3lame", "-b:a", bitrate] if suffix == "mp3" else ["-c:a", "pcm_s16le"]
    code, errors = _run_ffmpeg(
        [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
            "-f", "concat", "-safe", "0", "-i", str(listing),
            "-ar", "48000", "-ac", "2", *codec, str(partial),
        ],
        total,
        on_progress,
    )
    if code != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg concat failed: {errors.strip()[-500:]}")
    partial.replace(output)
    return output


# --- background image and video -------------------------------------------

def folder(set_id: str) -> Path:
    return SETS_DIR / str(set_id)


def save_background(set_id: str, filename: str, handle) -> Path:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image type {suffix or '(none)'}; use png, jpg or webp")
    target = folder(set_id)
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("background.*"):
        old.unlink(missing_ok=True)
    path = target / f"background{suffix}"
    with path.open("wb") as out:
        shutil.copyfileobj(handle, out)
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        raise ValueError("the uploaded image is empty")
    return path


def background_path(set_id: str) -> Path | None:
    target = folder(set_id)
    if not target.is_dir():
        return None
    found = sorted(p for p in target.glob("background.*") if p.is_file())
    return found[0] if found else None


def concat_audio_path(set_id: str) -> Path | None:
    for name in ("set.mp3", "set.wav"):
        path = folder(set_id) / name
        if path.is_file():
            return path
    return None


def video_path(set_id: str) -> Path | None:
    path = folder(set_id) / "set.mp4"
    return path if path.is_file() else None


def clear_renders(set_id: str) -> list[str]:
    """Drop the derived files; they belong to a tracklist that no longer exists."""
    removed = []
    for name in ("set.mp3", "set.wav", "set.mp4", "concat.txt"):
        path = folder(set_id) / name
        if path.is_file():
            path.unlink(missing_ok=True)
            removed.append(name)
    return removed


def tracklist_signature(songs) -> tuple:
    return tuple(
        (int(s.get("order") or 0), str(s.get("title") or ""), str(s.get("job_id") or ""))
        for s in sorted(songs or [], key=lambda s: int(s.get("order") or 0))
    )


def carry_rendered(old_songs, new_songs) -> None:
    """A re-draft keeps the audio of tracks whose title did not change."""
    by_title = {}
    for song in old_songs or []:
        if song.get("audio_seconds") and song.get("job_id"):
            by_title[str(song.get("title") or "").strip().lower()] = song
    for song in new_songs:
        previous = by_title.get(str(song.get("title") or "").strip().lower())
        if not previous:
            continue
        for key in ("job_id", "audio", "audio_seconds", "seed", "status"):
            song[key] = previous.get(key)


def _run_ffmpeg(command, total_seconds, on_progress=None):
    """Run ffmpeg, reporting percent from its -progress stream. Returns (code, stderr)."""
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=NO_CONSOLE,
    )
    captured = []

    def drain():
        captured.append(proc.stderr.read())

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    for line in proc.stdout:
        if not on_progress or total_seconds <= 0 or not line.startswith("out_time_ms="):
            continue
        try:
            seconds = int(line.split("=", 1)[1]) / 1_000_000
        except ValueError:
            continue
        # leave the last percent for the caller, which knows the encode finished
        on_progress(min(99.0, 100.0 * seconds / total_seconds))
    proc.wait()
    thread.join(timeout=10)
    return proc.returncode, "".join(captured)


def _probe_duration(ffmpeg: str, path: Path) -> float | None:
    probe = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    executable = str(probe) if probe.is_file() else "ffprobe"
    try:
        result = subprocess.run(
            [executable, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, creationflags=NO_CONSOLE,
        )
        return float(result.stdout.strip())
    except Exception:  # noqa: BLE001
        return None


def make_video(
    data: dict,
    ffmpeg: str,
    image: Path,
    *,
    width: int = 1920,
    height: int = 1080,
    fps: int = 2,
    crf: int = 23,
    on_progress=None,
) -> Path:
    """Still image + concatenated audio -> one upload-ready mp4."""
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: install it, or set YUE2_FFMPEG to its full path")
    audio = concat_audio_path(str(data["id"]))
    if audio is None:
        raise RuntimeError("concatenate the tracks first: there is no set audio yet")
    if not image.is_file():
        raise RuntimeError("background image not found")
    output = folder(str(data["id"])) / "set.mp4"
    partial = folder(str(data["id"])) / "set.part.mp4"
    partial.unlink(missing_ok=True)
    scale = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p"
    )
    command = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1",
        "-loop", "1", "-framerate", str(fps), "-i", str(image),
        "-i", str(audio),
        "-vf", scale, "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "320k",
    ]
    # -shortest alone overshoots on a looped still; bound the output explicitly.
    duration = _probe_duration(ffmpeg, audio)
    if duration:
        command += ["-t", f"{duration:.3f}"]
    command += ["-movflags", "+faststart", str(partial)]
    code, errors = _run_ffmpeg(command, duration or 0.0, on_progress)
    if code != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg video failed: {errors.strip()[-500:]}")
    partial.replace(output)
    return output
