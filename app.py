"""Standalone YuE2 webUI.

Single-process FastAPI app that wraps the official `yue2` inference package
(https://huggingface.co/m-a-p/YuE2-3B). One model instance, one generation at a
time, jobs queued and polled from the browser.

Run:  python app.py   (then open http://127.0.0.1:7860)
"""
from __future__ import annotations

import json
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("YUE2_MODEL", ROOT / "models/YuE2-3B"))
VAE_DIR = Path(os.environ.get("YUE2_VAE", ROOT / "models/YuE2-Vae"))
OUTPUTS = Path(os.environ.get("YUE2_OUTPUTS", ROOT / "outputs"))
MEMORY_GIB = float(os.environ.get("YUE2_MEMORY_GIB", "16"))
HOST = os.environ.get("YUE2_HOST", "127.0.0.1")
PORT = int(os.environ.get("YUE2_PORT", "7860"))
OUTPUTS.mkdir(parents=True, exist_ok=True)

# SheetSage2 (audio -> melody ABC) lives in its own conda env; see setup_sheetsage2.sh
SHEETSAGE2_DIR = Path(os.environ.get("YUE2_SHEETSAGE2", ROOT / "models/SheetSage2"))
MERT_DIR = Path(os.environ.get("YUE2_MERT", ROOT / "models/MERT-v2-FullSong"))
def _default_sheetsage2_py() -> Path:
    """Sibling conda env of the running interpreter, e.g. envs/yue2 -> envs/yue2-sheetsage2."""
    exe = Path(sys.executable).resolve()
    if exe.parent.name == "bin" and exe.parents[1].parent.name == "envs":
        return exe.parents[1].parent / "yue2-sheetsage2" / "bin" / "python"
    return Path("yue2-sheetsage2/bin/python")


SHEETSAGE2_PY = Path(os.environ.get("YUE2_SHEETSAGE2_PY", _default_sheetsage2_py()))
FRAMES_PER_SECOND = 25  # YuE2 codec frame rate; 1 semantic token = 0.04 s


def cover_available() -> bool:
    """Covers need SheetSage2, which lives in a separate env (see setup_sheetsage2.sh)."""
    return SHEETSAGE2_PY.exists() and (SHEETSAGE2_DIR / "config.json").exists()


STAGES = {
    "queued": "Queued",
    "loading": "Loading model",
    "transcribing": "Transcribing reference audio (SheetSage2)",
    "planning": "Planning score (ABC)",
    "generating": "Generating song tokens",
    "synthesizing": "Synthesizing latents",
    "decoding": "Decoding audio",
    "done": "Done",
    "error": "Error",
    "cancelled": "Cancelled",
}


def vram_usage() -> dict | None:
    """Process VRAM view, cheap enough to poll from the request thread."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        return {
            "used_gib": round(torch.cuda.memory_allocated() / 2**30, 1),
            "total_gib": round(total, 1),
        }
    except Exception:  # noqa: BLE001
        return None


# Set by the worker so the library's NAR/VAE counters reach the active job.
_DETAIL_SINK = None


def _install_progress_bridge() -> None:
    """Forward yue2's stage counters to the webUI.

    The pipeline only wires its on_progress callbacks when progress is enabled,
    and those callbacks land on yue2.progress._Stage. Wrapping update() is the
    only way to observe the NAR/VAE steps without reimplementing pipeline logic.
    """
    try:
        from yue2 import progress as yue2_progress

        stage = yue2_progress._Stage
    except Exception:  # noqa: BLE001
        return
    if getattr(stage.update, "_webui_bridge", False):
        return
    original = stage.update

    def update(self, completed, total=None):
        original(self, completed, total)
        sink = _DETAIL_SINK
        if sink is not None:
            sink(self.label, self.completed, self.total, self.unit)

    update._webui_bridge = True
    stage.update = update


_install_progress_bridge()


def note(job: "Job", text: str) -> None:
    job.events.append({"t": round(time.time() - (job.started or time.time()), 1), "text": text})
    del job.events[:-40]


def _job_sink(job: "Job"):
    def sink(label, done, total, unit):
        # AR stages report no total; only NAR steps and VAE chunks are useful.
        if total is None:
            return
        job.sub_label = label
        job.sub_done = done or 0
        job.sub_total = total
        job.sub_unit = unit or ""

    return sink


@dataclass
class Job:
    id: str
    params: dict
    status: str = "queued"
    tokens: int = 0
    message: str = ""
    error: str = ""
    seed: int = 0
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    audio: str | None = None
    audio_seconds: float | None = None
    score: str | None = None
    truncated: bool = False
    cover: bool = False
    generation_seconds: float | None = None
    cancel: bool = False
    # live detail
    stage_started: float | None = None
    stage_tokens: int = 0
    target_tokens: int | None = None
    sub_label: str = ""
    sub_done: int = 0
    sub_total: int | None = None
    sub_unit: str = ""
    events: list = field(default_factory=list)

    def eta_seconds(self) -> float | None:
        if not self.stage_started:
            return None
        elapsed = time.time() - self.stage_started
        if elapsed <= 0:
            return None
        if self.status in ("planning", "generating") and self.target_tokens and self.stage_tokens:
            if self.stage_tokens < 5:
                return None  # the first tokens carry prefill, so the rate is noise
            rate = self.stage_tokens / elapsed
            if rate > 0:
                return max(0.0, (self.target_tokens - self.stage_tokens) / rate)
        if self.sub_total and self.sub_done:
            rate = self.sub_done / elapsed
            if rate > 0:
                return max(0.0, (self.sub_total - self.sub_done) / rate)
        return None

    def public(self) -> dict:
        now = self.finished or time.time()
        return {
            "id": self.id,
            "status": self.status,
            "stage": STAGES.get(self.status, self.status),
            "tokens": self.tokens,
            "message": self.message,
            "error": self.error,
            "seed": self.seed,
            "cover": self.cover,
            "truncated": self.truncated,
            "elapsed": round((now - self.started), 1) if self.started else 0,
            "audio": f"/api/job/{self.id}/audio" if self.audio else None,
            "audio_seconds": self.audio_seconds,
            "score": self.score,
            "format": Path(self.audio).suffix.lstrip(".") if self.audio else self.params.get("format", "wav"),
            "generation_seconds": self.generation_seconds,
            "stage_tokens": self.stage_tokens,
            "target_tokens": self.target_tokens,
            "eta_seconds": self.eta_seconds(),
            "sub_progress": (
                {
                    "label": self.sub_label,
                    "completed": self.sub_done,
                    "total": self.sub_total,
                    "unit": self.sub_unit,
                }
                if self.sub_label
                else None
            ),
            "vram": vram_usage(),
            "events": self.events[-12:],
        }

    def meta(self) -> dict:
        data = self.public()
        data.update({"params": self.params, "created": self.created})
        return data


class GenerateRequest(BaseModel):
    style: str = Field(min_length=1, max_length=4000)
    lyrics: str = Field(default="", max_length=20000)
    cot: str = Field(default="full", pattern="^(full|melody|off)$")
    seed: int | None = None
    cfg_scale: float | None = Field(default=None, ge=0, le=20)
    abc: str | None = None
    max_duration: float | None = Field(default=None, ge=0.04, le=900)
    format: str = Field(default="wav", pattern="^(wav|flac)$")
    plan_only: bool = False
    # advanced sampling
    temperature: float | None = Field(default=None, ge=0, le=5)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=1)
    repetition_penalty: float | None = Field(default=None, gt=0)
    ode_steps: int | None = Field(default=None, ge=1, le=256)
    max_abc_tokens: int | None = Field(default=None, ge=1, le=20000)
    fast_decode: bool = False
    mp3: bool = False


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.load_lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.queue: queue.Queue[str] = queue.Queue()
        self.pipe = None
        self.pipe_error: str | None = None
        self.current: str | None = None

    def put(self, job: Job):
        with self.lock:
            self.jobs[job.id] = job
        self.queue.put(job.id)

    def get(self, job_id: str) -> Job:
        with self.lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "unknown job")
        return job

    def snapshot(self) -> dict:
        with self.lock:
            pending = sum(1 for j in self.jobs.values() if j.status == "queued")
        return {
            "model_loaded": self.pipe is not None,
            "model_error": self.pipe_error,
            "queued": pending,
            "current": self.current,
            "device": os.environ.get("YUE2_DEVICE", "cuda"),
            "cover_available": cover_available(),
        }


STATE = State()


def default_backend() -> str:
    """The CUDA-graph path uses flash-attention ops, which ROCm (HIP) lacks."""
    override = os.environ.get("YUE2_BACKEND")
    if override:
        return override
    import torch

    return "torch-eager" if torch.version.hip else "torch"


def load_pipeline():
    from yue2 import YuE2Pipeline

    local_model = (MODEL_DIR / "config.json").exists()
    local_vae = (VAE_DIR / "config.json").exists()
    pipe = YuE2Pipeline.from_pretrained(
        str(MODEL_DIR) if local_model else "m-a-p/YuE2-3B",
        vae=str(VAE_DIR) if local_vae else "m-a-p/YuE2-Vae",
        device=os.environ.get("YUE2_DEVICE", "cuda"),
        memory_budget_gib=MEMORY_GIB,
        backend=default_backend(),
        progress=os.environ.get("YUE2_PROGRESS", "1") == "1",
        local_files_only=local_model and local_vae,
    )
    return pipe


def build_sampling(params: dict):
    overrides = {}
    for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
        if params.get(key) is not None:
            overrides[key] = params[key]
    # Duration is a token cap on the semantic stage: 1 token = 1 frame = 1/25 s.
    max_duration = params.get("max_duration")
    if max_duration:
        frames = max(1, round(max_duration * FRAMES_PER_SECOND))
        overrides["max_tokens"] = frames
        overrides["min_tokens"] = min(200, frames)
    return overrides or None


def ffmpeg_path() -> str | None:
    return os.environ.get("YUE2_FFMPEG") or shutil.which("ffmpeg")


def encode_mp3(source: Path, destination: Path) -> Path:
    exe = ffmpeg_path()
    if not exe:
        raise RuntimeError(
            "ffmpeg not found: install it and put it on PATH, or set YUE2_FFMPEG to its full path"
        )
    proc = subprocess.run(
        [
            exe, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source), "-codec:a", "libmp3lame", "-b:a", "320k", str(destination),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not destination.exists():
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise RuntimeError(f"ffmpeg mp3 encode failed: {tail}")
    return destination


def transcribe_cover(audio_path: str, job: Job) -> str:
    """Run SheetSage2 in its own env to get a melody-only ABC score."""
    if not SHEETSAGE2_PY.exists():
        raise RuntimeError(
            f"SheetSage2 environment not found at {SHEETSAGE2_PY}. Run setup_sheetsage2.sh first."
        )
    if not (SHEETSAGE2_DIR / "config.json").exists():
        raise RuntimeError(f"SheetSage2 model not found at {SHEETSAGE2_DIR}. Run download_models.sh.")
    outdir = OUTPUTS / job.id / "transcription"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(SHEETSAGE2_PY),
        str(ROOT / "cover_transcribe.py"),
        audio_path,
        "--output",
        str(outdir),
        "--model",
        str(SHEETSAGE2_DIR),
        "--base-model",
        str(MERT_DIR),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(f"SheetSage2 transcription failed: {tail}")
    abc_path = outdir / "melody.abc"
    if not abc_path.exists() or not abc_path.read_text(encoding="utf-8").strip():
        raise RuntimeError("SheetSage2 produced no melody ABC")
    return abc_path.read_text(encoding="utf-8")


def run_job(job: Job):
    global _DETAIL_SINK
    job.status = "loading"
    job.started = time.time()
    job.stage_started = job.started
    if job.cancel:
        raise InterruptedError("cancelled")
    params = job.params
    note(job, "job started")

    if params.get("cover_audio"):
        # Transcribe first, with YuE2 off the GPU, to keep both pipelines sequential.
        if STATE.pipe is not None:
            STATE.pipe.close()
        job.cover = True
        job.status = "transcribing"
        note(job, "transcribing reference (SheetSage2)")
        params["abc"] = transcribe_cover(params["cover_audio"], job)
        params["cot"] = "melody"
        note(job, "transcription done")

    with STATE.load_lock:
        if STATE.pipe is None:
            if STATE.pipe_error:
                raise RuntimeError(STATE.pipe_error)
            load_started = time.time()
            note(job, "loading model (first run is the slow one)")
            try:
                STATE.pipe = load_pipeline()
            except Exception as exc:  # noqa: BLE001
                STATE.pipe_error = f"model load failed: {exc}"
                raise
            note(job, f"model ready in {time.time() - load_started:.1f}s")
    pipe = STATE.pipe
    _DETAIL_SINK = _job_sink(job)

    import dataclasses

    from yue2.protocol import GenerationConfig

    config = pipe.generation_config
    # ABC token budget lives on the planning sampler; ComfyUI defaults it to 8192.
    if params.get("max_abc_tokens") is not None:
        config = dataclasses.replace(
            config, abc=dataclasses.replace(config.abc, max_tokens=params["max_abc_tokens"])
        )
    if params.get("ode_steps") is not None:
        config = dataclasses.replace(config, ode_steps=params["ode_steps"])
    pipe.generation_config = config

    sampling = build_sampling(params)
    request_kwargs = {
        "style": params["style"],
        "lyrics": params["lyrics"],
        "cot": params["cot"],
        "seed": job.seed,
    }
    if params.get("cfg_scale") is not None:
        request_kwargs["cfg_scale"] = params["cfg_scale"]
    if params.get("abc"):
        if params["cot"] == "off":
            raise ValueError("An ABC score cannot be used with cot=off")
        request_kwargs["abc"] = params["abc"]

    cancelled = lambda: job.cancel  # noqa: E731
    abc_target = getattr(config.abc, "max_tokens", None)
    semantic_target = (sampling or {}).get("max_tokens") or getattr(config.semantic, "max_tokens", None)

    def start_stage(status, target):
        job.status = status
        job.stage_started = time.time()
        job.stage_tokens = 0
        job.target_tokens = target
        job.sub_label, job.sub_done, job.sub_total, job.sub_unit = "", 0, None, ""

    def on_token(phase, _token):
        wanted = "planning" if phase == "abc" else "generating"
        if job.status != wanted:
            start_stage(wanted, abc_target if phase == "abc" else semantic_target)
        job.tokens += 1
        job.stage_tokens += 1

    planning = params["cot"] != "off" and not params.get("abc")
    if planning:
        start_stage("planning", abc_target)
        note(job, f"planning score (ABC budget {abc_target} tokens)")
    else:
        note(job, "no symbolic plan (cot=off or supplied score)")
    plan = pipe.plan(**request_kwargs, cancelled=cancelled, on_token=on_token)
    job.score = plan.abc
    job.truncated = bool(plan.truncated)
    if planning:
        note(job, f"score ready: {len(plan.abc_ids)} tokens in {time.time() - job.stage_started:.1f}s")

    if params.get("plan_only"):
        job.status = "done"
        job.audio_seconds = None
        note(job, "plan only: no audio requested")
        return

    start_stage("generating", semantic_target)
    cap = f" (cap {semantic_target} tokens = {semantic_target / FRAMES_PER_SECOND:.0f}s)" if semantic_target else ""
    note(job, f"generating song tokens{cap}")
    semantic = pipe.generate_semantic(plan, sampling=sampling, cancelled=cancelled, on_token=on_token)
    job.truncated = job.truncated or bool(semantic.truncated)
    seconds = semantic.timing.get("seconds") or (time.time() - job.stage_started)
    rate = len(semantic.tokens) / seconds if seconds else 0.0
    note(job, f"semantic: {len(semantic.tokens)} tokens in {seconds:.1f}s ({rate:.1f} tok/s)")
    if job.cancel:
        raise InterruptedError("cancelled")

    start_stage("synthesizing", None)
    job.sub_label, job.sub_unit = "Synthesizing audio", "steps"
    note(job, "solving the acoustic flow (NAR)")
    synth_started = time.time()
    latents = pipe.synthesize(semantic, cancelled=cancelled)
    note(job, f"latents ready in {time.time() - synth_started:.1f}s")
    if job.cancel:
        raise InterruptedError("cancelled")

    start_stage("decoding", None)
    job.sub_label, job.sub_unit = "Decoding audio", "chunks"
    note(job, "decoding audio (VAE)")
    decode_started = time.time()
    audio = pipe.decode(latents, full=bool(params.get("fast_decode")))
    note(job, f"decoded {len(audio) / 48000:.1f}s of audio in {time.time() - decode_started:.1f}s")

    directory = OUTPUTS / job.id
    directory.mkdir(parents=True, exist_ok=True)
    suffix = params.get("format", "wav")
    if params.get("mp3"):
        # Encode from a temporary WAV; the lossless intermediate is not kept.
        wav_path = directory / "song.wav"
        sf.write(wav_path, audio, 48000, subtype="PCM_16")
        path = encode_mp3(wav_path, directory / "song.mp3")
        wav_path.unlink(missing_ok=True)
    else:
        path = directory / f"song.{suffix}"
        subtype = "PCM_24" if suffix == "flac" else "PCM_16"
        sf.write(path, audio, 48000, subtype=subtype)
    job.audio = path.name
    job.audio_seconds = round(len(audio) / 48000, 1)
    note(job, f"wrote {path.name} ({job.audio_seconds:.1f}s)")
    if job.score:
        (directory / "score.abc").write_text(job.score, encoding="utf-8")
    job.message = f"{job.tokens} tokens"


def worker():
    global _DETAIL_SINK
    while True:
        job_id = STATE.queue.get()
        job = STATE.get(job_id)
        STATE.current = job_id
        try:
            run_job(job)
            if job.cancel and job.status not in ("done",):
                job.status = "cancelled"
            elif job.status != "done":
                job.status = "done"
        except InterruptedError:
            job.status = "cancelled"
            job.error = ""
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            note(job, f"failed: {exc}")
            traceback.print_exc()
        finally:
            _DETAIL_SINK = None
            job.finished = time.time()
            if job.started:
                job.generation_seconds = round(job.finished - job.started, 1)
            directory = OUTPUTS / job.id
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "meta.json").write_text(
                json.dumps(job.meta(), indent=2, ensure_ascii=False), encoding="utf-8"
            )
            STATE.current = None
            STATE.queue.task_done()


def _preload():
    with STATE.load_lock:
        if STATE.pipe is not None:
            return
        try:
            STATE.pipe = load_pipeline()
        except Exception as exc:  # noqa: BLE001
            STATE.pipe_error = f"model load failed: {exc}"
            traceback.print_exc()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=worker, daemon=True).start()
    if os.environ.get("YUE2_PRELOAD", "0") == "1":
        threading.Thread(target=_preload, daemon=True).start()
    yield


app = FastAPI(title="YuE2", docs_url=None, redoc_url=None, lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static/index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def api_state():
    return STATE.snapshot()


@app.post("/api/load")
def api_load():
    if STATE.pipe is not None:
        return {"ok": True, "message": "already loaded"}
    if STATE.pipe_error:
        raise HTTPException(500, STATE.pipe_error)
    threading.Thread(target=_preload, daemon=True).start()
    return {"ok": True, "message": "loading started"}


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    if req.cot == "off" and req.abc:
        raise HTTPException(422, "ABC score requires cot=melody or cot=full")
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    job = Job(id=uuid.uuid4().hex[:12], params=req.model_dump(), seed=seed)
    STATE.put(job)
    return {"job_id": job.id, "seed": seed, "queued": STATE.snapshot()["queued"]}


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}


@app.post("/api/cover")
def api_cover(
    audio: UploadFile = File(...),
    style: str = Form(..., min_length=1, max_length=4000),
    lyrics: str = Form(""),
    cot: str = Form("melody"),
    seed: int | None = Form(None),
    cfg_scale: float | None = Form(None),
    max_duration: float | None = Form(None),
    fast_decode: bool = Form(False),
    format: str = Form("wav"),
    mp3: bool = Form(False),
):
    if format not in ("wav", "flac"):
        raise HTTPException(422, "format must be wav or flac")
    if cot not in ("melody", "full"):
        raise HTTPException(422, "covers use cot=melody (recommended) or cot=full")
    if not cover_available():
        raise HTTPException(
            503,
            "Covers are unavailable on this install: the SheetSage2 environment is missing "
            "(see setup_sheetsage2.sh). Text -> Song still works.",
        )
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix not in AUDIO_SUFFIXES:
        raise HTTPException(422, f"unsupported audio type {suffix or '(none)'}")
    seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    job = Job(id=uuid.uuid4().hex[:12], params={}, seed=seed)
    directory = OUTPUTS / job.id / "input"
    directory.mkdir(parents=True, exist_ok=True)
    reference = directory / f"reference{suffix}"
    with reference.open("wb") as handle:
        shutil.copyfileobj(audio.file, handle)
    if reference.stat().st_size == 0:
        raise HTTPException(422, "uploaded audio is empty")
    job.params = {
        "style": style,
        "lyrics": lyrics,
        "cot": cot,
        "seed": seed,
        "abc": None,
        "cover_audio": str(reference),
        "cfg_scale": cfg_scale,
        "max_duration": max_duration,
        "format": format,
        "plan_only": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": None,
        "ode_steps": None,
        "max_abc_tokens": None,
        "fast_decode": fast_decode,
        "mp3": mp3,
    }
    STATE.put(job)
    return {"job_id": job.id, "seed": seed, "queued": STATE.snapshot()["queued"]}


@app.get("/api/job/{job_id}")
def api_job(job_id: str):
    return STATE.get(job_id).public()


@app.post("/api/job/{job_id}/cancel")
def api_cancel(job_id: str):
    job = STATE.get(job_id)
    job.cancel = True
    return {"ok": True}


@app.get("/api/job/{job_id}/audio")
def api_audio(job_id: str):
    job = STATE.get(job_id)
    if not job.audio:
        raise HTTPException(404, "no audio for this job")
    path = OUTPUTS / job.id / job.audio
    if not path.exists():
        raise HTTPException(404, "audio file missing")
    return FileResponse(path, filename=f"yue2-{job.id}.{path.suffix.lstrip('.')}")


@app.get("/api/history")
def api_history():
    rows = []
    for meta in sorted(OUTPUTS.glob("*/meta.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(meta.read_text())
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "id": data.get("id"),
                "seed": data.get("seed"),
                "status": data.get("status"),
                "audio_seconds": data.get("audio_seconds"),
                "generation_seconds": data.get("generation_seconds"),
                "style": (data.get("params") or {}).get("style", "")[:80],
                "cover": data.get("cover", False),
                "created": data.get("created"),
                "audio": data.get("audio"),
            }
        )
    return rows[:50]


def main():
    if not (MODEL_DIR / "config.json").exists():
        print(f"! Local model not found at {MODEL_DIR}; it will download from the Hub on first run.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
