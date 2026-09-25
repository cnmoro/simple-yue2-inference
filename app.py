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
import re
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
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field

import sets
from abcscore import auto_seconds, FRAMES_PER_SECOND

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


# Observed cost of the NAR and VAE stages, as (semantic frames, seconds). The
# totals below scale the last observation, so the estimate self-calibrates.
_STAGE_COST: dict[str, tuple[int, float] | None] = {"nar": None, "vae": None}
TIMINGS_FILE = OUTPUTS / "stage_timings.json"


def _load_costs() -> None:
    try:
        data = json.loads(TIMINGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return
    for key in ("nar", "vae"):
        entry = data.get(key)
        if isinstance(entry, list) and len(entry) == 2:
            _STAGE_COST[key] = (int(entry[0]), float(entry[1]))


def _record_cost(stage: str, frames: int, seconds: float) -> None:
    if frames > 0 and seconds > 0:
        _STAGE_COST[stage] = (int(frames), float(seconds))
        try:
            TIMINGS_FILE.write_text(
                json.dumps({k: v for k, v in _STAGE_COST.items() if v}), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass


def _estimate_cost(stage: str, frames: int) -> float | None:
    known = _STAGE_COST.get(stage)
    if not known or frames <= 0 or known[0] <= 0:
        return None
    observed_frames, observed_seconds = known
    return observed_seconds * (frames / observed_frames)


_load_costs()


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

    def eta_total_seconds(self) -> float | None:
        """Remaining time for this stage plus the NAR/VAE work still ahead."""
        stage_eta = self.eta_seconds()
        if self.status == "decoding":
            return stage_eta
        frames = int(self.target_tokens or 0)
        if self.status == "synthesizing":
            nar = stage_eta if stage_eta is not None else _estimate_cost("nar", frames)
            vae = _estimate_cost("vae", frames)
            if nar is None and vae is None:
                return None
            return (nar or 0.0) + (vae or 0.0)
        if self.status in ("planning", "generating"):
            if stage_eta is None:
                return None
            return stage_eta + (_estimate_cost("nar", frames) or 0.0) + (_estimate_cost("vae", frames) or 0.0)
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
            "eta_total_seconds": self.eta_total_seconds(),
            "eta_calibrated": bool(_STAGE_COST["nar"] and _STAGE_COST["vae"]),
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
# ffmpeg joins run one at a time; a second request should say so instead of racing
MEDIA_LOCK = threading.Lock()
# percent of the running join/video encode, for the UI to poll
MEDIA_STATE: dict = {"kind": None, "percent": None}
# on-demand track conversions are independent of the set joins
CONVERT_LOCK = threading.Lock()


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
        creationflags=sets.NO_CONSOLE,
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
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=sets.NO_CONSOLE)
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

    if not params.get("max_duration"):
        # Blank duration: size the cap from what the score and lyrics actually need,
        # instead of guessing a number that would cut the song short.
        from yue2.protocol import CONTEXT

        raw = auto_seconds(plan.abc, params.get("lyrics"), margin=1.0)
        margin = 1.4 if plan.truncated else 1.15
        frames = max(1, min(int(round(raw * margin * FRAMES_PER_SECOND)), CONTEXT - len(plan.prefix) - 1))
        sampling = dict(sampling or {})
        sampling["max_tokens"] = frames
        sampling["min_tokens"] = min(200, frames)
        semantic_target = frames
        params["auto_seconds"] = round(raw, 1)
        params["auto_tokens"] = frames
        note(job, f"auto duration: score/lyrics need ~{raw:.0f}s -> cap {frames / FRAMES_PER_SECOND:.0f}s")

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
    synth_seconds = time.time() - synth_started
    _record_cost("nar", len(semantic.tokens), synth_seconds)
    note(job, f"latents ready in {synth_seconds:.1f}s")
    if job.cancel:
        raise InterruptedError("cancelled")

    start_stage("decoding", None)
    job.sub_label, job.sub_unit = "Decoding audio", "chunks"
    note(job, "decoding audio (VAE)")
    decode_started = time.time()
    audio = pipe.decode(latents, full=bool(params.get("fast_decode")))
    decode_seconds = time.time() - decode_started
    _record_cost("vae", len(semantic.tokens), decode_seconds)
    note(job, f"decoded {len(audio) / 48000:.1f}s of audio in {decode_seconds:.1f}s")

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
            set_id, song_id = job.params.get("set_id"), job.params.get("song_id")
            if set_id and song_id:
                sets.record_job_result(str(set_id), str(song_id), job)
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
    job = STATE.jobs.get(job_id)
    if job is not None:
        return job.public()
    # Jobs from a previous run are only on disk; the UI still polls them.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id):
        raise HTTPException(422, "bad job id")
    meta = OUTPUTS / job_id / "meta.json"
    if not meta.is_file():
        raise HTTPException(404, "unknown job")
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        raise HTTPException(404, "unknown job") from None


@app.post("/api/job/{job_id}/cancel")
def api_cancel(job_id: str):
    job = STATE.get(job_id)
    job.cancel = True
    return {"ok": True}


@app.get("/api/job/{job_id}/audio")
def api_audio(job_id: str, inline: bool = False, format: str | None = None):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id):
        raise HTTPException(422, "bad job id")
    path = sets.resolve_audio_file(job_id)
    if path is None:
        raise HTTPException(404, "no audio for this job")
    if format:
        if format not in ("wav", "flac", "mp3"):
            raise HTTPException(422, "format must be wav, flac or mp3")
        if path.suffix.lstrip(".").lower() != format:
            path = _converted_track(job_id, path, format)
    if inline:
        # No filename: omit Content-Disposition so the browser plays it in place.
        return FileResponse(path)
    return FileResponse(path, filename=f"yue2-{job_id}.{path.suffix.lstrip('.')}")


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


class SetCreate(BaseModel):
    name: str = Field(default="", max_length=200)
    target_minutes: float = Field(default=60, ge=1, le=600)
    brief: str = Field(default="", max_length=8000)


class SetDraft(BaseModel):
    api_key: str = Field(min_length=8)
    model: str = Field(min_length=1, max_length=200)
    brief: str = Field(default="", max_length=8000)
    target_minutes: float = Field(default=60, ge=1, le=600)
    track_count: int = Field(default=12, ge=1, le=40)
    language: str = Field(default="English", max_length=60)
    append: bool = False
    instrumental: bool = False


class SetTranslate(BaseModel):
    api_key: str = Field(min_length=8)
    model: str = Field(min_length=1, max_length=200)
    language: str = Field(min_length=2, max_length=60)
    song_ids: list[str] | None = Field(default=None, max_length=60)


class SetRender(BaseModel):
    only_missing: bool = True
    song_id: str | None = Field(default=None, max_length=40)
    ode_steps: int | None = Field(default=None, ge=1, le=256)
    max_abc_tokens: int = Field(default=12000, ge=1, le=20000)


class SetConcat(BaseModel):
    format: str = Field(default="mp3", pattern="^(mp3|wav)$")
    bitrate: str = Field(default="320k", max_length=12)


class SetVideo(BaseModel):
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=240, le=2160)
    fps: int = Field(default=2, ge=1, le=30)
    crf: int = Field(default=23, ge=0, le=51)
    bitrate: str = Field(default="320k", max_length=12)


def _load_set_or_404(set_id: str) -> dict:
    try:
        return sets.reconcile(sets.load_set(set_id))
    except FileNotFoundError:
        raise HTTPException(404, "unknown set") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/llm/models")
def api_llm_models(refresh: bool = False):
    try:
        return {"models": sets.openrouter_models(force=refresh)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, str(exc)) from exc


@app.get("/api/sets")
def api_sets():
    return sets.list_sets()


@app.post("/api/sets")
def api_set_create(req: SetCreate):
    return sets.create_set(req.name, req.target_minutes, req.brief)


def _enrich(set_id: str, data: dict) -> dict:
    """Attach live job state and the derived files the UI needs to show."""
    for song in data.get("songs") or []:
        job = STATE.jobs.get(str(song.get("job_id") or ""))
        song["job"] = job.public() if job is not None else None
        song["needs_rerender"] = False
        if song.get("audio_seconds") and song.get("job_id"):
            params = job.params if job is not None else None
            if params is None and re.fullmatch(r"[A-Za-z0-9_-]{1,40}", str(song["job_id"])):
                meta = sets.OUTPUTS / str(song["job_id"]) / "meta.json"
                if meta.is_file():
                    try:
                        params = json.loads(meta.read_text(encoding="utf-8")).get("params")
                    except (OSError, ValueError):
                        pass
            if isinstance(params, dict) and "style" in params and "lyrics" in params:
                song["needs_rerender"] = (
                    (song.get("style") or "instrumental") != params["style"]
                    or (song.get("lyrics") or "") != params["lyrics"]
                )
    for name in ("set.mp3", "set.wav"):
        path = sets.SETS_DIR / set_id / name
        if path.is_file():
            data["concat"] = {
                "file": name,
                "bytes": path.stat().st_size,
                "download": f"/api/sets/{set_id}/audio",
            }
            break
    image = sets.background_path(set_id)
    data["image"] = {"file": image.name, "bytes": image.stat().st_size} if image else None
    video = sets.video_path(set_id)
    data["video"] = (
        {"file": video.name, "bytes": video.stat().st_size, "download": f"/api/sets/{set_id}/video-file"}
        if video
        else None
    )
    data["media"] = dict(MEDIA_STATE) if MEDIA_STATE["kind"] else None
    return data


@app.get("/api/sets/{set_id}")
def api_set_get(set_id: str):
    return _enrich(set_id, _load_set_or_404(set_id))


@app.put("/api/sets/{set_id}")
def api_set_save(set_id: str, data: dict):
    previous = _load_set_or_404(set_id)
    data["id"] = set_id
    incoming = sets.normalize(data)
    current_by_id = {song["id"]: song for song in previous["songs"]}
    for song in incoming["songs"]:
        current = current_by_id.get(song["id"])
        if current:
            for field in ("job_id", "status", "audio", "audio_seconds"):
                song[field] = current.get(field)
    changed_audio = sets.audio_signature(previous["songs"]) != sets.audio_signature(incoming["songs"])
    if changed_audio and not MEDIA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "wait for the current join or video to finish before changing tracks")
    try:
        saved = sets.save_set(incoming)
        if changed_audio:
            sets.clear_renders(set_id)
    finally:
        if changed_audio:
            MEDIA_LOCK.release()
    return _enrich(set_id, saved)


@app.delete("/api/sets/{set_id}")
def api_set_delete(set_id: str):
    _load_set_or_404(set_id)
    sets.delete_set(set_id)
    return {"ok": True}


@app.post("/api/sets/{set_id}/draft")
def api_set_draft(set_id: str, req: SetDraft):
    data = _load_set_or_404(set_id)
    existing = data.get("songs") or []
    messages = sets.build_set_messages(
        req.brief or data.get("brief", ""),
        req.target_minutes,
        req.track_count,
        req.language,
        existing=existing if req.append else None,
        instrumental=req.instrumental,
    )
    try:
        content = sets.openrouter_chat(req.api_key, req.model, messages, max_tokens=12000)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, str(exc)) from exc
    try:
        name, background, songs = sets.songs_from_llm(sets.parse_json_object(content), req.target_minutes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            502,
            f"could not use the model reply ({exc}). Try a model with a larger output limit, "
            "or ask for fewer tracks.",
        ) from exc
    old_signature = sets.audio_signature(existing)
    if req.append:
        songs = sets.merge_tracks(existing, songs)
    else:
        sets.carry_rendered(existing, songs)
    data["songs"] = songs
    data["name"] = name or data.get("name") or "Untitled set"
    data["background"] = background or data.get("background", "")
    data["brief"] = req.brief or data.get("brief", "")
    data["target_minutes"] = req.target_minutes
    data["lyrics_language"] = req.language
    data["track_count"] = req.track_count
    data["instrumental"] = req.instrumental
    data["model"] = req.model
    if sets.audio_signature(songs) != old_signature:
        # the derived audio/video belong to the previous tracklist
        sets.clear_renders(set_id)
    return _enrich(set_id, sets.save_set(data))


@app.post("/api/sets/{set_id}/translate")
def api_set_translate(set_id: str, req: SetTranslate):
    data = _load_set_or_404(set_id)
    wanted = set(req.song_ids or [])
    targets = [song for song in data["songs"] if not wanted or song["id"] in wanted]
    titles = [str(song.get("title") or "").strip() or "Untitled" for song in targets]
    if not titles:
        raise HTTPException(422, "no tracks to translate")
    messages = sets.build_translation_messages(titles, req.language)
    try:
        content = sets.openrouter_chat(
            req.api_key, req.model, messages, max_tokens=2000, temperature=0.3
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, str(exc)) from exc
    try:
        sets.apply_translations(targets, sets.parse_json_object(content).get("translations"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"could not use the model reply ({exc})") from exc
    data["translation_language"] = req.language
    return _enrich(set_id, sets.save_set(data))


@app.post("/api/sets/{set_id}/render")
def api_set_render(set_id: str, req: SetRender):
    data = _load_set_or_404(set_id)
    auto = (data.get("duration_mode") or "auto") == "auto"
    queued = []
    for song in sorted(data["songs"], key=lambda s: int(s["order"])):
        if req.song_id and song["id"] != req.song_id:
            continue
        if not (song.get("style") or song.get("lyrics")):
            continue
        if req.only_missing and song.get("audio_seconds"):
            continue
        seed = random.randint(0, 2**31 - 1)
        params = {
            "style": song.get("style") or "instrumental",
            "lyrics": song.get("lyrics") or "",
            "cot": "full",
            "seed": seed,
            "abc": None,
            "cover_audio": None,
            "cfg_scale": None,
            # auto: leave the duration blank so the job sizes it from the score
            "max_duration": None if auto else float(song.get("target_seconds") or 300),
            "format": "wav",
            "plan_only": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": None,
            "ode_steps": req.ode_steps,
            "max_abc_tokens": req.max_abc_tokens,
            "fast_decode": False,
            "mp3": False,
            "set_id": set_id,
            "song_id": song["id"],
        }
        job = Job(id=uuid.uuid4().hex[:12], params=params, seed=seed)
        STATE.put(job)
        song.update(job_id=job.id, seed=seed, status="queued", audio=None, audio_seconds=None)
        queued.append({"song_id": song["id"], "job_id": job.id, "order": song["order"]})
    if queued:
        # re-rendering changes the audio, so the joined files are stale
        sets.clear_renders(set_id)
        sets.save_set(data)
    return {"queued": queued, "state": STATE.snapshot()}


@app.post("/api/sets/{set_id}/concat")
def api_set_concat(set_id: str, req: SetConcat):
    data = _load_set_or_404(set_id)
    if not MEDIA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another join or video is still running")
    MEDIA_STATE.update(kind="concat", percent=0.0)
    try:
        path = sets.concat_set(
            data,
            ffmpeg_path() or "",
            out_format=req.format,
            bitrate=req.bitrate,
            on_progress=lambda percent: MEDIA_STATE.update(percent=percent),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, str(exc)) from exc
    finally:
        MEDIA_STATE.update(kind=None, percent=None)
        MEDIA_LOCK.release()
    _, total = sets.timeline(data)
    return {
        "file": path.name,
        "download": f"/api/sets/{set_id}/audio",
        "bytes": path.stat().st_size,
        "seconds": round(total, 1),
    }


@app.get("/api/sets/{set_id}/audio")
def api_set_audio(set_id: str):
    data = _load_set_or_404(set_id)
    for name in ("set.mp3", "set.wav"):
        path = sets.SETS_DIR / set_id / name
        if path.is_file():
            return FileResponse(path, filename=f"{(data.get('name') or 'set').strip()}{path.suffix}")
    raise HTTPException(404, "no concatenated file yet")


def _converted_track(job_id: str, source: Path, fmt: str) -> Path:
    """Convert a rendered track on demand, caching the result next to it."""
    target = source.with_suffix("." + fmt)
    if target.is_file():
        return target
    if not CONVERT_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another conversion is running, try again in a moment")
    try:
        if not target.is_file():
            if fmt == "mp3":
                encode_mp3(source, target)
            else:
                raise HTTPException(422, f"cannot convert to {fmt} on demand")
    finally:
        CONVERT_LOCK.release()
    return target


PEAKS_CACHE: dict = {}


@app.get("/api/job/{job_id}/peaks")
def api_job_peaks(job_id: str, buckets: int = 64):
    """Waveform peaks for the UI, computed server side so the browser does not decode audio."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", job_id):
        raise HTTPException(422, "bad job id")
    buckets = max(16, min(256, buckets))
    path = sets.resolve_audio_file(job_id)
    if path is None:
        raise HTTPException(404, "no audio for this job")
    key = (job_id, buckets)
    if key in PEAKS_CACHE:
        return {"peaks": PEAKS_CACHE[key]}
    try:
        import numpy as np

        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"could not read the audio: {exc}") from exc
    mono = data.mean(axis=1) if data.ndim > 1 else data
    edges = np.linspace(0, len(mono), buckets + 1).astype(int)
    peaks = []
    for index in range(buckets):
        chunk = mono[edges[index]:edges[index + 1]]
        peaks.append(float(np.abs(chunk).max()) if chunk.size else 0.0)
    top = max(peaks) or 1.0
    normalized = [round(min(1.0, value / top * 1.1), 3) for value in peaks]
    PEAKS_CACHE[key] = normalized
    return {"peaks": normalized}


@app.get("/api/sets/{set_id}/export")
def api_set_export(set_id: str, format: str = "youtube"):
    data = _load_set_or_404(set_id)
    if format == "json":
        return Response(sets.export_json(data), media_type="application/json")
    if format == "md":
        return Response(sets.export_markdown(data), media_type="text/markdown; charset=utf-8")
    if format == "youtube":
        return Response(sets.export_youtube(data), media_type="text/plain; charset=utf-8")
    raise HTTPException(422, "format must be json, md or youtube")


@app.post("/api/sets/{set_id}/image")
def api_set_image(set_id: str, image: UploadFile = File(...)):
    _load_set_or_404(set_id)
    if not MEDIA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another join or video is still running")
    try:
        path = sets.save_background(set_id, image.filename or "", image.file)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        MEDIA_LOCK.release()
    return {"ok": True, "file": path.name, "bytes": path.stat().st_size}


@app.post("/api/sets/{set_id}/video")
def api_set_video(set_id: str, req: SetVideo):
    data = _load_set_or_404(set_id)
    ffmpeg = ffmpeg_path() or ""
    image = sets.background_path(set_id)
    if image is None:
        raise HTTPException(422, "upload a background image first")
    if not MEDIA_LOCK.acquire(blocking=False):
        raise HTTPException(409, "another join or video is still running")
    try:
        if sets.concat_audio_path(set_id) is None:
            MEDIA_STATE.update(kind="concat", percent=0.0)
            sets.concat_set(
                data, ffmpeg, out_format="mp3", bitrate=req.bitrate,
                on_progress=lambda percent: MEDIA_STATE.update(percent=percent),
            )
        MEDIA_STATE.update(kind="video", percent=0.0)
        path = sets.make_video(
            data, ffmpeg, image, width=req.width, height=req.height, fps=req.fps, crf=req.crf,
            on_progress=lambda percent: MEDIA_STATE.update(percent=percent),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, str(exc)) from exc
    finally:
        MEDIA_STATE.update(kind=None, percent=None)
        MEDIA_LOCK.release()
    _, total = sets.timeline(data)
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "seconds": round(total, 1),
        "download": f"/api/sets/{set_id}/video-file",
    }


@app.get("/api/sets/{set_id}/video-file")
def api_set_video_file(set_id: str, inline: bool = False):
    data = _load_set_or_404(set_id)
    path = sets.video_path(set_id)
    if path is None:
        raise HTTPException(404, "no video yet")
    name = (data.get("name") or "set").strip() or "set"
    if inline:
        return FileResponse(path, media_type="video/mp4")
    return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")


def main():
    if not (MODEL_DIR / "config.json").exists():
        print(f"! Local model not found at {MODEL_DIR}; it will download from the Hub on first run.")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()
