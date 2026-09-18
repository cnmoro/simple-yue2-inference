# YuE2 Studio

Standalone webUI for [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B), the open
music generation model from M-A-P. Type a style prompt and lyrics, get a full
song with vocals and accompaniment as 48 kHz stereo audio.

This is a thin wrapper around the official `yue2_infer` Python package. No
ComfyUI, no graph, no workflow JSON.

## How YuE2 works

YuE2 is an autoregressive + non-autoregressive model that turns text into a
song in three stages:

1. **Symbolic planning (AR).** Write an ABC score for the song — melody and,
   in `full` mode, chord symbols. This is the "editable score" part.
2. **Semantic token generation (AR).** Given the prompt + the score, generate
   music codec tokens.
3. **Synthesis + decode.** A flow-matching NAR model produces acoustic latents,
   and the YuE2 VAE decodes them to 48 kHz stereo audio.

The `cot` parameter picks the planning stage: `full` (melody + chords),
`melody` (melody only, recommended for covers), or `off` (skip the score).

There is no audio-reference argument anywhere in the YuE2 API. Covers work by
transcribing the reference recording to a **melody-only ABC score with
SheetSage2**, then conditioning YuE2 on that score. The original waveform and
the singer's voice are not transferred.

## Requirements

- Linux, Python 3.10–3.12 (the pinned `torch==2.10.0` has no 3.13+/3.14 wheels)
- NVIDIA GPU with BF16. 16 GB VRAM is enough; the reference peak is ~11 GiB for
  a full song. ~24 GB host RAM.
- `conda` for the environment bootstrap (or adapt `setup_env.sh` yourself).
- Covers additionally need the `yue2-sheetsage2` env (~3 GB extra weights), see
  below.

## Setup

```bash
bash setup_env.sh        # creates conda env "yue2", installs torch + yue2_infer + webUI deps
bash download_models.sh  # ~18 GB into ./models (YuE2-3B, YuE2-Vae, SheetSage2, MERT-v2)
conda activate yue2
python doctor.py         # sanity check: torch/cuda/bf16, package versions, model paths
```

Covers need SheetSage2 in a second env, because its pins (torch 2.8,
transformers 4.45, numpy 1.24) conflict with the generation env:

```bash
bash setup_sheetsage2.sh # creates conda env "yue2-sheetsage2"
```

Then:

```bash
python app.py            # http://127.0.0.1:7860
```

The model loads lazily on the first generation. You can also hit **Preload
model** in the UI, or start with `YUE2_PRELOAD=1 python app.py`.

## Using it

Fill in **Style / tags** and **Lyrics** (use `[Verse]` / `[Chorus]` markers),
pick a planning mode, hit Generate. First run includes model loading, so it is
slower than subsequent songs.

| Control | Meaning |
| --- | --- |
| Style / tags | Genre, mood, instruments, vocal gender/type, tempo. One long line works best. |
| Lyrics | Section markers guide song structure. |
| Instrumental | Clears and ignores lyrics. Best-effort — see the notes below. |
| Planning mode `cot` | `full` = melody+chords, `melody` = melody only, `off` = no score. |
| Seed | Same seed + same inputs = same song. Blank randomizes. |
| Max duration (s) | Upper bound on audio length, 1–900 s, blank = 360. The song can end earlier. |
| CFG scale | Text guidance. Default is 1.0 (`1.01` for `cot=off`). Try 1.2. |
| ABC score | Optional score to condition on. Requires `melody` or `full`. |
| Plan only | Returns the generated ABC score without rendering audio. |
| Advanced | Sampling (`temperature`, `top-p`, `top-k`, `repetition penalty`), flow-matching `steps`, ABC token budget, fast decode. |

Songs are written to `outputs/<job-id>/` as `song.wav|flac`, `score.abc`, and
`meta.json`. Covers also keep `input/reference.*` and the full
`transcription/` directory.

### Instrumental

YuE2 is a lyrics-to-song model and has no hard instrumental switch. The switches
that do exist, in order of how well they hold:

1. **Supply your own ABC score** with `cot=full` or `melody` and blank lyrics.
   The accompaniment follows the score, so this is the most controllable path.
2. **Blank lyrics + `cot=melody`**, with the instrumentation described in Style
   and no vocal words anywhere.
3. `[Instrumental]` / `[instrument solo]` section tags. These leak vocals often.

Even then vocals can appear, especially for vocal-heavy genres. Community
instrumental LoRAs (e.g. `Mothersuperior/YuE2-instrumental-cot-full-loras`)
target exactly this, but they need LoRA loading, which is not wired up here.

### Duration

YuE2's codec runs at 25 frames per second, and one semantic token is one frame,
so **seconds = semantic tokens / 25**. Max duration is applied as a token cap on
the semantic stage (`max_tokens = round(seconds * 25)`), which is what ComfyUI's
built-in node does as well. It is an upper bound, not a target: the model can
emit its end token earlier. When a cap is hit, the result shows "truncated
(token cap)".

There is a second budget on the planning stage: **Max ABC tokens**. The UI
prefills 8192, matching ComfyUI's built-in node; clear the field to fall back to
the package's 4096. For long songs the generated score can run past 4096 tokens
and get cut off, which then limits the song. Set it higher (up to 20000) if a
long plan looks truncated.

**Fast decode** skips VAE tiling (`decode(full=True)`). It is noticeably faster
when the card has room. On the 16 GB test card a 40 s clip decoded fine; a full
360 s song is more likely to OOM, so leave it off for long songs.

### Covers

Switch to **Audio -> Cover**, drop a reference recording, and write the new
style and lyrics. The backend runs SheetSage2 in the `yue2-sheetsage2` env to
get a melody-only ABC, then generates with `cot=melody` and that score. The
transcribed ABC is shown in the result so you can copy it back into the ABC
field and edit it.

Covers preserve the **melody**, not the voice or the waveform. Keep the new
lyrics close in length and phrasing to the original for a tighter melodic match.
For a fixed harmony, switch the planning mode to `full`.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `YUE2_MODEL` | `./models/YuE2-3B` | Model path or HF repo id |
| `YUE2_VAE` | `./models/YuE2-Vae` | Decoder path or HF repo id (or `YuE2-Vae-legacy`) |
| `YUE2_OUTPUTS` | `./outputs` | Where songs are written |
| `YUE2_MEMORY_GIB` | `16` | VRAM budget. Lower to 12 forces small VAE tiles. |
| `YUE2_DEVICE` | `cuda` | `cuda`, `mps`, or `cpu` |
| `YUE2_HOST` / `YUE2_PORT` | `127.0.0.1` / `7860` | Server bind |
| `YUE2_PRELOAD` | `0` | Set to `1` to load the model at startup |
| `YUE2_SHEETSAGE2` | `./models/SheetSage2` | SheetSage2 snapshot (covers) |
| `YUE2_MERT` | `./models/MERT-v2-FullSong` | MERT-v2 encoder for SheetSage2 |
| `YUE2_SHEETSAGE2_PY` | sibling env `yue2-sheetsage2` | Python of the transcription env |

## Verified on

RTX 5060 Ti 16 GB (sm_120), 44 GB RAM, torch 2.10.0+cu128, Python 3.11:

| Mode | Input | Audio | Time | Tokens |
| --- | --- | --- | --- | --- |
| `full` | short verse + chorus | 1:08 | 72 s | 2461 |
| `off` | `[Instrumental]` lyrics | 2:44 | 121 s | 4117 |
| `full`, plan only | one line | — (ABC only) | 14 s | 805 |
| `off`, instrumental, 40 s cap | blank lyrics | 0:40 (capped) | 37 s | 1000 |
| `off`, instrumental, fast decode | blank lyrics | 0:40 (capped) | 29 s | 1000 |
| cover, 60 s cap | 1:08 reference | 1:00 (capped) | 57 s | 1500 |

First call includes model load; the pipeline offloads the LM after decoding, so
per-job load time recurs. No OOM at `YUE2_MEMORY_GIB=16`. The cover row includes
SheetSage2 transcription (~3 GB weights, separate env).

## Notes and limits

- **One song at a time.** The pipeline holds one model and is not safe to call
  concurrently. Jobs are queued and run by a single worker thread.
- **No auth.** Binds to localhost by default. Don't expose it publicly.
- **Licensing.** Model weights are CC BY-NC 4.0 (research use). The ComfyUI
  repackage is `Comfy-Org/YuE2`; the upstream repo is `m-a-p/YuE2-3B`.
- Progress for the NAR/VAE stages is indeterminate — the public pipeline API
  does not expose a per-step callback there, only token counts for the AR
  stages. Transcription progress is just a window counter on stderr.
- **Covers do not transfer the voice** and cannot be cancelled mid-transcription;
  the SheetSage2 subprocess runs to completion.
- **Instrumental is best-effort.** The base checkpoint is lyric-first and has no
  negative prompt to suppress vocals.

## Layout

```
app.py               FastAPI server + single-worker job queue
static/index.html    single-page UI
cover_transcribe.py  SheetSage2 wrapper (runs in the sheetsage2 env)
doctor.py            environment check
setup_env.sh         generation env (torch 2.10 + yue2_infer + webUI)
setup_sheetsage2.sh  transcription env (torch 2.8 + SheetSage2)
download_models.sh   fetch weights into ./models
requirements.txt     webUI-only deps (torch + yue2_infer come from setup_env.sh)
```
