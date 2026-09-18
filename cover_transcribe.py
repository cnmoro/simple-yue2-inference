"""Transcribe a reference recording to a melody-only ABC score with SheetSage2.

Runs in the separate `yue2-sheetsage2` conda env (different torch / transformers
pins than the generation env). Invoked by the backend as a subprocess.

    python cover_transcribe.py song.mp3 --output out --model models/SheetSage2 \
        --base-model models/MERT-v2-FullSong
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="SheetSage2: audio -> melody-only ABC")
    parser.add_argument("audio")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="SheetSage2 snapshot directory")
    parser.add_argument("--base-model", required=True, help="MERT-v2-FullSong snapshot directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.model).resolve()))

    import torch
    from transformers import AutoModel

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=True,
        base_model_path=args.base_model,
    ).eval().to(device)

    def progress(value):
        if value.get("stage") == "encoding":
            print(f"window {value.get('window')}/{value.get('windows')}", file=sys.stderr, flush=True)

    result = model.transcribe(
        args.audio,
        output_dir=args.output,
        melody_only=True,
        dtype=args.dtype,
        progress=progress,
    )
    abc = result.get("abc")
    if not abc:
        raise SystemExit(f"no melody ABC produced: {result.get('abc_error')}")
    out = Path(args.output) / "melody.abc"
    out.write_text(abc, encoding="utf-8")
    print(json.dumps({"ok": True, "abc": str(out), "warnings": result.get("warnings", [])}))


if __name__ == "__main__":
    main()
