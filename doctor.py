"""Print environment readiness for the YuE2 webUI."""
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    print(f"python      : {sys.version.split()[0]} ({platform.machine()})")
    try:
        import torch
        print(f"torch       : {torch.__version__}")
        print(f"cuda build  : {torch.version.cuda}")
        print(f"cuda avail  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  gpu {i}      : {p.name}  {p.total_memory / 2**30:.1f} GiB  sm_{p.major}{p.minor}")
            print(f"bf16        : {torch.cuda.is_bf16_supported()}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch       : MISSING ({exc})")
    try:
        import yue2
        print(f"yue2-infer  : {yue2.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"yue2-infer  : MISSING ({exc})")
    try:
        import fastapi
        print(f"fastapi     : {fastapi.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"fastapi     : MISSING ({exc})")
    for name, path in (
        ("model", HERE / "models/YuE2-3B"),
        ("vae", HERE / "models/YuE2-Vae"),
        ("sheetsage2", HERE / "models/SheetSage2"),
        ("mert-v2", HERE / "models/MERT-v2-FullSong"),
    ):
        print(f"{name:<12}: {'found' if path.exists() else 'MISSING'}  {path}")
    sheetsage2_py = Path(sys.executable).resolve().parents[2] / "yue2-sheetsage2/bin/python"
    print(f"cover env   : {'found' if sheetsage2_py.exists() else 'MISSING'}  {sheetsage2_py}")
    print(f"ffmpeg      : {shutil.which('ffmpeg') or 'MISSING (only needed for mp3 export)'}")


if __name__ == "__main__":
    main()
