"""Launch the user's local YOLOv5 checkout without mutating that repository."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = Path(r"C:\Users\LENOVO\PycharmProjects\YOLOv5\yolov5")
LOCAL_SUPPORT_SITE = Path(r"C:\Users\LENOVO\anaconda3\envs\yolo\Lib\site-packages")


def configure_imports() -> Path:
    repo = Path(os.environ.get("PLATFORM_YOLOV5_REPO") or DEFAULT_REPO).resolve()
    if not (repo / "train.py").is_file():
        raise RuntimeError(f"YOLOv5 train.py 不存在：{repo}")
    for path in (PROJECT_ROOT, repo):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    # This host already has the small GitPython/thop helpers in another local
    # YOLO environment.  Append it after the active environment so Torch,
    # NumPy and all primary packages still come from the requested yolov5 env.
    if LOCAL_SUPPORT_SITE.is_dir() and str(LOCAL_SUPPORT_SITE) not in sys.path:
        sys.path.append(str(LOCAL_SUPPORT_SITE))
    return repo


def preflight(weights: str) -> None:
    repo = configure_imports()
    import torch
    import train

    checkpoint = torch.load(str(Path(weights).resolve()), map_location="cpu")
    model = (checkpoint.get("ema") or checkpoint.get("model")) if isinstance(checkpoint, dict) else checkpoint
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    result = {
        "ok": True,
        "repo": str(repo),
        "train_module": str(Path(train.__file__).resolve()),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda or ""),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
        if torch.cuda.is_available()
        else 0.0,
        "weight_names": list(names or []),
        "weight_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
    }
    print("PLATFORM_PREFLIGHT=" + json.dumps(result, ensure_ascii=False), flush=True)


def run_training() -> None:
    configure_imports()
    import train

    # The platform performs its own deterministic preflight.  Avoid YOLOv5's
    # network-based auto-installer and git status check on an offline site.
    train.check_requirements = lambda *args, **kwargs: True
    train.check_git_status = lambda *args, **kwargs: None
    options = train.parse_opt()
    train.main(options)


def main() -> None:
    if "--platform-preflight" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--platform-preflight", action="store_true")
        parser.add_argument("--weights", required=True)
        args = parser.parse_args()
        preflight(args.weights)
        return
    run_training()


if __name__ == "__main__":
    main()
