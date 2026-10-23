"""Push a local alarm image to the third-party alarm API.

Edit the configuration block below, then run:
    python push_alarm.py

Command-line arguments can still override the configuration for one-off tests:
    python push_alarm.py --image alarm_outputs/example.jpg --camera-id CAMERA001 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# =========================
# Editable configuration
# =========================

API_URL = "http://bim2.ztmbec.com:64307/api/KMJG/ViolationsAlarmThirdPlate/Forward/zidingyi"

# Fill this with the CameraID required by the third-party platform.
CAMERA_ID = ""

# Fill this with a local alarm image path, or leave empty and pass --image.
IMAGE_PATH = ""

VIOLATOR_LSNUMBER = "zidingyi"
EVENT_TYPE = "机械伤害预警"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
TIMEOUT_SECONDS = 10.0

# True prints the payload metadata without sending the HTTP request.
DRY_RUN = False


def image_to_base64(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    return base64.b64encode(path.read_bytes()).decode("ascii")


def parse_event_time(value: Optional[str]) -> datetime:
    return datetime.strptime(value, TIME_FORMAT) if value else datetime.now()


def build_payload(
    image_path: Path,
    camera_id: str,
    event_time: datetime,
    event_type: str = EVENT_TYPE,
) -> Dict[str, Any]:
    if not camera_id.strip():
        raise ValueError("CameraID is empty. Please set CAMERA_ID or pass --camera-id.")
    if not event_type.strip():
        raise ValueError("Event type is empty.")

    return {
        "Violator_LSNumber": VIOLATOR_LSNUMBER,
        "EventType": event_type.strip(),
        "time": event_time.strftime(TIME_FORMAT),
        "CameraID": camera_id,
        "ViolationPhoto": image_to_base64(image_path),
    }


def push_alarm(api_url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_text = response.read().decode("utf-8", errors="replace")
        return {
            "success": 200 <= int(response.status) < 300,
            "status": int(response.status),
            "response": response_text,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Push a local alarm image to the third-party API.")
    parser.add_argument("--image", help="Override IMAGE_PATH.")
    parser.add_argument("--camera-id", help="Override CAMERA_ID.")
    parser.add_argument("--api-url", help="Override API_URL.")
    parser.add_argument("--event-type", help="Override the default algorithm/alarm name.")
    parser.add_argument("--time", help=f"Override event time, format: {TIME_FORMAT}. Defaults to now.")
    parser.add_argument("--timeout", type=float, help="Override TIMEOUT_SECONDS.")
    parser.add_argument("--dry-run", action="store_true", help="Only print payload metadata, do not send.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path_text = args.image or IMAGE_PATH
    camera_id = args.camera_id or CAMERA_ID
    api_url = args.api_url or API_URL
    timeout = args.timeout if args.timeout is not None else TIMEOUT_SECONDS
    dry_run = bool(args.dry_run or DRY_RUN)

    if not image_path_text.strip():
        raise ValueError("Image path is empty. Please set IMAGE_PATH or pass --image.")
    if timeout <= 0:
        raise ValueError("Timeout must be greater than 0.")

    payload = build_payload(
        image_path=Path(image_path_text),
        camera_id=camera_id,
        event_time=parse_event_time(args.time),
        event_type=args.event_type or EVENT_TYPE,
    )

    if dry_run:
        preview = dict(payload)
        preview["ViolationPhoto"] = f"<base64 length={len(payload['ViolationPhoto'])}>"
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    result = push_alarm(api_url, payload, timeout)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
