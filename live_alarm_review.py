"""Serial local-VLM review queue for newly generated alarm events."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import config
from offline_alarm_review import (
    PROMPT_VERSION,
    AlarmCandidate,
    call_review_model,
    create_keyframe_contact_sheet,
    create_review_target_crop,
    extract_keyframes,
    load_prompt_bundle,
    read_json,
)


LOGGER = logging.getLogger(__name__)
MODEL_NAME = str(config.ALARM_REVIEW_MODEL_NAME)


class LiveAlarmReviewManager:
    """Own one model server and one worker so reviews never block detection."""

    def __init__(self, alarm_store: Any) -> None:
        self.alarm_store = alarm_store
        self.enabled = bool(config.ALARM_REVIEW_ENABLED)
        self._queue: queue.Queue[Optional[str]] = queue.Queue()
        self._queued: set[str] = set()
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._server_process: Optional[subprocess.Popen[Any]] = None
        self._server_stdout: Optional[Any] = None
        self._server_stderr: Optional[Any] = None
        self._current_event_id = ""
        self._last_error = ""
        self._completed = 0
        self._system_prompt = ""
        self._algorithm_prompts: Dict[str, str] = {}
        self._review_schema: Dict[str, Any] = {}

    def start(self) -> None:
        if not self.enabled:
            LOGGER.warning("[ALARM_REVIEW] disabled")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._system_prompt, self._algorithm_prompts, self._review_schema = load_prompt_bundle(
                Path(config.ALARM_REVIEW_PROMPT_DIR)
            )
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker,
                name="alarm-vlm-review",
                daemon=True,
            )
            self._thread.start()
        for event_id in self.alarm_store.pending_review_event_ids():
            self.enqueue(event_id)

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5.0)
        process = self._server_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8.0)
            except subprocess.TimeoutExpired:
                process.kill()
        for stream in (self._server_stdout, self._server_stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        self._server_process = None
        self._server_stdout = None
        self._server_stderr = None

    def enqueue(self, event_id: str) -> bool:
        normalized = str(event_id or "").strip()
        if not self.enabled or not normalized:
            return False
        with self._lock:
            if normalized in self._queued or normalized == self._current_event_id:
                return False
            self._queued.add(normalized)
        self._queue.put(normalized)
        return True

    def status(self) -> Dict[str, Any]:
        process = self._server_process
        return {
            "enabled": self.enabled,
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "worker_alive": bool(self._thread and self._thread.is_alive()),
            "model_server_owned": process is not None,
            "model_server_alive": bool(process is not None and process.poll() is None),
            "queued": self._queue.qsize(),
            "current_event_id": self._current_event_id or None,
            "completed": self._completed,
            "last_error": self._last_error,
        }

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                event_id = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if event_id is None:
                self._queue.task_done()
                break
            with self._lock:
                self._queued.discard(event_id)
                self._current_event_id = event_id
            try:
                self._review_event(event_id)
                self._completed += 1
                self._last_error = ""
            except Exception as exc:
                self._last_error = str(exc)
                LOGGER.exception("[ALARM_REVIEW] failed event=%s", event_id)
                try:
                    self.alarm_store.fail_review(
                        event_id,
                        str(exc),
                        model=MODEL_NAME,
                        prompt_version=PROMPT_VERSION,
                    )
                except Exception:
                    LOGGER.exception("[ALARM_REVIEW] failed to persist failure event=%s", event_id)
            finally:
                with self._lock:
                    self._current_event_id = ""
                self._queue.task_done()

    def _review_event(self, event_id: str) -> None:
        alarm = self.alarm_store.claim_review(event_id)
        if alarm is None:
            return
        metadata_path = Path(str(alarm.get("metadata_path") or ""))
        if not metadata_path.is_file():
            raise FileNotFoundError("告警 metadata.json 不存在")
        event_dir = metadata_path.parent
        self._wait_for_assets(event_dir, metadata_path)
        metadata = read_json(metadata_path)
        candidate = AlarmCandidate(
            event_id=str(metadata.get("event_id") or event_id),
            event_dir=event_dir,
            metadata_path=metadata_path,
            metadata=metadata,
        )
        review_dir = event_dir / "vlm_review"
        keyframes = extract_keyframes(
            event_dir / "alarm_clip.mp4",
            review_dir / "keyframes",
            int(config.ALARM_REVIEW_KEYFRAME_COUNT),
        )
        contact_sheet = create_keyframe_contact_sheet(
            keyframes,
            review_dir / "keyframes_contact.jpg",
        )
        target_crop = create_review_target_crop(
            candidate,
            review_dir / "review_target_crop.jpg",
        )
        self._ensure_model_server()
        started_at = time.monotonic()
        review, raw_content, usage = call_review_model(
            candidate=candidate,
            keyframes=keyframes,
            contact_sheet_path=contact_sheet,
            system_prompt=self._system_prompt,
            algorithm_prompts=self._algorithm_prompts,
            review_schema=self._review_schema,
            server_url=str(config.ALARM_REVIEW_SERVER_URL),
            timeout=float(config.ALARM_REVIEW_REQUEST_TIMEOUT_SECONDS),
            max_tokens=int(config.ALARM_REVIEW_MAX_TOKENS),
            target_crop_path=target_crop,
        )
        latency = round(time.monotonic() - started_at, 3)
        review_dir.mkdir(parents=True, exist_ok=True)
        (review_dir / "raw_response.txt").write_text(raw_content, encoding="utf-8")
        result = {
            "event_id": event_id,
            "status": "completed",
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
            "latency_seconds": latency,
            "usage": usage,
            "keyframes": [
                {
                    "filename": item.get("filename"),
                    "frame_index": item.get("frame_index"),
                    "seconds": item.get("seconds"),
                }
                for item in keyframes
            ],
            "review": review,
        }
        temporary = review_dir / "review.tmp"
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(review_dir / "review.json")
        if (metadata.get("scene_mode") == "crane_boom_person" and review.get("verdict") == "confirmed"
                and "pair_reviews" in review):
            from crane_target_review import render_confirmed_alarm
            render_confirmed_alarm(candidate, review)
        completed = self.alarm_store.complete_review(
            event_id,
            review,
            model=MODEL_NAME,
            prompt_version=PROMPT_VERSION,
        )
        if completed["verdict"] == "confirmed":
            self.alarm_store.report_alarm(int(completed["id"]))
        LOGGER.info(
            "[ALARM_REVIEW] completed event=%s verdict=%s latency=%.3f",
            event_id,
            completed["verdict"],
            latency,
        )

    def _wait_for_assets(self, event_dir: Path, metadata_path: Path) -> None:
        deadline = time.monotonic() + float(config.ALARM_REVIEW_ASSET_WAIT_SECONDS)
        required = [
            event_dir / "original.jpg",
            event_dir / "risk_crop.jpg",
            event_dir / "alarm.jpg",
            event_dir / "alarm_clip.mp4",
        ]
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            files_ready = all(path.is_file() and path.stat().st_size > 0 for path in required)
            video_ready = False
            try:
                metadata = read_json(metadata_path)
                video_ready = str(metadata.get("short_video_status") or "").lower() in {
                    "complete",
                    "completed",
                }
            except (OSError, ValueError, json.JSONDecodeError):
                pass
            if files_ready and video_ready:
                return
            self._stop_event.wait(0.5)
        if self._stop_event.is_set():
            raise RuntimeError("平台正在停止，复核已取消")
        missing = [path.name for path in required if not path.is_file() or path.stat().st_size <= 0]
        raise RuntimeError(f"告警复核素材未在限定时间内就绪：{', '.join(missing) or 'alarm_clip.mp4 未完成'}")

    @staticmethod
    def _server_health_url() -> str:
        return f"http://{config.ALARM_REVIEW_SERVER_HOST}:{int(config.ALARM_REVIEW_SERVER_PORT)}/health"

    def _server_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self._server_health_url(), timeout=2.0) as response:
                if not 200 <= int(response.status) < 300:
                    return False
            models_url = f"http://{config.ALARM_REVIEW_SERVER_HOST}:{int(config.ALARM_REVIEW_SERVER_PORT)}/v1/models"
            with urllib.request.urlopen(models_url, timeout=2.0) as response:
                models = json.loads(response.read().decode("utf-8"))
            return any(item.get("id") == MODEL_NAME for item in models.get("data", []))
        except (OSError, urllib.error.URLError, ValueError, AttributeError):
            return False

    def _ensure_model_server(self) -> None:
        if self._server_ready():
            return
        process = self._server_process
        if process is not None and process.poll() is not None:
            self._server_process = None
        if self._server_process is None:
            executable = Path(config.ALARM_REVIEW_LLAMA_SERVER_PATH)
            model_path = Path(config.ALARM_REVIEW_MODEL_PATH)
            mmproj_path = Path(config.ALARM_REVIEW_MMPROJ_PATH)
            for path, label in (
                (executable, "llama-server"),
                (model_path, "视觉大模型"),
                (mmproj_path, "视觉投影模型"),
            ):
                if not path.is_file():
                    raise FileNotFoundError(f"{label}文件不存在：{path}")
            Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
            self._server_stdout = (Path(config.LOG_DIR) / "alarm_vlm_live_stdout.log").open("ab")
            self._server_stderr = (Path(config.LOG_DIR) / "alarm_vlm_live_stderr.log").open("ab")
            command = [
                str(executable),
                "-m", str(model_path),
                "--mmproj", str(mmproj_path),
                "--host", str(config.ALARM_REVIEW_SERVER_HOST),
                "--port", str(int(config.ALARM_REVIEW_SERVER_PORT)),
                "--alias", MODEL_NAME,
                "--parallel", "1",
                "--reasoning", "off",
                "-c", str(int(config.ALARM_REVIEW_CONTEXT_SIZE)),
                "-ngl", str(int(config.ALARM_REVIEW_GPU_LAYERS)),
                "--image-max-tokens", str(int(config.ALARM_REVIEW_IMAGE_MAX_TOKENS)),
                "--flash-attn", "on",
            ]
            creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
            self._server_process = subprocess.Popen(
                command,
                cwd=str(model_path.parent),
                stdout=self._server_stdout,
                stderr=self._server_stderr,
                creationflags=creationflags,
            )
            LOGGER.info("[ALARM_REVIEW] starting local model server pid=%s", self._server_process.pid)
        deadline = time.monotonic() + float(config.ALARM_REVIEW_SERVER_START_TIMEOUT_SECONDS)
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            if self._server_ready():
                return
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RuntimeError(f"本地大模型服务提前退出，代码 {self._server_process.returncode}")
            self._stop_event.wait(1.0)
        raise RuntimeError("等待本地大模型服务启动超时")
