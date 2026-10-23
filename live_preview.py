"""Shared live-preview sessions and MJPEG Web routes."""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Set, Tuple

import cv2

import config
from video_channels import (
    ChannelNotFoundError,
    VideoChannelStore,
    channel_source,
    safe_error_message,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
PREVIEW_MAX_WIDTH = 1280
PREVIEW_JPEG_QUALITY = 75
PREVIEW_TARGET_FPS = 10.0
PREVIEW_IDLE_SECONDS = 8.0
PREVIEW_RECONNECT_SECONDS = 2.0


def timestamp_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class PreviewSession:
    channel_id: int
    condition: threading.Condition = field(default_factory=threading.Condition)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    last_unsubscribed_at: float = field(default_factory=time.monotonic)
    latest_frame: Optional[bytes] = None
    sequence: int = 0
    state: str = "connecting"
    error: str = ""
    last_frame_at: Optional[str] = None
    fps: float = 0.0
    width: int = 0
    height: int = 0
    reconnect_count: int = 0


class LivePreviewManager:
    """Open at most one RTSP connection per displayed channel and share JPEG frames."""

    def __init__(
        self,
        store: VideoChannelStore,
        resolve_capture_source: Callable[[str, Optional[float]], Tuple[Any, str]],
        make_camera_source: Callable[[str], str],
        stream_manager: Any = None,
    ) -> None:
        self.store = store
        self.resolve_capture_source = resolve_capture_source
        self.make_camera_source = make_camera_source
        self.stream_manager = stream_manager
        self._lock = threading.RLock()
        self._sessions: Dict[int, PreviewSession] = {}
        self._clients: Dict[str, Tuple[Set[int], float]] = {}

    def _ensure_session(self, channel_id: int) -> PreviewSession:
        channel = self.store.get_channel(channel_id, public=False)
        if not channel["enabled"]:
            raise ValueError("视频通道已停用")

        with self._lock:
            session = self._sessions.get(channel_id)
            if session is None or session.stop_event.is_set():
                session = PreviewSession(channel_id=channel_id)
                self._sessions[channel_id] = session
            if session.thread is None or not session.thread.is_alive():
                session.thread = threading.Thread(
                    target=self._run_session,
                    args=(session,),
                    daemon=True,
                    name="live-preview-{}".format(channel_id),
                )
                session.thread.start()
            return session

    def touch_client(self, client_id: str, channel_ids: Set[int]) -> None:
        client_key = str(client_id).strip()[:100]
        if not client_key:
            raise ValueError("预览客户端编号不能为空")
        now = time.monotonic()
        with self._lock:
            previous_ids = self._clients.get(client_key, (set(), now))[0]
            self._clients[client_key] = (set(channel_ids), now)
            for removed_id in previous_ids - channel_ids:
                session = self._sessions.get(removed_id)
                if session is not None:
                    session.last_unsubscribed_at = now
        for channel_id in channel_ids:
            self._ensure_session(channel_id)

    def _attach_client_channel(self, client_id: str, channel_id: int) -> PreviewSession:
        client_key = str(client_id).strip()[:100]
        if not client_key:
            client_key = "legacy-{}-{}".format(channel_id, time.monotonic_ns())
        now = time.monotonic()
        with self._lock:
            channel_ids = set(self._clients.get(client_key, (set(), now))[0])
            channel_ids.add(channel_id)
            self._clients[client_key] = (channel_ids, now)
        return self._ensure_session(channel_id)

    def _active_client_count(self, channel_id: int) -> int:
        now = time.monotonic()
        with self._lock:
            expired = [
                client_id
                for client_id, (_channel_ids, touched_at) in self._clients.items()
                if now - touched_at >= PREVIEW_IDLE_SECONDS
            ]
            for client_id in expired:
                self._clients.pop(client_id, None)
            return sum(1 for channel_ids, _touched_at in self._clients.values() if channel_id in channel_ids)

    def _is_idle(self, session: PreviewSession) -> bool:
        if self._active_client_count(session.channel_id) > 0:
            session.last_unsubscribed_at = time.monotonic()
            return False
        return time.monotonic() - session.last_unsubscribed_at >= PREVIEW_IDLE_SECONDS

    def _update_state(self, session: PreviewSession, state: str, error: str = "") -> None:
        with session.condition:
            session.state = state
            session.error = error
            session.condition.notify_all()

    def _publish_frame(
        self,
        session: PreviewSession,
        frame_bytes: bytes,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        with session.condition:
            session.latest_frame = frame_bytes
            session.sequence += 1
            session.state = "online"
            session.error = ""
            session.last_frame_at = timestamp_text()
            session.width = width
            session.height = height
            session.fps = round(fps, 1)
            session.condition.notify_all()

    def _prepare_frame(self, frame: Any) -> Tuple[bytes, int, int]:
        height, width = frame.shape[:2]
        if width > PREVIEW_MAX_WIDTH:
            scale = PREVIEW_MAX_WIDTH / float(width)
            frame = cv2.resize(
                frame,
                (PREVIEW_MAX_WIDTH, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = frame.shape[:2]
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), PREVIEW_JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("实时预览画面编码失败")
        return encoded.tobytes(), int(width), int(height)

    def _run_session(self, session: PreviewSession) -> None:
        frame_times = []
        try:
            while not session.stop_event.is_set():
                if self._is_idle(session):
                    break

                cap: Optional[Any] = None
                source_label = ""
                try:
                    channel = self.store.get_channel(session.channel_id, public=False)
                    if not channel["enabled"]:
                        self._update_state(session, "disabled", "视频通道已停用")
                        if session.stop_event.wait(1.0):
                            break
                        continue

                    self._update_state(
                        session,
                        "connecting" if session.latest_frame is None else "reconnecting",
                    )
                    source_label = channel_source(channel, self.make_camera_source)
                    if self.stream_manager is not None:
                        cap, _resolved = self.stream_manager.subscribe(session.channel_id)
                    else:
                        capture_source, _resolved = self.resolve_capture_source(source_label, 10.0)
                        cap = cv2.VideoCapture()
                        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000)
                        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
                        cap.open(capture_source)
                    if not cap.isOpened():
                        raise RuntimeError("无法打开视频源")

                    failures = 0
                    frame_interval = 1.0 / PREVIEW_TARGET_FPS
                    next_frame_at = time.monotonic()
                    while not session.stop_event.is_set():
                        if self._is_idle(session):
                            return
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            failures += 1
                            if failures >= 5:
                                raise RuntimeError("视频流连续取帧失败")
                            time.sleep(0.05)
                            continue

                        failures = 0
                        now = time.monotonic()
                        frame_times.append(now)
                        frame_times = [value for value in frame_times if now - value <= 2.0]
                        measured_fps = 0.0
                        if len(frame_times) >= 2:
                            duration = frame_times[-1] - frame_times[0]
                            if duration > 0:
                                measured_fps = (len(frame_times) - 1) / duration
                        frame_bytes, width, height = self._prepare_frame(frame)
                        self._publish_frame(session, frame_bytes, width, height, measured_fps)

                        next_frame_at = max(next_frame_at + frame_interval, time.monotonic())
                        wait_seconds = next_frame_at - time.monotonic()
                        if wait_seconds > 0 and session.stop_event.wait(wait_seconds):
                            return
                except ChannelNotFoundError:
                    self._update_state(session, "offline", "视频通道不存在")
                    break
                except Exception as exc:
                    session.reconnect_count += 1
                    message = safe_error_message(exc, [source_label])
                    self._update_state(session, "reconnecting", message)
                    if session.stop_event.wait(PREVIEW_RECONNECT_SECONDS):
                        break
                finally:
                    if cap is not None:
                        cap.release()
        finally:
            self._update_state(session, "idle")
            session.stop_event.set()
            with self._lock:
                if self._sessions.get(session.channel_id) is session:
                    self._sessions.pop(session.channel_id, None)

    def iter_mjpeg(self, channel_id: int, client_id: str = "") -> Iterator[bytes]:
        session = self._attach_client_channel(client_id, channel_id)
        last_sequence = -1
        while not session.stop_event.is_set():
            with session.condition:
                session.condition.wait_for(
                    lambda: session.sequence != last_sequence or session.stop_event.is_set(),
                    timeout=5.0,
                )
                if session.sequence == last_sequence or session.latest_frame is None:
                    continue
                frame = session.latest_frame
                last_sequence = session.sequence
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-cache\r\n\r\n"
                + frame
                + b"\r\n"
            )

    def statuses(self) -> Dict[int, Dict[str, Any]]:
        with self._lock:
            sessions = list(self._sessions.values())
        result: Dict[int, Dict[str, Any]] = {}
        for session in sessions:
            with session.condition:
                result[session.channel_id] = {
                    "state": session.state,
                    "error": session.error,
                    "last_frame_at": session.last_frame_at,
                    "fps": session.fps,
                    "width": session.width,
                    "height": session.height,
                    "reconnect_count": session.reconnect_count,
                    "viewers": self._active_client_count(session.channel_id),
                }
        return result

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop_event.set()
            with session.condition:
                session.condition.notify_all()
        with self._lock:
            self._clients.clear()


def register_live_preview_routes(app: Any, manager: LivePreviewManager, store: VideoChannelStore) -> None:
    from fastapi import HTTPException, Request
    from fastapi.responses import HTMLResponse, StreamingResponse
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @app.get("/preview", response_class=HTMLResponse, include_in_schema=False)
    def preview_page(request: Request) -> Any:
        return templates.TemplateResponse(
            request=request,
            name="preview.html",
            context={"page_title": "实时预览"},
        )

    @app.get("/api/live-preview/status")
    def preview_statuses() -> Dict[str, Any]:
        channels = [channel for channel in store.list_channels() if channel["enabled"]]
        statuses = manager.statuses()
        items = []
        for channel in channels:
            status = statuses.get(channel["id"], {"state": "idle"})
            items.append({"channel_id": channel["id"], **status})
        return {"items": items}

    @app.post("/api/live-preview/heartbeat")
    def preview_heartbeat(payload: Dict[str, Any]) -> Dict[str, bool]:
        client_id = str(payload.get("client_id", "")).strip()
        raw_channel_ids = payload.get("channel_ids", [])
        if not isinstance(raw_channel_ids, list):
            raise HTTPException(status_code=400, detail="channel_ids 必须是数组")
        try:
            channel_ids = {int(value) for value in raw_channel_ids}
            manager.touch_client(client_id, channel_ids)
        except (TypeError, ValueError, ChannelNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"success": True}

    @app.get("/api/live-preview/{channel_id}/stream")
    def live_preview_stream(channel_id: int, client_id: str = "") -> StreamingResponse:
        try:
            channel = store.get_channel(channel_id, public=False)
        except ChannelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if not channel["enabled"]:
            raise HTTPException(status_code=409, detail="视频通道已停用")
        return StreamingResponse(
            manager.iter_mjpeg(channel_id, client_id=client_id),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
