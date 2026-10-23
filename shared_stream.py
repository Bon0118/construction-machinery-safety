"""One upstream video connection per channel with fan-out frame subscriptions."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable, Deque, Dict, Optional, Tuple

import cv2

import config
from video_channels import (
    SOURCE_TYPE_CAMERA_INDEX_CODE,
    SOURCE_TYPE_LOCAL_FILE,
    STREAM_PROFILE_MAIN,
    VideoChannelStore,
    channel_source,
    safe_error_message,
)


LOGGER = logging.getLogger(__name__)


def timestamp_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class ChannelTrafficRecord:
    day: date = field(default_factory=date.today)
    media_bytes: int = 0
    active_seconds: float = 0.0

    def rollover(self) -> None:
        today = date.today()
        if self.day != today:
            self.day = today
            self.media_bytes = 0
            self.active_seconds = 0.0


@dataclass
class SharedStreamSession:
    channel_id: int
    condition: threading.Condition = field(default_factory=threading.Condition)
    stop_event: threading.Event = field(default_factory=threading.Event)
    restart_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    subscribers: int = 0
    sequence: int = 0
    frame: Any = None
    state: str = "connecting"
    error: str = ""
    source_label: str = ""
    last_frame_at: Optional[str] = None
    fps: float = 0.0
    width: int = 0
    height: int = 0
    bitrate_kbps: float = 0.0
    reconnect_count: int = 0
    packet_window: Deque[Tuple[float, int]] = field(default_factory=deque)
    last_accounted_at: float = field(default_factory=time.monotonic)


class SharedFrameCapture:
    """Small cv2.VideoCapture-compatible view over a shared channel session."""

    is_shared_stream = True

    def __init__(self, manager: "SharedStreamManager", session: SharedStreamSession) -> None:
        self._manager = manager
        self._session = session
        self._released = False
        self._sequence = -1

    def isOpened(self) -> bool:
        return not self._released and not self._session.stop_event.is_set()

    def read(self) -> Tuple[bool, Any]:
        if self._released:
            return False, None
        deadline = time.monotonic() + max(
            5.0,
            float(getattr(config, "VIDEO_READ_TIMEOUT_MILLISECONDS", 10000)) / 1000.0,
        )
        with self._session.condition:
            while not self._released and not self._session.stop_event.is_set():
                if self._session.frame is not None and self._session.sequence != self._sequence:
                    self._sequence = self._session.sequence
                    return True, self._session.frame.copy()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._session.condition.wait(timeout=min(1.0, remaining))
        return False, None

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return float(self._session.fps)
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._session.width)
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._session.height)
        if property_id == getattr(cv2, "CAP_PROP_BITRATE", -1):
            return float(self._session.bitrate_kbps)
        return 0.0

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._unsubscribe(self._session)


class SharedStreamManager:
    """Resolve, decode, measure, and distribute one stream for each channel."""

    def __init__(
        self,
        store: VideoChannelStore,
        resolve_capture_source: Callable[[str, Optional[float]], Tuple[Any, str]],
        make_camera_source: Callable[[str], str],
    ) -> None:
        self.store = store
        self.resolve_capture_source = resolve_capture_source
        self.make_camera_source = make_camera_source
        self._lock = threading.RLock()
        self._sessions: Dict[int, SharedStreamSession] = {}
        self._traffic: Dict[int, ChannelTrafficRecord] = {}
        self._profile_capture_locks: Dict[int, threading.Lock] = {}

    def subscribe(self, channel_id: int) -> Tuple[SharedFrameCapture, str]:
        channel_id = int(channel_id)
        channel = self.store.get_channel(channel_id, public=False)
        if not channel["enabled"]:
            raise RuntimeError("视频通道已停用")
        with self._lock:
            session = self._sessions.get(channel_id)
            if session is None or session.stop_event.is_set():
                session = SharedStreamSession(channel_id=channel_id)
                self._sessions[channel_id] = session
            session.subscribers += 1
            if session.thread is None or not session.thread.is_alive():
                session.thread = threading.Thread(
                    target=self._run_session,
                    args=(session,),
                    daemon=True,
                    name=f"shared-stream-{channel_id}",
                )
                session.thread.start()
        return SharedFrameCapture(self, session), f"shared-channel:{channel_id}"

    def _unsubscribe(self, session: SharedStreamSession) -> None:
        with self._lock:
            session.subscribers = max(0, session.subscribers - 1)
            should_stop = session.subscribers == 0
        if should_stop:
            session.stop_event.set()
            with session.condition:
                session.condition.notify_all()

    def restart_channel(self, channel_id: int) -> None:
        """Re-open the upstream while preserving every existing subscriber."""
        with self._lock:
            session = self._sessions.get(int(channel_id))
        if session is not None:
            session.restart_event.set()
            with self._lock:
                session.packet_window.clear()
                session.bitrate_kbps = 0.0
            self._set_state(session, "reconnecting")
            with session.condition:
                session.condition.notify_all()

    def stop_channel(self, channel_id: int) -> None:
        """Stop a channel permanently, for disable and delete operations."""
        with self._lock:
            session = self._sessions.get(int(channel_id))
        if session is not None:
            session.stop_event.set()
            session.restart_event.set()
            with session.condition:
                session.condition.notify_all()

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop_event.set()
            session.restart_event.set()
            with session.condition:
                session.condition.notify_all()
        for session in sessions:
            if session.thread is not None:
                session.thread.join(timeout=2.0)

    def capture_profile_frame(
        self,
        channel_id: int,
        stream_profile: str = STREAM_PROFILE_MAIN,
        timeout_seconds: float = 10.0,
    ) -> Any:
        """Read one on-demand frame from a selected profile and close it immediately."""
        channel_id = int(channel_id)
        channel = self.store.get_channel(channel_id, public=False)
        if channel.get("source_type") != SOURCE_TYPE_CAMERA_INDEX_CODE:
            raise RuntimeError("主码流二次确认仅支持 cameraIndexCode 通道")
        if not channel.get("enabled"):
            raise RuntimeError("视频通道已停用")
        profile_channel = dict(channel)
        profile_channel["stream_profile"] = str(stream_profile)
        source = channel_source(profile_channel, self.make_camera_source)
        capture_source, _resolved = self.resolve_capture_source(source, timeout_seconds)
        with self._lock:
            capture_lock = self._profile_capture_locks.setdefault(channel_id, threading.Lock())
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("主码流二次确认需要安装 PyAV") from exc

        with capture_lock:
            container = None
            try:
                options = {}
                if str(capture_source).lower().startswith("rtsp://"):
                    options["rtsp_transport"] = "tcp"
                timeout = max(1.0, float(timeout_seconds))
                container = av.open(
                    capture_source,
                    mode="r",
                    options=options,
                    timeout=(timeout, timeout),
                )
                video_stream = next((item for item in container.streams if item.type == "video"), None)
                if video_stream is None:
                    raise RuntimeError("主码流中没有可解码的视频轨道")
                for packet in container.demux():
                    size = max(0, int(packet.size or 0))
                    with self._lock:
                        record = self._traffic.setdefault(channel_id, ChannelTrafficRecord())
                        record.rollover()
                        record.media_bytes += size
                    if packet.stream.type != "video":
                        continue
                    for video_frame in packet.decode():
                        return video_frame.to_ndarray(format="bgr24")
                raise RuntimeError("主码流未返回可解码画面")
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:
                        pass

    def _record_packet(self, session: SharedStreamSession, packet_size: int, now: float) -> None:
        size = max(0, int(packet_size or 0))
        with self._lock:
            record = self._traffic.setdefault(session.channel_id, ChannelTrafficRecord())
            record.rollover()
            elapsed = max(0.0, now - session.last_accounted_at)
            record.active_seconds += min(elapsed, 2.0)
            record.media_bytes += size
            session.last_accounted_at = now
            session.packet_window.append((now, size))
            cutoff = now - 5.0
            while session.packet_window and session.packet_window[0][0] < cutoff:
                session.packet_window.popleft()
            if len(session.packet_window) >= 2:
                duration = session.packet_window[-1][0] - session.packet_window[0][0]
                if duration >= 0.25:
                    packet_bytes = sum(value for _timestamp, value in session.packet_window)
                    session.bitrate_kbps = packet_bytes * 8.0 / duration / 1000.0

    def _publish_frame(self, session: SharedStreamSession, frame: Any, fps: float) -> None:
        height, width = frame.shape[:2]
        with session.condition:
            session.frame = frame
            session.sequence += 1
            session.state = "online"
            session.error = ""
            session.last_frame_at = timestamp_text()
            session.width = int(width)
            session.height = int(height)
            if fps > 0:
                session.fps = float(fps)
            session.condition.notify_all()

    def _set_state(self, session: SharedStreamSession, state: str, error: str = "") -> None:
        with session.condition:
            session.state = state
            session.error = error
            session.condition.notify_all()

    def _run_session(self, session: SharedStreamSession) -> None:
        try:
            import av
        except ImportError as exc:
            self._set_state(session, "failed", "共享取流需要安装 PyAV")
            session.stop_event.set()
            LOGGER.exception("[SHARED_STREAM] PyAV is unavailable")
            return

        try:
            while not session.stop_event.is_set():
                # A restart requested before this iteration is now acknowledged;
                # the channel is fetched again below so the newest profile is used.
                session.restart_event.clear()
                container = None
                source = ""
                capture_source: Any = ""
                source_type = ""
                restart_requested = False
                try:
                    channel = self.store.get_channel(session.channel_id, public=False)
                    if not channel["enabled"]:
                        self._set_state(session, "disabled", "视频通道已停用")
                        break
                    source_type = str(channel["source_type"])
                    source = channel_source(channel, self.make_camera_source)
                    session.source_label = source
                    self._set_state(session, "connecting" if session.frame is None else "reconnecting")
                    capture_source, _resolved = self.resolve_capture_source(source, 10.0)
                    options = {}
                    if str(capture_source).lower().startswith("rtsp://"):
                        options["rtsp_transport"] = "tcp"
                    container = av.open(
                        capture_source,
                        mode="r",
                        options=options,
                        timeout=(
                            max(1.0, float(config.VIDEO_OPEN_TIMEOUT_MILLISECONDS) / 1000.0),
                            max(1.0, float(config.VIDEO_READ_TIMEOUT_MILLISECONDS) / 1000.0),
                        ),
                    )
                    video_stream = next((item for item in container.streams if item.type == "video"), None)
                    if video_stream is None:
                        raise RuntimeError("视频源中没有可解码的视频轨道")
                    average_rate = float(video_stream.average_rate) if video_stream.average_rate else 0.0
                    session.fps = average_rate
                    session.last_accounted_at = time.monotonic()
                    LOGGER.info(
                        "[SHARED_STREAM] channel=%s opened profile=%s subscribers=%s",
                        session.channel_id,
                        channel.get("stream_profile", "main"),
                        session.subscribers,
                    )
                    for packet in container.demux():
                        if session.stop_event.is_set():
                            break
                        if session.restart_event.is_set():
                            restart_requested = True
                            break
                        now = time.monotonic()
                        self._record_packet(session, int(packet.size or 0), now)
                        if packet.stream.type != "video":
                            continue
                        for video_frame in packet.decode():
                            if session.stop_event.is_set():
                                break
                            if session.restart_event.is_set():
                                restart_requested = True
                                break
                            self._publish_frame(
                                session,
                                video_frame.to_ndarray(format="bgr24"),
                                average_rate,
                            )
                        if restart_requested:
                            break
                    if restart_requested:
                        continue
                    if source_type == SOURCE_TYPE_LOCAL_FILE:
                        self._set_state(session, "completed")
                        break
                    if not session.stop_event.is_set():
                        raise RuntimeError("视频流已中断")
                except Exception as exc:
                    if session.stop_event.is_set():
                        break
                    if session.restart_event.is_set():
                        continue
                    session.reconnect_count += 1
                    message = safe_error_message(exc, [source, str(capture_source or "")])
                    self._set_state(session, "reconnecting", message)
                    LOGGER.warning(
                        "[SHARED_STREAM] channel=%s reconnect=%s error=%s",
                        session.channel_id,
                        session.reconnect_count,
                        message,
                    )
                    reconnect_deadline = time.monotonic() + float(
                        config.VIDEO_RECONNECT_INTERVAL_SECONDS
                    )
                    while not session.stop_event.is_set() and not session.restart_event.is_set():
                        remaining = reconnect_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        session.stop_event.wait(min(0.2, remaining))
                finally:
                    if container is not None:
                        try:
                            container.close()
                        except Exception:
                            pass
        finally:
            if session.state not in {"completed", "disabled", "failed"}:
                self._set_state(session, "idle")
            session.stop_event.set()
            with session.condition:
                session.condition.notify_all()
            with self._lock:
                if self._sessions.get(session.channel_id) is session:
                    self._sessions.pop(session.channel_id, None)
            LOGGER.info("[SHARED_STREAM] channel=%s closed", session.channel_id)

    def traffic_status(self, channel_id: int) -> Dict[str, Any]:
        channel_id = int(channel_id)
        with self._lock:
            session = self._sessions.get(channel_id)
            record = self._traffic.setdefault(channel_id, ChannelTrafficRecord())
            record.rollover()
            bitrate_kbps = float(session.bitrate_kbps) if session is not None else 0.0
            projected_bytes = bitrate_kbps * 1000.0 / 8.0 * 86400.0
            return {
                "state": session.state if session is not None else "idle",
                "shared_connection": session is not None and not session.stop_event.is_set(),
                "subscriber_count": int(session.subscribers) if session is not None else 0,
                "bitrate_kbps": round(bitrate_kbps, 1),
                "estimated_bytes_today": int(record.media_bytes),
                "estimated_gb_today": round(record.media_bytes / 1_000_000_000.0, 3),
                "projected_gb_per_day": round(projected_bytes / 1_000_000_000.0, 3),
                "active_seconds_today": round(record.active_seconds, 1),
                "last_frame_at": session.last_frame_at if session is not None else None,
                "fps": round(float(session.fps), 1) if session is not None else 0.0,
                "width": int(session.width) if session is not None else 0,
                "height": int(session.height) if session is not None else 0,
                "reconnect_count": int(session.reconnect_count) if session is not None else 0,
                "measurement": "encoded_media_payload_estimate",
            }

    def statuses(self) -> Dict[int, Dict[str, Any]]:
        channel_ids = {int(item["id"]) for item in self.store.list_channels()}
        with self._lock:
            channel_ids.update(self._traffic)
            channel_ids.update(self._sessions)
        return {channel_id: self.traffic_status(channel_id) for channel_id in channel_ids}
