"""Background execution for configured video detection tasks."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from video_channels import (
    SOURCE_TYPE_CAMERA_INDEX_CODE,
    STREAM_PROFILE_SUB,
    VideoChannelStore,
    channel_source,
)
from video_tasks import TASK_STATUS_RUNNING, TASK_STATUS_STOPPED, VideoTaskStore


LOGGER = logging.getLogger(__name__)


def timestamp_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass
class TaskRuntimeSession:
    task_id: int
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    state: str = "starting"
    error: str = ""
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    source: str = ""
    frames: int = 0
    alarm_count: int = 0
    stop_reason: str = ""
    preserve_desired_status: bool = False
    wake_event: threading.Event = field(default_factory=threading.Event)
    wake_until: float = 0.0
    wake_reason: str = ""
    confirmation_requests: int = 0
    confirmation_passed: int = 0
    confirmation_rejected: int = 0


class RuntimeGate:
    """Event-like token that also closes when a schedule window expires."""

    def __init__(self, stop_event: threading.Event, active: Callable[[], bool]) -> None:
        self.stop_event = stop_event
        self.active = active

    def is_set(self) -> bool:
        return self.stop_event.is_set() or not self.active()

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while not self.is_set():
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.stop_event.wait(min(0.2, remaining))
            else:
                self.stop_event.wait(0.2)
        return True


class TaskExecutionManager:
    """Run one detector loop per active task and expose runtime state."""

    def __init__(
        self,
        task_store: VideoTaskStore,
        channel_store: VideoChannelStore,
        detector_factory: Callable[[list[str]], Any],
        make_camera_source: Callable[[str], str],
        alarm_store: Any,
        stream_manager: Any = None,
        review_manager: Any = None,
    ) -> None:
        self.task_store = task_store
        self.channel_store = channel_store
        self.detector_factory = detector_factory
        self.make_camera_source = make_camera_source
        self.alarm_store = alarm_store
        self.stream_manager = stream_manager
        self.review_manager = review_manager
        self._lock = threading.RLock()
        self._sessions: Dict[int, TaskRuntimeSession] = {}

    def _session(self, task_id: int) -> Optional[TaskRuntimeSession]:
        with self._lock:
            return self._sessions.get(int(task_id))

    @staticmethod
    def _is_alive(session: Optional[TaskRuntimeSession]) -> bool:
        return bool(session and session.thread and session.thread.is_alive())

    def is_active(self, task_id: int) -> bool:
        return self._is_alive(self._session(task_id))

    def enrich(self, task: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(task)
        session = self._session(int(task["id"]))
        if session is None:
            data.update(
                runtime_state=("starting" if task["status"] == TASK_STATUS_RUNNING else "stopped"),
                runtime_error="",
                runtime_started_at=None,
                runtime_stopped_at=None,
                runtime_frames=0,
                runtime_alarm_count=0,
                runtime_stop_reason="",
                runtime_wake_until=None,
                runtime_wake_reason="",
                runtime_confirmation_requests=0,
                runtime_confirmation_passed=0,
                runtime_confirmation_rejected=0,
            )
            return data
        data.update(
            runtime_state=session.state,
            runtime_error=session.error,
            runtime_started_at=session.started_at,
            runtime_stopped_at=session.stopped_at,
            runtime_frames=session.frames,
            runtime_alarm_count=session.alarm_count,
            runtime_stop_reason=session.stop_reason,
            runtime_wake_until=(
                datetime.fromtimestamp(
                    time.time() + max(0.0, session.wake_until - time.monotonic())
                ).astimezone().isoformat(timespec="seconds")
                if session.wake_until > time.monotonic()
                else None
            ),
            runtime_wake_reason=session.wake_reason,
            runtime_confirmation_requests=session.confirmation_requests,
            runtime_confirmation_passed=session.confirmation_passed,
            runtime_confirmation_rejected=session.confirmation_rejected,
        )
        return data

    def list_tasks(self) -> list[Dict[str, Any]]:
        return [self.enrich(task) for task in self.task_store.list_tasks()]

    def get_task(self, task_id: int) -> Dict[str, Any]:
        return self.enrich(self.task_store.get_task(task_id))

    def start_task(self, task_id: int) -> Dict[str, Any]:
        task_id = int(task_id)
        with self._lock:
            existing = self._sessions.get(task_id)
            if self._is_alive(existing):
                return self.enrich(self.task_store.get_task(task_id))

            task = self.task_store.set_status(task_id, TASK_STATUS_RUNNING)
            session = TaskRuntimeSession(task_id=task_id)
            thread = threading.Thread(
                target=self._run_task,
                args=(session,),
                daemon=True,
                name=f"video-task-{task_id}",
            )
            session.thread = thread
            self._sessions[task_id] = session
            thread.start()
        return self.enrich(task)

    def stop_task(self, task_id: int, wait_seconds: float = 2.0) -> Dict[str, Any]:
        task_id = int(task_id)
        task = self.task_store.get_task(task_id)
        session = self._session(task_id)
        if session is not None and self._is_alive(session):
            session.state = "stopping"
            session.stop_event.set()
            session.wake_event.set()
            if session.thread is not None:
                session.thread.join(timeout=max(0.0, float(wait_seconds)))
        if task["status"] != TASK_STATUS_STOPPED:
            task = self.task_store.set_status(task_id, TASK_STATUS_STOPPED)
        return self.enrich(task)

    def start_saved_tasks(self) -> None:
        for task in self.task_store.list_tasks():
            if task["status"] != TASK_STATUS_RUNNING:
                continue
            try:
                self.start_task(int(task["id"]))
            except Exception as exc:
                self._record_start_failure(int(task["id"]), exc)

    def stop_all(self, preserve_desired_status: bool = True) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.preserve_desired_status = bool(preserve_desired_status)
            session.state = "stopping"
            session.stop_event.set()
            session.wake_event.set()
        for session in sessions:
            if session.thread is not None:
                session.thread.join(timeout=2.0)

    def _record_start_failure(self, task_id: int, exc: BaseException) -> None:
        session = TaskRuntimeSession(task_id=task_id, state="failed", error=str(exc), stopped_at=timestamp_text())
        with self._lock:
            self._sessions[task_id] = session
        try:
            self.task_store.set_status(task_id, TASK_STATUS_STOPPED)
        except Exception:
            pass

    def _on_alarm(self, record: Any) -> None:
        self.alarm_store.sync_from_output()
        if self.review_manager is not None:
            self.review_manager.enqueue(str(getattr(record, "event_id", "") or ""))

    @staticmethod
    def _schedule_contains(policy: Dict[str, Any], now: Optional[datetime] = None) -> bool:
        if not bool(policy.get("schedule_enabled")):
            return True
        current = (now or datetime.now().astimezone()).time().replace(tzinfo=None)
        start = datetime.strptime(str(policy.get("schedule_start", "00:00")), "%H:%M").time()
        end = datetime.strptime(str(policy.get("schedule_end", "23:59")), "%H:%M").time()
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _policy_active(
        self,
        policy: Dict[str, Any],
        session: TaskRuntimeSession,
    ) -> tuple[bool, bool]:
        scheduled = self._schedule_contains(policy)
        event_override = bool(
            policy.get("event_wakeup_enabled") and session.wake_until > time.monotonic()
        )
        return scheduled or event_override, event_override and not scheduled

    def wake_from_camera_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Wake armed tasks by channel id, CameraID, or cameraIndexCode."""
        channel_id_value = payload.get("channel_id")
        camera_id = str(payload.get("camera_id") or "").strip()
        camera_index_code = str(payload.get("camera_index_code") or "").strip()
        if channel_id_value in (None, "") and not camera_id and not camera_index_code:
            raise ValueError("channel_id、camera_id、camera_index_code 至少填写一个")
        matched_channel_ids = set()
        for channel in self.channel_store.list_channels():
            if channel_id_value not in (None, "") and int(channel["id"]) == int(channel_id_value):
                matched_channel_ids.add(int(channel["id"]))
            if camera_id and str(channel.get("camera_id") or "") == camera_id:
                matched_channel_ids.add(int(channel["id"]))
            if camera_index_code and str(channel.get("camera_index_code") or "") == camera_index_code:
                matched_channel_ids.add(int(channel["id"]))
        event_type = str(payload.get("event_type") or "camera_event").strip()[:100]
        awakened = []
        for task in self.task_store.list_tasks():
            if task.get("channel_id") not in matched_channel_ids:
                continue
            if task.get("status") != TASK_STATUS_RUNNING:
                continue
            policy = dict(task.get("runtime_policy") or {})
            if not policy.get("event_wakeup_enabled"):
                continue
            session = self._session(int(task["id"]))
            if not self._is_alive(session):
                self.start_task(int(task["id"]))
                session = self._session(int(task["id"]))
            if session is None:
                continue
            duration = int(policy.get("event_wakeup_seconds", 300))
            session.wake_until = max(session.wake_until, time.monotonic() + duration)
            session.wake_reason = event_type
            session.wake_event.set()
            awakened.append({"task_id": int(task["id"]), "task_code": task["task_code"], "seconds": duration})
        return {
            "success": True,
            "matched_channel_ids": sorted(matched_channel_ids),
            "awakened_tasks": awakened,
            "awakened_count": len(awakened),
        }

    def _confirmation_callback(
        self,
        session: TaskRuntimeSession,
        channel: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Optional[Callable[..., Dict[str, Any]]]:
        if not bool(policy.get("main_stream_confirmation_enabled")):
            return None
        if channel.get("source_type") != SOURCE_TYPE_CAMERA_INDEX_CODE:
            return None
        if channel.get("stream_profile") != STREAM_PROFILE_SUB:
            return None
        if self.stream_manager is None:
            return None
        area_threshold = float(policy.get("small_target_area_ratio", 0.02))
        timeout = float(policy.get("confirmation_timeout_seconds", 10))

        def confirm(**candidate: Any) -> Dict[str, Any]:
            frame = candidate.get("frame")
            boxes = candidate.get("candidate_boxes") or []
            verify = candidate.get("verify")
            if frame is None or not callable(verify):
                return {"confirmed": True, "frame": frame}
            height, width = frame.shape[:2]
            frame_area = max(1.0, float(width * height))
            is_small = any(
                max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))
                / frame_area
                <= area_threshold
                for box in boxes
                if len(box) >= 4
            )
            if not is_small:
                return {"confirmed": True, "frame": frame}
            session.confirmation_requests += 1
            try:
                main_frame = self.stream_manager.capture_profile_frame(
                    int(channel["id"]), "main", timeout
                )
                result = dict(verify(main_frame) or {})
                confirmed = bool(result.get("confirmed"))
                if confirmed:
                    session.confirmation_passed += 1
                else:
                    session.confirmation_rejected += 1
                result.setdefault("frame", main_frame)
                return result
            except Exception as exc:
                session.confirmation_rejected += 1
                LOGGER.warning(
                    "[MAIN_STREAM_CONFIRM] task=%s channel=%s failed=%s",
                    session.task_id,
                    channel.get("id"),
                    exc,
                )
                return {"confirmed": False, "error": str(exc)}

        return confirm

    def _run_task(self, session: TaskRuntimeSession) -> None:
        try:
            task = self.task_store.get_task(session.task_id)
            if task["channel_id"] is None:
                raise RuntimeError("视频任务未关联有效的视频通道")
            channel = self.channel_store.get_channel(int(task["channel_id"]), public=False)
            if not channel["enabled"]:
                raise RuntimeError("视频通道已停用")
            if not task["algorithm_enabled"]:
                raise RuntimeError("视频任务未启用检测算法")

            session.source = channel_source(channel, self.make_camera_source)
            session.started_at = timestamp_text()
            algorithms = task.get("algorithms") or []
            algorithm_codes = [str(item.get("code") or "").strip() for item in algorithms if item.get("code")]
            if not algorithm_codes:
                raise RuntimeError("视频任务未选择检测算法")
            primary_algorithm = algorithms[0]
            policy = dict(task.get("runtime_policy") or {})
            while not session.stop_event.is_set():
                active, event_override = self._policy_active(policy, session)
                if not active:
                    session.state = "waiting_event" if policy.get("event_wakeup_enabled") else "scheduled_idle"
                    session.stop_reason = "outside_detection_schedule"
                    session.wake_event.wait(timeout=1.0)
                    session.wake_event.clear()
                    continue

                detector = self.detector_factory(algorithm_codes)
                config_snapshot = getattr(detector, "config_snapshot", None)
                algorithm_config = dict(config_snapshot()) if callable(config_snapshot) else {}
                session.state = "event_running" if event_override else "running"
                gate = RuntimeGate(
                    session.stop_event,
                    lambda policy=policy, session=session: self._policy_active(policy, session)[0],
                )
                process_kwargs = {
                    "show": False,
                    "save_video": False,
                    "stop_event": gate,
                    "event_context": {
                        "task_id": int(task["id"]),
                        "task_code": task["task_code"],
                        "channel_id": int(channel["id"]),
                        "channel_name": channel["name"],
                        "camera_id": channel.get("camera_id", ""),
                        "report_url": task.get("report_url", ""),
                        "algorithm_codes": algorithm_codes,
                        "algorithm_code": str(primary_algorithm.get("code") or ""),
                        "algorithm_name": str(primary_algorithm.get("name") or ""),
                        "algorithm_config": algorithm_config,
                    },
                    "event_callback": self._on_alarm,
                    "algorithm_alarm_intervals": dict(
                        task.get("algorithm_alarm_intervals") or {}
                    ),
                    "algorithm_parameters": dict(task.get("algorithm_parameters") or {}),
                }
                confirmation = self._confirmation_callback(session, channel, policy)
                if confirmation is not None:
                    process_kwargs["alarm_confirmation"] = confirmation
                if self.stream_manager is not None:
                    channel_id = int(channel["id"])
                    process_kwargs["capture_factory"] = (
                        lambda channel_id=channel_id: self.stream_manager.subscribe(channel_id)
                    )
                result = detector.process_video(session.source, **process_kwargs)
                session.frames += int(result.get("frames", 0) or 0)
                session.alarm_count += int(result.get("alarm_count", 0) or 0)
                session.stop_reason = str(result.get("stop_reason", "") or "")
                if session.stop_event.is_set():
                    break
                still_active, _event_override = self._policy_active(policy, session)
                if not still_active:
                    continue
                session.state = "completed"
                break
            if session.stop_event.is_set():
                session.state = "stopped"
        except Exception as exc:
            session.state = "failed"
            session.error = str(exc)
        finally:
            session.stopped_at = timestamp_text()
            if not session.preserve_desired_status:
                try:
                    self.task_store.set_status(session.task_id, TASK_STATUS_STOPPED)
                except Exception:
                    pass
