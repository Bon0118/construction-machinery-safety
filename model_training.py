"""Visual YOLOv5 fine-tuning jobs for human-reviewed safety-helmet data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import threading
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import config
import yaml
from starlette.requests import Request
from video_channels import DEFAULT_DATABASE_PATH, TEMPLATES_DIR


CLASS_IDS = {"person": 0, "head": 1, "helmet": 2}
ACTIVE_JOB_STATUSES = {"starting", "running", "stopping"}


class TrainingValidationError(ValueError):
    pass


class TrainingNotFoundError(LookupError):
    pass


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class ModelTrainingManager:
    """Prepare reviewed crops and supervise one isolated YOLOv5 process."""

    def __init__(
        self,
        database_path: Path = DEFAULT_DATABASE_PATH,
        training_root: Optional[Path] = None,
        active_task_provider: Optional[Callable[[], List[Dict[str, Any]]]] = None,
    ) -> None:
        self.database_path = Path(database_path)
        self.training_root = Path(training_root or config.MODEL_TRAINING_ROOT)
        self.datasets_root = self.training_root / "safety_helmet" / "datasets"
        self.runs_root = self.training_root / "safety_helmet" / "runs"
        self.logs_root = self.training_root / "logs"
        self.active_task_provider = active_task_provider
        self._lock = threading.RLock()
        self._processes: Dict[int, subprocess.Popen[Any]] = {}
        self._environment_cache: Optional[Tuple[float, Dict[str, Any]]] = None
        for path in (self.datasets_root, self.runs_root, self.logs_root, Path(config.MODEL_TRAINING_YOLO_CONFIG_DIR)):
            path.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database_path), timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS model_training_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm_code TEXT NOT NULL,
                    run_name TEXT NOT NULL UNIQUE,
                    dataset_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    command_json TEXT NOT NULL DEFAULT '[]',
                    pid INTEGER,
                    output_dir TEXT NOT NULL,
                    log_path TEXT NOT NULL,
                    best_model_path TEXT NOT NULL DEFAULT '',
                    metrics_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    ended_at TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON model_training_jobs(status, created_at)"
            )
            connection.execute(
                """
                UPDATE model_training_jobs
                SET status = 'interrupted', ended_at = ?, error = CASE WHEN error = '' THEN '平台重启导致训练进程状态中断' ELSE error END
                WHERE status IN ('starting', 'running', 'stopping')
                """,
                (now_text(),),
            )

    @staticmethod
    def _json(value: Any, fallback: Any) -> Any:
        if isinstance(value, type(fallback)):
            return value
        try:
            parsed = json.loads(value or "")
        except (TypeError, ValueError, json.JSONDecodeError):
            return fallback
        return parsed if isinstance(parsed, type(fallback)) else fallback

    @staticmethod
    def _box(value: Any) -> Optional[List[float]]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return None
        try:
            box = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return box if box[2] > box[0] and box[3] > box[1] else None

    @staticmethod
    def _overlap_smaller(first: List[float], second: List[float]) -> float:
        intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
            0.0, min(first[3], second[3]) - max(first[1], second[1])
        )
        smaller = min((first[2] - first[0]) * (first[3] - first[1]), (second[2] - second[0]) * (second[3] - second[1]))
        return intersection / smaller if smaller > 0 else 0.0

    @staticmethod
    def _center_in(box: List[float], container: List[float]) -> bool:
        center_x, center_y = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        return container[0] <= center_x <= container[2] and container[1] <= center_y <= container[3]

    def _runner_environment(self) -> Dict[str, str]:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "KMP_DUPLICATE_LIB_OK": "TRUE",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "YOLO_CONFIG_DIR": str(config.MODEL_TRAINING_YOLO_CONFIG_DIR),
                "PLATFORM_YOLOV5_REPO": str(config.MODEL_TRAINING_YOLOV5_REPO),
            }
        )
        return environment

    def environment_status(self, force: bool = False) -> Dict[str, Any]:
        import time

        if not force and self._environment_cache and time.monotonic() - self._environment_cache[0] < 60:
            return dict(self._environment_cache[1])
        checks = {
            "python_exists": Path(config.MODEL_TRAINING_PYTHON).is_file(),
            "yolov5_train_exists": (Path(config.MODEL_TRAINING_YOLOV5_REPO) / "train.py").is_file(),
            "runner_exists": Path(config.MODEL_TRAINING_RUNNER).is_file(),
            "weights_exists": Path(config.SAFETY_HELMET_TRAINING_WEIGHTS).is_file(),
        }
        result: Dict[str, Any] = {
            "ready": False,
            "checks": checks,
            "python": str(config.MODEL_TRAINING_PYTHON),
            "repo": str(config.MODEL_TRAINING_YOLOV5_REPO),
            "weights": str(config.SAFETY_HELMET_TRAINING_WEIGHTS),
            "error": "",
        }
        if not all(checks.values()):
            result["error"] = "训练环境路径不完整"
        else:
            command = [
                str(config.MODEL_TRAINING_PYTHON),
                str(config.MODEL_TRAINING_RUNNER),
                "--platform-preflight",
                "--weights",
                str(config.SAFETY_HELMET_TRAINING_WEIGHTS),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(config.MODEL_TRAINING_YOLOV5_REPO),
                    env=self._runner_environment(),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    check=False,
                )
                line = next((item for item in completed.stdout.splitlines() if item.startswith("PLATFORM_PREFLIGHT=")), "")
                if completed.returncode == 0 and line:
                    detail = json.loads(line.split("=", 1)[1])
                    result.update(detail)
                    result["ready"] = bool(detail.get("cuda_available")) and detail.get("weight_names") == ["person", "head", "helmet"]
                    if not result["ready"]:
                        result["error"] = "需要 CUDA，并且初始模型类别必须为 person/head/helmet"
                else:
                    result["error"] = (completed.stderr or completed.stdout or "训练环境检查失败")[-2000:]
            except Exception as exc:
                result["error"] = str(exc)
        self._environment_cache = (time.monotonic(), dict(result))
        return result

    def review_overview(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT workflow_status, decision, annotations
                FROM manual_reviews
                """
            ).fetchall()
        completed = [row for row in rows if row["workflow_status"] == "completed"]
        eligible = [row for row in completed if row["decision"] in {"confirmed", "corrected", "no_target"}]
        class_counts = {name: 0 for name in CLASS_IDS}
        for row in eligible:
            for item in self._json(row["annotations"], []):
                class_name = str(item.get("class_name") or "")
                if class_name in class_counts:
                    class_counts[class_name] += 1
        return {
            "total_reviews": len(rows),
            "completed": len(completed),
            "eligible_events": len(eligible),
            "class_counts": class_counts,
            "recommended_events": int(config.MODEL_TRAINING_RECOMMENDED_SAMPLES),
            "can_prepare": len(eligible) >= 2,
        }

    def _training_records(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT mr.id, mr.alarm_id, mr.decision, mr.annotations, mr.source_annotation_path,
                       mr.version, mr.completed_at, va.event_id
                FROM manual_reviews mr
                JOIN video_alarms va ON va.id = mr.alarm_id
                WHERE mr.workflow_status = 'completed'
                  AND mr.decision IN ('confirmed', 'corrected', 'no_target')
                ORDER BY mr.completed_at, mr.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _mapped_crop_labels(self, record: Dict[str, Any], crop_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if record["decision"] == "no_target":
            return []
        annotations = self._json(record["annotations"], [])
        transform = crop_data.get("crop_transform") if isinstance(crop_data.get("crop_transform"), dict) else {}
        crop_box = self._box(transform.get("crop_box_original"))
        person_box = self._box(transform.get("source_person_box_original"))
        output_size = transform.get("output_size")
        if crop_box is None or person_box is None or not isinstance(output_size, (list, tuple)) or len(output_size) != 2:
            return []
        people = [item for item in annotations if item.get("class_name") == "person" and self._box(item.get("box"))]
        matches = sorted(
            ((self._overlap_smaller(self._box(item["box"]) or [0, 0, 0, 0], person_box), item) for item in people),
            key=lambda pair: pair[0],
            reverse=True,
        )
        if not matches or matches[0][0] < 0.35:
            return []
        selected_person = matches[0][1]
        selected_person_box = self._box(selected_person["box"]) or person_box
        selected = [selected_person]
        for item in annotations:
            if item.get("class_name") not in {"head", "helmet"}:
                continue
            box = self._box(item.get("box"))
            if box and (self._center_in(box, selected_person_box) or self._overlap_smaller(box, selected_person_box) >= 0.25):
                selected.append(item)
        width, height = float(output_size[0]), float(output_size[1])
        scale_x, scale_y = width / (crop_box[2] - crop_box[0]), height / (crop_box[3] - crop_box[1])
        mapped: List[Dict[str, Any]] = []
        for item in selected:
            box = self._box(item.get("box"))
            if box is None:
                continue
            transformed = [
                max(0.0, min(width, (box[0] - crop_box[0]) * scale_x)),
                max(0.0, min(height, (box[1] - crop_box[1]) * scale_y)),
                max(0.0, min(width, (box[2] - crop_box[0]) * scale_x)),
                max(0.0, min(height, (box[3] - crop_box[1]) * scale_y)),
            ]
            if transformed[2] - transformed[0] < 2 or transformed[3] - transformed[1] < 2:
                continue
            if item.get("class_name") in {"head", "helmet"} and min(
                transformed[2] - transformed[0], transformed[3] - transformed[1]
            ) < 20:
                continue
            mapped.append({"class_name": item["class_name"], "box": transformed, "width": width, "height": height})
        return mapped

    @staticmethod
    def _yolo_lines(labels: List[Dict[str, Any]]) -> List[str]:
        lines = []
        for item in labels:
            x1, y1, x2, y2 = item["box"]
            width, height = item["width"], item["height"]
            lines.append(
                f"{CLASS_IDS[item['class_name']]} {((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
                f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
            )
        return lines

    def prepare_dataset(self) -> Dict[str, Any]:
        records = self._training_records()
        if len(records) < 2:
            raise TrainingValidationError("至少需要完成人工复核的 2 个有效事件，才能划分训练集和验证集")
        entries: List[Dict[str, Any]] = []
        skipped = []
        for record in records:
            teacher_path = Path(str(record.get("source_annotation_path") or ""))
            try:
                teacher_data = json.loads(teacher_path.read_text(encoding="utf-8"))
            except Exception:
                skipped.append({"event_id": record["event_id"], "reason": "教师标注文件不可读取"})
                continue
            run_dir = teacher_path.parent.parent
            crop_ids = teacher_data.get("crop_ids") if isinstance(teacher_data.get("crop_ids"), list) else []
            for crop_id_raw in crop_ids:
                crop_id = str(crop_id_raw or "")
                if not crop_id or Path(crop_id).name != crop_id:
                    continue
                image_path = run_dir / "crops" / "images" / f"{crop_id}.jpg"
                annotation_path = run_dir / "crops" / "annotations" / f"{crop_id}.json"
                try:
                    crop_data = json.loads(annotation_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                labels = self._mapped_crop_labels(record, crop_data)
                if record["decision"] != "no_target" and not any(item["class_name"] == "person" for item in labels):
                    continue
                if image_path.is_file():
                    entries.append({"event_id": record["event_id"], "crop_id": crop_id, "image": image_path, "labels": labels})
        events = sorted({entry["event_id"] for entry in entries})
        if len(events) < 2:
            raise TrainingValidationError("可映射到头肩裁剪的已复核事件不足 2 个，请先继续人工复核")
        ordered_events = sorted(events, key=lambda value: hashlib.sha1(value.encode("utf-8")).hexdigest())
        validation_count = max(1, round(len(ordered_events) * 0.2))
        validation_events = set(ordered_events[:validation_count])
        dataset_id = "safety_helmet_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        dataset_dir = self.datasets_root / dataset_id
        class_counts = {name: 0 for name in CLASS_IDS}
        split_counts = {"train": 0, "val": 0}
        manifest_entries = []
        for entry in entries:
            split = "val" if entry["event_id"] in validation_events else "train"
            image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)
            destination_image = image_dir / f"{entry['crop_id']}.jpg"
            shutil.copy2(entry["image"], destination_image)
            lines = self._yolo_lines(entry["labels"])
            (label_dir / f"{entry['crop_id']}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            for item in entry["labels"]:
                class_counts[item["class_name"]] += 1
            split_counts[split] += 1
            manifest_entries.append(
                {"event_id": entry["event_id"], "crop_id": entry["crop_id"], "split": split, "label_count": len(lines)}
            )
        data_yaml = dataset_dir / "data.yaml"
        yaml_path = dataset_dir.as_posix()
        data_yaml.write_text(
            f"path: {yaml_path}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: head\n  2: helmet\n",
            encoding="utf-8",
        )
        training_ready = all(class_counts[name] > 0 for name in CLASS_IDS) and all(split_counts[name] > 0 for name in split_counts)
        warnings = []
        if len(events) < int(config.MODEL_TRAINING_RECOMMENDED_SAMPLES):
            warnings.append(f"当前仅 {len(events)} 个事件，建议累计到 {config.MODEL_TRAINING_RECOMMENDED_SAMPLES} 个左右再正式调优")
        if not training_ready:
            warnings.append("数据集必须同时包含 person、head、helmet，且训练集与验证集均不能为空")
        manifest = {
            "dataset_id": dataset_id,
            "algorithm_code": "safety_helmet",
            "created_at": now_text(),
            "source": "completed_manual_reviews",
            "event_count": len(events),
            "image_count": len(entries),
            "split_counts": split_counts,
            "class_counts": class_counts,
            "training_ready": training_ready,
            "recommended_events": int(config.MODEL_TRAINING_RECOMMENDED_SAMPLES),
            "warnings": warnings,
            "skipped": skipped,
            "data_yaml": str(data_yaml),
            "entries": manifest_entries,
        }
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest

    @staticmethod
    def _inside(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    @staticmethod
    def _dataset_names(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value]
        if isinstance(value, dict):
            try:
                return [str(value[key]).strip() for key in sorted(value, key=lambda item: int(item))]
            except (TypeError, ValueError):
                return []
        return []

    def _resolve_dataset_images(self, value: Any, yaml_dir: Path, dataset_base: Path, source_root: Path) -> List[Path]:
        values = value if isinstance(value, list) else [value]
        images: List[Path] = []
        extensions = {item.lower() for item in config.IMAGE_EXTENSIONS}
        for raw in values:
            text = str(raw or "").strip()
            if not text:
                continue
            candidate = Path(text)
            if candidate.is_absolute():
                raise TrainingValidationError("data.yaml 不能引用压缩包外的绝对路径")
            path = (dataset_base / candidate).resolve()
            if not self._inside(path, source_root):
                raise TrainingValidationError("data.yaml 中的数据路径超出压缩包范围")
            if path.is_dir():
                images.extend(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in extensions)
            elif path.is_file() and path.suffix.lower() == ".txt":
                for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
                    item_text = line.strip()
                    if not item_text:
                        continue
                    item_path = Path(item_text)
                    if item_path.is_absolute():
                        raise TrainingValidationError("图片清单不能引用压缩包外的绝对路径")
                    item = ((path.parent if item_text.startswith(".") else yaml_dir) / item_path).resolve()
                    if not self._inside(item, source_root):
                        raise TrainingValidationError("图片清单路径超出压缩包范围")
                    if item.is_file() and item.suffix.lower() in extensions:
                        images.append(item)
            elif path.is_file() and path.suffix.lower() in extensions:
                images.append(path)
        return sorted(set(images), key=lambda item: item.as_posix().lower())

    @staticmethod
    def _source_label_path(image_path: Path) -> Optional[Path]:
        parts = list(image_path.parts)
        image_indexes = [index for index, item in enumerate(parts) if item.lower() == "images"]
        if not image_indexes:
            return None
        parts[image_indexes[-1]] = "labels"
        return Path(*parts).with_suffix(".txt")

    @staticmethod
    def _validated_label_lines(label_path: Optional[Path], split: str, image_name: str) -> Tuple[List[str], Dict[str, int]]:
        counts = {name: 0 for name in CLASS_IDS}
        if label_path is None or not label_path.is_file():
            return [], counts
        result: List[str] = []
        for line_number, raw in enumerate(label_path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 5:
                raise TrainingValidationError(f"{split}/{image_name} 的第 {line_number} 行不是 YOLO 检测框格式")
            try:
                class_id = int(fields[0])
                values = [float(item) for item in fields[1:]]
            except (TypeError, ValueError) as exc:
                raise TrainingValidationError(f"{split}/{image_name} 的第 {line_number} 行包含无效数值") from exc
            if class_id not in CLASS_IDS.values():
                raise TrainingValidationError(f"{split}/{image_name} 包含超出 person/head/helmet 范围的类别编号 {class_id}")
            if any(value < 0.0 or value > 1.0 for value in values) or values[2] <= 0.0 or values[3] <= 0.0:
                raise TrainingValidationError(f"{split}/{image_name} 的第 {line_number} 行坐标不在 0~1 范围内")
            class_name = next(name for name, value in CLASS_IDS.items() if value == class_id)
            counts[class_name] += 1
            result.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
        return result, counts

    def import_dataset_archive(self, archive_path: Path, original_name: str = "dataset.zip") -> Dict[str, Any]:
        archive_path = Path(archive_path)
        if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
            raise TrainingValidationError("请选择有效的 ZIP 数据集压缩包")
        upload_id = uuid.uuid4().hex
        work_dir = self.datasets_root / f".upload_{upload_id}"
        source_root = work_dir / "source"
        dataset_dir: Optional[Path] = None
        try:
            source_root.mkdir(parents=True, exist_ok=False)
            with zipfile.ZipFile(archive_path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if not members:
                    raise TrainingValidationError("ZIP 压缩包为空")
                extracted_bytes = sum(max(0, int(item.file_size)) for item in members)
                if extracted_bytes > int(config.MODEL_TRAINING_MAX_EXTRACTED_BYTES):
                    raise TrainingValidationError("数据集解压后体积超过平台限制")
                for item in members:
                    destination = (source_root / item.filename).resolve()
                    if not self._inside(destination, source_root):
                        raise TrainingValidationError("ZIP 中包含不安全的越界路径")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            yaml_candidates = sorted(source_root.rglob("data.yaml"))
            if not yaml_candidates:
                yaml_candidates = sorted(source_root.rglob("*.yaml")) + sorted(source_root.rglob("*.yml"))
            if not yaml_candidates:
                raise TrainingValidationError("压缩包中未找到 data.yaml")
            yaml_path = yaml_candidates[0]
            try:
                definition = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
            except Exception as exc:
                raise TrainingValidationError("data.yaml 无法解析") from exc
            if not isinstance(definition, dict):
                raise TrainingValidationError("data.yaml 内容格式不正确")
            names = self._dataset_names(definition.get("names"))
            if names != list(CLASS_IDS):
                raise TrainingValidationError("类别必须按 0:person、1:head、2:helmet 的顺序配置")
            yaml_dir = yaml_path.parent.resolve()
            raw_base = Path(str(definition.get("path") or "."))
            if raw_base.is_absolute():
                raise TrainingValidationError("data.yaml 的 path 必须使用压缩包内相对路径")
            dataset_base = (yaml_dir / raw_base).resolve()
            if not self._inside(dataset_base, source_root):
                raise TrainingValidationError("data.yaml 的 path 超出压缩包范围")
            split_images = {
                split: self._resolve_dataset_images(definition.get(split), yaml_dir, dataset_base, source_root)
                for split in ("train", "val")
            }
            if not split_images["train"] or not split_images["val"]:
                raise TrainingValidationError("数据集必须同时提供非空的 train 和 val 图片")
            all_source_images = split_images["train"] + split_images["val"]
            if len(set(all_source_images)) != len(all_source_images):
                raise TrainingValidationError("同一图片不能同时出现在 train 和 val 中")
            dataset_id = "safety_helmet_upload_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dataset_dir = self.datasets_root / dataset_id
            class_counts = {name: 0 for name in CLASS_IDS}
            split_counts = {split: len(items) for split, items in split_images.items()}
            labeled_images = 0
            negative_images = 0
            for split, images in split_images.items():
                image_dir, label_dir = dataset_dir / "images" / split, dataset_dir / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                for index, source_image in enumerate(images, 1):
                    digest = hashlib.sha1(source_image.as_posix().encode("utf-8")).hexdigest()[:10]
                    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", source_image.stem).strip("_")[:60] or f"image_{index:06d}"
                    file_name = f"{index:06d}_{stem}_{digest}{source_image.suffix.lower()}"
                    label_lines, item_counts = self._validated_label_lines(
                        self._source_label_path(source_image), split, source_image.name
                    )
                    shutil.copy2(source_image, image_dir / file_name)
                    (label_dir / Path(file_name).with_suffix(".txt").name).write_text(
                        "\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8"
                    )
                    if label_lines:
                        labeled_images += 1
                    else:
                        negative_images += 1
                    for class_name, count in item_counts.items():
                        class_counts[class_name] += count
            data_yaml = dataset_dir / "data.yaml"
            data_yaml.write_text(
                f"path: {dataset_dir.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: head\n  2: helmet\n",
                encoding="utf-8",
            )
            training_ready = all(class_counts[name] > 0 for name in CLASS_IDS)
            warnings: List[str] = []
            if sum(split_counts.values()) < int(config.MODEL_TRAINING_RECOMMENDED_SAMPLES):
                warnings.append(
                    f"当前共 {sum(split_counts.values())} 张图片，正式调优建议累计约 {config.MODEL_TRAINING_RECOMMENDED_SAMPLES} 张"
                )
            if not training_ready:
                warnings.append("数据集必须至少包含一个 person、head、helmet 标注")
            manifest = {
                "dataset_id": dataset_id,
                "algorithm_code": "safety_helmet",
                "created_at": now_text(),
                "source": "local_zip_upload",
                "original_name": Path(original_name).name,
                "event_count": 0,
                "image_count": sum(split_counts.values()),
                "labeled_images": labeled_images,
                "negative_images": negative_images,
                "split_counts": split_counts,
                "class_counts": class_counts,
                "training_ready": training_ready,
                "warnings": warnings,
                "data_yaml": str(data_yaml),
            }
            (dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            return manifest
        except Exception:
            if dataset_dir is not None and dataset_dir.is_dir() and not (dataset_dir / "manifest.json").is_file():
                shutil.rmtree(dataset_dir, ignore_errors=True)
            raise
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = []
        for path in sorted(self.datasets_root.glob("*/manifest.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("source") != "local_zip_upload":
                    continue
                data["directory"] = str(path.parent)
                datasets.append(data)
            except Exception:
                continue
        return datasets

    def _active_video_tasks(self) -> List[Dict[str, Any]]:
        if self.active_task_provider is None:
            return []
        active_states = {"starting", "running", "event_running", "reconnecting"}
        try:
            return [item for item in self.active_task_provider() if item.get("runtime_state") in active_states]
        except Exception:
            return []

    def _job_row(self, job_id: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM model_training_jobs WHERE id = ?", (int(job_id),)).fetchone()
        if row is None:
            raise TrainingNotFoundError("训练任务不存在")
        return row

    def _results_metrics(self, row: sqlite3.Row) -> Dict[str, Any]:
        config_data = self._json(row["config_json"], {})
        results_path = Path(str(row["output_dir"])) / "results.csv"
        result: Dict[str, Any] = {"epoch": 0, "progress": 0.0}
        if results_path.is_file():
            try:
                with results_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    latest = {str(key).strip(): value for key, value in rows[-1].items()}
                    epoch = int(float(latest.get("epoch") or len(rows) - 1)) + 1
                    result.update(
                        epoch=epoch,
                        progress=round(min(1.0, epoch / max(1, int(config_data.get("epochs") or 1))), 4),
                        precision=self._metric(latest, "metrics/precision"),
                        recall=self._metric(latest, "metrics/recall"),
                        map50=self._metric(latest, "metrics/mAP_0.5"),
                        map5095=self._metric(latest, "metrics/mAP_0.5:0.95"),
                        train_box_loss=self._metric(latest, "train/box_loss"),
                        train_obj_loss=self._metric(latest, "train/obj_loss"),
                        train_cls_loss=self._metric(latest, "train/cls_loss"),
                    )
            except Exception:
                pass
        return result

    @staticmethod
    def _metric(row: Dict[str, Any], key: str) -> Optional[float]:
        try:
            return round(float(row.get(key) or 0.0), 6)
        except (TypeError, ValueError):
            return None

    def _job_public(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        metrics = self._results_metrics(row)
        return {
            "id": int(data["id"]),
            "algorithm_code": data["algorithm_code"],
            "run_name": data["run_name"],
            "dataset_id": data["dataset_id"],
            "status": data["status"],
            "config": self._json(data["config_json"], {}),
            "command": self._json(data["command_json"], []),
            "pid": data["pid"],
            "output_dir": data["output_dir"],
            "log_path": data["log_path"],
            "best_model_path": data["best_model_path"],
            "metrics": metrics,
            "error": data["error"],
            "created_at": data["created_at"],
            "started_at": data["started_at"],
            "ended_at": data["ended_at"],
        }

    def list_jobs(self) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM model_training_jobs ORDER BY id DESC LIMIT 100").fetchall()
        return [self._job_public(row) for row in rows]

    def get_job(self, job_id: int) -> Dict[str, Any]:
        return self._job_public(self._job_row(job_id))

    def create_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        environment = self.environment_status()
        if not environment.get("ready"):
            raise TrainingValidationError(f"训练环境未就绪：{environment.get('error') or '未知错误'}")
        active_tasks = self._active_video_tasks()
        if active_tasks:
            raise TrainingValidationError("存在正在运行的视频检测任务，请先停止检测任务，释放 GPU 后再训练")
        with self._connect() as connection:
            active_job = connection.execute(
                "SELECT id FROM model_training_jobs WHERE status IN ('starting', 'running', 'stopping') LIMIT 1"
            ).fetchone()
        if active_job is not None:
            raise TrainingValidationError("已有训练任务正在运行，本机同一时间只允许一个训练进程")
        dataset_id = str(payload.get("dataset_id") or "").strip()
        manifest_path = self.datasets_root / dataset_id / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TrainingValidationError("请选择已准备的数据集") from exc
        if manifest.get("source") != "local_zip_upload":
            raise TrainingValidationError("当前模块只允许使用本地上传的数据集")
        if not manifest.get("training_ready"):
            raise TrainingValidationError("该数据集未通过类别和训练/验证集完整性检查")
        epochs = max(1, min(1000, int(payload.get("epochs") or config.MODEL_TRAINING_DEFAULT_EPOCHS)))
        batch_size = max(1, min(64, int(payload.get("batch_size") or config.MODEL_TRAINING_DEFAULT_BATCH_SIZE)))
        image_size = int(payload.get("image_size") or config.MODEL_TRAINING_DEFAULT_IMAGE_SIZE)
        if image_size not in {320, 416, 512, 640, 768, 960, 1280}:
            raise TrainingValidationError("训练尺寸不受支持")
        workers = max(0, min(8, int(payload.get("workers") or config.MODEL_TRAINING_DEFAULT_WORKERS)))
        patience = max(0, min(300, int(payload.get("patience") or config.MODEL_TRAINING_DEFAULT_PATIENCE)))
        freeze_layers = 10 if bool(payload.get("freeze_backbone", True)) else 0
        raw_name = str(payload.get("run_name") or "").strip()
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_name).strip("_")[:48]
        run_name = slug or ("safety_helmet_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        output_dir = self.runs_root / run_name
        log_path = self.logs_root / f"{run_name}.log"
        job_config = {
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "workers": workers,
            "patience": patience,
            "freeze_backbone": bool(freeze_layers),
            "device": "0",
            "weights": str(config.SAFETY_HELMET_TRAINING_WEIGHTS),
            "data_yaml": str(manifest_path.parent / "data.yaml"),
        }
        command = [
            str(config.MODEL_TRAINING_PYTHON),
            str(config.MODEL_TRAINING_RUNNER),
            "--weights", str(config.SAFETY_HELMET_TRAINING_WEIGHTS),
            "--data", job_config["data_yaml"],
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--imgsz", str(image_size),
            "--device", "0",
            "--workers", str(workers),
            "--project", str(self.runs_root),
            "--name", run_name,
            "--patience", str(patience),
            "--freeze", str(freeze_layers),
            "--exist-ok",
        ]
        timestamp = now_text()
        with self._connect() as connection:
            try:
                cursor = connection.execute(
                    """
                    INSERT INTO model_training_jobs (
                        algorithm_code, run_name, dataset_id, status, config_json, command_json,
                        output_dir, log_path, created_at
                    ) VALUES (?, ?, ?, 'starting', ?, ?, ?, ?, ?)
                    """,
                    (
                        "safety_helmet", run_name, dataset_id,
                        json.dumps(job_config, ensure_ascii=False), json.dumps(command, ensure_ascii=False),
                        str(output_dir), str(log_path), timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise TrainingValidationError("训练名称已存在，请更换名称") from exc
            job_id = int(cursor.lastrowid)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8", errors="replace") as log_stream:
                process = subprocess.Popen(
                    command,
                    cwd=str(config.MODEL_TRAINING_YOLOV5_REPO),
                    env=self._runner_environment(),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            with self._lock:
                self._processes[job_id] = process
            with self._connect() as connection:
                connection.execute(
                    "UPDATE model_training_jobs SET status = 'running', pid = ?, started_at = ? WHERE id = ?",
                    (int(process.pid), now_text(), job_id),
                )
            thread = threading.Thread(target=self._monitor_job, args=(job_id, process), daemon=True, name=f"model-training-{job_id}")
            thread.start()
        except Exception as exc:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE model_training_jobs SET status = 'failed', error = ?, ended_at = ? WHERE id = ?",
                    (str(exc), now_text(), job_id),
                )
            raise TrainingValidationError(f"训练进程启动失败：{exc}") from exc
        return self.get_job(job_id)

    def _monitor_job(self, job_id: int, process: subprocess.Popen[Any]) -> None:
        return_code = process.wait()
        with self._lock:
            self._processes.pop(job_id, None)
        row = self._job_row(job_id)
        current_status = str(row["status"])
        best_path = Path(str(row["output_dir"])) / "weights" / "best.pt"
        if current_status == "stopping":
            status, error = "stopped", "用户停止训练"
        elif return_code == 0 and best_path.is_file():
            status, error = "completed", ""
        else:
            status, error = "failed", f"训练进程退出码：{return_code}"
        with self._connect() as connection:
            connection.execute(
                "UPDATE model_training_jobs SET status = ?, best_model_path = ?, error = ?, ended_at = ? WHERE id = ?",
                (status, str(best_path) if best_path.is_file() else "", error, now_text(), int(job_id)),
            )

    def stop_job(self, job_id: int) -> Dict[str, Any]:
        row = self._job_row(job_id)
        if row["status"] not in ACTIVE_JOB_STATUSES:
            return self._job_public(row)
        with self._connect() as connection:
            connection.execute("UPDATE model_training_jobs SET status = 'stopping' WHERE id = ?", (int(job_id),))
        with self._lock:
            process = self._processes.get(int(job_id))
        if process is not None and process.poll() is None:
            process.terminate()
        return self.get_job(job_id)

    def stop_all(self) -> None:
        with self._lock:
            items = list(self._processes.items())
        for job_id, process in items:
            if process.poll() is None:
                with self._connect() as connection:
                    connection.execute("UPDATE model_training_jobs SET status = 'stopping' WHERE id = ?", (job_id,))
                process.terminate()

    def log_tail(self, job_id: int, line_count: int = 200) -> Dict[str, Any]:
        row = self._job_row(job_id)
        path = Path(str(row["log_path"]))
        lines: List[str] = []
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max(1, min(1000, int(line_count))):]
            except OSError:
                lines = []
        return {"job_id": int(job_id), "status": row["status"], "lines": lines}

    def overview(self, force_environment: bool = False) -> Dict[str, Any]:
        return {
            "algorithm": {"code": "safety_helmet", "name": "安全帽佩戴检测", "classes": list(CLASS_IDS)},
            "input_mode": "local_zip_upload",
            "environment": self.environment_status(force=force_environment),
            "datasets": self.list_datasets(),
            "jobs": self.list_jobs(),
            "defaults": {
                "epochs": int(config.MODEL_TRAINING_DEFAULT_EPOCHS),
                "batch_size": int(config.MODEL_TRAINING_DEFAULT_BATCH_SIZE),
                "image_size": int(config.MODEL_TRAINING_DEFAULT_IMAGE_SIZE),
                "workers": int(config.MODEL_TRAINING_DEFAULT_WORKERS),
                "patience": int(config.MODEL_TRAINING_DEFAULT_PATIENCE),
                "freeze_backbone": True,
            },
            "active_video_tasks": len(self._active_video_tasks()),
        }


def register_model_training_routes(app: Any, manager: ModelTrainingManager) -> None:
    from fastapi import HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def handle(exc: BaseException) -> None:
        if isinstance(exc, TrainingNotFoundError):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, TrainingValidationError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise exc

    @app.get("/training", response_class=HTMLResponse, include_in_schema=False)
    def model_training_page(request: Request) -> Any:
        return templates.TemplateResponse(request=request, name="model_training.html", context={"page_title": "模型训练"})

    @app.get("/api/training/overview")
    def model_training_overview(refresh_environment: bool = False) -> Dict[str, Any]:
        return manager.overview(force_environment=refresh_environment)

    @app.post("/api/training/datasets/upload")
    async def upload_model_training_dataset(request: Request) -> Dict[str, Any]:
        upload_dir = manager.training_root / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temp_path = upload_dir / f"upload_{uuid.uuid4().hex}.zip"
        size = 0
        try:
            with temp_path.open("wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > int(config.MODEL_TRAINING_MAX_UPLOAD_BYTES):
                        raise TrainingValidationError("上传数据集超过平台允许的最大体积")
                    stream.write(chunk)
            if size == 0:
                raise TrainingValidationError("请选择需要上传的 ZIP 数据集")
            encoded_name = request.headers.get("x-dataset-filename", "dataset.zip")
            from urllib.parse import unquote

            return manager.import_dataset_archive(temp_path, unquote(encoded_name))
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")
        finally:
            temp_path.unlink(missing_ok=True)

    @app.get("/api/training/jobs")
    def list_model_training_jobs() -> Dict[str, Any]:
        return {"items": manager.list_jobs()}

    @app.post("/api/training/jobs")
    def create_model_training_job(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return manager.create_job(payload)
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")

    @app.get("/api/training/jobs/{job_id}")
    def get_model_training_job(job_id: int) -> Dict[str, Any]:
        try:
            return manager.get_job(job_id)
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")

    @app.post("/api/training/jobs/{job_id}/stop")
    def stop_model_training_job(job_id: int) -> Dict[str, Any]:
        try:
            return manager.stop_job(job_id)
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")

    @app.get("/api/training/jobs/{job_id}/log")
    def get_model_training_log(job_id: int, lines: int = 200) -> Dict[str, Any]:
        try:
            return manager.log_tail(job_id, lines)
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")

    @app.get("/api/training/jobs/{job_id}/best-model")
    def download_best_model(job_id: int) -> FileResponse:
        try:
            job = manager.get_job(job_id)
            path = Path(str(job.get("best_model_path") or ""))
            if job["status"] != "completed" or not path.is_file():
                raise TrainingValidationError("该训练任务尚未生成可用的 best.pt")
            return FileResponse(path=str(path), filename=f"{job['run_name']}_best.pt", media_type="application/octet-stream")
        except (TrainingNotFoundError, TrainingValidationError) as exc:
            handle(exc)
            raise AssertionError("unreachable")
