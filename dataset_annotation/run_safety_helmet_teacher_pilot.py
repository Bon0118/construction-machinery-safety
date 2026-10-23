"""Generate safety-helmet v2 teacher prelabels with independent model evidence."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from dataset_annotation.annotation.head_shoulder_crop import HeadShoulderCrop, create_head_shoulder_crop
from dataset_annotation.annotation.rules import PrelabelBox, attach_truncation, eligible_head_size
from dataset_annotation.annotation.teacher_models import (
    GroundingDinoLocator,
    IndependentPersonDetector,
    QwenSemanticSelector,
    Sam2Refiner,
    SemanticDecision,
    TeacherCandidate,
    build_label_candidates,
    candidate_matches_target_person,
    draw_candidate_overlay,
    fallback_decision,
)
from dataset_annotation.export.yolo_exporter import (
    write_annotation_json,
    write_preview,
    write_yolo,
)
from dataset_annotation.quality.image_quality import evaluate_image
from dataset_annotation.run_safety_helmet_pilot import (
    build_html_report,
    choose_candidates,
    create_contact_sheets,
    write_jsonl,
)
from dataset_annotation.source.alarm_event_source import AlarmCandidate, scan_safety_helmet_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "safety_helmet_v2.json"


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_jpeg(path: Path, image: np.ndarray, quality: int = 95) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"无法编码图片：{path}")
    encoded.tofile(str(path))


def select_source_candidates(
    alarm_root: Path,
    config: Dict[str, Any],
    source_manifest: Path | None,
    count: int,
) -> List[Tuple[AlarmCandidate, Any]]:
    candidates = scan_safety_helmet_events(alarm_root)
    by_id = {item.event_id: item for item in candidates}
    if source_manifest is None:
        accepted = []
        for candidate in candidates:
            quality, _image = evaluate_image(candidate.original_path, config["quality"])
            if quality.accepted and candidate.event_id not in config.get("manual_quality_rejections", {}):
                accepted.append((candidate, quality))
        config["sampling"]["count"] = count
        selected, _skips = choose_candidates(accepted, config)
        return list(selected)

    # A source manifest already fixes the sample population. Avoid decoding and
    # evaluating every historical alarm image before validating those rows.
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = []
    for row in rows[:count]:
        event_id = str(row["event_id"])
        candidate = by_id.get(event_id)
        if candidate is None:
            raise RuntimeError(f"源试标清单中的告警已不存在：{event_id}")
        quality, _image = evaluate_image(candidate.original_path, config["quality"])
        if not quality.accepted:
            raise RuntimeError(f"源试标图片未通过当前质量规则：{event_id}")
        selected.append((candidate, quality))
    if len(selected) != count:
        raise RuntimeError(f"源试标清单只有{len(selected)}条，要求{count}条")
    return selected


def load_person_detector(pipeline: Mapping[str, Any]):
    preferred = str(pipeline["device"])
    try:
        return IndependentPersonDetector(pipeline["person_detector"], preferred), preferred, ""
    except Exception as exc:
        if preferred == "cpu" or not bool(pipeline.get("allow_cpu_fallback", True)):
            raise
        return IndependentPersonDetector(pipeline["person_detector"], "cpu"), "cpu", str(exc)


def load_grounding_dino(pipeline: Mapping[str, Any]):
    preferred = str(pipeline["device"])
    try:
        return GroundingDinoLocator(pipeline["grounding_dino"], preferred), preferred, ""
    except Exception as exc:
        if preferred == "cpu" or not bool(pipeline.get("allow_cpu_fallback", True)):
            raise
        return GroundingDinoLocator(pipeline["grounding_dino"], "cpu"), "cpu", str(exc)


def load_sam2(pipeline: Mapping[str, Any]):
    if not bool(pipeline["sam2"].get("enabled", True)):
        return None, "disabled", "sam2_disabled_by_config"
    preferred = str(pipeline["device"])
    try:
        return Sam2Refiner(pipeline["sam2"], preferred), preferred, ""
    except Exception as exc:
        if preferred != "cpu" and bool(pipeline.get("allow_cpu_fallback", True)):
            try:
                return Sam2Refiner(pipeline["sam2"], "cpu"), "cpu", str(exc)
            except Exception as cpu_exc:
                return None, "unavailable", f"cuda={exc}; cpu={cpu_exc}"
        return None, "unavailable", str(exc)


def candidate_from_dict(value: Mapping[str, Any]) -> TeacherCandidate:
    box = tuple(int(item) for item in value["box"])
    return TeacherCandidate(
        candidate_id=str(value["candidate_id"]),
        kind=str(value["kind"]),
        score=float(value["score"]),
        box=box,
        source=str(value["source"]),
        prompt=str(value.get("prompt") or ""),
        sam_refined=bool(value.get("sam_refined")),
        mask_score=(float(value["mask_score"]) if value.get("mask_score") is not None else None),
    )


def final_crop_labels(
    person: TeacherCandidate,
    crop: HeadShoulderCrop,
    candidates: Sequence[TeacherCandidate],
    decision: SemanticDecision,
    config: Mapping[str, Any],
) -> Tuple[List[PrelabelBox], List[str]]:
    issues: List[str] = []
    if not decision.is_target_person or decision.head_state == "no_person":
        return [], ["qwen_rejected_person_detection"]
    labels = [
        PrelabelBox(
            class_id=0,
            class_name="person",
            confidence=float(person.score),
            box=crop.person_box,
            truncated=crop.person_truncated,
            source=person.source,
        )
    ]
    if decision.head_state in {"person_only", "ignore"}:
        issues.append(f"qwen_{decision.head_state}")
        return labels, issues

    candidate_map = {item.candidate_id: item for item in candidates}
    selected = candidate_map.get(decision.selected_candidate_id)
    if selected is None or selected.kind != decision.head_state:
        issues.append("qwen_candidate_selection_invalid")
        return labels, issues
    if not candidate_matches_target_person(selected.box, crop.person_box):
        issues.append("head_candidate_outside_target_person_upper_region")
        return labels, issues
    if decision.head_state == "helmet" and decision.visibility_ratio < float(config["helmet_visible_ratio_minimum"]):
        issues.append("helmet_visibility_below_50_percent")
        return labels, issues
    if not eligible_head_size(
        selected.box,
        crop.output_size,
        crop.output_size,
        int(config["training_width"]),
        int(config["training_height"]),
        float(config["minimum_head_short_side"]),
    ):
        issues.append(f"{decision.head_state}_below_20px_at_training_size")
        return labels, issues
    class_id = 1 if decision.head_state == "head" else 2
    labels.append(
        PrelabelBox(
            class_id=class_id,
            class_name=decision.head_state,
            confidence=min(float(decision.confidence), float(selected.score)),
            box=selected.box,
            truncated=decision.truncated,
            source=f"{selected.source}+{decision.source}",
        )
    )
    if "geometry_fallback" in selected.source:
        issues.append("geometry_head_candidate_requires_manual_review")
    if not selected.sam_refined:
        issues.append("selected_candidate_not_sam_refined")
    if decision.confidence < float(config["teacher_pipeline"]["qwen"]["minimum_auto_confidence"]):
        issues.append("qwen_confidence_below_auto_threshold")
    return attach_truncation(labels, crop.output_size, crop.output_size), sorted(set(issues))


def run(args: argparse.Namespace) -> Path:
    config = read_config(args.config)
    pipeline = config["teacher_pipeline"]
    selected = select_source_candidates(args.alarm_root, config, args.source_manifest, args.count)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or (PROJECT_ROOT / "dataset_outputs" / f"safety_helmet_teacher_{args.count}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=False)
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    annotations_dir = output_dir / "annotations"
    previews_dir = output_dir / "previews"
    crops_dir = output_dir / "crops"
    crop_images_dir = crops_dir / "images"
    crop_labels_dir = crops_dir / "labels"
    crop_annotations_dir = crops_dir / "annotations"
    crop_previews_dir = crops_dir / "previews"
    crop_candidates_dir = crops_dir / "candidates"
    for directory in (
        images_dir,
        labels_dir,
        annotations_dir,
        previews_dir,
        crop_images_dir,
        crop_labels_dir,
        crop_annotations_dir,
        crop_previews_dir,
        crop_candidates_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / args.config.name)

    runtime: Dict[str, Any] = {
        "requested_device": pipeline["device"],
        "student_safety_helmet_model_used_for_labels": False,
    }
    known_non_person = config.get("known_non_person_events", {})
    samples: List[Dict[str, Any]] = []

    person_detector, person_device, person_fallback = load_person_detector(pipeline)
    runtime["person_detector"] = {"device": person_device, "fallback_error": person_fallback}
    try:
        for index, (candidate, quality) in enumerate(selected, 1):
            sample_stem = f"sample_{index:03d}_{candidate.event_id}"
            image_path = images_dir / f"{sample_stem}.jpg"
            shutil.copy2(candidate.original_path, image_path)
            data = np.fromfile(str(image_path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            override = str(known_non_person.get(candidate.event_id) or "")
            people = [] if override else person_detector.predict(image)
            samples.append(
                {
                    "index": index,
                    "candidate": candidate,
                    "quality": quality,
                    "sample_stem": sample_stem,
                    "image_path": image_path,
                    "people": people,
                    "non_person_override": override,
                    "crops": [],
                }
            )
            print(f"[PERSON] {index}/{len(selected)} {candidate.event_id} people={len(people)}", flush=True)
    finally:
        person_detector.close()

    crop_records: List[Dict[str, Any]] = []
    for sample in samples:
        image = cv2.imdecode(np.fromfile(str(sample["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        for person_index, person in enumerate(sample["people"], 1):
            person_label = PrelabelBox(0, "person", person.score, person.box, source=person.source)
            crop = create_head_shoulder_crop(image, person_label, config["crop"])
            crop_id = f"sample_{sample['index']:03d}_person_{person_index:02d}_{sample['candidate'].event_id}"
            crop_image_path = crop_images_dir / f"{crop_id}.jpg"
            encode_jpeg(crop_image_path, crop.image)
            record = {
                "crop_id": crop_id,
                "sample": sample,
                "person": person,
                "crop": crop,
                "image_path": crop_image_path,
                "grounded": [],
                "label_candidates": [],
                "pipeline_issues": [],
            }
            sample["crops"].append(record)
            crop_records.append(record)

    grounding, grounding_device, grounding_fallback = load_grounding_dino(pipeline)
    runtime["grounding_dino"] = {"device": grounding_device, "fallback_error": grounding_fallback}
    try:
        for index, record in enumerate(crop_records, 1):
            image = cv2.imdecode(np.fromfile(str(record["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            record["grounded"] = grounding.predict(image)
            print(f"[DINO] {index}/{len(crop_records)} candidates={len(record['grounded'])}", flush=True)
    finally:
        grounding.close()

    sam2, sam_device, sam_error = load_sam2(pipeline)
    runtime["sam2"] = {"device": sam_device, "error": sam_error}
    try:
        for index, record in enumerate(crop_records, 1):
            image = cv2.imdecode(np.fromfile(str(record["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            grounded = record["grounded"]
            if sam2 is not None:
                try:
                    grounded = sam2.refine(image, grounded)
                except Exception as exc:
                    record["pipeline_issues"].append(f"sam2_refine_failed:{exc}")
            else:
                record["pipeline_issues"].append("sam2_unavailable")
            record["grounded"] = grounded
            record["label_candidates"] = build_label_candidates(image, record["crop"].person_box, grounded)
            overlay = draw_candidate_overlay(image, record["crop"].person_box, record["label_candidates"])
            overlay_path = crop_candidates_dir / f"{record['crop_id']}.jpg"
            encode_jpeg(overlay_path, overlay, 92)
            record["overlay_path"] = overlay_path
            write_annotation_json(
                crop_candidates_dir / f"{record['crop_id']}.json",
                {
                    "raw_grounding_candidates": [item.to_dict() for item in grounded],
                    "label_candidates": [item.to_dict() for item in record["label_candidates"]],
                    "issues": record["pipeline_issues"],
                },
            )
            print(f"[SAM] {index}/{len(crop_records)} labels={len(record['label_candidates'])}", flush=True)
    finally:
        if sam2 is not None:
            sam2.close()

    qwen = QwenSemanticSelector(pipeline["qwen"], PROJECT_ROOT)

    def review_crop(record: Dict[str, Any]) -> Dict[str, Any]:
        image = cv2.imdecode(np.fromfile(str(record["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        overlay = cv2.imdecode(np.fromfile(str(record["overlay_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        candidate = record["sample"]["candidate"]
        try:
            historical_review = {
                "verdict": candidate.review.get("verdict"),
                "evidence_quality": candidate.review.get("evidence_quality"),
                "summary": candidate.review.get("summary"),
                "reason_codes": candidate.review.get("reason_codes") or [],
            }
            decision, qwen_runtime = qwen.review(
                image,
                overlay,
                record["crop"].person_box,
                record["label_candidates"],
                historical_review,
            )
        except Exception as exc:
            decision = fallback_decision(candidate.semantic_bucket, record["label_candidates"], str(exc))
            qwen_runtime = {"status": "fallback", "error": str(exc)}
            record["pipeline_issues"].append("qwen_review_failed")
        if (
            decision.head_state in {"head", "helmet"}
            and not decision.selected_candidate_id
            and len(decision.direct_box) == 4
        ):
            height, width = image.shape[:2]
            coordinates = decision.direct_box
            # Crops are exported at a fixed 640x640 size and the local model is
            # more reliable with pixel coordinates. Retain compatibility with
            # an occasional normalized 0..1000 response when values exceed the
            # actual crop dimensions.
            normalized_response = max(coordinates) > max(width, height)
            if normalized_response:
                coordinates = tuple(round(value * size / 1000) for value, size in zip(coordinates, (width, height, width, height)))
            direct_box = (
                max(0, min(width - 1, int(coordinates[0]))),
                max(0, min(height - 1, int(coordinates[1]))),
                max(1, min(width, int(coordinates[2]))),
                max(1, min(height, int(coordinates[3]))),
            )
            direct_candidate = TeacherCandidate(
                candidate_id="Q01",
                kind=decision.head_state,
                score=decision.confidence,
                box=direct_box,
                source="qwen3vl4b_direct_grounding",
                prompt="direct normalized box",
            )
            record["label_candidates"].append(direct_candidate)
            decision = replace(decision, selected_candidate_id="Q01")
            record["pipeline_issues"].append("qwen_direct_box_requires_manual_review")
        record["decision"] = decision
        record["qwen_runtime"] = qwen_runtime
        return record

    qwen_workers = max(1, int(pipeline["qwen"].get("concurrency", 1)))
    runtime["qwen"] = {"concurrency": qwen_workers}
    if qwen_workers == 1:
        for index, record in enumerate(crop_records, 1):
            completed = review_crop(record)
            decision = completed["decision"]
            print(
                f"[QWEN] {index}/{len(crop_records)} state={decision.head_state} "
                f"confidence={decision.confidence:.2f}",
                flush=True,
            )
    else:
        with ThreadPoolExecutor(max_workers=qwen_workers, thread_name_prefix="qwen-teacher") as executor:
            pending = {executor.submit(review_crop, record): record for record in crop_records}
            for completed_count, future in enumerate(as_completed(pending), 1):
                completed = future.result()
                decision = completed["decision"]
                print(
                    f"[QWEN] {completed_count}/{len(crop_records)} crop={completed['crop_id']} "
                    f"state={decision.head_state} confidence={decision.confidence:.2f}",
                    flush=True,
                )

    full_manifest: List[Dict[str, Any]] = []
    crop_manifest: List[Dict[str, Any]] = []
    full_previews: List[Path] = []
    crop_previews: List[Path] = []
    full_counts = Counter()
    crop_counts = Counter()
    issue_counts = Counter()
    decision_counts = Counter()

    for sample in samples:
        candidate = sample["candidate"]
        quality = sample["quality"]
        image = cv2.imdecode(np.fromfile(str(sample["image_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        full_labels: List[PrelabelBox] = []
        full_issues: List[str] = []
        for record in sample["crops"]:
            crop = record["crop"]
            person = record["person"]
            decision = record["decision"]
            labels, issues = final_crop_labels(person, crop, record["label_candidates"], decision, config)
            issues = sorted(set(issues + record["pipeline_issues"] + ["teacher_prelabel_requires_manual_review"]))
            decision_counts[decision.head_state] += 1
            issue_counts.update(issues)
            crop_counts.update(item.class_name for item in labels)
            crop_label_path = crop_labels_dir / f"{record['crop_id']}.txt"
            crop_annotation_path = crop_annotations_dir / f"{record['crop_id']}.json"
            crop_preview_path = crop_previews_dir / f"{record['crop_id']}.jpg"
            write_yolo(crop_label_path, labels, crop.output_size, crop.output_size)
            write_preview(crop_preview_path, crop.image, labels, f"teacher: {decision.head_state} {decision.confidence:.2f}")
            crop_previews.append(crop_preview_path)
            annotation = {
                "schema_version": config["schema_version"],
                "crop_id": record["crop_id"],
                "source_event_id": candidate.event_id,
                "source_original_path": str(candidate.original_path),
                "annotation_status": "teacher_prelabeled_pending_manual_review",
                "manual_review": True,
                "manual_review_reasons": issues,
                "crop_transform": crop.metadata(),
                "independent_person_detection": person.to_dict(),
                "grounding_candidates": [item.to_dict() for item in record["grounded"]],
                "label_candidates": [item.to_dict() for item in record["label_candidates"]],
                "qwen_semantic_decision": decision.to_dict(),
                "qwen_runtime": record["qwen_runtime"],
                "labels": [item.to_dict() for item in labels],
                "training_eligibility": "manual_review_required",
            }
            write_annotation_json(crop_annotation_path, annotation)
            label_counter = Counter(item.class_name for item in labels)
            crop_manifest.append(
                {
                    "sample_index": len(crop_manifest) + 1,
                    "source_sample_index": sample["index"],
                    "event_id": record["crop_id"],
                    "source_event_id": candidate.event_id,
                    "semantic_bucket": decision.head_state,
                    "review_verdict": decision.head_state,
                    "review_summary": decision.reason,
                    "image_path": str(record["image_path"]),
                    "label_path": str(crop_label_path),
                    "annotation_path": str(crop_annotation_path),
                    "preview_path": str(crop_preview_path),
                    "candidate_overlay_path": str(record["overlay_path"]),
                    "label_counts": dict(label_counter),
                    "issues": issues,
                    "manual_review": True,
                    "training_eligibility": "manual_review_required",
                }
            )
            if labels:
                full_labels.append(
                    PrelabelBox(
                        class_id=0,
                        class_name="person",
                        confidence=person.score,
                        box=person.box,
                        source=person.source,
                    )
                )
            for label in labels:
                if label.class_name not in {"head", "helmet"}:
                    continue
                mapped = crop.remap_crop_box(label.box)
                if mapped is None or not eligible_head_size(
                    mapped,
                    quality.width,
                    quality.height,
                    int(config["training_width"]),
                    int(config["training_height"]),
                    float(config["minimum_head_short_side"]),
                ):
                    full_issues.append(f"full_frame_{label.class_name}_below_20px_or_invalid")
                    continue
                full_labels.append(
                    PrelabelBox(
                        class_id=label.class_id,
                        class_name=label.class_name,
                        confidence=label.confidence,
                        box=mapped,
                        source=label.source + "+inverse_crop_transform",
                    )
                )
        if sample["non_person_override"]:
            full_labels = []
            full_issues.extend(["known_non_person_override", sample["non_person_override"]])
        full_labels = attach_truncation(full_labels, quality.width, quality.height)
        full_counts.update(item.class_name for item in full_labels)
        issue_counts.update(full_issues)
        label_path = labels_dir / f"{sample['sample_stem']}.txt"
        annotation_path = annotations_dir / f"{sample['sample_stem']}.json"
        preview_path = previews_dir / f"{sample['sample_stem']}.jpg"
        write_yolo(label_path, full_labels, quality.width, quality.height)
        write_preview(preview_path, image, full_labels, f"teacher #{sample['index']:02d}")
        full_previews.append(preview_path)
        counts = Counter(item.class_name for item in full_labels)
        annotation = {
            "schema_version": config["schema_version"],
            "sample_index": sample["index"],
            "event_id": candidate.event_id,
            "source_original_path": str(candidate.original_path),
            "annotation_status": "teacher_prelabeled_pending_manual_review",
            "manual_review": True,
            "manual_review_reasons": sorted(set(["teacher_prelabel_requires_manual_review"] + full_issues)),
            "historical_alarm_review": candidate.review,
            "quality": quality.to_dict(),
            "non_person_override": sample["non_person_override"] or None,
            "labels": [item.to_dict() for item in full_labels],
            "crop_ids": [record["crop_id"] for record in sample["crops"]],
            "training_eligibility": "traceability_original_not_primary_crop_training",
        }
        write_annotation_json(annotation_path, annotation)
        full_manifest.append(
            {
                "sample_index": sample["index"],
                "event_id": candidate.event_id,
                "semantic_bucket": candidate.semantic_bucket,
                "review_verdict": str(candidate.review.get("verdict") or ""),
                "review_summary": str(candidate.review.get("summary") or ""),
                "image_path": str(sample["image_path"]),
                "label_path": str(label_path),
                "annotation_path": str(annotation_path),
                "preview_path": str(preview_path),
                "label_counts": dict(counts),
                "issues": sorted(set(full_issues)),
                "manual_review": True,
                "training_eligibility": annotation["training_eligibility"],
                "non_person_override": sample["non_person_override"],
            }
        )

    write_jsonl(output_dir / "manifest.jsonl", full_manifest)
    write_jsonl(crops_dir / "manifest.jsonl", crop_manifest)
    for directory in (output_dir, crops_dir):
        (directory / "classes.txt").write_text("person\nhead\nhelmet\n", encoding="ascii")
        (directory / "data.yaml").write_text(
            "path: .\ntrain: images\nval: images\nnames:\n  0: person\n  1: head\n  2: helmet\n",
            encoding="utf-8",
        )
    summary = {
        "schema_version": config["schema_version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_alarm_root": str(args.alarm_root),
        "source_manifest": str(args.source_manifest or ""),
        "selected_original_count": len(samples),
        "head_shoulder_crop_count": len(crop_records),
        "known_non_person_sample_count": sum(bool(item["non_person_override"]) for item in samples),
        "full_frame_prelabel_counts": dict(full_counts),
        "crop_prelabel_counts": dict(crop_counts),
        "qwen_head_state_counts": dict(decision_counts),
        "issue_counts": dict(issue_counts),
        "runtime": runtime,
        "student_safety_helmet_model_used_for_labels": False,
        "primary_training_images": "crops/images",
        "original_images_role": "traceability_only",
        "training_readiness": "not_ready_until_manual_review_completed",
        "status": "teacher_prelabeled_pending_manual_review",
    }
    write_annotation_json(output_dir / "summary.json", summary)
    write_annotation_json(crops_dir / "summary.json", summary)
    full_sheets = create_contact_sheets(full_previews, output_dir)
    crop_sheets = create_contact_sheets(crop_previews, crops_dir)
    build_html_report(output_dir, summary, full_manifest, full_sheets, title="安全帽教师流水线：原图溯源")
    build_html_report(crops_dir, summary, crop_manifest, crop_sheets, title="安全帽教师流水线：头肩裁剪标注")
    (output_dir / "PIPELINE_NOTE.md").write_text(
        "# safety-helmet-v2 teacher pipeline\n\n"
        "本批次未使用待调优的安全帽YOLOv5模型生成标签。person来自COCO通用人员模型，"
        "head/helmet候选来自Grounding DINO，SAM2.1用于边界细化，Qwen3-VL-4B负责语义类别与候选框选择。"
        "所有结果仍为教师预标注，完成人工审核后才能作为训练真值。\n",
        encoding="utf-8",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行安全帽独立教师模型50张试标")
    parser.add_argument("--alarm-root", type=Path, default=PROJECT_ROOT / "alarm_outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    destination = run(parse_args())
    print(f"[DONE] {destination}", flush=True)
