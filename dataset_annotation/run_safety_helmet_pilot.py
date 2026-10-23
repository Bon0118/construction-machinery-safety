"""Create a read-only 50-image safety-helmet pre-label pilot from alarm events."""

from __future__ import annotations

import argparse
import html
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from dataset_annotation.annotation.prelabel import SafetyHelmetPrelabeler
from dataset_annotation.annotation.head_shoulder_crop import (
    associated_with_person_head_region,
    create_head_shoulder_crop,
)
from dataset_annotation.annotation.rules import PrelabelBox, eligible_head_size
from dataset_annotation.export.yolo_exporter import (
    write_annotation_json,
    write_preview,
    write_yolo,
)
from dataset_annotation.quality.image_quality import (
    ImageQualityResult,
    evaluate_image,
    hash_distance,
)
from dataset_annotation.source.alarm_event_source import AlarmCandidate, scan_safety_helmet_events


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "safety_helmet_v1.json"


def read_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_candidates(
    accepted: Sequence[Tuple[AlarmCandidate, ImageQualityResult]],
    config: Dict[str, Any],
) -> Tuple[List[Tuple[AlarmCandidate, ImageQualityResult]], Counter]:
    seed = int(config["sampling"]["seed"])
    target_count = int(config["sampling"]["count"])
    targets = {key: int(value) for key, value in config["sampling"]["bucket_targets"].items()}
    duplicate_distance = int(config["quality"]["duplicate_hash_distance"])
    randomizer = random.Random(seed)
    grouped: Dict[str, List[Tuple[AlarmCandidate, ImageQualityResult]]] = defaultdict(list)
    for item in accepted:
        grouped[item[0].semantic_bucket].append(item)
    for values in grouped.values():
        randomizer.shuffle(values)

    selected: List[Tuple[AlarmCandidate, ImageQualityResult]] = []
    selected_hashes: List[str] = []
    skipped = Counter()

    def try_add(item: Tuple[AlarmCandidate, ImageQualityResult]) -> bool:
        image_hash = item[1].perceptual_hash
        if any(hash_distance(image_hash, previous) <= duplicate_distance for previous in selected_hashes):
            skipped["near_duplicate"] += 1
            return False
        selected.append(item)
        selected_hashes.append(image_hash)
        return True

    for bucket in ("head", "helmet", "difficult"):
        wanted = targets.get(bucket, 0)
        for item in grouped.get(bucket, []):
            if sum(value[0].semantic_bucket == bucket for value in selected) >= wanted:
                break
            try_add(item)

    remainder = list(accepted)
    randomizer.shuffle(remainder)
    for item in remainder:
        if len(selected) >= target_count:
            break
        if item in selected:
            continue
        try_add(item)
    if len(selected) < target_count:
        raise RuntimeError(f"质量过滤和去重后只能选择 {len(selected)} 张，目标为 {target_count} 张")
    return selected[:target_count], skipped


def _fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def create_contact_sheets(preview_paths: Sequence[Path], destination: Path) -> List[Path]:
    sheets: List[Path] = []
    tile_width, tile_height = 640, 390
    for sheet_index, offset in enumerate(range(0, len(preview_paths), 10), start=1):
        canvas = np.full((tile_height * 5, tile_width * 2, 3), 255, dtype=np.uint8)
        for position, path in enumerate(preview_paths[offset:offset + 10]):
            data = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            tile = _fit_tile(image, tile_width, tile_height)
            row, column = divmod(position, 2)
            canvas[
                row * tile_height:(row + 1) * tile_height,
                column * tile_width:(column + 1) * tile_width,
            ] = tile
        output_path = destination / f"contact_sheet_{sheet_index:02d}.jpg"
        ok, encoded = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("无法生成质检联系表")
        encoded.tofile(str(output_path))
        sheets.append(output_path)
    return sheets


def build_html_report(
    output_dir: Path,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    contact_sheets: Sequence[Path],
    title: str = "安全帽告警原图试标",
) -> None:
    crop_report_link = (
        '<p><a href="crops/index.html">查看人员头肩裁剪试标报告</a></p>'
        if (output_dir / "crops").is_dir()
        else ""
    )
    cards = []
    for row in rows:
        preview = html.escape(str(Path(row["preview_path"]).relative_to(output_dir)).replace("\\", "/"))
        original = html.escape(str(Path(row["image_path"]).relative_to(output_dir)).replace("\\", "/"))
        issues = ", ".join(row["issues"]) or "无"
        cards.append(
            f"""
            <article class="card {html.escape(row['semantic_bucket'])}">
              <a href="{original}"><img loading="lazy" src="{preview}"></a>
              <h3>#{row['sample_index']:02d} {html.escape(row['event_id'])}</h3>
              <p>语义分组：{html.escape(row['semantic_bucket'])}；复核：{html.escape(row['review_verdict'] or 'missing')}</p>
              <p>预标注：{html.escape(str(row['label_counts']))}</p>
              <p>问题：{html.escape(issues)}</p>
              <p>{html.escape(row['review_summary'][:220])}</p>
            </article>
            """
        )
    sheets = "".join(
        f'<p><a href="{path.name}">{html.escape(path.name)}</a></p>' for path in contact_sheets
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f4f6f8;color:#172033}}
.summary,.card{{background:white;border-radius:10px;padding:16px;box-shadow:0 2px 10px #0001}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:18px}}
.card{{border-left:7px solid #d97706}}.card.head{{border-left-color:#dc2626}}.card.helmet{{border-left-color:#16a34a}}
.card img{{width:100%;height:360px;object-fit:contain;background:#111;border-radius:6px}}
h3{{font-size:15px;word-break:break-all}}p{{font-size:13px;line-height:1.5}}
pre{{white-space:pre-wrap}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<section class="summary"><h1>{html.escape(title)}</h1><pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
{crop_report_link}<h2>联系表</h2>{sheets}<p>绿色=helmet语义候选，红色=head语义候选，橙色=困难样本。所有框均为模型预标注，尚未成为人工真值。</p></section>
<section class="grid">{''.join(cards)}</section></body></html>"""
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    config = read_config(args.config)
    config["sampling"]["count"] = int(args.count)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or (PROJECT_ROOT / "dataset_outputs" / f"safety_helmet_pilot_{args.count}_{timestamp}")
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
    for directory in (images_dir, labels_dir, annotations_dir, previews_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (crop_images_dir, crop_labels_dir, crop_annotations_dir, crop_previews_dir):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "safety_helmet_v1.json")

    candidates = scan_safety_helmet_events(args.alarm_root)
    accepted: List[Tuple[AlarmCandidate, ImageQualityResult]] = []
    rejected_rows: List[Dict[str, Any]] = []
    quality_counts = Counter()
    manual_quality_rejections = config.get("manual_quality_rejections", {})
    for candidate in candidates:
        quality, _image = evaluate_image(candidate.original_path, config["quality"])
        manual_reason = str(manual_quality_rejections.get(candidate.event_id) or "")
        if manual_reason:
            quality_counts[manual_reason] += 1
            rejected_rows.append(
                {
                    "event_id": candidate.event_id,
                    "original_path": str(candidate.original_path),
                    "semantic_bucket": candidate.semantic_bucket,
                    "quality": quality.to_dict(),
                    "manual_quality_rejection": manual_reason,
                }
            )
            continue
        quality_counts[quality.rejection_reason or "accepted"] += 1
        if quality.accepted:
            accepted.append((candidate, quality))
        else:
            rejected_rows.append(
                {
                    "event_id": candidate.event_id,
                    "original_path": str(candidate.original_path),
                    "semantic_bucket": candidate.semantic_bucket,
                    "quality": quality.to_dict(),
                }
            )
    selected, dedup_skips = choose_candidates(accepted, config)
    prelabeler = SafetyHelmetPrelabeler(config)
    manifest_rows: List[Dict[str, Any]] = []
    crop_manifest_rows: List[Dict[str, Any]] = []
    preview_paths: List[Path] = []
    crop_preview_paths: List[Path] = []
    label_counts = Counter()
    crop_label_counts = Counter()
    issue_counts = Counter()
    crop_issue_counts = Counter()
    known_non_person_events = config.get("known_non_person_events", {})

    for index, (candidate, quality) in enumerate(selected, start=1):
        sample_stem = f"sample_{index:03d}_{candidate.event_id}"
        image_path = images_dir / f"{sample_stem}.jpg"
        label_path = labels_dir / f"{sample_stem}.txt"
        annotation_path = annotations_dir / f"{sample_stem}.json"
        preview_path = previews_dir / f"{sample_stem}.jpg"
        shutil.copy2(candidate.original_path, image_path)
        quality_check, image = evaluate_image(image_path, config["quality"])
        assert image is not None and quality_check.accepted
        raw_source_labels = prelabeler.raw_labels(image)
        labels, issues = prelabeler.predict(image, candidate.semantic_bucket, raw_source_labels)
        non_person_reason = str(known_non_person_events.get(candidate.event_id) or "")
        if non_person_reason:
            labels = []
            issues = sorted(set(issues + ["known_non_person_override", non_person_reason]))
        counts = Counter(item.class_name for item in labels)
        label_counts.update(counts)
        issue_counts.update(issues)
        review = candidate.review
        manual_reasons = sorted(
            set(
                (["all_pilot_prelabels_require_review"])
                + (["difficult_semantic_bucket"] if candidate.semantic_bucket == "difficult" else [])
                + issues
            )
        )
        annotation = {
            "schema_version": config["schema_version"],
            "sample_index": index,
            "event_id": candidate.event_id,
            "source_original_path": str(candidate.original_path),
            "image_filename": image_path.name,
            "image_width": quality.width,
            "image_height": quality.height,
            "training_size": [config["training_width"], config["training_height"]],
            "semantic_bucket": candidate.semantic_bucket,
            "semantic_reason": candidate.semantic_reason,
            "annotation_status": "prelabeled_pending_manual_review",
            "manual_review": True,
            "manual_review_reasons": manual_reasons,
            "helmet_color": candidate.helmet_color,
            "review": review,
            "quality": quality.to_dict(),
            "labels": [item.to_dict() for item in labels],
            "training_eligibility": (
                "negative_non_person"
                if non_person_reason
                else
                "full_three_class_candidate"
                if counts.get("head", 0) or counts.get("helmet", 0)
                else "person_only_or_manual_review"
            ),
            "non_person_override": non_person_reason or None,
        }
        write_yolo(label_path, labels, quality.width, quality.height)
        write_annotation_json(annotation_path, annotation)
        write_preview(
            preview_path,
            image,
            labels,
            f"#{index:02d} {candidate.semantic_bucket} {str(review.get('verdict') or 'missing')}",
        )
        preview_paths.append(preview_path)
        manifest_rows.append(
            {
                "sample_index": index,
                "event_id": candidate.event_id,
                "semantic_bucket": candidate.semantic_bucket,
                "semantic_reason": candidate.semantic_reason,
                "review_verdict": str(review.get("verdict") or ""),
                "review_evidence_quality": str(review.get("evidence_quality") or ""),
                "review_summary": str(review.get("summary") or ""),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "annotation_path": str(annotation_path),
                "preview_path": str(preview_path),
                "label_counts": dict(counts),
                "issues": issues,
                "quality": quality.to_dict(),
                "manual_review": True,
                "training_eligibility": annotation["training_eligibility"],
                "non_person_override": non_person_reason,
            }
        )

        for person_index, person in enumerate(
            (item for item in labels if item.class_name == "person"), start=1
        ):
            crop = create_head_shoulder_crop(image, person, config["crop"])
            mapped_source_candidates: List[PrelabelBox] = []
            for source_status in raw_source_labels:
                if source_status.class_name not in {"head", "helmet"}:
                    continue
                if not associated_with_person_head_region(
                    source_status.box, person.box, config["crop"]
                ):
                    continue
                remapped = crop.remap_original_box(source_status.box)
                if remapped is None or not eligible_head_size(
                    remapped,
                    crop.output_size,
                    crop.output_size,
                    int(config["training_width"]),
                    int(config["training_height"]),
                    float(config["minimum_head_short_side"]),
                ):
                    continue
                mapped_source_candidates.append(
                    PrelabelBox(
                        class_id=source_status.class_id,
                        class_name=source_status.class_name,
                        confidence=source_status.confidence,
                        box=remapped,
                        source="model1_best.pt_full_frame_remapped",
                    )
                )
            crop_labels, crop_issues = prelabeler.predict_crop(
                crop.image,
                crop.person_box,
                mapped_source_candidates,
            )
            crop_counts = Counter(item.class_name for item in crop_labels)
            crop_label_counts.update(crop_counts)
            crop_issue_counts.update(crop_issues)
            crop_id = f"sample_{index:03d}_person_{person_index:02d}_{candidate.event_id}"
            crop_image_path = crop_images_dir / f"{crop_id}.jpg"
            crop_label_path = crop_labels_dir / f"{crop_id}.txt"
            crop_annotation_path = crop_annotations_dir / f"{crop_id}.json"
            crop_preview_path = crop_previews_dir / f"{crop_id}.jpg"
            ok, encoded = cv2.imencode(
                ".jpg", crop.image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            )
            if not ok:
                raise RuntimeError(f"无法编码头肩裁剪图：{crop_id}")
            encoded.tofile(str(crop_image_path))
            write_yolo(crop_label_path, crop_labels, crop.output_size, crop.output_size)
            crop_annotation = {
                "schema_version": config["schema_version"],
                "crop_id": crop_id,
                "source_sample_index": index,
                "source_event_id": candidate.event_id,
                "source_original_path": str(candidate.original_path),
                "source_copied_image_path": str(image_path),
                "image_filename": crop_image_path.name,
                "annotation_status": "prelabeled_pending_manual_review",
                "manual_review": True,
                "manual_review_reasons": sorted(
                    set(["all_pilot_prelabels_require_review"] + crop_issues)
                ),
                "semantic_bucket": candidate.semantic_bucket,
                "helmet_color": candidate.helmet_color,
                "crop_transform": crop.metadata(),
                "source_person_detection": person.to_dict(),
                "labels": [item.to_dict() for item in crop_labels],
                "training_eligibility": (
                    "full_three_class_candidate"
                    if crop_counts.get("head", 0) or crop_counts.get("helmet", 0)
                    else "person_only_or_manual_review"
                ),
            }
            write_annotation_json(crop_annotation_path, crop_annotation)
            write_preview(
                crop_preview_path,
                crop.image,
                crop_labels,
                f"#{index:02d}.{person_index:02d} head-shoulder crop",
            )
            crop_preview_paths.append(crop_preview_path)
            crop_manifest_rows.append(
                {
                    "sample_index": len(crop_manifest_rows) + 1,
                    "source_sample_index": index,
                    "event_id": crop_id,
                    "source_event_id": candidate.event_id,
                    "semantic_bucket": candidate.semantic_bucket,
                    "review_verdict": str(review.get("verdict") or ""),
                    "review_summary": str(review.get("summary") or ""),
                    "image_path": str(crop_image_path),
                    "label_path": str(crop_label_path),
                    "annotation_path": str(crop_annotation_path),
                    "preview_path": str(crop_preview_path),
                    "label_counts": dict(crop_counts),
                    "issues": crop_issues,
                    "manual_review": True,
                    "training_eligibility": crop_annotation["training_eligibility"],
                    "crop_transform": crop.metadata(),
                }
            )

    write_jsonl(output_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(crops_dir / "manifest.jsonl", crop_manifest_rows)
    write_jsonl(output_dir / "rejected_quality_manifest.jsonl", rejected_rows)
    (output_dir / "classes.txt").write_text("person\nhead\nhelmet\n", encoding="ascii")
    (output_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: person\n  1: head\n  2: helmet\n",
        encoding="utf-8",
    )
    (crops_dir / "classes.txt").write_text("person\nhead\nhelmet\n", encoding="ascii")
    (crops_dir / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: person\n  1: head\n  2: helmet\n",
        encoding="utf-8",
    )
    contact_sheets = create_contact_sheets(preview_paths, output_dir)
    crop_contact_sheets = create_contact_sheets(crop_preview_paths, crops_dir)
    summary = {
        "schema_version": config["schema_version"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_alarm_root": str(args.alarm_root),
        "source_candidates": len(candidates),
        "quality_counts": dict(quality_counts),
        "near_duplicate_skips_during_sampling": int(dedup_skips["near_duplicate"]),
        "selected_count": len(manifest_rows),
        "selected_semantic_buckets": dict(Counter(row["semantic_bucket"] for row in manifest_rows)),
        "selected_review_verdicts": dict(Counter(row["review_verdict"] or "missing" for row in manifest_rows)),
        "prelabel_counts": dict(label_counts),
        "prelabel_issue_counts": dict(issue_counts),
        "full_three_class_candidate_count": sum(
            row["training_eligibility"] == "full_three_class_candidate" for row in manifest_rows
        ),
        "person_only_or_manual_review_count": sum(
            row["training_eligibility"] == "person_only_or_manual_review" for row in manifest_rows
        ),
        "known_non_person_sample_count": sum(bool(row["non_person_override"]) for row in manifest_rows),
        "head_shoulder_crop_count": len(crop_manifest_rows),
        "head_shoulder_crop_prelabel_counts": dict(crop_label_counts),
        "head_shoulder_crop_issue_counts": dict(crop_issue_counts),
        "head_shoulder_full_three_class_candidate_count": sum(
            row["training_eligibility"] == "full_three_class_candidate"
            for row in crop_manifest_rows
        ),
        "training_readiness": "not_ready_until_manual_review_completed",
        "head_size_rule": "以等比例 letterbox 到 640x640 后计算；head/helmet 短边小于20像素不标注。",
        "camera_limitation": "当前安全帽告警候选均来自 Camera 01，试标结果不能代表多摄像头泛化能力。",
        "status": "prelabeled_pending_manual_review",
    }
    write_annotation_json(output_dir / "summary.json", summary)
    build_html_report(output_dir, summary, manifest_rows, contact_sheets)
    crop_summary = {
        "schema_version": config["schema_version"],
        "source_output": str(output_dir),
        "crop_count": len(crop_manifest_rows),
        "prelabel_counts": dict(crop_label_counts),
        "issue_counts": dict(crop_issue_counts),
        "full_three_class_candidate_count": sum(
            row["training_eligibility"] == "full_three_class_candidate"
            for row in crop_manifest_rows
        ),
        "person_only_pending_head_status_review_count": sum(
            row["training_eligibility"] == "person_only_or_manual_review"
            for row in crop_manifest_rows
        ),
        "training_readiness": "not_ready_until_manual_review_completed",
        "status": "prelabeled_pending_manual_review",
    }
    write_annotation_json(crops_dir / "summary.json", crop_summary)
    build_html_report(
        crops_dir,
        crop_summary,
        crop_manifest_rows,
        crop_contact_sheets,
        title="安全帽人员头肩裁剪试标",
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从告警原图创建安全帽50张预标注试验集")
    parser.add_argument("--alarm-root", type=Path, default=PROJECT_ROOT / "alarm_outputs")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    destination = run(parse_args())
    print(destination)
