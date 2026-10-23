"""Recheck the explicitly selected seven alarms and update their platform records."""
import json
import shutil
import sqlite3
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import config
from offline_alarm_review import (AlarmCandidate, PROMPT_VERSION, call_review_model,
    load_prompt_bundle, extract_keyframes, create_keyframe_contact_sheet, create_review_target_crop)
from video_alarms import VideoAlarmStore


def main():
    root = Path(__file__).resolve().parent
    batch = root / 'review_outputs' / ('recheck_20260915_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    batch.mkdir(parents=True)
    def save(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    with sqlite3.connect(root / 'data/platform.db') as con:
        con.row_factory = sqlite3.Row
        rows = [dict(r) for r in con.execute("SELECT * FROM video_alarms WHERE channel_id=29 AND alarm_type=(SELECT alarm_type FROM video_alarms WHERE id=10361) AND alarm_time >= '2026-09-15T15:22:54' AND id<=10407 ORDER BY alarm_time")]
    assert len(rows) == 7, 'Selection changed; inspect before proceeding'
    save(batch / 'original_records.json', rows)
    with urllib.request.urlopen('http://127.0.0.1:8010/v1/models') as response:
        model = json.load(response)['data'][0]['id']
    system, prompts, schemas = load_prompt_bundle(Path(config.ALARM_REVIEW_PROMPT_DIR))
    store = VideoAlarmStore()
    results = []
    print('BATCH=' + str(batch), flush=True)
    for row in rows:
        event_id = row['event_id']
        metadata_path = Path(row['metadata_path'])
        event_dir = metadata_path.parent
        backup = batch / event_id
        backup.mkdir()
        shutil.copy2(metadata_path, backup / 'metadata.json')
        review_dir = event_dir / 'vlm_review'
        if review_dir.exists():
            shutil.copytree(review_dir, backup / 'vlm_review')
        work = backup / 'new_review'
        work.mkdir()
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        candidate = AlarmCandidate(event_id, event_dir, metadata_path, metadata)
        started = time.monotonic()
        frames = extract_keyframes(event_dir / 'alarm_clip.mp4', work / 'keyframes', 5)
        sheet = create_keyframe_contact_sheet(frames, work / 'keyframes_contact.jpg')
        crop = create_review_target_crop(candidate, work / 'review_target_crop.jpg')
        review, raw, usage = call_review_model(candidate, frames, sheet, system, prompts, schemas,
            str(config.ALARM_REVIEW_SERVER_URL), 240, int(config.ALARM_REVIEW_MAX_TOKENS), crop, model)
        result = {'event_id': event_id, 'status': 'completed', 'model': model,
            'prompt_version': PROMPT_VERSION, 'latency_seconds': round(time.monotonic()-started, 3),
            'usage': usage, 'review': review,
            'keyframes': [{k:v for k,v in f.items() if k!='path'} for f in frames]}
        save(work / 'review.json', result)
        (work / 'raw_response.txt').write_text(raw, encoding='utf-8')
        review_dir.mkdir(exist_ok=True)
        for name in ('review.json', 'raw_response.txt', 'keyframes_contact.jpg', 'review_target_crop.jpg'):
            shutil.copy2(work / name, review_dir / name)
        store.complete_review(event_id, review, model=model, prompt_version=PROMPT_VERSION)
        with sqlite3.connect(root / 'data/platform.db') as con:
            con.execute('UPDATE video_alarms SET review_attempts=review_attempts+1 WHERE id=?', (row['id'],))
            actual = con.execute('SELECT review_status,review_summary,review_prompt_version FROM video_alarms WHERE id=?', (row['id'],)).fetchone()
        assert actual == (review['verdict'], review['summary'][:2000], PROMPT_VERSION)
        updated = json.loads(metadata_path.read_text(encoding='utf-8'))
        assert updated['review_status'] == review['verdict']
        results.append({'id':row['id'], 'alarm_time':row['alarm_time'], 'before':row['review_status'], 'after': review['verdict'], 'summary':review['summary']})
        save(batch / 'results.json', results)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    print('VERIFIED=' + str(len(results)), flush=True)


if __name__ == '__main__':
    main()
