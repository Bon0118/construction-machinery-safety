"""Image evidence for body translation; detection-box movement alone is insufficient."""
from collections import deque
from dataclasses import dataclass, field
import math

import cv2
import numpy as np

from . import defaults as cfg


@dataclass
class BodyVisualEvidence:
    samples: deque = field(default_factory=deque)
    reliable: bool = False
    support: float = 0.0
    cells: int = 0
    points: int = 0
    last_at: float = -1.0

    def clear(self):
        self.samples.clear()
        self.reliable = False
        self.support = 0.0
        self.cells = self.points = 0
        self.last_at = -1.0

    def observe(self, previous, aligned, box, persons, timestamp, camera_reliable):
        self.last_at = timestamp
        self.reliable = False
        self.support = 0.0
        self.cells = self.points = 0
        if not camera_reliable:
            self.clear()
            return
        height, width = previous.shape
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        if min(x2 - x1, y2 - y1) < 16:
            self.clear()
            return
        a, b = previous[y1:y2, x1:x2], aligned[y1:y2, x1:x2]
        scale = min(1.0, 240.0 / max(a.shape))
        if scale < 1:
            size = (max(16, round(a.shape[1] * scale)), max(16, round(a.shape[0] * scale)))
            a, b = cv2.resize(a, size), cv2.resize(b, size)
        mask = np.full(a.shape, 255, np.uint8)
        mask[:3, :] = mask[-3:, :] = 0
        mask[:, :3] = mask[:, -3:] = 0
        for px1, py1, px2, py2 in persons:
            left, top = max(0, int((px1-x1)*scale)-3), max(0, int((py1-y1)*scale)-3)
            right, bottom = min(mask.shape[1], int((px2-x1)*scale)+3), min(mask.shape[0], int((py2-y1)*scale)+3)
            if right > left and bottom > top:
                mask[top:bottom, left:right] = 0
        points = cv2.goodFeaturesToTrack(a, 100, 0.02, 5, mask=mask)
        if points is None or len(points) < cfg.BODY_VISUAL_MIN_POINTS:
            self.clear()
            return
        forward, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None)
        if forward is None or status is None:
            self.clear()
            return
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(b, a, forward, None)
        if backward is None or back_status is None:
            self.clear()
            return
        p, q, r = points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
        valid = (status.ravel() == 1) & (back_status.ravel() == 1)
        valid &= np.isfinite(q).all(axis=1) & (np.linalg.norm(p-r, axis=1) < 0.8)
        p, q = p[valid], q[valid]
        self.points = len(p)
        if len(p) < cfg.BODY_VISUAL_MIN_POINTS:
            self.clear()
            return
        self.reliable = True
        vectors = (q-p) / scale
        median = np.median(vectors, axis=0)
        magnitude = float(np.linalg.norm(median))
        consistent = np.linalg.norm(vectors-median, axis=1) <= max(0.5/scale, magnitude*0.5)
        moving = np.linalg.norm(vectors, axis=1) >= cfg.BODY_VISUAL_MIN_STEP_PIXELS
        consistent &= moving
        self.support = float(np.mean(consistent))
        coords = p[consistent]
        cells = {(min(2, int(x*3/a.shape[1])), min(2, int(y*3/a.shape[0]))) for x, y in coords}
        self.cells = len(cells)
        diagonal = max(1.0, math.hypot(x2-x1, y2-y1))
        accepted = (magnitude >= cfg.BODY_VISUAL_MIN_STEP_PIXELS
                    and self.support >= cfg.BODY_VISUAL_MIN_SUPPORT
                    and self.cells >= cfg.BODY_VISUAL_MIN_CELLS)
        self.samples.append((timestamp, float(median[0])/diagonal,
                             float(median[1])/diagonal, accepted))
        self._expire(timestamp)

    def _expire(self, timestamp):
        while self.samples and timestamp-self.samples[0][0] > cfg.MACHINERY_MOTION_WINDOW_SECONDS:
            self.samples.popleft()

    def confirms(self, timestamp, direction):
        self._expire(timestamp)
        if not self.reliable or timestamp-self.last_at > cfg.BODY_VISUAL_MAX_AGE_SECONDS:
            return False
        recent = [s for s in self.samples if s[3]]
        if len(recent) < cfg.BODY_VISUAL_MIN_VOTES:
            return False
        # Require current evidence too, instead of reusing old motion indefinitely.
        if not self.samples or not self.samples[-1][3]:
            return False
        dx, dy = sum(s[1] for s in recent), sum(s[2] for s in recent)
        net = math.hypot(dx, dy)
        travel = sum(math.hypot(s[1], s[2]) for s in recent)
        dot = dx*direction[0] + dy*direction[1]
        return (net >= cfg.BODY_VISUAL_MIN_NET_RATIO
                and net/max(travel, 1e-9) >= cfg.MACHINERY_DIRECTION_CONSISTENCY
                and dot/max(net, 1e-9) >= 0.5)
