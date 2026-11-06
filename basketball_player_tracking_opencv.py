import cv2
import numpy as np
from collections import deque


# ============================================================
# Improved Basketball + Player Tracking using only OpenCV + NumPy
# ------------------------------------------------------------
# Main upgrades over the earlier version:
# 1. Court-region gating to suppress bench/crowd detections.
# 2. Multi-cue ball scoring: orange color + motion + local edge energy
#    + circularity + proximity to previous predicted state.
# 3. Ball search uses BOTH global candidates and local ROI candidates.
# 4. Better player tracking with size-aware association.
# 5. Team assignment only on confirmed on-court tracks.
# ============================================================

VIDEO_PATH = "input.mp4"
OUTPUT_PATH = "output_tracked_v2.mp4"
DISPLAY = True

FONT = cv2.FONT_HERSHEY_SIMPLEX

# -----------------------------
# Tunable parameters
# -----------------------------
MIN_PLAYER_AREA = 850
MAX_PLAYER_AREA = 22000
PLAYER_MIN_W = 16
PLAYER_MIN_H = 55
PLAYER_MAX_W = 150
PLAYER_MAX_H = 300
PLAYER_MAX_MATCH_DIST = 85
PLAYER_MAX_MISSED = 16
PLAYER_CONFIRM_FRAMES = 3

BALL_MIN_AREA = 18
BALL_MAX_AREA = 380
BALL_MAX_MATCH_DIST = 95
BALL_MAX_MISSED = 8
BALL_TRAIL = 20

TEAM_HIST_BINS = 18
TEAM_ASSIGN_MIN_SAT = 45
TEAM_ASSIGN_MIN_VAL = 45

# Suppress bench / crowd / broadcast graphics
COURT_TOP_FRAC = 0.22
COURT_BOTTOM_FRAC = 0.93
COURT_LEFT_FRAC = 0.01
COURT_RIGHT_FRAC = 0.99

# Ball search ROI size around predicted center
BALL_LOCAL_SEARCH_RADIUS = 110


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def center_of(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


def box_from_rect(x, y, w, h):
    return (int(x), int(y), int(x + w), int(y + h))


def crop_safe(img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = clamp(int(x1), 0, w - 1)
    y1 = clamp(int(y1), 0, h - 1)
    x2 = clamp(int(x2), 0, w)
    y2 = clamp(int(y2), 0, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def circularity(cnt):
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (peri * peri))


def draw_label(img, text, org, color=(255, 255, 255), bg=(0, 0, 0)):
    (tw, th), baseline = cv2.getTextSize(text, FONT, 0.5, 1)
    x, y = org
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 8, y + 4), bg, -1)
    cv2.putText(img, text, (x + 4, y - 3), FONT, 0.5, color, 1, cv2.LINE_AA)


def within_court_region(box, frame_shape):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return (
        COURT_LEFT_FRAC * w <= cx <= COURT_RIGHT_FRAC * w
        and COURT_TOP_FRAC * h <= cy <= COURT_BOTTOM_FRAC * h
    )


def point_in_any_box(pt, boxes, margin=0):
    x, y = pt
    for b in boxes:
        x1, y1, x2, y2 = b
        if (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin):
            return True
    return False


def bottom_anchor(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, y2], dtype=np.float32)


# ------------------------------------------------------------
# Track classes
# ------------------------------------------------------------
class PlayerTrack:
    def __init__(self, tid, box, frame_idx):
        self.id = tid
        self.box = box
        self.center = center_of(box)
        self.anchor = bottom_anchor(box)
        self.prev_center = self.center.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_seen = frame_idx
        self.missed = 0
        self.hits = 1
        self.confirmed = False
        self.team = -1
        self.team_votes = []

    def predict_anchor(self):
        return self.anchor + self.velocity

    def update(self, box, frame_idx, team=-1):
        new_center = center_of(box)
        new_anchor = bottom_anchor(box)
        self.velocity = new_anchor - self.anchor
        self.prev_center = self.center.copy()
        self.center = new_center
        self.anchor = new_anchor
        self.box = box
        self.last_seen = frame_idx
        self.missed = 0
        self.hits += 1
        if self.hits >= PLAYER_CONFIRM_FRAMES:
            self.confirmed = True

        if team != -1:
            self.team_votes.append(team)
            if len(self.team_votes) > 24:
                self.team_votes.pop(0)
            vals, counts = np.unique(np.array(self.team_votes), return_counts=True)
            self.team = int(vals[np.argmax(counts)])

    def mark_missed(self):
        self.missed += 1
        self.anchor = self.anchor + 0.65 * self.velocity
        self.center = self.center + 0.65 * self.velocity


class BallTrack:
    def __init__(self, box, frame_idx, confidence=1.0):
        self.box = box
        self.center = center_of(box)
        self.prev_center = self.center.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_seen = frame_idx
        self.missed = 0
        self.hits = 1
        self.confidence = confidence
        self.trail = deque(maxlen=BALL_TRAIL)
        self.trail.append(tuple(self.center.astype(int)))

    def predict(self):
        return self.center + self.velocity

    def update(self, box, frame_idx, confidence=1.0):
        new_center = center_of(box)
        raw_velocity = new_center - self.center
        self.velocity = 0.55 * self.velocity + 0.45 * raw_velocity
        self.prev_center = self.center.copy()
        self.center = new_center
        self.box = box
        self.last_seen = frame_idx
        self.missed = 0
        self.hits += 1
        self.confidence = confidence
        self.trail.append(tuple(self.center.astype(int)))

    def mark_missed(self):
        self.missed += 1
        self.center = self.center + 0.85 * self.velocity
        self.trail.append(tuple(self.center.astype(int)))
        self.confidence *= 0.85


# ------------------------------------------------------------
# Player detector
# ------------------------------------------------------------
def detect_players(frame, fgmask):
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    mask = cv2.medianBlur(fgmask, 5)
    _, mask = cv2.threshold(mask, 190, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel7)
    mask = cv2.dilate(mask, kernel5, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_PLAYER_AREA or area > MAX_PLAYER_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < PLAYER_MIN_W or h < PLAYER_MIN_H:
            continue
        if w > PLAYER_MAX_W or h > PLAYER_MAX_H:
            continue

        aspect = h / float(max(w, 1))
        if aspect < 1.15 or aspect > 6.5:
            continue

        box = box_from_rect(x, y, w, h)
        if not within_court_region(box, frame.shape):
            continue

        detections.append(box)

    return detections, mask


# ------------------------------------------------------------
# Team color estimation
# ------------------------------------------------------------
def jersey_patch(frame, box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    px1 = x1 + int(0.20 * w)
    px2 = x1 + int(0.80 * w)
    py1 = y1 + int(0.18 * h)
    py2 = y1 + int(0.52 * h)
    return crop_safe(frame, (px1, py1, px2, py2))


def hue_hist_feature(frame, box):
    patch = jersey_patch(frame, box)
    if patch is None or patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    valid = (s > TEAM_ASSIGN_MIN_SAT) & (v > TEAM_ASSIGN_MIN_VAL)
    if np.count_nonzero(valid) < 20:
        return None

    hist = cv2.calcHist([h], [0], valid.astype(np.uint8), [TEAM_HIST_BINS], [0, 180])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total <= 1e-6:
        return None
    hist /= total
    return hist


def assign_teams_from_tracks(frame, tracks):
    feats, ids = [], []
    for t in tracks:
        if not t.confirmed or t.missed > 2:
            continue
        if not within_court_region(t.box, frame.shape):
            continue
        feat = hue_hist_feature(frame, t.box)
        if feat is None:
            continue
        feats.append(feat)
        ids.append(t.id)

    if len(feats) < 2:
        return {}

    Z = np.array(feats, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, _ = cv2.kmeans(Z, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    mapping = {}
    for i, tid in enumerate(ids):
        mapping[tid] = int(labels[i, 0])
    return mapping


# ------------------------------------------------------------
# Ball detection helpers
# ------------------------------------------------------------
def orange_ball_mask(hsv):
    lower1 = np.array([4, 85, 70], dtype=np.uint8)
    upper1 = np.array([24, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower1, upper1)
    return mask


def ball_roi_box(pred_center, frame_shape, radius=BALL_LOCAL_SEARCH_RADIUS):
    h, w = frame_shape[:2]
    cx, cy = pred_center
    x1 = clamp(int(cx - radius), 0, w - 1)
    y1 = clamp(int(cy - radius), 0, h - 1)
    x2 = clamp(int(cx + radius), 0, w)
    y2 = clamp(int(cy + radius), 0, h)
    return (x1, y1, x2, y2)


def edge_energy(gray, box):
    patch = crop_safe(gray, box)
    if patch is None or patch.size == 0:
        return 0.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def orange_ratio(hsv, box):
    patch = crop_safe(hsv, box)
    if patch is None or patch.size == 0:
        return 0.0
    mask = orange_ball_mask(patch)
    return float(np.count_nonzero(mask)) / float(mask.size)


def motion_ratio(motion_mask, box):
    patch = crop_safe(motion_mask, box)
    if patch is None or patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def is_near_player(box, player_boxes, margin=8):
    cx, cy = center_of(box)
    return point_in_any_box((cx, cy), player_boxes, margin=margin)


def build_ball_candidate_mask(frame, fgmask, roi_box=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, motion = cv2.threshold(fgmask, 190, 255, cv2.THRESH_BINARY)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, k3)
    motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, k5)

    orange = orange_ball_mask(hsv)
    orange = cv2.medianBlur(orange, 5)

    edges = cv2.Canny(gray, 70, 150)
    edges = cv2.dilate(edges, k3, iterations=1)

    combined = cv2.bitwise_and(orange, motion)
    combined = cv2.bitwise_or(combined, cv2.bitwise_and(orange, edges))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, k3)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k5)

    if roi_box is not None:
        roi_mask = np.zeros_like(combined)
        x1, y1, x2, y2 = roi_box
        roi_mask[y1:y2, x1:x2] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

    return combined, hsv, gray, motion


def extract_ball_candidates(frame, fgmask, player_boxes, roi_box=None):
    cand_mask, hsv, gray, motion = build_ball_candidate_mask(frame, fgmask, roi_box=roi_box)

    contours, _ = cv2.findContours(cand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < BALL_MIN_AREA or area > BALL_MAX_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / float(max(h, 1))
        if ratio < 0.55 or ratio > 1.75:
            continue

        circ = circularity(cnt)
        if circ < 0.32:
            continue

        box = box_from_rect(x, y, w, h)
        if not within_court_region(box, frame.shape):
            continue

        # Do not hard reject if near a player; just penalize later.
        cand = {
            "box": box,
            "area": float(area),
            "circ": float(circ),
            "orange_ratio": orange_ratio(hsv, box),
            "motion_ratio": motion_ratio(motion, box),
            "edge_energy": edge_energy(gray, box),
            "near_player": is_near_player(box, player_boxes, margin=10),
        }
        candidates.append(cand)

    return candidates, cand_mask


def score_ball_candidate(cand, ball_track):
    box = cand["box"]
    center = center_of(box)
    area = cand["area"]
    circ = cand["circ"]
    orange_r = cand["orange_ratio"]
    motion_r = cand["motion_ratio"]
    edge_e = cand["edge_energy"]
    near_player = cand["near_player"]

    # Preferred basketball blob area in this broadcast scale is often around this range.
    area_penalty = abs(area - 95.0) / 95.0
    circ_penalty = 1.0 - min(circ, 1.0)
    orange_penalty = max(0.0, 0.18 - orange_r) * 3.5
    motion_penalty = max(0.0, 0.10 - motion_r) * 2.5
    edge_penalty = max(0.0, 16.0 - edge_e) / 16.0
    player_penalty = 0.16 if near_player else 0.0

    score = 1.0 * area_penalty + 1.4 * circ_penalty + orange_penalty + motion_penalty + 0.5 * edge_penalty + player_penalty

    if ball_track is not None:
        pred = ball_track.predict()
        dist = float(np.linalg.norm(pred - center))
        score += 0.015 * dist

        # Reward motion continuity a bit.
        vel = center - ball_track.center
        if np.linalg.norm(ball_track.velocity) > 1.0 and np.linalg.norm(vel) > 1.0:
            dir1 = ball_track.velocity / (np.linalg.norm(ball_track.velocity) + 1e-6)
            dir2 = vel / (np.linalg.norm(vel) + 1e-6)
            alignment = float(np.dot(dir1, dir2))
            score += 0.18 * (1.0 - alignment)

    return score


def choose_best_ball_candidate(global_cands, local_cands, ball_track):
    merged = []

    for c in local_cands:
        c2 = dict(c)
        c2["source_bonus"] = -0.10
        merged.append(c2)
    for c in global_cands:
        c2 = dict(c)
        c2["source_bonus"] = 0.0
        merged.append(c2)

    if not merged:
        return None, None

    best = None
    best_score = 1e18
    for cand in merged:
        score = score_ball_candidate(cand, ball_track) + cand["source_bonus"]
        cand["final_score"] = score
        if score < best_score:
            best_score = score
            best = cand

    return best, best_score


# ------------------------------------------------------------
# Player association
# ------------------------------------------------------------
def match_player_tracks(tracks, detections):
    matches = []
    unmatched_tracks = set(range(len(tracks)))
    unmatched_dets = set(range(len(detections)))

    if len(tracks) == 0 or len(detections) == 0:
        return matches, unmatched_tracks, unmatched_dets

    pairs = []
    for ti, tr in enumerate(tracks):
        pred_anchor = tr.predict_anchor()
        old_area = box_area(tr.box)
        for di, det in enumerate(detections):
            det_anchor = bottom_anchor(det)
            dist = float(np.linalg.norm(pred_anchor - det_anchor))
            ov = iou(tr.box, det)
            new_area = box_area(det)
            size_change = abs(new_area - old_area) / float(max(old_area, 1))
            cost = dist - 35.0 * ov + 18.0 * size_change
            pairs.append((cost, dist, ti, di))

    pairs.sort(key=lambda x: x[0])
    used_t, used_d = set(), set()

    for _, dist, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        if dist > PLAYER_MAX_MATCH_DIST:
            continue
        matches.append((ti, di))
        used_t.add(ti)
        used_d.add(di)

    unmatched_tracks -= used_t
    unmatched_dets -= used_d
    return matches, unmatched_tracks, unmatched_dets


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------
def team_color(team):
    if team == 0:
        return (255, 120, 0)
    if team == 1:
        return (0, 220, 255)
    return (200, 200, 200)


def draw_players(frame, tracks):
    team_counts = {0: 0, 1: 0}

    for tr in tracks:
        if not tr.confirmed or tr.missed > PLAYER_MAX_MISSED:
            continue
        if not within_court_region(tr.box, frame.shape):
            continue

        x1, y1, x2, y2 = map(int, tr.box)
        color = team_color(tr.team)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID {tr.id}"
        if tr.team != -1:
            label += f" T{tr.team + 1}"
            team_counts[tr.team] += 1

        draw_label(frame, label, (x1, y1), color=(255, 255, 255), bg=color)
        cx, cy = map(int, tr.center)
        cv2.circle(frame, (cx, cy), 3, color, -1)

    return team_counts


def draw_ball(frame, ball_track):
    if ball_track is None or ball_track.missed > BALL_MAX_MISSED:
        return

    x1, y1, x2, y2 = map(int, ball_track.box)
    cx, cy = map(int, ball_track.center)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
    cv2.circle(frame, (cx, cy), 5, (0, 140, 255), -1)
    draw_label(frame, f"BALL {ball_track.confidence:.2f}", (x1, y1), color=(255, 255, 255), bg=(0, 140, 255))

    pts = list(ball_track.trail)
    for i in range(1, len(pts)):
        p0, p1 = pts[i - 1], pts[i]
        thickness = max(1, int(5 - 4 * (i / max(len(pts), 1))))
        cv2.line(frame, p0, p1, (0, 165, 255), thickness)


def draw_overlay(frame, frame_idx, team_counts):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 100), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    cv2.putText(frame, f"Frame: {frame_idx}", (20, 32), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Team 1 Count: {team_counts.get(0, 0)}", (20, 60), FONT, 0.65, (255, 120, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Team 2 Count: {team_counts.get(1, 0)}", (20, 88), FONT, 0.65, (0, 220, 255), 2, cv2.LINE_AA)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    bgsub = cv2.createBackgroundSubtractorMOG2(history=800, varThreshold=28, detectShadows=True)

    player_tracks = []
    next_player_id = 1
    ball_track = None

    frame_idx = 0
    warmup_frames = 25

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        fgmask = bgsub.apply(frame_blur)

        if frame_idx <= warmup_frames:
            writer.write(frame)
            if DISPLAY:
                cv2.imshow("tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
            continue

        # --------------------
        # Player detection / tracking
        # --------------------
        player_dets, _ = detect_players(frame_blur, fgmask)
        matches, unmatched_tracks, unmatched_dets = match_player_tracks(player_tracks, player_dets)

        for ti, di in matches:
            player_tracks[ti].update(player_dets[di], frame_idx)

        for ti in unmatched_tracks:
            player_tracks[ti].mark_missed()

        for di in unmatched_dets:
            tr = PlayerTrack(next_player_id, player_dets[di], frame_idx)
            player_tracks.append(tr)
            next_player_id += 1

        player_tracks = [t for t in player_tracks if t.missed <= PLAYER_MAX_MISSED]

        # Team labeling only after track formation
        team_map = assign_teams_from_tracks(frame, player_tracks)
        for tr in player_tracks:
            if tr.id in team_map:
                tr.update(tr.box, frame_idx, team=team_map[tr.id])

        active_player_boxes = [t.box for t in player_tracks if t.confirmed or t.hits >= 2]

        # --------------------
        # Ball detection / tracking
        # --------------------
        local_candidates = []
        local_mask = None
        if ball_track is not None and ball_track.missed <= BALL_MAX_MISSED:
            roi = ball_roi_box(ball_track.predict(), frame.shape, BALL_LOCAL_SEARCH_RADIUS)
            local_candidates, local_mask = extract_ball_candidates(frame_blur, fgmask, active_player_boxes, roi_box=roi)

        global_candidates, global_mask = extract_ball_candidates(frame_blur, fgmask, active_player_boxes, roi_box=None)
        best_cand, best_score = choose_best_ball_candidate(global_candidates, local_candidates, ball_track)

        if best_cand is not None:
            best_box = best_cand["box"]
            confidence = float(max(0.0, 1.4 - best_score))
            if ball_track is None:
                ball_track = BallTrack(best_box, frame_idx, confidence=confidence)
            else:
                # Reject wild jumps if confidence is weak.
                pred = ball_track.predict()
                jump = float(np.linalg.norm(center_of(best_box) - pred))
                if jump <= BALL_MAX_MATCH_DIST + 20 or confidence > 0.72:
                    ball_track.update(best_box, frame_idx, confidence=confidence)
                else:
                    ball_track.mark_missed()
        else:
            if ball_track is not None:
                ball_track.mark_missed()
                if ball_track.missed > BALL_MAX_MISSED:
                    ball_track = None

        # --------------------
        # Draw
        # --------------------
        out = frame.copy()
        team_counts = draw_players(out, player_tracks)
        draw_ball(out, ball_track)
        draw_overlay(out, frame_idx, team_counts)

        writer.write(out)

        if DISPLAY:
            cv2.imshow("tracking", out)
            # Uncomment for debugging:
            # cv2.imshow("global_ball_mask", global_mask)
            # if local_mask is not None:
            #     cv2.imshow("local_ball_mask", local_mask)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Player + ball tracking with OpenCV and NumPy only.")
    ap.add_argument("--input", default=VIDEO_PATH)
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--no-display", action="store_true", help="don't open a preview window")
    args = ap.parse_args()
    VIDEO_PATH, OUTPUT_PATH = args.input, args.output
    DISPLAY = not args.no_display
    main()
