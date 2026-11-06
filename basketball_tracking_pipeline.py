import cv2
import numpy as np
from collections import deque
import math
import argparse


# --------------------------- Utility helpers ---------------------------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / float(area_a + area_b - inter)


def center_xyxy(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0], dtype=np.float32)


def bbox_from_contour(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return [x, y, x + w, y + h]


def circularity(cnt):
    area = cv2.contourArea(cnt)
    per = cv2.arcLength(cnt, True)
    if per <= 1e-6:
        return 0.0
    return (4.0 * math.pi * area) / (per * per)


def dominant_team_from_roi(frame_bgr, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = clamp(x1, 0, w - 1), clamp(y1, 0, h - 1), clamp(x2, 1, w), clamp(y2, 1, h)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, 0.0

    # Focus on torso region to reduce skin / shorts / court leakage.
    hh, ww = roi.shape[:2]
    tx1 = int(0.2 * ww)
    tx2 = int(0.8 * ww)
    ty1 = int(0.15 * hh)
    ty2 = int(0.55 * hh)
    torso = roi[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        torso = roi

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Tuned for this broadcast: yellow Lakers vs blue Warriors.
    yellow = cv2.inRange(hsv, (15, 70, 90), (40, 255, 255))
    blue = cv2.inRange(hsv, (90, 70, 60), (140, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 170), (180, 70, 255))

    total = torso.shape[0] * torso.shape[1]
    if total == 0:
        return None, 0.0

    yellow_score = (np.count_nonzero(yellow) + 0.25 * np.count_nonzero(white)) / total
    blue_score = (np.count_nonzero(blue) + 0.10 * np.count_nonzero(white)) / total

    if max(yellow_score, blue_score) < 0.05:
        return None, max(yellow_score, blue_score)
    return ("Team 1" if yellow_score >= blue_score else "Team 2"), max(yellow_score, blue_score)


# --------------------------- Player tracker ---------------------------

class Track:
    def __init__(self, track_id, box, frame_idx):
        self.id = track_id
        self.box = np.array(box, dtype=np.float32)
        self.last_box = np.array(box, dtype=np.float32)
        self.center = center_xyxy(box)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.hits = 1
        self.misses = 0
        self.last_seen = frame_idx
        self.history = deque(maxlen=24)
        self.history.append(self.center.copy())
        self.team_votes = {"Team 1": 0.0, "Team 2": 0.0}
        self.team = None

    def predict(self):
        pred_center = self.center + self.velocity
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        return np.array([pred_center[0] - w / 2, pred_center[1] - h / 2,
                         pred_center[0] + w / 2, pred_center[1] + h / 2], dtype=np.float32)

    def update(self, box, frame_idx):
        self.last_box = self.box.copy()
        new_center = center_xyxy(box)
        self.velocity = 0.7 * self.velocity + 0.3 * (new_center - self.center)
        self.box = np.array(box, dtype=np.float32)
        self.center = new_center
        self.hits += 1
        self.misses = 0
        self.last_seen = frame_idx
        self.history.append(self.center.copy())

    def mark_missed(self):
        self.misses += 1
        self.center = self.center + self.velocity
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        self.box = np.array([self.center[0] - w / 2, self.center[1] - h / 2,
                             self.center[0] + w / 2, self.center[1] + h / 2], dtype=np.float32)
        self.history.append(self.center.copy())

    def is_confirmed(self):
        return self.hits >= 3 and self.misses <= 8


class MultiObjectTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, frame_idx):
        assigned_tracks = set()
        assigned_dets = set()

        if self.tracks and detections:
            pairs = []
            for ti, track in enumerate(self.tracks):
                pred = track.predict()
                pc = center_xyxy(pred)
                for di, det in enumerate(detections):
                    i = iou_xyxy(pred, det)
                    dc = center_xyxy(det)
                    dist = np.linalg.norm(pc - dc)
                    score = i - 0.0015 * dist
                    pairs.append((score, ti, di, i, dist))
            pairs.sort(reverse=True, key=lambda x: x[0])

            for score, ti, di, i, dist in pairs:
                if ti in assigned_tracks or di in assigned_dets:
                    continue
                # Relax matching when motion is moderate.
                if i >= 0.08 or dist <= 45:
                    self.tracks[ti].update(detections[di], frame_idx)
                    assigned_tracks.add(ti)
                    assigned_dets.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in assigned_tracks:
                track.mark_missed()

        for di, det in enumerate(detections):
            if di not in assigned_dets:
                self.tracks.append(Track(self.next_id, det, frame_idx))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.misses <= 15]
        return self.tracks


# --------------------------- Ball tracker ---------------------------

class BallTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.8
        self.initialized = False
        self.missed = 0
        self.last_pos = None
        self.trail = deque(maxlen=30)

    def predict(self):
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return np.array([pred[0, 0], pred[1, 0]], dtype=np.float32)

    def correct(self, pt):
        m = np.array([[np.float32(pt[0])], [np.float32(pt[1])]])
        if not self.initialized:
            self.kf.statePre = np.array([[pt[0]], [pt[1]], [0], [0]], dtype=np.float32)
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        self.kf.correct(m)
        self.last_pos = np.array(pt, dtype=np.float32)
        self.trail.append(tuple(map(int, pt)))
        self.missed = 0

    def mark_missed(self):
        self.missed += 1
        pred = self.predict()
        if pred is not None:
            self.last_pos = pred
            self.trail.append(tuple(map(int, pred)))


def detect_players(frame_bgr, fgmask, court_mask):
    h, w = frame_bgr.shape[:2]
    mask = cv2.bitwise_and(fgmask, court_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 120:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / max(1.0, float(bw))
        # Broadcast player blobs are roughly vertical; let partial occlusions pass.
        if bh < 22 or bw < 8 or bh > 180 or bw > 90:
            continue
        if aspect < 1.0 or aspect > 5.5:
            continue
        # Ignore sideline crowd and scorebug edges.
        if y < 70 or y + bh > h - 18:
            continue
        detections.append([x, y, x + bw, y + bh])
    return detections, mask


def detect_ball(frame_bgr, fgmask, court_mask, player_tracks, predicted_ball):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Orange basketball mask; tuned to be reasonably tolerant.
    orange1 = cv2.inRange(hsv, (4, 90, 90), (18, 255, 255))
    orange2 = cv2.inRange(hsv, (0, 60, 120), (25, 255, 255))
    orange = cv2.bitwise_or(orange1, orange2)

    moving = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)[1]
    search = cv2.bitwise_and(orange, moving)
    search = cv2.bitwise_and(search, court_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    search = cv2.morphologyEx(search, cv2.MORPH_OPEN, kernel, iterations=1)
    search = cv2.dilate(search, kernel, iterations=1)

    contours, _ = cv2.findContours(search, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    player_boxes = [t.box for t in player_tracks if t.is_confirmed()]
    best = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8 or area > 250:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 3 or h < 3 or w > 25 or h > 25:
            continue
        circ = circularity(cnt)
        if circ < 0.25:
            continue
        cx = x + w / 2.0
        cy = y + h / 2.0

        # Penalize candidates that are deep inside player boxes.
        overlap_penalty = 0.0
        for pb in player_boxes:
            if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                overlap_penalty += 2.0

        dist_penalty = 0.0
        if predicted_ball is not None:
            dist_penalty = 0.03 * np.linalg.norm(np.array([cx, cy]) - predicted_ball)

        score = 2.5 * circ + 0.01 * area - overlap_penalty - dist_penalty
        if score > best_score:
            best_score = score
            best = (int(cx), int(cy), x, y, w, h)

    return best, search


def build_court_mask(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Hardwood floor is bright and moderately saturated.
    wood = cv2.inRange(hsv, (8, 15, 80), (40, 180, 255))

    mask = np.zeros((h, w), dtype=np.uint8)
    # Keep the broadcast play area and avoid upper crowd / lower audience.
    cv2.rectangle(mask, (0, 72), (w, h - 28), 255, -1)

    court = cv2.bitwise_and(wood, mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    court = cv2.morphologyEx(court, cv2.MORPH_CLOSE, kernel, iterations=2)
    court = cv2.morphologyEx(court, cv2.MORPH_OPEN, kernel, iterations=1)

    # If mask is too sparse on a frame, fall back to broad ROI.
    if np.count_nonzero(court) < 0.12 * h * w:
        court = mask
    return court


# --------------------------- Main processing ---------------------------

def process_video(input_path, output_path, display_scale=1.0):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    backsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=28, detectShadows=True)
    tracker = MultiObjectTracker()
    ball_tracker = BallTracker()

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        court_mask = build_court_mask(frame)
        fgmask = backsub.apply(frame)
        fgmask = cv2.GaussianBlur(fgmask, (5, 5), 0)
        _, fgmask = cv2.threshold(fgmask, 170, 255, cv2.THRESH_BINARY)

        detections, cleaned_mask = detect_players(frame, fgmask, court_mask)
        tracks = tracker.update(detections, frame_idx)

        # Team assignment on confirmed tracks.
        team_counts = {"Team 1": 0, "Team 2": 0}
        active_tracks = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            label, conf = dominant_team_from_roi(frame, t.box)
            if label is not None:
                t.team_votes[label] += conf
            if sum(t.team_votes.values()) > 0:
                t.team = max(t.team_votes, key=t.team_votes.get)
            active_tracks.append(t)
            if t.team in team_counts and t.misses <= 4:
                team_counts[t.team] += 1

        predicted_ball = ball_tracker.predict()
        ball_candidate, ball_mask = detect_ball(frame, fgmask, court_mask, active_tracks, predicted_ball)
        if ball_candidate is not None:
            ball_tracker.correct((ball_candidate[0], ball_candidate[1]))
        else:
            ball_tracker.mark_missed()

        # --------------------------- Draw overlays ---------------------------
        vis = frame.copy()

        # Players
        for t in active_tracks:
            x1, y1, x2, y2 = map(int, t.box)
            if t.team == "Team 1":
                color = (0, 255, 255)   # yellow
            elif t.team == "Team 2":
                color = (255, 0, 0)     # blue
            else:
                color = (255, 255, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"ID {t.id}"
            if t.team is not None:
                text += f" | {t.team}"
            cv2.putText(vis, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

            hist = list(t.history)
            for i in range(1, len(hist)):
                p1 = tuple(map(int, hist[i - 1]))
                p2 = tuple(map(int, hist[i]))
                cv2.line(vis, p1, p2, color, 1, cv2.LINE_AA)

        # Ball
        if ball_tracker.last_pos is not None and ball_tracker.missed <= 12:
            bx, by = map(int, ball_tracker.last_pos)
            cv2.circle(vis, (bx, by), 10, (0, 140, 255), 2)
            cv2.drawMarker(vis, (bx, by), (0, 140, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
            cv2.putText(vis, "Ball", (bx + 10, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2, cv2.LINE_AA)
            trail = list(ball_tracker.trail)
            for i in range(1, len(trail)):
                cv2.line(vis, trail[i - 1], trail[i], (0, 140, 255), 2, cv2.LINE_AA)

        # HUD
        hud = vis.copy()
        cv2.rectangle(hud, (10, 10), (260, 105), (25, 25, 25), -1)
        vis = cv2.addWeighted(hud, 0.35, vis, 0.65, 0)
        cv2.putText(vis, "Custom OpenCV Tracking", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Team 1 (yellow): {team_counts['Team 1']}", (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Team 2 (blue):   {team_counts['Team 2']}", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 50), 2, cv2.LINE_AA)
        cv2.putText(vis, f"Frame: {frame_idx}", (18, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

        out.write(vis)
        frame_idx += 1

    cap.release()
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Basketball and player tracking using only OpenCV + NumPy.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="tracked_output.mp4", help="Path to output annotated video")
    args = parser.parse_args()
    process_video(args.input, args.output)
    print(f"Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()
