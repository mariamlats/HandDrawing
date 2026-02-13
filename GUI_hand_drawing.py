import os
import time
import math
import base64
import threading
import webbrowser
import subprocess
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
import requests
from dotenv import load_dotenv

# =========================
# SETTINGS / CONSTANTS
# =========================
WINDOW_NAME = "Hand Drawing"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()
MESHY_API_KEY = os.getenv("MESHY_API_KEY", "")  # expects .env next to the script (or project root)

# Drawing state
draw_color = (0, 255, 0)      # BGR (OpenCV uses BGR, not RGB)
brush_thickness = 6
eraser_thickness = 50         # pixel-eraser size (two-finger gesture)

BRUSH_MIN, BRUSH_MAX = 1, 60
ERASER_MIN, ERASER_MAX = 10, 140

# UI panel (left-side tool panel area)
PANEL_X0, PANEL_Y0 = 0, 120
PANEL_X1, PANEL_Y1 = 520, 520

# Selection
select_mode = False
selected_stroke_idx: Optional[int] = None

# Cooldown to prevent rapid “re-click” when finger stays on UI / selecting
click_cooldown = 0
CLICK_DELAY = 14  # tweak if UI feels too “spammy”

# Gesture thresholds (in pixels)
PINCH_THRESH_PX = 45         # index-middle close -> erase
SELECT_THRESH_PX = 35        # distance to stroke for selection

# UI click gesture (thumb+middle pinch)
UI_CLICK_THRESH_PX = 38
ui_click_armed = True        # edge-trigger: fires once per pinch (prevents holding = repeated clicks)

# Straightening toggle
straighten_enabled = True

# =========================
# SIMPLE UI WIDGETS
# =========================
class Button:
    def __init__(self, x, y, w, h, color, text=""):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.color = color
        self.text = text
        self.active = False

    def draw(self, frame):
        # Filled rect + a border (thicker if active)
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), self.color, -1)
        border = 4 if self.active else 2
        cv2.rectangle(frame, (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 255, 255), border)

        if self.text:
            fs = 0.45
            th = 1
            (tw, thh), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx = self.x + (self.w - tw) // 2
            ty = self.y + (self.h + thh) // 2
            cv2.putText(frame, self.text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    def is_clicked(self, x, y):
        # simple bbox hit-test
        return self.x <= x <= self.x + self.w and self.y <= y <= self.y + self.h


# =========================
# COLOR BUTTONS (ADD BROWN)
# =========================
color_buttons = [
    Button(10, 140, 55, 55, (0, 0, 255),   "Red"),
    Button(70, 140, 55, 55, (0, 255, 0),   "Green"),
    Button(130, 140, 55, 55, (255, 0, 0),  "Blue"),
    Button(190, 140, 55, 55, (0, 255, 255), "Yel"),
    Button(250, 140, 55, 55, (255, 0, 255), "Pink"),
    Button(310, 140, 55, 55, (255, 255, 255), "White"),
    Button(370, 140, 55, 55, (0, 0, 0),    "Black"),
    Button(430, 140, 55, 55, (19, 69, 139), "Brown"),  # BGR brown
]
clear_button = Button(10, 205, 110, 50, (200, 200, 200), "Clear")
btn_3d = Button(130, 205, 110, 50, (70, 120, 255), "3D")
btn_straight = Button(250, 205, 235, 50, (120, 120, 120), "Straighten: ON")

color_buttons[1].active = True  # default green

# =========================
# STROKES + ERASER MASK
# =========================
# Each stroke: {"color": (b,g,r), "thickness": int, "points": [(x,y), ...]}
strokes: List[dict] = []
_current_stroke: Optional[dict] = None

# Pixel-eraser mask: 0 = keep, 255 = erase (so we can "erase" without deleting stroke geometry)
eraser_mask: Optional[np.ndarray] = None
_prev_erase_pt: Optional[Tuple[int, int]] = None  # used to draw continuous erase lines

# =========================
# MESHY JOB STATE (background)
# =========================
meshy_lock = threading.Lock()
meshy_job = {
    "running": False,
    "task_id": None,
    "progress": 0,
    "status": "",
    "error": "",
    "glb_url": "",
    "thumb_url": "",
    "glb_path": "",
}

# =========================
# MEDIAPIPE HANDS
# =========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def _lm_px(hand_landmarks, idx: int, frame_shape) -> Tuple[int, int]:
    # convert normalized landmark coords -> pixel coords
    h, w, _ = frame_shape
    lm = hand_landmarks.landmark[idx]
    return int(lm.x * w), int(lm.y * h)

def get_index_tip(hand_landmarks, frame_shape) -> Tuple[int, int]:
    return _lm_px(hand_landmarks, 8, frame_shape)

def is_finger_up(hand_landmarks, finger_tip_id, finger_pip_id) -> bool:
    # naive "up" test: tip is above PIP in image coordinates
    tip = hand_landmarks.landmark[finger_tip_id]
    pip = hand_landmarks.landmark[finger_pip_id]
    return tip.y < pip.y

def fingers_close(hand_landmarks, frame_shape, threshold_px: int) -> Tuple[bool, Tuple[int, int]]:
    ix, iy = _lm_px(hand_landmarks, 8, frame_shape)
    mx, my = _lm_px(hand_landmarks, 12, frame_shape)
    dist = math.hypot(ix - mx, iy - my)
    cx, cy = (ix + mx) // 2, (iy + my) // 2
    return dist < threshold_px, (cx, cy)

def get_gesture(hand_landmarks, frame_shape) -> str:
    # simple state machine: draw with index, erase with index+middle pinch
    index_up = is_finger_up(hand_landmarks, 8, 6)
    middle_up = is_finger_up(hand_landmarks, 12, 10)
    close, _ = fingers_close(hand_landmarks, frame_shape, PINCH_THRESH_PX)

    if index_up and middle_up and close:
        return "erase"
    if index_up and not middle_up:
        return "paint"
    return "none"

def ui_pinch_click(hand_landmarks, frame_shape) -> bool:
    """
    UI click gesture: thumb tip (4) + middle tip (12) pinched.
    This is separate from erase pinch (index+middle).
    """
    tx, ty = _lm_px(hand_landmarks, 4, frame_shape)    # thumb tip
    mx, my = _lm_px(hand_landmarks, 12, frame_shape)   # middle tip
    return math.hypot(tx - mx, ty - my) < UI_CLICK_THRESH_PX


# =========================
# UI INTERACTION
# =========================
def _update_straight_button():
    # keep the label in sync so it doesn't lie to us :)
    btn_straight.text = f"Straighten: {'ON' if straighten_enabled else 'OFF'}"
    btn_straight.color = (120, 120, 120) if straighten_enabled else (80, 80, 80)

def check_ui_interaction(x: int, y: int, clicked_event: bool, current_canvas_bgr: Optional[np.ndarray]) -> str:
    """
    UI buttons only trigger when clicked_event is True (edge-triggered pinch click).
    Returns:
      "color" | "clear" | "3d" | "straight" | ""
    """
    global draw_color, selected_stroke_idx, straighten_enabled

    if not clicked_event:
        return ""

    # color palette buttons
    for btn in color_buttons:
        if btn.is_clicked(x, y):
            draw_color = btn.color
            for b in color_buttons:
                b.active = False
            btn.active = True
            return "color"

    # clear everything
    if clear_button.is_clicked(x, y):
        strokes.clear()
        selected_stroke_idx = None
        if eraser_mask is not None:
            eraser_mask[:] = 0
        return "clear"

    # 3D export
    if btn_3d.is_clicked(x, y):
        if current_canvas_bgr is not None:
            start_meshy_3d_from_canvas(current_canvas_bgr)
        return "3d"

    # straighten toggle
    if btn_straight.is_clicked(x, y):
        straighten_enabled = not straighten_enabled
        _update_straight_button()
        return "straight"

    return ""

def draw_ui(frame):
    # semi-transparent tool panel overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (PANEL_X0, PANEL_Y0), (PANEL_X1, PANEL_Y1), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "Tools", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    for btn in color_buttons:
        btn.draw(frame)
    clear_button.draw(frame)
    btn_3d.draw(frame)
    btn_straight.draw(frame)

    cv2.putText(frame, "Current:", (10, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(frame, (110, 257), (150, 297), draw_color, -1)
    cv2.rectangle(frame, (110, 257), (150, 297), (255, 255, 255), 2)

    cv2.putText(frame, f"Brush: {brush_thickness}  (-/=)", (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Eraser: {eraser_thickness}  ([/])", (10, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    mode_txt = "SELECT MODE (m)" if select_mode else "DRAW MODE (m)"
    cv2.putText(frame, mode_txt, (10, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255) if select_mode else (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame, "Index = draw/select | Index+Middle close = pixel erase",
                (10, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, "q quit | s save PNG | v save SVG | 3 generate 3D | x delete selected | t toggle straighten",
                (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Meshy status line (simple, but helps debugging)
    with meshy_lock:
        if meshy_job["running"]:
            txt = f"Meshy: {meshy_job['status']} {meshy_job['progress']}%  (task {str(meshy_job['task_id'])[:8]}...)"
        elif meshy_job["error"]:
            txt = f"Meshy ERROR: {meshy_job['error']}"
        elif meshy_job["glb_path"]:
            txt = f"Meshy DONE: saved {os.path.basename(meshy_job['glb_path'])}"
        else:
            txt = "Meshy: idle"
    cv2.putText(frame, txt, (10, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# =========================
# GEOMETRY HELPERS
# =========================
def _rdp(points: List[Tuple[int, int]], eps: float) -> List[Tuple[int, int]]:
    # Ramer–Douglas–Peucker simplification
    if len(points) < 3:
        return points[:]

    def perp_dist(p, a, b) -> float:
        (px, py), (ax, ay), (bx, by) = p, a, b
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - ax, py - ay)
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - bx, py - by)
        t = c1 / c2
        projx = ax + t * vx
        projy = ay + t * vy
        return math.hypot(px - projx, py - projy)

    a = points[0]
    b = points[-1]
    max_d = -1.0
    idx = 0
    for i in range(1, len(points) - 1):
        d = perp_dist(points[i], a, b)
        if d > max_d:
            max_d = d
            idx = i

    if max_d > eps:
        left = _rdp(points[: idx + 1], eps)
        right = _rdp(points[idx:], eps)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]

def _moving_average(points: List[Tuple[int, int]], k: int = 5) -> List[Tuple[int, int]]:
    # quick smoothing to reduce hand jitter (cheap + good enough)
    if len(points) < k:
        return points[:]
    out = []
    half = k // 2
    for i in range(len(points)):
        xs = ys = cnt = 0
        for j in range(i - half, i + half + 1):
            if 0 <= j < len(points):
                xs += points[j][0]
                ys += points[j][1]
                cnt += 1
        out.append((int(xs / cnt), int(ys / cnt)))
    return out

def _stroke_length(pts: List[Tuple[int, int]]) -> float:
    if len(pts) < 2:
        return 0.0
    p = np.array(pts, dtype=np.float32)
    d = p[1:] - p[:-1]
    return float(np.sum(np.linalg.norm(d, axis=1)))

def _perp_dist_to_line(points_xy: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-6:
        return np.full((points_xy.shape[0],), 1e9, dtype=np.float32)
    pa = points_xy - a
    cross = np.abs(pa[:, 0] * ab[1] - pa[:, 1] * ab[0])
    return cross / denom

def _total_turning_angle(points_xy: np.ndarray) -> float:
    # how “curvy” the stroke is overall (used to avoid snapping curves)
    if len(points_xy) < 3:
        return 0.0
    v1 = points_xy[1:-1] - points_xy[:-2]
    v2 = points_xy[2:] - points_xy[1:-1]
    n1 = np.linalg.norm(v1, axis=1) + 1e-9
    n2 = np.linalg.norm(v2, axis=1) + 1e-9
    v1u = v1 / n1[:, None]
    v2u = v2 / n2[:, None]
    dot = np.clip(np.sum(v1u * v2u, axis=1), -1.0, 1.0)
    ang = np.arccos(dot)
    return float(np.sum(np.abs(ang)))

def _smart_snap_to_line_if_intended(points: List[Tuple[int, int]], thickness: int) -> Optional[List[Tuple[int, int]]]:
    # heuristic to guess if user intended a straight line (then snap)
    if len(points) < 8:
        return None
    p = np.array(points, dtype=np.float32)

    L = _stroke_length(points)
    if L < max(60.0, 8.0 * thickness):
        return None

    start = p[0]
    end = p[-1]
    end_to_end = float(np.linalg.norm(end - start))
    if end_to_end < max(35.0, 6.0 * thickness):
        return None

    efficiency = end_to_end / (L + 1e-9)
    if efficiency < 0.90:
        return None

    min_xy = p.min(axis=0)
    max_xy = p.max(axis=0)
    bw = float(max_xy[0] - min_xy[0])
    bh = float(max_xy[1] - min_xy[1])
    small = min(bw, bh)
    large = max(bw, bh)
    aspect = large / (small + 1e-9)
    if aspect < 2.2 and small > 3.0 * thickness:
        return None

    mean = p.mean(axis=0)
    cov = np.cov((p - mean).T)
    eigvals, eigvecs = np.linalg.eig(cov)
    eigvals = np.real(eigvals).astype(np.float32)
    eigvecs = np.real(eigvecs).astype(np.float32)
    linearity = float(np.max(eigvals)) / (float(np.sum(eigvals)) + 1e-9)
    if linearity < 0.93:
        return None

    v = eigvecs[:, int(np.argmax(eigvals))]
    v = v / (np.linalg.norm(v) + 1e-9)

    t = (p - mean) @ v
    a = mean + v * t.min()
    b = mean + v * t.max()

    d = _perp_dist_to_line(p, a, b)
    mean_d = float(np.mean(d))
    max_d = float(np.max(d))

    if not (mean_d <= max(1.8, 0.45 * thickness) and max_d <= max(5.0, 1.25 * thickness)):
        return None

    turn = _total_turning_angle(p)
    if turn > 0.9:
        return None

    return [(int(a[0]), int(a[1])), (int(b[0]), int(b[1]))]

def _auto_straighten(points: List[Tuple[int, int]], thickness: int) -> List[Tuple[int, int]]:
    """
    If straightening is enabled:
      - smooth
      - snap only if intended straight
      - preserve curves (especially closed strokes)
    If disabled:
      - only smooth lightly (no snapping / heavy simplification)
    """
    sm = _moving_average(points, k=5)

    if not straighten_enabled:
        return sm

    if len(sm) < 2:
        return sm

    p = np.array(sm, dtype=np.float32)
    p0, p1 = p[0], p[-1]
    end_dist = float(np.linalg.norm(p1 - p0))
    min_xy = p.min(axis=0)
    max_xy = p.max(axis=0)
    bw = float(max_xy[0] - min_xy[0])
    bh = float(max_xy[1] - min_xy[1])
    size = max(1.0, min(bw, bh))
    is_closed = (len(sm) >= 25) and (end_dist < 0.12 * size)

    snapped = _smart_snap_to_line_if_intended(sm, thickness)
    if snapped is not None:
        return snapped

    L = _stroke_length(sm)
    chord = float(np.linalg.norm(p1 - p0))
    efficiency = chord / (L + 1e-9)

    if is_closed:
        # closed loops usually mean shapes; don't destroy them by over-simplifying
        eps = max(0.25, thickness * 0.05)
        simp = _rdp(sm, eps)
        if len(simp) < 40:
            return sm
        return simp

    if efficiency < 0.90:
        eps = max(0.35, thickness * 0.08)
    else:
        eps = max(0.9, thickness * 0.20)

    simp = _rdp(sm, eps)
    if len(simp) < max(16, len(sm) // 5):
        return sm
    return simp

def _point_segment_distance(px, py, ax, ay, bx, by) -> float:
    # classic projection-based distance
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - ax, py - ay)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - bx, py - by)
    t = c1 / c2
    projx = ax + t * vx
    projy = ay + t * vy
    return math.hypot(px - projx, py - projy)

def _distance_to_stroke(pt: Tuple[int, int], stroke_pts: List[Tuple[int, int]]) -> float:
    # min distance from point -> polyline
    if len(stroke_pts) == 0:
        return 1e9
    if len(stroke_pts) == 1:
        return math.hypot(pt[0] - stroke_pts[0][0], pt[1] - stroke_pts[0][1])

    px, py = pt
    best = 1e9
    for i in range(1, len(stroke_pts)):
        ax, ay = stroke_pts[i - 1]
        bx, by = stroke_pts[i]
        d = _point_segment_distance(px, py, ax, ay, bx, by)
        best = min(best, d)
    return best

# =========================
# ERASER MASK FIX: allow repaint
# =========================
def unerase_on_draw(pt_prev: Optional[Tuple[int, int]], pt_now: Tuple[int, int], thickness: int):
    # when we draw over erased areas, clear the mask so paint shows up again
    global eraser_mask
    if eraser_mask is None:
        return
    r = max(1, thickness // 2)
    if pt_prev is None:
        cv2.circle(eraser_mask, pt_now, r, 0, -1, cv2.LINE_AA)
    else:
        cv2.line(eraser_mask, pt_prev, pt_now, 0, thickness=r * 2, lineType=cv2.LINE_AA)

# =========================
# RENDERING
# =========================
def render_canvas(frame_shape) -> np.ndarray:
    global eraser_mask, _current_stroke

    h, w, _ = frame_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # finished strokes
    for idx, st in enumerate(strokes):
        pts = st["points"]
        if len(pts) < 2:
            continue
        color = st["color"]
        thick = int(st["thickness"])

        # selected stroke gets a white "halo" behind it so it's obvious
        if selected_stroke_idx is not None and idx == selected_stroke_idx:
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), thick + 8, cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, thick, cv2.LINE_AA)

    # in-progress stroke (draw slightly highlighted so it feels responsive)
    if _current_stroke is not None:
        pts = _current_stroke.get("points", [])
        if len(pts) >= 2:
            color = _current_stroke["color"]
            thick = int(_current_stroke["thickness"])
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, (255, 255, 255), thick + 2, cv2.LINE_AA)
            cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, thick, cv2.LINE_AA)
        elif len(pts) == 1:
            color = _current_stroke["color"]
            thick = int(_current_stroke["thickness"])
            cv2.circle(canvas, pts[0], max(1, thick // 2), color, -1, cv2.LINE_AA)

    # keep mask sized to the current camera frame
    if eraser_mask is None or eraser_mask.shape[:2] != (h, w):
        eraser_mask = np.zeros((h, w), dtype=np.uint8)

    # apply eraser as "holes" in the canvas
    canvas[eraser_mask > 0] = (0, 0, 0)
    return canvas

def draw_eraser_indicator(frame: np.ndarray, center: Tuple[int, int], radius: int):
    # translucent red circle so user knows where they're erasing
    x, y = center
    overlay = frame.copy()
    cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.circle(frame, (x, y), radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

# =========================
# SAVE + VECTORIZATION
# =========================
def _timestamp_name(prefix: str, ext: str) -> str:
    # quick unique-ish filename; fine for a demo app
    return os.path.join(OUTPUT_DIR, f"{prefix}_{int(time.time())}.{ext}")

def save_png(canvas_bgr: np.ndarray) -> str:
    fname = _timestamp_name("drawing", "png")
    cv2.imwrite(fname, canvas_bgr)
    return fname

def _kmeans_quantize(img_bgr: np.ndarray, max_colors: int = 8) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
    # reduce to a small color palette so SVG export isn't insane
    img = img_bgr.copy()
    bg = np.all(img < 15, axis=2)
    pts = img[~bg].reshape(-1, 3)
    if pts.shape[0] == 0:
        return img_bgr, []

    # heuristic for choosing K based on how many colored pixels exist
    K = int(min(max_colors, max(2, int(np.sqrt(pts.shape[0] / 2000.0) + 2))))
    K = max(2, min(K, max_colors))

    Z = np.float32(pts)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)

    q = img.copy()
    q[~bg] = centers[labels.flatten()]
    palette = [tuple(map(int, c)) for c in centers.tolist()]
    return q, palette

def vectorize_to_svg(canvas_bgr: np.ndarray, out_svg_path: str, max_colors: int = 10, eps: float = 2.0):
    h, w, _ = canvas_bgr.shape
    quant, palette = _kmeans_quantize(canvas_bgr, max_colors=max_colors)
    bg = np.all(quant < 15, axis=2)

    svg_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>'
    ]

    for bgr in palette:
        b, g, r = bgr
        # skip near-black (usually background-ish)
        if b < 20 and g < 20 and r < 20:
            continue

        mask = np.all(quant == np.array([b, g, r], dtype=np.uint8), axis=2).astype(np.uint8) * 255
        mask[bg] = 0
        if cv2.countNonZero(mask) == 0:
            continue

        # small close to fill tiny gaps
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        fill = f'rgb({r},{g},{b})'
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            approx = cv2.approxPolyDP(cnt, epsilon=eps, closed=True)
            if len(approx) < 3:
                continue
            pts = approx.reshape(-1, 2)
            d = f'M {pts[0][0]} {pts[0][1]} ' + " ".join([f'L {x} {y}' for x, y in pts[1:]]) + " Z"
            svg_parts.append(f'<path d="{d}" fill="{fill}" stroke="none"/>')

    svg_parts.append("</svg>")
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_parts))

def save_svg_only(canvas_bgr: np.ndarray) -> str:
    svg_path = _timestamp_name("vector", "svg")
    vectorize_to_svg(canvas_bgr, svg_path, max_colors=10, eps=2.0)
    return svg_path

def save_png_and_svg(canvas_bgr: np.ndarray) -> Tuple[str, str]:
    png_path = save_png(canvas_bgr)
    svg_path = png_path.replace(".png", ".svg")
    vectorize_to_svg(canvas_bgr, svg_path, max_colors=10, eps=2.0)
    return png_path, svg_path

# =========================
# STROKE LIFECYCLE
# =========================
def start_stroke(pt: Tuple[int, int]):
    global _current_stroke
    _current_stroke = {
        "color": draw_color,
        "thickness": int(brush_thickness),
        "points": [pt],
    }

def append_stroke_point(pt: Tuple[int, int]):
    global _current_stroke
    if _current_stroke is None:
        start_stroke(pt)
        return
    _current_stroke["points"].append(pt)

def finish_stroke():
    global _current_stroke
    if _current_stroke is None:
        return
    pts = _current_stroke["points"]
    if len(pts) >= 2:
        # do smoothing / snapping at the end so drawing feels natural while moving
        cleaned = _auto_straighten(pts, int(_current_stroke["thickness"]))
        _current_stroke["points"] = cleaned
        strokes.append(_current_stroke)
    _current_stroke = None

def delete_selected_stroke():
    global selected_stroke_idx
    if selected_stroke_idx is None:
        return
    if 0 <= selected_stroke_idx < len(strokes):
        strokes.pop(selected_stroke_idx)
    selected_stroke_idx = None

def select_nearest_stroke(pt: Tuple[int, int]) -> Optional[int]:
    # naive nearest-stroke selection (good enough for small stroke counts)
    if not strokes:
        return None
    best_i = None
    best_d = 1e9
    for i, st in enumerate(strokes):
        d = _distance_to_stroke(pt, st["points"])
        if d < best_d:
            best_d = d
            best_i = i
    if best_i is not None and best_d <= SELECT_THRESH_PX:
        return best_i
    return None

# =========================
# MESHY INTEGRATION
# =========================
def _file_to_data_uri_png(path: str) -> str:
    # Meshy API can take a data URI; avoids needing a public URL
    with open(path, "rb") as f:
        b = f.read()
    b64 = base64.b64encode(b).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _download_file(url: str, out_path: str):
    # simple download helper; enough for model files
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def _open_file_or_url(path: str, fallback_url: str = ""):
    # best-effort: open file locally (mac uses "open"), otherwise open URL
    try:
        if os.path.exists(path):
            subprocess.run(["open", path], check=False)
            return
    except Exception:
        pass
    if fallback_url:
        webbrowser.open(fallback_url)

def _meshy_worker(png_path: str):
    global meshy_job
    try:
        if not MESHY_API_KEY:
            raise RuntimeError("MESHY_API_KEY is missing in .env")

        # update UI state before network calls
        with meshy_lock:
            meshy_job.update({
                "running": True, "task_id": None, "progress": 0, "status": "STARTING",
                "error": "", "glb_url": "", "thumb_url": "", "glb_path": ""
            })

        data_uri = _file_to_data_uri_png(png_path)

        url = "https://api.meshy.ai/openapi/v1/image-to-3d"
        headers = {"Authorization": f"Bearer {MESHY_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "image_url": data_uri,
            "enable_pbr": True,
            "should_remesh": True,
            "should_texture": True,
            "save_pre_remeshed_model": True,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        task_id = resp.json().get("result")
        if not task_id:
            raise RuntimeError(f"Meshy create task returned no id: {resp.text}")

        with meshy_lock:
            meshy_job["task_id"] = task_id
            meshy_job["status"] = "PENDING"
            meshy_job["progress"] = 0

        get_url = f"https://api.meshy.ai/openapi/v1/image-to-3d/{task_id}"
        while True:
            # polling loop (2s is a decent balance)
            r = requests.get(get_url, headers={"Authorization": f"Bearer {MESHY_API_KEY}"}, timeout=60)
            r.raise_for_status()
            obj = r.json()

            status = obj.get("status", "")
            progress = int(obj.get("progress", 0) or 0)

            thumb = obj.get("thumbnail_url", "") or ""
            model_urls = obj.get("model_urls", {}) or {}
            glb_url = model_urls.get("glb", "") or ""

            with meshy_lock:
                meshy_job["status"] = status
                meshy_job["progress"] = progress
                meshy_job["thumb_url"] = thumb
                meshy_job["glb_url"] = glb_url

            if status == "SUCCEEDED":
                if not glb_url:
                    raise RuntimeError("Task succeeded but no GLB URL returned.")
                out_glb = os.path.join(OUTPUT_DIR, f"meshy_{task_id}.glb")
                _download_file(glb_url, out_glb)
                with meshy_lock:
                    meshy_job["glb_path"] = out_glb
                    meshy_job["running"] = False
                _open_file_or_url(out_glb, fallback_url=thumb)
                return

            if status in ("FAILED", "CANCELED", "CANCELLED"):
                err = obj.get("task_error", {}) or {}
                msg = err.get("message", "") or "Unknown Meshy error"
                raise RuntimeError(f"{status}: {msg}")

            time.sleep(2.0)

    except Exception as e:
        with meshy_lock:
            meshy_job["running"] = False
            meshy_job["error"] = str(e)

def start_meshy_3d_from_canvas(canvas_bgr: np.ndarray):
    # avoid starting multiple jobs at once
    with meshy_lock:
        if meshy_job["running"]:
            return

    png_path, svg_path = save_png_and_svg(canvas_bgr)
    print(f"Saved: {png_path}")
    print(f"Vector: {svg_path}")
    print("Sending to Meshy...")

    t = threading.Thread(target=_meshy_worker, args=(png_path,), daemon=True)
    t.start()

# =========================
# MAIN LOOP
# =========================
def main():
    global eraser_mask, _prev_erase_pt, click_cooldown
    global brush_thickness, eraser_thickness, select_mode, selected_stroke_idx
    global straighten_enabled, ui_click_armed

    _update_straight_button()

    cap = cv2.VideoCapture(0)
    # camera resolution (not guaranteed, but helps on most webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    if not ret:
        print("Failed to access camera.")
        return
    frame = cv2.flip(frame, 1)  # mirror view feels more natural for drawing
    h, w, _ = frame.shape
    eraser_mask = np.zeros((h, w), dtype=np.uint8)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if click_cooldown > 0:
                click_cooldown -= 1

            if eraser_mask is None or eraser_mask.shape[:2] != frame.shape[:2]:
                eraser_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            # render canvas BEFORE checking UI clicks (needed for 3D button)
            canvas_now = render_canvas(frame.shape)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style(),
                    )

                    x, y = get_index_tip(hand_landmarks, frame.shape)
                    gesture = get_gesture(hand_landmarks, frame.shape)

                    # ---------- UI pinch click (thumb+middle), edge-triggered ----------
                    clicked_now = ui_pinch_click(hand_landmarks, frame.shape)
                    ui_clicked_event = False
                    if ui_click_armed and clicked_now:
                        ui_clicked_event = True
                        ui_click_armed = False
                    elif not clicked_now:
                        ui_click_armed = True

                    # visual feedback when click is held
                    if clicked_now:
                        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)

                    if gesture == "paint":
                        _prev_erase_pt = None

                        # UI interaction only triggers on click event
                        ui_action = check_ui_interaction(x, y, ui_clicked_event, canvas_now)
                        if ui_action:
                            finish_stroke()
                            selected_stroke_idx = None
                            if click_cooldown == 0:
                                click_cooldown = CLICK_DELAY
                            continue

                        if select_mode:
                            finish_stroke()
                            if click_cooldown == 0:
                                idx = select_nearest_stroke((x, y))
                                selected_stroke_idx = idx
                                click_cooldown = CLICK_DELAY
                            cv2.circle(frame, (x, y), 10, (255, 255, 0), 2, cv2.LINE_AA)
                        else:
                            pt = (x, y)
                            cv2.circle(frame, pt, max(1, brush_thickness // 2), draw_color, -1, cv2.LINE_AA)

                            if _current_stroke is None:
                                start_stroke(pt)
                                unerase_on_draw(None, pt, brush_thickness)
                            else:
                                prev = _current_stroke["points"][-1] if _current_stroke["points"] else None
                                append_stroke_point(pt)
                                unerase_on_draw(prev, pt, brush_thickness)

                    elif gesture == "erase":
                        # stop drawing when erasing (keeps strokes clean)
                        finish_stroke()
                        selected_stroke_idx = None

                        close, center = fingers_close(hand_landmarks, frame.shape, PINCH_THRESH_PX)
                        if close:
                            ex, ey = center
                            r = max(ERASER_MIN, min(eraser_thickness, ERASER_MAX)) // 2
                            if _prev_erase_pt is None:
                                cv2.circle(eraser_mask, (ex, ey), r, 255, -1, cv2.LINE_AA)
                            else:
                                cv2.line(eraser_mask, _prev_erase_pt, (ex, ey), 255, thickness=r * 2, lineType=cv2.LINE_AA)
                            _prev_erase_pt = (ex, ey)
                            draw_eraser_indicator(frame, (ex, ey), r)
                        else:
                            _prev_erase_pt = None

                    else:
                        finish_stroke()
                        _prev_erase_pt = None
            else:
                finish_stroke()
                _prev_erase_pt = None

            # Re-render after any updates
            canvas = render_canvas(frame.shape)

            combined = cv2.addWeighted(frame, 0.70, canvas, 0.30, 0)
            draw_ui(combined)

            if selected_stroke_idx is not None:
                cv2.putText(combined, f"Selected stroke: {selected_stroke_idx} (press x to delete)",
                            (540, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, combined)
            key = cv2.waitKey(1) & 0xFF

            # keyboard shortcuts (mostly for debugging / quick control)
            if key == ord("q"):
                break
            elif key == ord("m"):
                select_mode = not select_mode
                finish_stroke()
                click_cooldown = CLICK_DELAY
            elif key == ord("x"):
                delete_selected_stroke()
                click_cooldown = CLICK_DELAY
            elif key == ord("s"):
                p = save_png(canvas)
                print(f"Saved: {p}")
            elif key == ord("v"):
                svg = save_svg_only(canvas)
                print(f"Vector (SVG only): {svg}")
            elif key == ord("3"):
                start_meshy_3d_from_canvas(canvas)
            elif key == ord("t"):
                straighten_enabled = not straighten_enabled
                _update_straight_button()
                print(f"Straighten: {'ON' if straighten_enabled else 'OFF'}")
            elif key == ord("-"):
                brush_thickness = max(BRUSH_MIN, brush_thickness - 1)
            elif key == ord("="):
                brush_thickness = min(BRUSH_MAX, brush_thickness + 1)
            elif key == ord("["):
                eraser_thickness = max(ERASER_MIN, eraser_thickness - 5)
            elif key == ord("]"):
                eraser_thickness = min(ERASER_MAX, eraser_thickness + 5)
            elif key == ord("c"):
                strokes.clear()
                selected_stroke_idx = None
                eraser_mask[:] = 0
                click_cooldown = CLICK_DELAY
            elif key == ord("r"):
                # quick reset for the mask only (useful if erase got messy)
                eraser_mask[:] = 0
                click_cooldown = CLICK_DELAY

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()