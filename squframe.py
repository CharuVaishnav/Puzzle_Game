# pip install opencv-python mediapipe
import cv2
import mediapipe as mp
import numpy as np
import time
import random
import os
import urllib.request
from mediapipe.tasks.python import BaseOptions, vision

CAM_INDEX = 0
PINCH_THRESH = 40
STEADY_THRESH = 40
HOLD_TIME = 1.5
MIN_RECT = 150
BOARD = 480
GRID = 3
TILE = BOARD // GRID

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading hand landmark model (one-time)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    num_hands=2, running_mode=vision.RunningMode.VIDEO,
    min_hand_detection_confidence=0.6, min_tracking_confidence=0.6))
last_ts = 0
cap = cv2.VideoCapture(CAM_INDEX)

mode = "CAPTURE"
hold_start = None
hold_rect = None

tiles = []
order = []
selected = None
prev_pinch = False
move_count = 0
solved = False
solved_time = 0.0


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def hand_points(lm, w, h):
    thumb = (int(lm[4].x * w), int(lm[4].y * h))
    index = (int(lm[8].x * w), int(lm[8].y * h))
    mid = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
    pinching = dist(thumb, index) < PINCH_THRESH
    return pinching, mid, index


def make_tiles(img):
    global tiles, order, move_count, solved
    resized = cv2.resize(img, (BOARD, BOARD))
    tiles = []
    for r in range(GRID):
        for c in range(GRID):
            tiles.append(resized[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE].copy())
    order = list(range(9))
    while order == list(range(9)):
        random.shuffle(order)
    move_count = 0
    solved = False


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    clean = frame.copy()
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = max(last_ts + 1, int(time.time() * 1000))
    last_ts = ts
    result = landmarker.detect_for_video(mp_image, ts)

    hands_data = []
    for hlm in result.hand_landmarks:
        hands_data.append(hand_points(hlm, w, h))
        for lm in hlm:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 200, 0), -1)

    if mode == "CAPTURE":
        if len(hands_data) >= 2:
            p1, p2 = hands_data[0][1], hands_data[1][1]
            pinching = hands_data[0][0] and hands_data[1][0]
            x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
            x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])
            rw, rh = x2 - x1, y2 - y1

            if pinching:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                if (hold_rect is None or
                        abs(x1 - hold_rect[0]) > STEADY_THRESH or abs(y1 - hold_rect[1]) > STEADY_THRESH or
                        abs(x2 - hold_rect[2]) > STEADY_THRESH or abs(y2 - hold_rect[3]) > STEADY_THRESH):
                    hold_start = time.time()
                hold_rect = (x1, y1, x2, y2)

                if rw >= MIN_RECT and rh >= MIN_RECT:
                    elapsed = time.time() - hold_start
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    radius = int(max(rw, rh) / 2 * max(0.0, 1 - elapsed / HOLD_TIME))
                    cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 3)
                    if elapsed >= HOLD_TIME:
                        roi = clean[y1:y2, x1:x2].copy()
                        if roi.size > 0:
                            make_tiles(roi)
                            mode = "PUZZLE"
                            hold_start = None
                            hold_rect = None
                else:
                    hold_start = time.time()
            else:
                hold_start = None
                hold_rect = None
        else:
            hold_start = None
            hold_rect = None
        cv2.putText(frame, "Pinch both hands to draw rectangle (min 150x150, hold 1.5s)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    else:  # PUZZLE mode
        bx, by = (w - BOARD) // 2, (h - BOARD) // 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 10, by - 10), (bx + BOARD + 10, by + BOARD + 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        for i in range(9):
            r, c = divmod(i, GRID)
            tx, ty = bx + c * TILE, by + r * TILE
            frame[ty:ty + TILE, tx:tx + TILE] = tiles[order[i]]
            cv2.rectangle(frame, (tx, ty), (tx + TILE, ty + TILE), (50, 50, 50), 1)

        cursor, pinching = None, False
        if hands_data:
            pinching, _, cursor = hands_data[0]

        hover_idx = None
        if cursor and bx <= cursor[0] < bx + BOARD and by <= cursor[1] < by + BOARD:
            c = (cursor[0] - bx) // TILE
            r = (cursor[1] - by) // TILE
            hover_idx = r * GRID + c

        if not solved:
            if pinching and not prev_pinch and hover_idx is not None:
                selected = hover_idx
            if not pinching and prev_pinch:
                if selected is not None and hover_idx is not None and hover_idx != selected:
                    order[selected], order[hover_idx] = order[hover_idx], order[selected]
                    move_count += 1
                    if order == list(range(9)):
                        solved = True
                        solved_time = time.time()
                selected = None
        prev_pinch = pinching

        if selected is not None:
            r, c = divmod(selected, GRID)
            tx, ty = bx + c * TILE, by + r * TILE
            cv2.rectangle(frame, (tx, ty), (tx + TILE, ty + TILE), (0, 255, 255), 4)
        if hover_idx is not None and hover_idx != selected:
            r, c = divmod(hover_idx, GRID)
            tx, ty = bx + c * TILE, by + r * TILE
            cv2.rectangle(frame, (tx, ty), (tx + TILE, ty + TILE), (255, 255, 0), 2)

        if cursor:
            cv2.circle(frame, cursor, 8, (0, 0, 255), -1)

        cv2.putText(frame, f"Moves: {move_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "r=recapture  q=quit", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if solved:
            flash_elapsed = time.time() - solved_time
            if flash_elapsed < 1.0 and int(flash_elapsed * 8) % 2 == 0:
                green = np.full_like(frame, (0, 255, 0))
                frame = cv2.addWeighted(green, 0.3, frame, 0.7, 0)

            pcy = by + BOARD // 2
            cv2.rectangle(frame, (bx, pcy - 55), (bx + BOARD, pcy + 55), (0, 0, 0), -1)

            msg = "Yay! You solved it!"
            (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(frame, msg, (bx + BOARD // 2 - mw // 2, pcy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            mv = f"Moves: {move_count}"
            (vw, _), _ = cv2.getTextSize(mv, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(frame, mv, (bx + BOARD // 2 - vw // 2, pcy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("SquFrame Puzzle", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('r'):
        mode = "CAPTURE"
        hold_start = None
        hold_rect = None
        selected = None
        prev_pinch = False
        solved = False

cap.release()
cv2.destroyAllWindows()
