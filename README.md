# squframe

A webcam puzzle game controlled entirely with your hands — no mouse, no keyboard (well, almost). Pinch to capture a piece of the world in front of your camera, then solve it as a shuffled tile puzzle using nothing but your fingertips.

Built with **OpenCV** and **MediaPipe** hand tracking, in a single Python file.

## How it works

**1. Capture mode** — pinch (thumb tip to index fingertip) with both hands to draw a rectangle in mid-air over whatever you want to turn into a puzzle. Hold it steady for ~1.5s and watch the countdown ring shrink — it then snaps a photo of that region.

**2. Puzzle mode** — the captured image is sliced into a shuffled 3×3 grid. Use one hand as a cursor: pinch on a tile to pick it up, move to another tile, release to swap. Keep swapping until the image is back in order.

## Requirements

- Windows, Python 3.12+
- A webcam

## Setup

```bash
pip install opencv-python mediapipe
python squframe.py
```

First run auto-downloads MediaPipe's hand landmark model (~8MB, one-time).

## Controls

| Input | Action |
|---|---|
| Pinch both hands (capture mode) | Draw / resize the capture rectangle |
| Hold rectangle steady ~1.5s | Capture and start the puzzle |
| Pinch a tile (puzzle mode) | Select it |
| Release pinch over another tile | Swap the two tiles |
| `r` | Discard and recapture a new image |
| `q` | Quit |

## Tech

- [OpenCV](https://opencv.org/) for video capture, drawing, and image processing
- [MediaPipe](https://developers.google.com/mediapipe) Hand Landmarker (Tasks API) for real-time hand/finger tracking
- Pure Python, one file, no other dependencies
