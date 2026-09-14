"""
Main application script for Real-Time Object Detection & Counting.
Run with:
    python main.py
    python main.py --source 0 --class bottle
    python main.py --mode line_crossing
"""
import os
import sys
import time
import argparse
from typing import List, Optional
import cv2
import torch

import config
from detector import ObjectDetector
from tracker_counter import ObjectCounter
from utils.drawing import (
    draw_bounding_box,
    draw_counting_line,
    draw_hud
)


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Counting System with YOLO & GPU")
    parser.add_argument("--source", type=str, default=str(config.DEFAULT_SOURCE),
                        help="Camera index (e.g. 0, 1) or path to video file (e.g. video.mp4)")
    parser.add_argument("--model", type=str, default=config.DEFAULT_MODEL,
                        help="YOLO model path/name (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--class", dest="classes", nargs="+", default=None,
                        help="Target object class(es) to count (e.g. bottle cup person)")
    parser.add_argument("--mode", type=str,
                        choices=[config.MODE_IN_FRAME, config.MODE_TRAY, config.MODE_LINE_CROSSING],
                        default=config.DEFAULT_MODE,
                        help="Counting mode: in_frame, tray, or line_crossing")
    parser.add_argument("--conf", type=float, default=config.DEFAULT_CONF,
                        help="Confidence threshold (0.1 - 0.95)")
    parser.add_argument("--device", type=str, default=config.DEVICE,
                        help="Hardware device to run on ('cuda:0' or 'cpu')")
    parser.add_argument("--line-y", type=float, default=0.55,
                        help="Vertical position ratio of counting line (0.1 to 0.9)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine source (int for camera index, str for file path)
    source = int(args.source) if args.source.isdigit() else args.source

    # Target classes list
    target_classes: Optional[List[str]] = args.classes
    current_popular_idx = -1  # -1 means whatever was initially passed or All

    # Quick selection cycling list
    cycle_classes = [None] + [[c] for c in config.POPULAR_CLASSES]

    # Initialize YOLO Detector
    print("=" * 60)
    print("  OBJECT DETECTION & COUNTING SYSTEM")
    print(f"  Device: {args.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Model:  {args.model}")
    print(f"  Source: {source}")
    print("=" * 60)

    detector = ObjectDetector(
        model_name_or_path=args.model,
        device=args.device,
        conf_threshold=args.conf
    )

    # Initialize Counter
    counter = ObjectCounter(mode=args.mode)
    line_y_ratio = max(0.1, min(0.9, args.line_y))
    counter.line_normalized = ((0.05, line_y_ratio), (0.95, line_y_ratio))

    # Initialize Video Capture
    print(f"[Main] Opening video source: {source}...")
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) and sys.platform.startswith("win") else cv2.CAP_ANY)
    
    if not cap.isOpened() and isinstance(source, int):
        # Fallback without CAP_DSHOW
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        print("Please check if your camera is connected or verify the file path.")
        sys.exit(1)

    # Try setting HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Real-Time Object Counter (Press 'Q' to exit, 'C' to change class)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Prepare screenshot directory
    os.makedirs("screenshots", exist_ok=True)

    prev_time = time.time()
    fps = 30.0
    conf_current = args.conf

    print("\n[Controls]")
    print("  [Q] or [ESC] : Exit")
    print("  [C]          : Cycle target object (All -> bottle -> cup -> person -> ...)")
    print("  [M]          : Toggle mode (In-Frame <-> Line-Crossing)")
    print("  [R]          : Reset all counts")
    print("  [S]          : Save screenshot")
    print("  [+] / [-]    : Adjust confidence threshold")
    print("  [W] / [S]    : Move counting line Up / Down (hold Shift or tap)\n")

    device_label = "GPU" if "cuda" in args.device else "CPU"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Main] Video stream ended or camera disconnected.")
                break

            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Detect / Track objects
            if counter.mode in (config.MODE_TRAY, config.MODE_LINE_CROSSING):
                # Tracking required for unique IDs (Tray mode & Line crossing)
                detections = detector.track(
                    frame=frame,
                    target_classes=target_classes,
                    conf=conf_current,
                    persist=True
                )
            else:
                # In-frame detection
                detections = detector.detect(
                    frame=frame,
                    target_classes=target_classes,
                    conf=conf_current
                )

            # Update counting logic
            stats = counter.update(detections, frame.shape)

            # Draw counting line if in line-crossing mode
            if counter.mode == config.MODE_LINE_CROSSING:
                draw_counting_line(
                    frame,
                    stats["line_pixel"],
                    is_active=stats["is_line_active"]
                )

            # Draw bounding boxes
            for det in detections:
                frame = draw_bounding_box(frame, det)

            # Draw HUD
            frame = draw_hud(
                frame=frame,
                stats=stats,
                fps=fps,
                target_classes=target_classes,
                conf_threshold=conf_current,
                device_info=device_label
            )

            cv2.imshow(window_name, frame)

            # Handle Keyboard Events
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("[Main] Exiting...")
                break

            elif key in (ord('r'), ord('R')):  # Reset
                counter.reset_counts()

            elif key in (ord('m'), ord('M')):  # Toggle Mode
                new_mode = counter.toggle_mode()
                print(f"[Main] Switched to mode: {new_mode}")

            elif key in (ord('c'), ord('C')):  # Cycle popular target classes
                current_popular_idx = (current_popular_idx + 1) % len(cycle_classes)
                target_classes = cycle_classes[current_popular_idx]
                name_disp = target_classes[0] if target_classes else "All Classes"
                th_name = config.COCO_THAI_NAMES.get(name_disp, name_disp)
                print(f"[Main] Target class changed to: {th_name} ({name_disp})")

            elif key in (ord('+'), ord('=')):  # Increase Confidence
                conf_current = min(0.95, conf_current + 0.05)
                print(f"[Main] Confidence threshold: {conf_current:.2f}")

            elif key in (ord('-'), ord('_')):  # Decrease Confidence
                conf_current = max(0.10, conf_current - 0.05)
                print(f"[Main] Confidence threshold: {conf_current:.2f}")

            elif key in (ord('w'), ord('W')):  # Move line up
                line_y_ratio = max(0.1, line_y_ratio - 0.04)
                counter.line_normalized = ((0.05, line_y_ratio), (0.95, line_y_ratio))
                print(f"[Main] Line moved up: {line_y_ratio:.2f}")

            elif key in (ord('x'), ord('X')):  # Move line down
                line_y_ratio = min(0.9, line_y_ratio + 0.04)
                counter.line_normalized = ((0.05, line_y_ratio), (0.95, line_y_ratio))
                print(f"[Main] Line moved down: {line_y_ratio:.2f}")

            elif key in (ord('s'), ord('S')):  # Save screenshot
                ts = time.strftime("%Y%m%d_%H%M%S")
                fn = os.path.join("screenshots", f"count_{ts}.jpg")
                cv2.imwrite(fn, frame)
                print(f"[Main] Screenshot saved to: {fn}")

    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[Main] Cleanup complete.")


if __name__ == "__main__":
    main()
