"""
Automated unit & pipeline verification script for Object Counting System.
Tests detector, counter modes (In-Frame, Tray Counter, Line Crossing), and drawing modules.
"""
import numpy as np
import cv2
import config
from detector import ObjectDetector, Detection
from tracker_counter import ObjectCounter
from utils.drawing import draw_bounding_box, draw_counting_line, draw_hud


def test_full_pipeline():
    print("[Test] 1. Initializing ObjectDetector on", config.DEVICE)
    detector = ObjectDetector(
        model_name_or_path="yolov8n.pt",
        device=config.DEVICE,
        conf_threshold=0.25
    )

    print("[Test] 2. Testing available classes count:", len(detector.names))
    assert len(detector.names) >= 80, "Expected at least 80 COCO classes"
    assert detector.get_thai_name("bottle") == "ขวดน้ำ"

    # Synthetic frame
    frame = np.full((720, 1280, 3), 40, dtype=np.uint8)

    # -------------------------------------------------------------
    # FEATURE 1: In-Frame Live Count (Decreases when objects leave)
    # -------------------------------------------------------------
    print("\n--- Testing Feature 1: In-Frame Live Count (Decreases when object leaves) ---")
    counter_in_frame = ObjectCounter(mode=config.MODE_IN_FRAME)

    det_bottle1 = Detection(39, "bottle", "ขวดน้ำ", 0.95, (200, 200, 300, 400), track_id=101)
    det_bottle2 = Detection(39, "bottle", "ขวดน้ำ", 0.91, (500, 200, 600, 400), track_id=102)

    # Step A: 2 bottles in frame
    res_a = counter_in_frame.update([det_bottle1, det_bottle2], frame.shape)
    assert res_a["total_in_frame"] == 2, f"Expected 2, got {res_a['total_in_frame']}"
    print(f"[Pass] 2 bottles in frame -> Count = {res_a['total_in_frame']}")

    # Step B: 1 bottle leaves (only bottle 1 remains)
    res_b = counter_in_frame.update([det_bottle1], frame.shape)
    assert res_b["total_in_frame"] == 1, f"Expected 1, got {res_b['total_in_frame']}"
    print(f"[Pass] 1 bottle left frame -> Count reduced to = {res_b['total_in_frame']}")

    # Step C: All bottles leave frame
    res_c = counter_in_frame.update([], frame.shape)
    assert res_c["total_in_frame"] == 0, f"Expected 0, got {res_c['total_in_frame']}"
    print(f"[Pass] All bottles left frame -> Count reduced to = {res_c['total_in_frame']}")

    # -------------------------------------------------------------
    # FEATURE 2: Tray / Cumulative Inventory Count (Persists when objects leave)
    # -------------------------------------------------------------
    print("\n--- Testing Feature 2: Tray / Cumulative Count (Persists when object leaves) ---")
    counter_tray = ObjectCounter(mode=config.MODE_TRAY)

    # Step A: Bottle 1 appears (frame 1)
    t1 = counter_tray.update([det_bottle1], frame.shape)
    # Step B: Bottle 1 confirmed (frame 2)
    t2 = counter_tray.update([det_bottle1], frame.shape)
    assert t2["tray_total"] == 1, f"Expected tray_total 1, got {t2['tray_total']}"
    assert t2["total_in_frame"] == 1
    print(f"[Pass] Bottle 1 detected -> Tray Count = {t2['tray_total']} (In-Frame: {t2['total_in_frame']})")

    # Step C: Bottle 1 leaves the frame! (Out of view)
    t3 = counter_tray.update([], frame.shape)
    assert t3["tray_total"] == 1, f"Expected tray_total to STAY at 1, got {t3['tray_total']}"
    assert t3["total_in_frame"] == 0, f"Expected in-frame 0, got {t3['total_in_frame']}"
    print(f"[Pass] Bottle 1 left frame -> Tray Count PERSISTS at = {t3['tray_total']} (In-Frame = {t3['total_in_frame']})")

    # Step D: Bottle 2 enters the tray (frame 1 & 2)
    counter_tray.update([det_bottle2], frame.shape)
    t4 = counter_tray.update([det_bottle2], frame.shape)
    assert t4["tray_total"] == 2, f"Expected tray_total 2, got {t4['tray_total']}"
    assert t4["total_in_frame"] == 1
    print(f"[Pass] Bottle 2 entered tray -> Tray Count increased to = {t4['tray_total']}")

    # Step E: Both bottles leave frame
    t5 = counter_tray.update([], frame.shape)
    assert t5["tray_total"] == 2, f"Expected tray_total to STAY at 2, got {t5['tray_total']}"
    assert t5["total_in_frame"] == 0
    print(f"[Pass] Frame empty -> Tray Count PERSISTS at = {t5['tray_total']} (In-Frame = {t5['total_in_frame']})")

    # Step F: Reset
    counter_tray.reset_counts()
    t6 = counter_tray.update([], frame.shape)
    assert t6["tray_total"] == 0
    print(f"[Pass] Reset counter -> Tray Count = {t6['tray_total']}")

    # -------------------------------------------------------------
    # FEATURE 3: Line-Crossing Logic
    # -------------------------------------------------------------
    print("\n--- Testing Feature 3: Line-Crossing Logic ---")
    counter_line = ObjectCounter(mode=config.MODE_LINE_CROSSING)
    det_cross1 = Detection(39, "bottle", "ขวดน้ำ", 0.92, (600, 250, 680, 350), track_id=1)
    det_cross2 = Detection(39, "bottle", "ขวดน้ำ", 0.92, (600, 450, 680, 550), track_id=1)
    counter_line.update([det_cross1], frame.shape)
    lc = counter_line.update([det_cross2], frame.shape)
    assert lc["total_crossed"] == 1, f"Expected 1 crossed, got {lc['total_crossed']}"
    print(f"[Pass] Line Crossing verified! Total crossed: {lc['total_crossed']}")

    # Test HUD rendering for Tray mode
    out_frame = draw_bounding_box(frame.copy(), det_bottle1)
    out_frame = draw_hud(
        frame=out_frame,
        stats=t4,
        fps=58.2,
        target_classes=["bottle"],
        conf_threshold=0.40,
        device_info="RTX 3050 Ti"
    )
    assert out_frame.shape == (720, 1280, 3)
    cv2.imwrite("test_output.jpg", out_frame)
    print("[Test] Saved HUD render to 'test_output.jpg'")

    print("\n✅ ALL 3 COUNTING MODES VERIFIED SUCCESSFULLY!")


if __name__ == "__main__":
    test_full_pipeline()
