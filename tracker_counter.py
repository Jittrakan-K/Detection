"""
Object Counting and Tracking Logic module.
Supports:
  1. In-Frame Real-Time Counting (decreases immediately when objects leave screen)
  2. Tray / Cumulative Inventory Counting (counts upon detection, persists after objects leave)
  3. Line-Crossing Counting (counts when objects cross a designated virtual line)
"""
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import time
from detector import Detection
import config


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """Check if three points are listed in counterclockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def lines_intersect(
    A: Tuple[float, float],
    B: Tuple[float, float],
    C: Tuple[float, float],
    D: Tuple[float, float]
) -> bool:
    """Return True if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


def signed_distance_to_line(
    P: Tuple[float, float],
    A: Tuple[float, float],
    B: Tuple[float, float]
) -> float:
    """
    Calculate the cross-product signed distance of point P relative to directed line AB.
    Positive: Point P is on the left side of line A->B.
    Negative: Point P is on the right side of line A->B.
    """
    return (P[0] - A[0]) * (B[1] - A[1]) - (P[1] - A[1]) * (B[0] - A[0])


class ObjectCounter:
    """
    Manages counting statistics for:
    - MODE_IN_FRAME: counts only objects currently visible on screen.
    - MODE_TRAY: counts each new unique object once and retains count permanently until reset.
    - MODE_LINE_CROSSING: counts objects passing through a virtual line.
    """
    def __init__(self, mode: str = config.DEFAULT_MODE):
        self.mode = mode

        # 1. In-Frame statistics (Live: decreases when objects leave)
        self.current_in_frame_counts: Dict[str, int] = defaultdict(int)
        self.peak_in_frame_counts: Dict[str, int] = defaultdict(int)

        # 2. Tray / Inventory Cumulative statistics (persists after objects leave)
        self.tray_counts_by_class: Dict[str, int] = defaultdict(int)
        self.tray_counted_ids: Set[int] = set()
        self.id_consecutive_frames: Dict[int, int] = defaultdict(int)
        self.min_confirm_frames: int = 2  # Must be visible >= 2 frames to avoid false flicker
        self.last_tray_add_time: float = 0.0

        # 3. Line-Crossing cumulative statistics
        self.in_counts: Dict[str, int] = defaultdict(int)
        self.out_counts: Dict[str, int] = defaultdict(int)
        self.total_crossed: Dict[str, int] = defaultdict(int)
        self.counted_ids: Set[int] = set()

        # Tracking state
        self.track_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.last_seen: Dict[int, float] = {}
        self.last_cross_time: float = 0.0

        # Region of Interest (ROI) box: (x1, y1, x2, y2) normalized [0..1]
        self.roi_normalized: Optional[Tuple[float, float, float, float]] = None

        # Counting Line in normalized coordinates ((x1, y1), (x2, y2))
        self.line_normalized = config.DEFAULT_LINE_COORDS

    def set_mode(self, mode: str) -> None:
        """Switch counting mode ('in_frame', 'tray', or 'line_crossing')."""
        if mode in (config.MODE_IN_FRAME, config.MODE_TRAY, config.MODE_LINE_CROSSING):
            self.mode = mode
            print(f"[Counter] Mode changed to: {self.mode}")

    def toggle_mode(self) -> str:
        """Cycle through available counting modes."""
        if self.mode == config.MODE_IN_FRAME:
            self.set_mode(config.MODE_TRAY)
        elif self.mode == config.MODE_TRAY:
            self.set_mode(config.MODE_LINE_CROSSING)
        else:
            self.set_mode(config.MODE_IN_FRAME)
        return self.mode

    def reset_counts(self) -> None:
        """Reset all counters and tracking history."""
        self.current_in_frame_counts.clear()
        self.peak_in_frame_counts.clear()
        
        # Reset Tray
        self.tray_counts_by_class.clear()
        self.tray_counted_ids.clear()
        self.id_consecutive_frames.clear()

        # Reset Line
        self.in_counts.clear()
        self.out_counts.clear()
        self.total_crossed.clear()
        self.counted_ids.clear()

        # Reset History
        self.track_history.clear()
        self.last_seen.clear()
        print("[Counter] All counts reset.")

    def get_pixel_line(self, frame_w: int, frame_h: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Convert normalized line coordinates to absolute pixel coordinates."""
        (x1_r, y1_r), (x2_r, y2_r) = self.line_normalized
        p1 = (int(x1_r * frame_w), int(y1_r * frame_h))
        p2 = (int(x2_r * frame_w), int(y2_r * frame_h))
        return p1, p2

    def is_point_in_roi(self, pt: Tuple[int, int], frame_w: int, frame_h: int) -> bool:
        """Check if a pixel point falls inside the defined ROI (if set)."""
        if self.roi_normalized is None:
            return True
        rx1, ry1, rx2, ry2 = self.roi_normalized
        x, y = pt
        return (rx1 * frame_w <= x <= rx2 * frame_w) and (ry1 * frame_h <= y <= ry2 * frame_h)

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        Process detections for current frame, update counts, and return summary dict.
        """
        frame_h, frame_w = frame_shape[:2]
        now = time.time()

        # 1. Update In-Frame Counts (Always computed: decreases when objects leave)
        in_frame_counts: Dict[str, int] = defaultdict(int)
        for det in detections:
            if self.is_point_in_roi(det.center, frame_w, frame_h):
                in_frame_counts[det.class_name] += 1

        self.current_in_frame_counts = in_frame_counts
        for cname, count in in_frame_counts.items():
            if count > self.peak_in_frame_counts[cname]:
                self.peak_in_frame_counts[cname] = count

        # 2. Update Tray / Cumulative Counter (if in tray mode)
        tray_just_added = False
        if self.mode == config.MODE_TRAY:
            for det in detections:
                tid = det.track_id
                if tid is None:
                    continue

                self.last_seen[tid] = now
                if tid not in self.tray_counted_ids:
                    self.id_consecutive_frames[tid] += 1
                    if self.id_consecutive_frames[tid] >= self.min_confirm_frames:
                        self.tray_counted_ids.add(tid)
                        self.tray_counts_by_class[det.class_name] += 1
                        self.last_tray_add_time = now
                        tray_just_added = True
                        print(
                            f"[Counter] Tray +1: {det.class_name} (#{tid}) -> "
                            f"Total in tray: {sum(self.tray_counts_by_class.values())}"
                        )

        # 3. Update Line-Crossing (if in line_crossing mode)
        line_p1, line_p2 = self.get_pixel_line(frame_w, frame_h)
        just_crossed = False

        if self.mode == config.MODE_LINE_CROSSING:
            for det in detections:
                tid = det.track_id
                if tid is None:
                    continue

                curr_pt = det.center
                self.last_seen[tid] = now

                # Record history (keep up to 30 points)
                history = self.track_history[tid]
                history.append(curr_pt)
                if len(history) > 30:
                    history.pop(0)

                # Check for line crossing if we have at least 2 points and not already counted
                if len(history) >= 2 and tid not in self.counted_ids:
                    prev_pt = history[-2]
                    
                    # Check segment intersection with counting line
                    if lines_intersect(prev_pt, curr_pt, line_p1, line_p2):
                        d_prev = signed_distance_to_line(prev_pt, line_p1, line_p2)
                        d_curr = signed_distance_to_line(curr_pt, line_p1, line_p2)

                        if d_prev < 0 and d_curr >= 0:
                            direction = "IN"
                            self.in_counts[det.class_name] += 1
                        else:
                            direction = "OUT"
                            self.out_counts[det.class_name] += 1

                        self.total_crossed[det.class_name] += 1
                        self.counted_ids.add(tid)
                        self.last_cross_time = now
                        just_crossed = True
                        print(
                            f"[Counter] Object crossed line! Class: {det.class_name} "
                            f"ID: #{tid} Direction: {direction} (Total: {sum(self.total_crossed.values())})"
                        )

        # 4. Clean up stale track histories (> 5 seconds inactive)
        stale_threshold = 5.0
        stale_tids = [tid for tid, last_t in self.last_seen.items() if now - last_t > stale_threshold]
        for tid in stale_tids:
            self.track_history.pop(tid, None)
            self.last_seen.pop(tid, None)

        total_in_frame = sum(self.current_in_frame_counts.values())
        tray_total = sum(self.tray_counts_by_class.values())
        total_in = sum(self.in_counts.values())
        total_out = sum(self.out_counts.values())
        total_crossed = sum(self.total_crossed.values())

        return {
            "mode": self.mode,
            "in_frame_counts": dict(self.current_in_frame_counts),
            "total_in_frame": total_in_frame,
            "peak_in_frame": dict(self.peak_in_frame_counts),
            
            # Tray data
            "tray_counts": dict(self.tray_counts_by_class),
            "tray_total": tray_total,
            "is_tray_active": (now - self.last_tray_add_time) < 0.6,
            "tray_just_added": tray_just_added,

            # Line crossing data
            "in_counts": dict(self.in_counts),
            "out_counts": dict(self.out_counts),
            "total_crossed_by_class": dict(self.total_crossed),
            "total_in": total_in,
            "total_out": total_out,
            "total_crossed": total_crossed,
            "just_crossed": just_crossed,
            "line_pixel": (line_p1, line_p2),
            "is_line_active": (now - self.last_cross_time) < 0.6
        }
