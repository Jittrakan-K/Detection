"""
Object Detector module using Ultralytics YOLO with GPU acceleration.
"""
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO
import config


class Detection:
    """Represents a single detected object in a frame."""
    def __init__(
        self,
        class_id: int,
        class_name: str,
        thai_name: str,
        confidence: float,
        box: Tuple[int, int, int, int],  # x1, y1, x2, y2
        track_id: Optional[int] = None
    ):
        self.class_id = class_id
        self.class_name = class_name
        self.thai_name = thai_name
        self.confidence = confidence
        self.box = box
        self.track_id = track_id
        # Calculate center point
        x1, y1, x2, y2 = box
        self.center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def label(self) -> str:
        """Formatted label string for display."""
        tid_str = f"#{self.track_id} " if self.track_id is not None else ""
        return f"{tid_str}{self.thai_name} ({self.confidence:.0%})"


class ObjectDetector:
    """Wrapper around YOLO model providing detection, tracking, and class filtering."""
    def __init__(
        self,
        model_name_or_path: str = config.DEFAULT_MODEL,
        device: str = config.DEVICE,
        conf_threshold: float = config.DEFAULT_CONF,
        iou_threshold: float = config.DEFAULT_IOU,
    ):
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model_path = model_name_or_path

        print(f"[Detector] Loading YOLO model: '{model_name_or_path}' on {device}...")
        self.model = YOLO(model_name_or_path)
        # Class mapping
        self.names: Dict[int, str] = self.model.names
        self.name_to_id: Dict[str, int] = {v.lower(): k for k, v in self.names.items()}
        print(f"[Detector] Model loaded successfully. Total classes: {len(self.names)}")

    def get_class_id(self, class_name: str) -> Optional[int]:
        """Convert a class name (e.g. 'bottle') to YOLO class ID."""
        return self.name_to_id.get(class_name.lower().strip())

    def get_thai_name(self, class_name: str) -> str:
        """Get Thai translation for class name if available, else return original."""
        return config.COCO_THAI_NAMES.get(class_name.lower().strip(), class_name)

    def resolve_target_classes(self, target_classes: Optional[List[str]]) -> Optional[List[int]]:
        """
        Convert list of class names (e.g. ['bottle', 'cup']) into list of class IDs.
        If target_classes is None or empty or contains 'all', returns None (detect all).
        """
        if not target_classes:
            return None
        
        # Check if user wants all
        if any(c.lower() in ("all", "*", "ทั้งหมด") for c in target_classes):
            return None

        class_ids = []
        for name in target_classes:
            cid = self.get_class_id(name)
            if cid is not None:
                class_ids.append(cid)
            else:
                # Try matching by thai name
                for en, th in config.COCO_THAI_NAMES.items():
                    if th == name:
                        cid = self.get_class_id(en)
                        if cid is not None:
                            class_ids.append(cid)
                        break
        return class_ids if class_ids else None

    def track(
        self,
        frame: np.ndarray,
        target_classes: Optional[List[str]] = None,
        conf: Optional[float] = None,
        tracker: str = config.DEFAULT_TRACKER,
        persist: bool = True
    ) -> List[Detection]:
        """
        Run tracking on a single frame. Returns list of Detection instances with track_id.
        """
        conf_val = conf if conf is not None else self.conf_threshold
        classes_arg = self.resolve_target_classes(target_classes)

        # Run Ultralytics tracker
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=tracker,
            conf=conf_val,
            iou=self.iou_threshold,
            classes=classes_arg,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        track_ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(boxes)

        for box, score, cid, tid in zip(boxes, confidences, class_ids, track_ids):
            cname = self.names.get(cid, f"class_{cid}")
            th_name = self.get_thai_name(cname)
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=cname,
                    thai_name=th_name,
                    confidence=float(score),
                    box=(x1, y1, x2, y2),
                    track_id=int(tid) if tid is not None else None
                )
            )

        return detections

    def detect(
        self,
        frame: np.ndarray,
        target_classes: Optional[List[str]] = None,
        conf: Optional[float] = None
    ) -> List[Detection]:
        """
        Standard detection without persistent tracking.
        """
        conf_val = conf if conf is not None else self.conf_threshold
        classes_arg = self.resolve_target_classes(target_classes)

        results = self.model(
            source=frame,
            conf=conf_val,
            iou=self.iou_threshold,
            classes=classes_arg,
            device=self.device,
            verbose=False
        )

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes.xyxy.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, score, cid in zip(boxes, confidences, class_ids):
            cname = self.names.get(cid, f"class_{cid}")
            th_name = self.get_thai_name(cname)
            x1, y1, x2, y2 = [int(v) for v in box]
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=cname,
                    thai_name=th_name,
                    confidence=float(score),
                    box=(x1, y1, x2, y2),
                    track_id=None
                )
            )

        return detections
