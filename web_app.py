"""
Flask Web Application for Real-Time Object Detection & Counting System.
Provides MJPEG video stream, REST APIs, and serves modern HTML5 Dashboard.
"""
import os
import sys
import time
import threading
import json
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, request, jsonify, send_file

import config
from detector import ObjectDetector
from tracker_counter import ObjectCounter
from utils.drawing import draw_bounding_box, draw_counting_line, draw_hud

app = Flask(__name__)

# Global State Container
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.device = config.DEVICE
        self.source = config.DEFAULT_SOURCE
        self.model_name = config.DEFAULT_MODEL
        self.mode = config.DEFAULT_MODE
        self.conf = config.DEFAULT_CONF
        self.target_classes: Optional[List[str]] = ["bottle"]  # default target: bottle
        self.line_y_ratio = 0.55
        
        # Detector & Counter
        self.detector = ObjectDetector(
            model_name_or_path=self.model_name,
            device=self.device,
            conf_threshold=self.conf
        )
        self.counter = ObjectCounter(mode=self.mode)
        self.counter.line_normalized = ((0.05, self.line_y_ratio), (0.95, self.line_y_ratio))

        # Runtime metrics
        self.fps = 30.0
        self.latest_stats: Dict[str, Any] = {}
        self.latest_frame: Optional[np.ndarray] = None
        self.is_camera_connected = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = True

state = AppState()


def camera_loop():
    """Background thread that continuously reads camera frames and runs inference."""
    global state
    print(f"[Camera Thread] Starting video capture on source: {state.source}")

    def open_camera(src):
        if isinstance(src, int) and sys.platform.startswith("win"):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return cap

    cap = open_camera(state.source)
    state.cap = cap
    state.is_camera_connected = cap.isOpened()

    prev_time = time.time()
    current_source = state.source

    while state.running:
        # Check if source changed
        if state.source != current_source:
            print(f"[Camera Thread] Switching source from {current_source} to {state.source}")
            if cap and cap.isOpened():
                cap.release()
            cap = open_camera(state.source)
            state.cap = cap
            current_source = state.source
            state.is_camera_connected = cap.isOpened()

        if not cap or not cap.isOpened():
            state.is_camera_connected = False
            # Generate animated placeholder frame
            placeholder = np.full((720, 1280, 3), 30, dtype=np.uint8)
            cv2.putText(
                placeholder,
                f"Camera source [{state.source}] not connected",
                (320, 340),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 165, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                placeholder,
                "Please check camera index or select a video file in Settings.",
                (300, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                1,
                cv2.LINE_AA
            )
            with state.lock:
                state.latest_frame = placeholder
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            # If video file ended, loop back to start
            if isinstance(state.source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            state.is_camera_connected = False
            time.sleep(0.05)
            continue

        state.is_camera_connected = True
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        if dt > 0:
            state.fps = 0.9 * state.fps + 0.1 * (1.0 / dt)

        # Process inference with detector & counter
        with state.lock:
            mode = state.mode
            target_classes = state.target_classes
            conf = state.conf

        if mode in (config.MODE_TRAY, config.MODE_LINE_CROSSING):
            detections = state.detector.track(
                frame=frame,
                target_classes=target_classes,
                conf=conf,
                persist=True
            )
        else:
            detections = state.detector.detect(
                frame=frame,
                target_classes=target_classes,
                conf=conf
            )

        # Update stats
        stats = state.counter.update(detections, frame.shape)

        # Visual Drawing
        if mode == config.MODE_LINE_CROSSING:
            draw_counting_line(frame, stats["line_pixel"], is_active=stats["is_line_active"])

        for det in detections:
            frame = draw_bounding_box(frame, det)

        device_label = "GPU" if "cuda" in state.device else "CPU"
        frame = draw_hud(
            frame=frame,
            stats=stats,
            fps=state.fps,
            target_classes=target_classes,
            conf_threshold=conf,
            device_info=device_label
        )

        with state.lock:
            state.latest_frame = frame
            state.latest_stats = stats

    if cap and cap.isOpened():
        cap.release()
    print("[Camera Thread] Stopped.")


# Start camera capture thread on module load
cam_thread = threading.Thread(target=camera_loop, daemon=True)
cam_thread.start()


def generate_mjpeg_stream():
    """Generator yielding MJPEG multipart frames for HTML5 <img> or video element."""
    while True:
        with state.lock:
            frame = state.latest_frame
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.025)  # ~40 fps max stream rate


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    return response


@app.route('/')
@app.route('/index.html')
def index():
    """Serve main HTML5 web application page."""
    root_index = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(root_index):
        return send_file(root_index)
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """MJPEG stream endpoint."""
    return Response(
        generate_mjpeg_stream(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Returns current detection and counting statistics in JSON."""
    with state.lock:
        s = dict(state.latest_stats)
        s['fps'] = round(state.fps, 1)
        s['device'] = state.device
        s['device_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        s['is_connected'] = state.is_camera_connected
        s['mode'] = state.mode
        s['conf'] = state.conf
        s['target_classes'] = state.target_classes
        s['line_y'] = state.line_y_ratio
    return jsonify(s)


@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Return all available classes with English & Thai names."""
    popular = []
    for c in config.POPULAR_CLASSES:
        popular.append({
            "id": c,
            "en": c,
            "th": config.COCO_THAI_NAMES.get(c, c)
        })

    all_classes = []
    for cid, cname in state.detector.names.items():
        all_classes.append({
            "id": cname,
            "en": cname,
            "th": config.COCO_THAI_NAMES.get(cname, cname)
        })

    return jsonify({
        "popular": popular,
        "all": all_classes
    })


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update runtime configuration parameters."""
    data = request.get_json() or {}
    with state.lock:
        if 'mode' in data and data['mode'] in (config.MODE_IN_FRAME, config.MODE_TRAY, config.MODE_LINE_CROSSING):
            state.mode = data['mode']
            state.counter.set_mode(state.mode)

        if 'target_classes' in data:
            tc = data['target_classes']
            if tc is None or (isinstance(tc, list) and len(tc) == 0) or 'all' in tc:
                state.target_classes = None
            else:
                state.target_classes = tc

        if 'conf' in data:
            state.conf = max(0.1, min(0.95, float(data['conf'])))

        if 'line_y' in data:
            ly = max(0.1, min(0.9, float(data['line_y'])))
            state.line_y_ratio = ly
            state.counter.line_normalized = ((0.05, ly), (0.95, ly))

        if 'source' in data:
            src = str(data['source']).strip()
            state.source = int(src) if src.isdigit() else src

    return jsonify({"status": "success", "settings": {
        "mode": state.mode,
        "target_classes": state.target_classes,
        "conf": state.conf,
        "line_y": state.line_y_ratio,
        "source": state.source
    }})


@app.route('/api/reset', methods=['POST'])
def reset_counts():
    """Reset counters to zero."""
    with state.lock:
        state.counter.reset_counts()
    return jsonify({"status": "success", "message": "Counts reset."})


@app.route('/api/snapshot', methods=['GET'])
def take_snapshot():
    """Capture current annotated frame and download it as an image."""
    with state.lock:
        frame = state.latest_frame.copy() if state.latest_frame is not None else None
    if frame is None:
        return jsonify({"error": "No frame available"}), 400

    os.makedirs("screenshots", exist_ok=True)
    fn = os.path.join("screenshots", f"web_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(fn, frame)
    return send_file(os.path.abspath(fn), mimetype='image/jpeg', as_attachment=True, download_name=os.path.basename(fn))


@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """Endpoint for processing client-side camera frames sent via HTTP POST."""
    import base64
    try:
        data = request.get_json() or {}
        img_b64 = data.get('image', '')
        if ',' in img_b64:
            img_b64 = img_b64.split(',', 1)[1]

        if not img_b64:
            return jsonify({"error": "No image data provided"}), 400

        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "Failed to decode frame"}), 400

        with state.lock:
            mode = state.mode
            target_classes = state.target_classes
            conf = state.conf

        if mode in (config.MODE_TRAY, config.MODE_LINE_CROSSING):
            detections = state.detector.track(frame=frame, target_classes=target_classes, conf=conf, persist=True)
        else:
            detections = state.detector.detect(frame=frame, target_classes=target_classes, conf=conf)

        stats = state.counter.update(detections, frame.shape)

        det_list = []
        for d in detections:
            det_list.append({
                "class_id": d.class_id,
                "class_name": d.class_name,
                "thai_name": d.thai_name,
                "conf": round(d.confidence, 2),
                "box": list(d.box),
                "center": list(d.center),
                "track_id": d.track_id
            })

        return jsonify({
            "status": "success",
            "detections": det_list,
            "stats": {
                "mode": mode,
                "total_in_frame": stats.get("total_in_frame", 0),
                "tray_total": stats.get("tray_total", 0),
                "total_crossed": stats.get("total_crossed", 0),
                "total_in": stats.get("total_in", 0),
                "total_out": stats.get("total_out", 0)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"\n=======================================================")
    print(f"  AI Object Counter HTML5 Web Dashboard running at:")
    print(f"  --> http://localhost:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
