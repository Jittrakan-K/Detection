"""
app.py - Backend Server สำหรับระบบตรวจจับและนับจำนวนน็อต (YOLOv11 Counting Studio)
พัฒนาด้วย Flask + Ultralytics YOLOv11 + PyTorch CUDA
รองรับการเก็บ Dataset, สั่งเทรนโมเดล, และรัน Inference ตรวจจับแบบ Real-time
"""

import os
import sys
import time
import json
import base64
import threading
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string
import cv2

# Ultralytics & PyTorch
import torch
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__, static_folder=".", static_url_path="")

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PUT, DELETE'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Training state
train_state = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "progress_pct": 0.0,
    "loss": 0.0,
    "mAP50": 0.0,
    "logs": [],
    "status": "idle", # "idle", "training", "completed", "error"
    "error_message": "",
    "model_path": ""
}

# Current loaded YOLO model for inference
current_yolo_model = None
current_yolo_name = "yolo11n.pt"

def get_yolo_model(model_name="yolo11n.pt"):
    global current_yolo_model, current_yolo_name
    best_path = MODELS_DIR / "best.pt"
    
    # Priority: custom trained model > requested model
    target = str(best_path) if best_path.exists() and model_name == "custom" else model_name
    
    if current_yolo_model is None or current_yolo_name != target:
        try:
            device = 0 if torch.cuda.is_available() else "cpu"
            print(f"[AI] Loading YOLO model: {target} on {device}...")
            current_yolo_model = YOLO(target)
            current_yolo_name = target
        except Exception as e:
            print(f"[AI] Error loading {target}: {e}, falling back to yolo11n.pt")
            current_yolo_model = YOLO("yolo11n.pt")
            current_yolo_name = "yolo11n.pt"
            
    return current_yolo_model

# --- Helper: Setup dataset structure & data.yaml ---
def ensure_dataset_yaml(class_names=None):
    DATASET_DIR.mkdir(exist_ok=True)
    (DATASET_DIR / "images" / "train").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "labels" / "train").mkdir(parents=True, exist_ok=True)
    
    yaml_path = DATASET_DIR / "data.yaml"
    existing_classes = ["nut"]
    
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                ydata = yaml.safe_load(f)
                if ydata and "names" in ydata:
                    if isinstance(ydata["names"], dict):
                        existing_classes = list(ydata["names"].values())
                    elif isinstance(ydata["names"], list):
                        existing_classes = ydata["names"]
        except Exception:
            pass
            
    if class_names:
        for c in class_names:
            if c not in existing_classes:
                existing_classes.append(c)

    yaml_content = f"""# Ultralytics YOLOv11 Dataset Configuration
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/train

names:
"""
    for idx, name in enumerate(existing_classes):
        yaml_content += f"  {idx}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    return existing_classes

# --- API: System Information ---
@app.route("/api/system/info", methods=["GET"])
def system_info():
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU)"
    vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if has_cuda else 0
    
    best_pt = (MODELS_DIR / "best.pt").exists()
    best_onnx = (MODELS_DIR / "best.onnx").exists()

    return jsonify({
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "vram_gb": vram_gb,
        "pytorch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "has_trained_model": best_pt,
        "has_onnx_model": best_onnx
    })

@app.route("/api/system/shutdown", methods=["POST"])
def shutdown_app():
    def kill_soon():
        time.sleep(1)
        os._exit(0)
    threading.Thread(target=kill_soon, daemon=True).start()
    return jsonify({"success": True, "message": "Server shutting down..."})

# --- API: Dataset Stats & Sample Management ---
@app.route("/api/dataset/stats", methods=["GET"])
def dataset_stats():
    images_dir = DATASET_DIR / "images" / "train"
    labels_dir = DATASET_DIR / "labels" / "train"
    
    img_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    label_count = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0
    
    classes = ensure_dataset_yaml()
    
    # Get recent 8 thumbnails (base64)
    samples = []
    if images_dir.exists():
        recent_imgs = sorted(list(images_dir.glob("*.jpg")), key=lambda p: p.stat().st_mtime, reverse=True)[:8]
        for p in recent_imgs:
            try:
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                    samples.append({
                        "filename": p.name,
                        "data": f"data:image/jpeg;base64,{b64}"
                    })
            except Exception:
                pass

    return jsonify({
        "image_count": img_count,
        "label_count": label_count,
        "classes": classes,
        "recent_samples": samples
    })

@app.route("/api/dataset/capture", methods=["POST"])
def capture_dataset_sample():
    """
    รับภาพและ Bounding Boxes จากหน้าเว็บแล้วเซฟเป็นไฟล์ภาพและ YOLO label txt
    """
    try:
        data = request.json
        img_b64 = data.get("image")
        bboxes = data.get("bboxes", [])
        class_names = data.get("class_names", ["nut"])
        
        if not img_b64:
            return jsonify({"success": False, "error": "No image data"}), 400
            
        classes = ensure_dataset_yaml(class_names)
        
        # Decode base64
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        
        # Unique timestamp filename
        ts = int(time.time() * 1000)
        img_name = f"nut_{ts}.jpg"
        txt_name = f"nut_{ts}.txt"
        
        img_path = DATASET_DIR / "images" / "train" / img_name
        txt_path = DATASET_DIR / "labels" / "train" / txt_name
        
        image.save(img_path, "JPEG", quality=95)
        
        # Save YOLO annotation lines: class_id cx cy w h (normalized 0.0-1.0)
        lines = []
        for b in bboxes:
            cname = b.get("class_name", "nut")
            cid = classes.index(cname) if cname in classes else 0
            cx = max(0.0, min(1.0, float(b.get("cx", 0.5))))
            cy = max(0.0, min(1.0, float(b.get("cy", 0.5))))
            w = max(0.01, min(1.0, float(b.get("w", 0.1))))
            h = max(0.01, min(1.0, float(b.get("h", 0.1))))
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            
        return jsonify({
            "success": True,
            "filename": img_name,
            "bbox_count": len(bboxes)
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset/clear", methods=["POST"])
def clear_dataset():
    try:
        import shutil
        images_dir = DATASET_DIR / "images" / "train"
        labels_dir = DATASET_DIR / "labels" / "train"
        if images_dir.exists():
            for f in images_dir.glob("*.jpg"):
                f.unlink()
        if labels_dir.exists():
            for f in labels_dir.glob("*.txt"):
                f.unlink()
        return jsonify({"success": True, "message": "Dataset cleared"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/dataset/generate_synthetic", methods=["POST"])
def generate_synthetic():
    """
    สร้างตัวอย่างภาพน็อตจำลองแบบอัตโนมัติ (30 ตัวอย่าง)
    ช่วยให้ผู้ใช้ทดสอบกดเทรนโมเดลได้ทันทีแม้ยังไม่ได้ต่อกล้อง
    """
    try:
        classes = ensure_dataset_yaml(["nut"])
        images_dir = DATASET_DIR / "images" / "train"
        labels_dir = DATASET_DIR / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        count = 30
        for i in range(count):
            w_img, h_img = 640, 480
            # Background: light grey/white tray with slight noise
            bg_val = np.random.randint(235, 250)
            img = np.full((h_img, w_img, 3), bg_val, dtype=np.uint8)
            
            # Add random tray border lines
            cv2.line(img, (50, 0), (50, h_img), (210, 210, 210), 2)
            cv2.line(img, (w_img - 50, 0), (w_img - 50, h_img), (210, 210, 210), 2)
            
            # Draw 1-3 random nuts
            num_nuts = np.random.randint(1, 4)
            labels = []
            for _ in range(num_nuts):
                nx = np.random.randint(100, w_img - 100)
                ny = np.random.randint(60, h_img - 60)
                radius = np.random.randint(22, 38)
                angle_offset = np.random.uniform(0, np.pi)
                
                # Draw Hexagon
                pts = []
                for k in range(6):
                    ang = angle_offset + (k * np.pi / 3)
                    px = int(nx + radius * np.cos(ang))
                    py = int(ny + radius * np.sin(ang))
                    pts.append([px, py])
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                
                # Dark metallic color
                color_val = np.random.randint(30, 55)
                cv2.fillPoly(img, [pts], (color_val, color_val, color_val))
                
                # Center hole
                cv2.circle(img, (nx, ny), int(radius * 0.45), (bg_val, bg_val, bg_val), -1)
                
                # Compute bounding box
                bx1, by1 = max(0, nx - radius), max(0, ny - radius)
                bx2, by2 = min(w_img, nx + radius), min(h_img, ny + radius)
                bw = bx2 - bx1
                bh = by2 - by1
                cx = (bx1 + bw / 2) / w_img
                cy = (by1 + bh / 2) / h_img
                norm_w = bw / w_img
                norm_h = bh / h_img
                
                labels.append(f"0 {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}")
                
            ts = int(time.time() * 1000) + i
            img_path = images_dir / f"syn_nut_{ts}.jpg"
            txt_path = labels_dir / f"syn_nut_{ts}.txt"
            
            cv2.imwrite(str(img_path), img)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(labels))
                
        return jsonify({"success": True, "created": count})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- Training Worker Thread ---
def training_worker(epochs=20, batch=8, imgsz=640, base_model="yolo11n.pt"):
    global train_state
    try:
        train_state["is_training"] = True
        train_state["status"] = "training"
        train_state["total_epochs"] = epochs
        train_state["current_epoch"] = 0
        train_state["progress_pct"] = 0.0
        train_state["logs"] = [f"[INIT] เริ่มต้นการฝึกสอน YOLOv11 ด้วยโมเดลฐาน: {base_model}"]
        
        yaml_path = DATASET_DIR / "data.yaml"
        if not yaml_path.exists():
            ensure_dataset_yaml()
            
        device = 0 if torch.cuda.is_available() else "cpu"
        train_state["logs"].append(f"[DEVICE] ใช้งาน: {torch.cuda.get_device_name(0) if device==0 else 'CPU'}")
        
        model = YOLO(base_model)
        
        # Callback for epoch end
        def on_train_epoch_end(trainer):
            epoch = trainer.epoch + 1
            train_state["current_epoch"] = epoch
            train_state["progress_pct"] = round((epoch / epochs) * 100, 1)
            loss_val = float(trainer.loss.item()) if hasattr(trainer, "loss") and trainer.loss is not None else 0.0
            train_state["loss"] = round(loss_val, 4)
            train_state["logs"].append(f"[EPOCH {epoch}/{epochs}] Box Loss: {train_state['loss']}")
            if len(train_state["logs"]) > 50:
                train_state["logs"].pop(0)

        model.add_callback("on_train_epoch_end", on_train_epoch_end)

        train_state["logs"].append(f"[TRAIN] กำลังเริ่มการคำนวณ Epoch 1 ถึง {epochs}...")
        
        # Train
        project_dir = BASE_DIR / "runs" / "train"
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(project_dir),
            name="bolt_nut_detector",
            exist_ok=True,
            verbose=False
        )

        best_pt = project_dir / "bolt_nut_detector" / "weights" / "best.pt"
        if best_pt.exists():
            target_pt = MODELS_DIR / "best.pt"
            import shutil
            shutil.copy2(best_pt, target_pt)
            train_state["logs"].append(f"[SUCCESS] บันทึกโมเดลสำเร็จ: {target_pt}")
            
            # Export to ONNX
            train_state["logs"].append("[EXPORT] กำลังแปลงโมเดลเป็น ONNX สำหรับใช้งานบน Web...")
            try:
                trained_m = YOLO(str(target_pt))
                onnx_p = trained_m.export(format="onnx", imgsz=imgsz, dynamic=False)
                onnx_target = MODELS_DIR / "best.onnx"
                if Path(onnx_p).exists() and Path(onnx_p) != onnx_target:
                    shutil.copy2(onnx_p, onnx_target)
                train_state["logs"].append(f"[SUCCESS] สร้างไฟล์ ONNX สำเร็จ: {onnx_target}")
            except Exception as ex:
                train_state["logs"].append(f"[WARN] ONNX export warning: {ex}")

        train_state["is_training"] = False
        train_state["status"] = "completed"
        train_state["progress_pct"] = 100.0
        train_state["logs"].append("[COMPLETE] การเทรนเสร็จสมบูรณ์ 100%! พร้อมนำไปใช้งานนับวัตถุจริง")
        
        # Reload global model
        global current_yolo_model
        current_yolo_model = None

    except Exception as e:
        train_state["is_training"] = False
        train_state["status"] = "error"
        train_state["error_message"] = str(e)
        train_state["logs"].append(f"[ERROR] เกิดข้อผิดพลาด: {e}")

@app.route("/api/train/start", methods=["POST"])
def start_training():
    global train_state
    if train_state["is_training"]:
        return jsonify({"success": False, "error": "Training already in progress"}), 400
        
    data = request.json or {}
    epochs = int(data.get("epochs", 20))
    batch = int(data.get("batch", 8))
    imgsz = int(data.get("imgsz", 640))
    base_model = data.get("base_model", "yolo11n.pt")
    
    # Check if dataset has at least 1 image
    images_dir = DATASET_DIR / "images" / "train"
    img_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
    if img_count == 0:
        return jsonify({
            "success": False, 
            "error": "ยังไม่มีรูปใน Dataset กรุณากดถ่ายภาพหรือกดปุ่ม 'สร้างภาพตัวอย่างอัตโนมัติ' ก่อนกดเทรน"
        }), 400

    th = threading.Thread(target=training_worker, args=(epochs, batch, imgsz, base_model), daemon=True)
    th.start()
    
    return jsonify({"success": True, "message": "Training started"})

@app.route("/api/train/status", methods=["GET"])
def training_status():
    return jsonify(train_state)

# --- Real-time Inference API ---
@app.route("/api/inference/detect", methods=["POST"])
def run_inference():
    """
    รับภาพ Frame จาก Canvas แล้วรันโมเดล YOLOv11 ตรวจจับตำแหน่งน็อต
    ส่งผลลัพธ์ Bounding Box, Confidence และพิกัด Centroid กลับไปให้ Frontend นับ
    """
    try:
        data = request.json
        img_b64 = data.get("image")
        conf_thresh = float(data.get("confidence", 0.45))
        model_type = data.get("model", "custom")
        
        if not img_b64:
            return jsonify({"success": False, "error": "No image"}), 400
            
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
            
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        h_orig, w_orig = img.shape[:2]
        
        model = get_yolo_model(model_type)
        results = model.predict(source=img, conf=conf_thresh, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results[0].names.get(cls_id, f"class_{cls_id}")
                
                x1, y1, x2, y2 = xyxy
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                bw = x2 - x1
                bh = y2 - y1
                
                detections.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [round(x1, 1), round(y1, 1), round(bw, 1), round(bh, 1)],
                    "centroid": [round(cx, 1), round(cy, 1)],
                    "area": round(bw * bh, 1)
                })
                
        return jsonify({
            "success": True,
            "detections": detections,
            "model_used": current_yolo_name
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# --- Serve Web App ---
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/models/<path:filename>")
def serve_models(filename):
    return send_from_directory("models", filename)

if __name__ == "__main__":
    ensure_dataset_yaml()
    print("=" * 60)
    print("🚀 YOLOv11 Object Counting & Training Studio")
    print(f"  - Server Address: http://localhost:5000")
    print(f"  - GPU Active: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    # Auto open browser window in App Mode
    import subprocess
    import webbrowser
    def open_browser_window():
        time.sleep(1.2)
        chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
        edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
        try:
            if os.path.exists(chrome):
                subprocess.Popen([chrome, "--app=http://localhost:5000", "--window-size=1440,920"])
            elif os.path.exists(edge):
                subprocess.Popen([edge, "--app=http://localhost:5000", "--window-size=1440,920"])
            else:
                webbrowser.open("http://localhost:5000")
        except Exception:
            webbrowser.open("http://localhost:5000")

    threading.Thread(target=open_browser_window, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
