"""
train_yolo11.py - สคริปต์ฝึกสอนโมเดล YOLOv11 สำหรับตรวจจับวัตถุเฉพาะทาง (เช่น น็อต)
รองรับการรันบน GPU (CUDA) และส่งออกผลลัพธ์เป็น ONNX อัตโนมัติ
"""

import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

def setup_directories():
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "dataset"
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
    return dataset_dir

def ensure_data_yaml(dataset_dir, class_names=None):
    if class_names is None or len(class_names) == 0:
        class_names = ["nut"]
    
    yaml_path = dataset_dir / "data.yaml"
    content = f"""# Ultralytics YOLOv11 Dataset Configuration
path: {dataset_dir.as_posix()}
train: images/train
val: images/train  # ใช้ train เป็น val หากข้อมูลน้อย

names:
"""
    for idx, name in enumerate(class_names):
        content += f"  {idx}: {name}\n"
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[INFO] Created/Updated data.yaml at {yaml_path}")
    return yaml_path

def train_model(data_yaml, model_name="yolo11n.pt", epochs=30, batch_size=8, imgsz=640, device=""):
    # Detect device
    if not device:
        device = 0 if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print(f"🚀 เริ่มการฝึกสอน YOLOv11 Model")
    print(f"  - Base Model: {model_name}")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Image Size: {imgsz}")
    print(f"  - Hardware Device: {device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"  - GPU Name: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Initialize model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(Path(__file__).resolve().parent / "runs" / "train"),
        name="bolt_nut_detector",
        exist_ok=True,
        verbose=True
    )

    print("\n[SUCCESS] การฝึกสอนสำเร็จเรียบร้อยแล้ว!")
    best_pt = Path(__file__).resolve().parent / "runs" / "train" / "bolt_nut_detector" / "weights" / "best.pt"
    
    if best_pt.exists():
        print(f"[INFO] พบไฟล์โมเดลที่แม่นยำที่สุด: {best_pt}")
        # Copy to models directory
        models_dir = Path(__file__).resolve().parent / "models"
        models_dir.mkdir(exist_ok=True)
        target_pt = models_dir / "best.pt"
        import shutil
        shutil.copy2(best_pt, target_pt)
        print(f"[INFO] คัดลอกไปยัง {target_pt}")

        # Export to ONNX for HTML5 web inference
        print("[INFO] กำลังส่งออกโมเดลเป็นรูปแบบ ONNX สำหรับรันบน Web Browser...")
        try:
            trained_model = YOLO(str(target_pt))
            onnx_path = trained_model.export(format="onnx", imgsz=imgsz, dynamic=False)
            print(f"[SUCCESS] สร้างไฟล์ ONNX สำเร็จที่: {onnx_path}")
        except Exception as e:
            print(f"[WARNING] ไม่สามารถ export เป็น ONNX ได้: {e}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv11 Custom Training for Nut/Bolt Counter")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (yolo11n.pt, yolo11s.pt, etc.)")
    parser.add_argument("--device", type=str, default="", help="Device: 0 or cpu")
    args = parser.parse_args()

    ds_dir = setup_directories()
    yaml_file = ds_dir / "data.yaml"
    if not yaml_file.exists():
        yaml_file = ensure_data_yaml(ds_dir, ["nut"])

    train_model(
        data_yaml=yaml_file,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
