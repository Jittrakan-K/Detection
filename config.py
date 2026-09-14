"""
Configuration module for Real-Time Object Detection & Counting System.
"""
import torch

# GPU / Hardware device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# YOLO Model path or name
DEFAULT_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt or custom.pt

# Detection & Tracking thresholds
DEFAULT_CONF = 0.40
DEFAULT_IOU = 0.45
DEFAULT_TRACKER = "bytetrack.yaml"

# Camera Source (0 for built-in webcam, 1 for external, or "path/to/video.mp4")
DEFAULT_SOURCE = 0

# Counting Modes
MODE_IN_FRAME = "in_frame"          # 1. Counts visible objects in the current frame (decreases when object leaves)
MODE_TRAY = "tray"                  # 2. Cumulative Tray/Inventory Counter (counts once, persists when object leaves)
MODE_LINE_CROSSING = "line_crossing" # 3. Counts objects crossing a virtual counting line
DEFAULT_MODE = MODE_IN_FRAME

# Default normalized counting line ((x1_ratio, y1_ratio), (x2_ratio, y2_ratio))
# Default is horizontal line across the middle of the screen
DEFAULT_LINE_COORDS = ((0.05, 0.55), (0.95, 0.55))

# Common COCO Classes dictionary with English & Thai names
COCO_THAI_NAMES = {
    "person": "คน",
    "bicycle": "จักรยาน",
    "car": "รถยนต์",
    "motorcycle": "มอเตอร์ไซค์",
    "airplane": "เครื่องบิน",
    "bus": "รถบัส",
    "train": "รถไฟ",
    "truck": "รถบรรทุก",
    "boat": "เรือ",
    "traffic light": "สัญญาณไฟจราจร",
    "fire hydrant": "หัวดับเพลิง",
    "stop sign": "ป้ายหยุด",
    "bench": "ม้านั่ง",
    "bird": "นก",
    "cat": "แมว",
    "dog": "สุนัข",
    "horse": "ม้า",
    "sheep": "แกะ",
    "cow": "วัว",
    "elephant": "ช้าง",
    "bear": "หมี",
    "zebra": "ม้าลาย",
    "giraffe": "ยีราฟ",
    "backpack": "กระเป๋าเป้",
    "umbrella": "ร่ม",
    "handbag": "กระเป๋าถือ",
    "tie": "เนกไท",
    "suitcase": "กระเป๋าเดินทาง",
    "frisbee": "จานร่อน",
    "skis": "สกี",
    "snowboard": "สโนว์บอร์ด",
    "sports ball": "ลูกบอล",
    "kite": "ว่าว",
    "baseball bat": "ไม้เบสบอล",
    "baseball glove": "ถุงมือเบสบอล",
    "skateboard": "สเก็ตบอร์ด",
    "surfboard": "กระดานโต้คลื่น",
    "tennis racket": "ไม้เทนนิส",
    "bottle": "ขวดน้ำ",
    "wine glass": "แก้วไวน์",
    "cup": "แก้วน้ำ/ถ้วย",
    "fork": "ส้อม",
    "knife": "มีด",
    "spoon": "ช้อน",
    "bowl": "ชาม",
    "banana": "กล้วย",
    "apple": "แอปเปิ้ล",
    "sandwich": "แซนด์วิช",
    "orange": "ส้ม",
    "broccoli": "บรอกโคลี",
    "carrot": "แครอท",
    "hot dog": "ฮอทด็อก",
    "pizza": "พิซซ่า",
    "donut": "โดนัท",
    "cake": "เค้ก",
    "chair": "เก้าอี้",
    "couch": "โซฟา",
    "potted plant": "ต้นไม้ในกระถาง",
    "bed": "เตียง",
    "dining table": "โต๊ะอาหาร",
    "toilet": "ชักโครก",
    "tv": "โทรทัศน์/จอภาพ",
    "laptop": "โน้ตบุ๊ก",
    "mouse": "เมาส์",
    "remote": "รีโมท",
    "keyboard": "คีย์บอร์ด",
    "cell phone": "โทรศัพท์มือถือ",
    "microwave": "ไมโครเวฟ",
    "oven": "เตาอบ",
    "toaster": "เครื่องปิ้งขนมปัง",
    "sink": "อ่างล้างจาน",
    "refrigerator": "ตู้เย็น",
    "book": "หนังสือ",
    "clock": "นาฬิกา",
    "vase": "แจกัน",
    "scissors": "กรรไกร",
    "teddy bear": "ตุ๊กตาหมี",
    "hair drier": "ไดร์เป่าผม",
    "toothbrush": "แปรงสีฟัน",
}

# Popular Quick Select categories
POPULAR_CLASSES = [
    "bottle",       # ขวดน้ำ
    "cup",          # แก้วน้ำ
    "person",       # คน
    "cell phone",   # โทรศัพท์มือถือ
    "book",         # หนังสือ
    "chair",        # เก้าอี้
    "car",          # รถยนต์
    "laptop",       # โน้ตบุ๊ก
    "mouse",        # เมาส์
    "apple",        # แอปเปิ้ล
]

# Color Palette (BGR for OpenCV)
COLORS = {
    "primary": (255, 140, 0),     # Dark Orange / Amber
    "secondary": (0, 200, 255),   # Bright Yellow
    "accent": (50, 220, 50),      # Bright Green
    "danger": (60, 60, 255),      # Bright Red
    "background": (20, 20, 20),   # Dark background
    "card_bg": (35, 35, 35),      # Card background
    "text": (245, 245, 245),      # Light text
    "line": (0, 165, 255),        # Orange Counting Line
    "line_active": (0, 255, 0),   # Green when crossed
}
