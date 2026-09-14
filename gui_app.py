"""
Modern Desktop GUI Launcher for Object Counting System.
Built with Tkinter & TTK.
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import config
from detector import ObjectDetector
from tracker_counter import ObjectCounter
from utils.drawing import draw_bounding_box, draw_counting_line, draw_hud
import time


class ObjectCounterGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ระบบนับจำนวนสิ่งของจากกล้อง (AI Object Counter)")
        self.root.geometry("620x720")
        self.root.minsize(580, 680)

        # Styling
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # State
        self.is_running = False
        self.stop_event = threading.Event()
        self.capture_thread = None

        self._build_ui()

    def _build_ui(self):
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_lbl = ttk.Label(
            main_frame,
            text="🎯 AI Real-Time Object Counter",
            font=("Segoe UI", 16, "bold")
        )
        title_lbl.pack(pady=(0, 4))

        sub_lbl = ttk.Label(
            main_frame,
            text="โปรแกรมนับจำนวนสิ่งของเป้าหมายจากกล้องด้วย YOLOv8 + GPU",
            font=("Leelawadee UI", 10),
            foreground="#555555"
        )
        sub_lbl.pack(pady=(0, 14))

        # --- 1. Video Source Section ---
        src_group = ttk.LabelFrame(main_frame, text=" 1. แหล่งที่มาของภาพ (Video Source) ", padding=10)
        src_group.pack(fill=tk.X, pady=6)

        self.src_type_var = tk.StringVar(value="webcam")
        r_cam0 = ttk.Radiobutton(src_group, text="Webcam ในตัว (Index 0)", variable=self.src_type_var, value="webcam_0")
        r_cam0.grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        r_cam1 = ttk.Radiobutton(src_group, text="กล้อง USB นอก (Index 1)", variable=self.src_type_var, value="webcam_1")
        r_cam1.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        r_file = ttk.Radiobutton(src_group, text="ไฟล์วิดีโอ (MP4/AVI):", variable=self.src_type_var, value="file")
        r_file.grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)

        self.file_path_var = tk.StringVar(value="")
        self.file_entry = ttk.Entry(src_group, textvariable=self.file_path_var, width=32)
        self.file_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)

        btn_browse = ttk.Button(src_group, text="เลือกไฟล์...", command=self._browse_file)
        btn_browse.grid(row=1, column=2, padx=5, pady=4)

        # --- 2. Target Object Section ---
        obj_group = ttk.LabelFrame(main_frame, text=" 2. สิ่งของที่ต้องการนับ (Target Objects) ", padding=10)
        obj_group.pack(fill=tk.X, pady=6)

        self.target_mode_var = tk.StringVar(value="popular")

        r_all = ttk.Radiobutton(obj_group, text="ตรวจจับทุกอย่าง (All 80 Classes)", variable=self.target_mode_var, value="all")
        r_all.grid(row=0, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        r_pop = ttk.Radiobutton(obj_group, text="เลือกสิ่งของยอดนิยม:", variable=self.target_mode_var, value="popular")
        r_pop.grid(row=1, column=0, sticky=tk.W, padx=5, pady=4)

        # Popular combo
        popular_display = [f"{config.COCO_THAI_NAMES.get(c, c)} ({c})" for c in config.POPULAR_CLASSES]
        self.popular_combo = ttk.Combobox(obj_group, values=popular_display, state="readonly", width=25)
        self.popular_combo.current(0)  # Default: bottle
        self.popular_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=4)

        r_custom = ttk.Radiobutton(obj_group, text="ระบุชื่อ Class ภาษาอังกฤษ (คั่นด้วยวรรค):", variable=self.target_mode_var, value="custom")
        r_custom.grid(row=2, column=0, sticky=tk.W, padx=5, pady=4)

        self.custom_class_var = tk.StringVar(value="bottle cup")
        custom_entry = ttk.Entry(obj_group, textvariable=self.custom_class_var, width=28)
        custom_entry.grid(row=2, column=1, sticky=tk.W, padx=5, pady=4)

        # --- 3. Counting Mode & Sensitivity ---
        mode_group = ttk.LabelFrame(main_frame, text=" 3. โหมดการนับ & ความแม่นยำ ", padding=10)
        mode_group.pack(fill=tk.X, pady=6)

        lbl_mode = ttk.Label(mode_group, text="โหมดการนับ:")
        lbl_mode.grid(row=0, column=0, sticky=tk.W, padx=5, pady=4)

        self.mode_var = tk.StringVar(value=config.MODE_IN_FRAME)
        c_mode_in = ttk.Radiobutton(mode_group, text="1. นับเฉพาะในภาพ (In-Frame Live - ออกจอลดลง)", variable=self.mode_var, value=config.MODE_IN_FRAME)
        c_mode_in.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        c_mode_tray = ttk.Radiobutton(mode_group, text="2. นับสะสมของในถาด (Tray Counter - ออกจอเลขไม่ลด)", variable=self.mode_var, value=config.MODE_TRAY)
        c_mode_tray.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        c_mode_line = ttk.Radiobutton(mode_group, text="3. นับข้ามเส้นสะสม (Line-Crossing)", variable=self.mode_var, value=config.MODE_LINE_CROSSING)
        c_mode_line.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Confidence Slider
        lbl_conf = ttk.Label(mode_group, text="Confidence Threshold:")
        lbl_conf.grid(row=3, column=0, sticky=tk.W, padx=5, pady=6)

        self.conf_val_lbl = ttk.Label(mode_group, text="40%")
        self.conf_val_lbl.grid(row=2, column=2, padx=5, pady=6)

        self.conf_slider = ttk.Scale(mode_group, from_=0.15, to=0.90, value=0.40, command=self._on_slider_change)
        self.conf_slider.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=6)

        # Model size
        lbl_model = ttk.Label(mode_group, text="ขนาดโมเดล YOLO:")
        lbl_model.grid(row=3, column=0, sticky=tk.W, padx=5, pady=4)

        self.model_var = tk.StringVar(value="yolov8n.pt")
        model_combo = ttk.Combobox(mode_group, textvariable=self.model_var, values=["yolov8n.pt (เร็วสุด)", "yolov8s.pt (แม่นยำขึ้น)"], state="readonly", width=22)
        model_combo.current(0)
        model_combo.grid(row=3, column=1, sticky=tk.W, padx=5, pady=4)

        # --- Status & Actions ---
        btn_frame = ttk.Frame(main_frame, padding=10)
        btn_frame.pack(fill=tk.X, pady=10)

        self.btn_start = tk.Button(
            btn_frame,
            text="▶  เริ่มเปิดกล้องตรวจจับ (Start)",
            font=("Segoe UI", 12, "bold"),
            bg="#28a745",
            fg="white",
            activebackground="#218838",
            activeforeground="white",
            relief=tk.FLAT,
            height=2,
            command=self.start_counting
        )
        self.btn_start.pack(fill=tk.X)

        self.status_lbl = ttk.Label(
            main_frame,
            text="สถานะ: พร้อมใช้งาน (กดปุ่ม Start เพื่อเริ่มเปิดหน้าต่างกล้อง)",
            font=("Leelawadee UI", 9),
            foreground="#28a745"
        )
        self.status_lbl.pack(pady=4)

    def _browse_file(self):
        f = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All Files", "*.*")]
        )
        if f:
            self.file_path_var.set(f)
            self.src_type_var.set("file")

    def _on_slider_change(self, val):
        self.conf_val_lbl.config(text=f"{float(val)*100:.0f}%")

    def get_source(self):
        st = self.src_type_var.get()
        if st == "webcam_0":
            return 0
        elif st == "webcam_1":
            return 1
        elif st == "file":
            fp = self.file_path_var.get().strip()
            if not fp or not os.path.exists(fp):
                messagebox.showerror("Error", "กรุณาเลือกไฟล์วิดีโอที่ถูกต้อง")
                return None
            return fp
        return 0

    def get_target_classes(self):
        tm = self.target_mode_var.get()
        if tm == "all":
            return None
        elif tm == "popular":
            idx = self.popular_combo.current()
            if 0 <= idx < len(config.POPULAR_CLASSES):
                return [config.POPULAR_CLASSES[idx]]
            return ["bottle"]
        elif tm == "custom":
            raw = self.custom_class_var.get().strip()
            if not raw:
                return None
            return raw.split()
        return None

    def start_counting(self):
        if self.is_running:
            return

        source = self.get_source()
        if source is None:
            return

        target_classes = self.get_target_classes()
        mode = self.mode_var.get()
        conf = float(self.conf_slider.get())
        model_sel = self.model_var.get().split()[0]  # Take "yolov8n.pt"

        self.is_running = True
        self.stop_event.clear()
        self.btn_start.config(text="⏳ กำลังเปิดกล้อง...", state=tk.DISABLED, bg="#6c757d")
        self.status_lbl.config(text="กำลังโหลดโมเดลและเปิดกล้อง...", foreground="#007bff")

        # Start processing loop in background thread
        self.capture_thread = threading.Thread(
            target=self._run_loop,
            args=(source, target_classes, mode, conf, model_sel),
            daemon=True
        )
        self.capture_thread.start()

    def _run_loop(self, source, target_classes, mode, conf, model_name):
        try:
            detector = ObjectDetector(
                model_name_or_path=model_name,
                device=config.DEVICE,
                conf_threshold=conf
            )
            counter = ObjectCounter(mode=mode)

            cap = cv2.VideoCapture(
                source,
                cv2.CAP_DSHOW if isinstance(source, int) and sys.platform.startswith("win") else cv2.CAP_ANY
            )
            if not cap.isOpened() and isinstance(source, int):
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                self.root.after(0, lambda: messagebox.showerror("Error", f"ไม่สามารถเปิดกล้อง/วิดีโอ (Source: {source})"))
                self.root.after(0, self._on_stop)
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            window_name = "AI Object Counter (Press 'Q' on window to stop)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            self.root.after(0, lambda: self.status_lbl.config(
                text="● กำลังตรวจจับแบบ Real-time (กด 'Q' ในหน้าต่างกล้องเพื่อหยุด)",
                foreground="#28a745"
            ))

            prev_time = time.time()
            fps = 30.0
            device_label = "GPU" if "cuda" in config.DEVICE else "CPU"

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt)

                if counter.mode in (config.MODE_TRAY, config.MODE_LINE_CROSSING):
                    detections = detector.track(
                        frame=frame,
                        target_classes=target_classes,
                        conf=conf,
                        persist=True
                    )
                else:
                    detections = detector.detect(
                        frame=frame,
                        target_classes=target_classes,
                        conf=conf
                    )

                stats = counter.update(detections, frame.shape)

                if counter.mode == config.MODE_LINE_CROSSING:
                    draw_counting_line(frame, stats["line_pixel"], is_active=stats["is_line_active"])

                for det in detections:
                    frame = draw_bounding_box(frame, det)

                frame = draw_hud(
                    frame=frame,
                    stats=stats,
                    fps=fps,
                    target_classes=target_classes,
                    conf_threshold=conf,
                    device_info=device_label
                )

                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('r'), ord('R')):
                    counter.reset_counts()
                elif key in (ord('m'), ord('M')):
                    counter.toggle_mode()
                elif key in (ord('s'), ord('S')):
                    os.makedirs("screenshots", exist_ok=True)
                    fn = os.path.join("screenshots", f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(fn, frame)

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"[Error in loop] {e}")
            self.root.after(0, lambda: messagebox.showerror("Runtime Error", str(e)))

        finally:
            self.root.after(0, self._on_stop)

    def _on_stop(self):
        self.is_running = False
        self.btn_start.config(text="▶  เริ่มเปิดกล้องตรวจจับ (Start)", state=tk.NORMAL, bg="#28a745")
        self.status_lbl.config(text="สถานะ: พร้อมใช้งาน (หยุดการทำงานแล้ว)", foreground="#555555")


def main():
    root = tk.Tk()
    app = ObjectCounterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
