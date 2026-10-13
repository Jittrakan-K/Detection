@echo off
title ปิดระบบ YOLOv11 Counter
echo กำลังปิดระบบและคืนหน่วยความจำ GPU ทั้งหมด...
taskkill /f /im pythonw.exe >nul 2>&1
echo ปิดระบบเรียบร้อยแล้ว
timeout /t 2 >nul
