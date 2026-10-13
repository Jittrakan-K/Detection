@echo off
title YOLOv11 Industrial Object Counter & Training Studio
color 0B
echo ================================================================
echo    YOLOv11 Industrial Object Counter & Training Studio
echo ================================================================
echo กำลังเปิดระบบ AI Object Counter & Training Studio...
echo URL: http://localhost:5000
echo.
timeout /t 2 >nul
start http://localhost:5000
python app.py
pause
