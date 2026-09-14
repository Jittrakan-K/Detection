@echo off
title AI Object Counter Pro
chcp 65001 > nul
echo ========================================================
echo   กำลังเริ่มต้นระบบ AI Object Counter Pro...
echo ========================================================
echo.
python index.py
if errorlevel 1 (
    echo.
    echo [ERROR] เกิดข้อผิดพลาด กรุณาลองรัน setup.bat ก่อน
    pause
)
