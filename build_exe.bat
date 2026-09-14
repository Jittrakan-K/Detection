@echo off
title Build Standalone Executable (.exe)
chcp 65001 > nul
echo ================================================================
echo   กำลังแปลงโปรเจกต์เป็นโปรแกรมสำเร็จรูป (.exe) แบบ Standalone
echo   (เครื่องอื่นจะสามารถเปิดใช้งานได้ทันทีโดยไม่ต้องติดตั้ง Python)
echo ================================================================
echo.
echo [1/2] กำลังรัน PyInstaller เพื่อรวมไฟล์ทั้งหมด...
echo      (ขั้นตอนนี้อาจใช้เวลาประมาณ 2-5 นาทีเนื่องจากมี PyTorch + AI Model)
echo.

pyinstaller --noconfirm AI_Object_Counter.spec

if errorlevel 1 (
    echo.
    echo [ERROR] การแปลงไฟล์ล้มเหลว กรุณาตรวจสอบข้อความด้านบน
    pause
    exit /b 1
)

echo.
echo ================================================================
echo   [สำเร็จ!] สร้างโปรแกรมสำเร็จรูปเรียบร้อยแล้ว
echo   โฟลเดอร์ผลลัพธ์อยู่ที่: dist\AI_Object_Counter\
echo.
echo   วิธีนำไปใช้ในเครื่องอื่น:
echo   - ก๊อปปี้โฟลเดอร์ dist\AI_Object_Counter ไปไว้ในเครื่องไหนก็ได้
echo   - ดับเบิลคลิกไฟล์ AI_Object_Counter.exe ใช้งานได้ทันที 100%%
echo ================================================================
pause
