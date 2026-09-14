@echo off
title AI Object Counter - Installation
chcp 65001 > nul
echo ========================================================
echo   กำลังติดตั้งแพ็กเกจที่จำเป็นสำหรับ AI Object Counter
echo ========================================================
echo.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo ========================================================
echo   ติดตั้งเรียบร้อยแล้ว! คุณสามารถดับเบิลคลิก run.bat เพื่อเปิดใช้งานได้ทันที
echo ========================================================
pause
