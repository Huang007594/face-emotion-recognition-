@echo off
cd /d "%~dp0"
echo ========================================
echo   人脸表情识别 - 摄像头实时识别
echo   按 Q 退出, S 截图, H 帮助
echo ========================================
"%~dp0venv\Scripts\python.exe" main.py camera --model saved_model/best_model.h5
pause
