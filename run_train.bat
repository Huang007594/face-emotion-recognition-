@echo off
cd /d "%~dp0"
echo ========================================
echo   人脸表情识别 - 训练脚本
echo   50 epochs, batch_size=64
echo ========================================
"%~dp0venv\Scripts\python.exe" main.py train --epochs 50 --batch_size 64
echo.
echo 训练完成！模型保存在 saved_model/ 目录
pause
