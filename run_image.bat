@echo off
cd /d "%~dp0"
echo ========================================
echo   人脸表情识别 - 单张图片识别
echo ========================================
if "%~1"=="" (
    echo 用法: run_image.bat 图片路径 [输出路径]
    echo 示例: run_image.bat test.jpg result.jpg
    pause
    exit /b
)
"%~dp0venv\Scripts\python.exe" main.py image --model saved_model/best_model.h5 --input %1 --output %2
pause
