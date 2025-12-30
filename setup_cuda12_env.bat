@echo off
REM Setup CUDA 12 environment for ONNX models
REM Run this script once to create the environment

echo ============================================
echo Setting up cuda12_env for ONNX GPU acceleration
echo ============================================

echo.
echo Step 1: Creating conda environment...
call conda create -n cuda12_env python=3.10 -y

echo.
echo Step 2: Installing PyTorch with CUDA 12.1...
call conda run -n cuda12_env pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo.
echo Step 3: Installing dependencies from requirements_cuda12.txt...
call conda run -n cuda12_env pip install -r requirements_cuda12.txt

echo.
echo ============================================
echo Setup complete!
echo.
echo To verify CUDA is working, run:
echo   conda activate cuda12_env
echo   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo.
echo The pipeline will automatically use cuda12_env for ONNX models.
echo ============================================
