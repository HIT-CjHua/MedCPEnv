@echo off
REM MedAgent Agentic RL Multi-GPU Training Launch Script (Windows)
REM
REM 使用 accelerate 进行多卡并行训练
REM
REM 使用方法:
REM   scripts\train_multi_gpu.bat [OPTIONS]
REM
REM 示例:
REM   scripts\train_multi_gpu.bat --model Qwen/Qwen2.5-3B --max-steps 1000
REM   scripts\train_multi_gpu.bat --config deepspeed_zero2 --model Qwen/Qwen2.5-7B --disable-kb

setlocal enabledelayedexpansion

REM 默认配置
set CONFIG=multi_gpu
set NUM_GPUS=2
set MODEL=Qwen/Qwen2.5-3B
set MAX_STEPS=500
set BATCH_SIZE=4
set DATA=data/train.jsonl
set DISABLE_KB=0

REM 解析命令行参数
:parse_args
if "%~1"=="" goto :run
if "%~1"=="--config" (
    set CONFIG=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--num-gpus" (
    set NUM_GPUS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--model" (
    set MODEL=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max-steps" (
    set MAX_STEPS=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--data" (
    set DATA=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--disable-kb" (
    set DISABLE_KB=1
    shift
    goto :parse_args
)
if "%~1"=="--help" (
    echo Usage: scripts\train_multi_gpu.bat [OPTIONS]
    echo.
    echo Options:
    echo   --config        Accelerate config: multi_gpu ^| deepspeed_zero2 ^| deepspeed_zero3
    echo   --num-gpus      Number of GPUs
    echo   --model         Base model name or path
    echo   --max-steps     Maximum training steps
    echo   --batch-size    Per-device batch size
    echo   --data          Training data path
    echo   --disable-kb    Disable knowledge base (use simulated responses)
    echo   --help          Show this help message
    exit /b 0
)
echo Unknown option: %~1
exit /b 1

:run
REM 配置文件路径
set CONFIG_FILE=scripts\accelerate_configs\%CONFIG%.yaml

if not exist "%CONFIG_FILE%" (
    echo Error: Config file not found: %CONFIG_FILE%
    exit /b 1
)

echo ============================================================
echo MedAgent Agentic RL Multi-GPU Training
echo ============================================================
echo Config:     %CONFIG%
echo Config File: %CONFIG_FILE%
echo Num GPUs:   %NUM_GPUS%
echo Model:      %MODEL%
echo Max Steps:  %MAX_STEPS%
echo Batch Size: %BATCH_SIZE% (per device)
echo Data:       %DATA%
echo Disable KB: %DISABLE_KB%
echo ============================================================

REM 构建额外参数
set EXTRA_ARGS=
if %DISABLE_KB%==1 (
    set EXTRA_ARGS=--disable-kb
)

REM 启动训练
accelerate launch ^
    --config_file "%CONFIG_FILE%" ^
    --num_processes %NUM_GPUS% ^
    scripts\agentic_rl.py ^
    --model "%MODEL%" ^
    --data "%DATA%" ^
    --max-steps %MAX_STEPS% ^
    --batch-size %BATCH_SIZE% ^
    --use-lora ^
    --lora-r 16 ^
    --lora-alpha 32 ^
    --gradient-accumulation-steps 4 ^
    --output-dir "output\agentic_rl_multi_gpu" ^
    %EXTRA_ARGS%

echo ============================================================
echo Training completed!
echo ============================================================

endlocal