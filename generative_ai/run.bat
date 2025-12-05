@echo off
setlocal enabledelayedexpansion

REM =====================================
REM 参数空间
REM =====================================

set latent_size=4 8 16 32 128
set component_num=16 32 48 64
set scale_weight=1 0.5 1.5 0.25

REM 最大并行任务数
set MAX_JOBS=2

echo 开始批量实验（并行=%MAX_JOBS%）...

for %%N in (%latent_size%) do (
    for %%G in (%component_num%) do (
        for %%W in (%scale_weight%) do (

            echo 启动任务: latent_size=%%N component_num=%%G scale_weight=%%W

            REM ------------------------------
            REM 后台运行任务
            REM ------------------------------
            start "" /B python step2_train_vaev2.py ^
                --latent_size %%N ^
                --n_components %%G ^
                --scale_weight %%W

            REM ------------------------------
            REM 控制并行任务数 = MAX_JOBS
            REM ------------------------------
            :WAIT_SLOT_%%N_%%G_%%W
            for /f "tokens=2" %%P in ('tasklist ^| find /C "python.exe"') do set JOB_COUNT=%%P

            if !JOB_COUNT! GEQ %MAX_JOBS% (
                timeout /t 1 >nul
                goto WAIT_SLOT_%%N_%%G_%%W
            )

        )
    )
)

echo 等待所有任务结束...

:WAIT_ALL
for /f "tokens=2" %%P in ('tasklist ^| find /C "python.exe"') do set JOB_COUNT=%%P
if !JOB_COUNT! EQU 0 goto END_WAIT
timeout /t 2 >nul
goto WAIT_ALL

:END_WAIT
echo 全部任务完成！

endlocal
