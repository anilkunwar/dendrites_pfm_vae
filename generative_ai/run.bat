@echo off
setlocal enabledelayedexpansion

:: ============================
:: 批量实验参数范围
:: ============================

:: 图像噪声概率
set noise_list=0.1 0.3 0.5

:: Edge 权重
set edge_weights=0.001 0.01

:: FFT 权重
set fft_weights=0.001 0.01

:: TV 权重
set tv_weights=0.0005 0.001 0.005

:: Smoothness 权重
set smooth_weights=0.005 0.01

:: Grad 权重
set grad_weights=0.001 0.01 0.05

:: 输出根目录
set OUTROOT=results
mkdir %OUTROOT%

:: ============================
:: 依次启动实验
:: ============================

for %%N in (%noise_list%) do (
    for %%E in (%edge_weights%) do (
        for %%F in (%fft_weights%) do (
            for %%T in (%tv_weights%) do (
                for %%S in (%smooth_weights%) do (
                    for %%G in (%grad_weights%) do (

                        echo -----------------------------------------
                        echo  启动实验:
                        echo     Noise  = %%N
                        echo     Edge   = %%E
                        echo     FFT    = %%F
                        echo     TV     = %%T
                        echo     Smooth = %%S
                        echo     Grad   = %%G
                        echo     输出目录 = !OUTROOT!
                        echo -----------------------------------------

                        python step2_train_vae.py ^
                            --noise_prob %%N ^
                            --w_edge %%E ^
                            --w_fft %%F ^
                            --w_tv %%T ^
                            --w_smooth %%S ^
                            --w_grad %%G ^
                            --fig_root !OUTROOT!

                    )
                )
            )
        )
    )
)

pause
