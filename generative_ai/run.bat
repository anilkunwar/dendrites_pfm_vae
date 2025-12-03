@echo off
setlocal enabledelayedexpansion

:: ============================
:: 批量实验参数范围
:: ============================

:: 隐空间大小
set latent_size=64 128 256

:: 图像噪声概率
set noise_list=0.5 0.7 0.9

:: kl 权重
set kl_weights=0.1 0.2 0.5 0.7

:: Grad 权重
set grad_weights=0.01 0.1 0.5 1.0

:: 输出根目录
set OUTROOT=results
mkdir %OUTROOT%

:: ============================
:: 依次启动实验
:: ============================

for %%N in (%noise_list%) do (
    for %%F in (%kl_weights%) do (
        for %%T in (%latent_size%) do (
            for %%G in (%grad_weights%) do (

                echo -----------------------------------------
                echo  启动实验:
                echo     Noise  = %%N
                echo     KL    = %%F
                echo     latent_size     = %%T
                echo     Grad   = %%G
                echo     输出目录 = !OUTROOT!
                echo -----------------------------------------

                python step2_train_vae.py ^
                    --noise_prob %%N ^
                    --latent_size %%T ^
                    --beta_start %%F ^
                    --w_grad %%G ^
                    --fig_root !OUTROOT!

            )
        )
    )
)

pause
