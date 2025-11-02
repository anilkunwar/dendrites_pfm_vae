this is work is based on Pytorch

- CVAE: https://github.com/timbmg/VAE-CVAE-MNIST

## Usage:
1. First run the simulations in the directory **Data**
2. Run **Data/step0_make_cvae_dataset.py** to build the dataset for CVAE model
3. Run **step1_train_cvae.py** to train the Conditional VAE (or --conditional=False to train a normal VAE)
3. Run **step2_predict_cvae.py** with control variable specified


Hao Tang, Oct 20th, 2025