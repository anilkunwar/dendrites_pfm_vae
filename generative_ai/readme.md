this is work is based on Pytorch

- CVAE: https://github.com/timbmg/VAE-CVAE-MNIST

## Usage:
1. First run the simulations, store the results in the directory **data/case_xxx/exodus_files**
2. Run **Data/step0_make_cvae_dataset_exodus.py** to transfer the moose results to npy files, you may need post-processing to filter physically incorrect results
3. Run **step1_split_dataset.py** to split and build the dataset
4. Run **step2_train_cvae.py** to train the Conditional VAE (or --conditional=False to train a normal VAE)
3. Run **step3_predict_cvae.py** with control variable specified


Hao Tang, Oct 20th, 2025