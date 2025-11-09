# according to the result of MOOSE simulation, make datasets for CVAE training
import glob
import os
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from natsort import natsorted

def makeDataset(path_sim, train_ratio=0.8, val_ratio=0.1):
    path_save_npy = os.path.join(path_sim, 'npy_files')
    if not os.path.exists(path_save_npy):
        os.makedirs(path_save_npy)

    sim_no = os.path.basename(path_sim)  # Simulation No
    csv_files = natsorted(glob.glob(path_sim + 'csv_files/*.csv'))  # Importing csv files of simulation naturally sorted

    filenames = []  # List of lists of compositions for all coordinate points for all time step
    for idx, file in enumerate(csv_files):  # Looping for all csv files (total csv files = number of time steps)
        df = pd.read_csv(file)
        # extract coordinates
        x = np.sort(df["Points:0"].unique())
        y = np.sort(df["Points:1"].unique())
        W = len(x)
        H = len(y)
        # build arrays
        data = np.zeros((H, W, 3), dtype=np.float32)
        # build idxs
        x_to_i = {v: i for i, v in enumerate(x)}
        y_to_j = {v: j for j, v in enumerate(y)}
        # fill in data
        for _, row in df.iterrows():
            i = y_to_j[row["Points:1"]]
            j = x_to_i[row["Points:0"]]
            data[i, j, 0] = row["eta"]
            data[i, j, 1] = row["c"]
            data[i, j, 2] = row["pot"]

        # save as npy files
        save_path = os.path.join(os.path.abspath(path_save_npy), f"{idx}.npy")
        np.save(save_path, data)
        filenames.append(save_path)

    # split dataset
    # 打乱
    files = list(filenames)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(os.path.join(path_sim, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    ### Plotting last frame ###
    frame_no = idx
    plt.imshow(data[..., 0], cmap="rainbow")
    plt.title(f'Showing eta of frame no: {frame_no + 1} of Simulation {sim_no}')
    plt.gca().invert_yaxis()
    plt.show()
    plt.imshow(data[..., 1], cmap="rainbow")
    plt.title(f'Showing c of frame no: {frame_no + 1} of Simulation {sim_no}')
    plt.gca().invert_yaxis()
    plt.show()
    plt.imshow(data[..., 2], cmap="rainbow")
    plt.title(f'Showing pot of frame no: {frame_no + 1} of Simulation {sim_no}')
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == '__main__':

    makeDataset("data/case_000/")
    makeDataset("data/case_001/")
    # makeDataset("sim_2/")