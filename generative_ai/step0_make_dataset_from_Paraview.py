# according to the result of MOOSE simulation, make datasets for training
# DO NOT RECOMMEND
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from natsort import natsorted

def makeDataset(path_sim):
    path_save_npy = os.path.join(path_sim, 'npy_files')
    if not os.path.exists(path_save_npy):
        os.makedirs(path_save_npy)

    sim_no = os.path.basename(path_sim)  # Simulation No
    csv_files = natsorted(glob.glob(path_sim + '/csv_files/*.csv'))  # Importing csv files of simulation naturally sorted

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
        save_path = os.path.join(os.path.abspath(path_save_npy), f"{row['Time']}.npy")
        np.save(save_path, data)
        filenames.append(save_path)

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

    for n in os.listdir("data"):
        makeDataset(os.path.join("data", n))