import random
import json
import glob, os

if __name__ == '__main__':

    train_ratio = 0.9
    # val_ratio = 0.1

    data_root = "data"

    # # split dataset
    # for vn in os.listdir(data_root):
    filenames = glob.glob(os.path.join(data_root, "case_00*", "npy_files", "*.npy"))
    files = list(filenames)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_ratio)
    # n_val = int(n * val_ratio)

    train_files = files[:n_train]
    # val_files = files[n_train:n_train + n_val]
    # test_files = files[n_train + n_val:]
    test_files = files[n_train:]

    splits = {
        "train": train_files,
        # "val": val_files,
        "test": test_files
    }

    with open(os.path.join(data_root, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    # print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")