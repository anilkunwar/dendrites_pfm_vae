import random
import json
import glob, os

if __name__ == '__main__':

    train_ratio = 0.9
    test_ratio = 0.1

    data_root = "data"

    # split test and train-val dataset
    bc = os.listdir(data_root)
    test_case_num = int(len(bc) * test_ratio)
    random.shuffle(bc)

    tests = []
    case_tests = bc[:test_case_num]
    for case in case_tests:
        case_dir = os.path.join(data_root, case)
        tests.extend(glob.glob(os.path.join(case_dir, "npy_files", "*.npy")))
    random.shuffle(tests)

    # split dataset
    train_vals = []
    case_train_vals = bc[test_case_num:]
    for case in case_train_vals:
        case_dir = os.path.join(data_root, case)
        train_vals.extend(glob.glob(os.path.join(case_dir, "npy_files", "*.npy")))
    random.shuffle(train_vals)

    n = len(train_vals)
    n_train = int(n * train_ratio)

    train_files = train_vals[:n_train]
    val_files = train_vals[n_train:]
    test_files = tests

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(os.path.join(data_root, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")