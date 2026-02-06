import os
import shutil
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALL_DIR = os.path.join(BASE_DIR, "data", "processed", "_all_antspy")
FINAL_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORT_DIR = os.path.join(FINAL_DIR, "reports")

SPLITS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15
}

SEED = 42
random.seed(SEED)

os.makedirs(REPORT_DIR, exist_ok=True)

def clear_old_splits():
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(FINAL_DIR, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)



def ensure_dirs():
    for split in SPLITS:
        for label in ["CN", "MCI", "AD"]:
            os.makedirs(
                os.path.join(FINAL_DIR, split, label),
                exist_ok=True
            )


def split_class(label):
    src_dir = os.path.join(ALL_DIR, label)
    files = [f for f in os.listdir(src_dir) if f.endswith(".npy")]

    random.shuffle(files)

    n = len(files)
    n_train = int(SPLITS["train"] * n)
    n_val = int(SPLITS["val"] * n)

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return train_files, val_files, test_files


def move_files(files, label, split):
    for f in files:
        src = os.path.join(ALL_DIR, label, f)
        dst = os.path.join(FINAL_DIR, split, label, f)
        shutil.copy(src, dst)


def main():
    clear_old_splits() 
    ensure_dirs()

    summary = []

    for label in ["CN", "MCI", "AD"]:
        train_f, val_f, test_f = split_class(label)

        move_files(train_f, label, "train")
        move_files(val_f, label, "val")
        move_files(test_f, label, "test")

        summary.append({
            "class": label,
            "train": len(train_f),
            "val": len(val_f),
            "test": len(test_f)
        })

    # Save report
    df = pd.DataFrame(summary)
    df.to_csv(
        os.path.join(REPORT_DIR, "class_distribution.csv"),
        index=False
    )

    with open(os.path.join(REPORT_DIR, "split_summary.txt"), "w") as f:
        f.write("Dataset split completed\n")
        f.write(f"Random seed: {SEED}\n")
        f.write(str(df))

    print("Dataset split completed successfully.")
    print(df)


if __name__ == "__main__":
    main()
