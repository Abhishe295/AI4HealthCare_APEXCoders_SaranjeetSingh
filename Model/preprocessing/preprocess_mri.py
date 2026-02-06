import os
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NIFTI_DIR = os.path.join(BASE_DIR, "data", "curated", "nifti")
LABELS_CSV = os.path.join(BASE_DIR, "data", "curated", "labels.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "_all")

TARGET_SHAPE = (128, 128, 128)
NUM_SLICES = 64

os.makedirs(OUT_DIR, exist_ok=True)


# ------------------ utility functions ------------------

def normalize_intensity(volume):
    v_min, v_max = volume.min(), volume.max()
    return (volume - v_min) / (v_max - v_min + 1e-8)


def simple_skull_strip(volume):
    threshold = np.percentile(volume, 5)
    volume[volume < threshold] = 0
    return volume


def resize_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def extract_middle_slices(volume, num_slices):
    z = volume.shape[2]
    center = z // 2
    half = num_slices // 2
    return volume[:, :, center - half:center + half]


# ------------------ main preprocessing ------------------

def main():
    labels_df = pd.read_csv(LABELS_CSV)

    for label in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

    print("Starting MRI preprocessing...")

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        subject_id = row["subject_id"]
        label = row["label"]

        nifti_path = os.path.join(NIFTI_DIR, f"{subject_id}.nii.gz")
        if not os.path.exists(nifti_path):
            continue

        # Load MRI
        img = nib.load(nifti_path)
        volume = img.get_fdata()

        # Preprocessing steps
        volume = simple_skull_strip(volume)
        volume = normalize_intensity(volume)
        volume = resize_volume(volume, TARGET_SHAPE)
        volume = extract_middle_slices(volume, NUM_SLICES)

        # Save
        out_path = os.path.join(
            OUT_DIR, label, f"{subject_id}.npy"
        )
        np.save(out_path, volume)

    print("Preprocessing completed.")
    print("Output saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
