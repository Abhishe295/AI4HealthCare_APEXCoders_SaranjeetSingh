import os
import numpy as np
import pandas as pd
import ants
from scipy.ndimage import zoom
from tqdm import tqdm

# ---------------- CONFIG ----------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NIFTI_DIR = os.path.join(BASE_DIR, "data", "curated", "nifti")
LABELS_CSV = os.path.join(BASE_DIR, "data", "curated", "labels.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "_all_antspy")

TARGET_SHAPE = (128, 128, 64)   # match your CNN
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- HELPERS ----------------

def normalize_intensity(volume):
    mean = volume.mean()
    std = volume.std() + 1e-6
    return (volume - mean) / std


def resize_volume(volume, target_shape):
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)


def center_crop(volume, target_shape):
    cx, cy, cz = [s // 2 for s in volume.shape]
    tx, ty, tz = [t // 2 for t in target_shape]

    return volume[
        cx - tx: cx + tx,
        cy - ty: cy + ty,
        cz - tz: cz + tz
    ]


# ---------------- MAIN PIPELINE ----------------

def process_subject(nifti_path):
    # Load MRI
    img = ants.image_read(nifti_path)

    # 1️⃣ Bias field correction
    img = ants.n4_bias_field_correction(img)

    # 2️⃣ Brain masking (ANTsPy correct way)
    mask = ants.get_mask(img)
    brain = img * mask

    # 3️⃣ Register to MNI (FAST: affine only)
    mni = ants.image_read(ants.get_ants_data("mni"))
    reg = ants.registration(
        fixed=mni,
        moving=brain,
        type_of_transform="Affine"
    )
    warped = reg["warpedmovout"]

    # Convert to numpy
    volume = warped.numpy()

    # 4️⃣ Normalize intensity
    mean = volume.mean()
    std = volume.std() + 1e-6
    volume = (volume - mean) / std

    # 5️⃣ Center crop + resize
    volume = volume[
        volume.shape[0]//2 - 80 : volume.shape[0]//2 + 80,
        volume.shape[1]//2 - 80 : volume.shape[1]//2 + 80,
        volume.shape[2]//2 - 40 : volume.shape[2]//2 + 40
    ]

    volume = zoom(volume, (128/160, 128/160, 64/80), order=1)

    return volume.astype(np.float32)



def main():
    labels_df = pd.read_csv(LABELS_CSV)

    for label in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(OUT_DIR, label), exist_ok=True)

    print("🚀 Starting ANTsPy MRI preprocessing...")

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
        subject_id = row["subject_id"]
        label = row["label"]

        nifti_path = os.path.join(NIFTI_DIR, f"{subject_id}.nii.gz")
        if not os.path.exists(nifti_path):
            continue

        try:
            volume = process_subject(nifti_path)
            out_path = os.path.join(OUT_DIR, label, f"{subject_id}.npy")
            np.save(out_path, volume)

        except Exception as e:
            print(f"[SKIP] {subject_id} | {e}")

    print("✅ ANTsPy preprocessing complete.")
    print("Saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
