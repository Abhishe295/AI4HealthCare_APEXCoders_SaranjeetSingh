import os
import pandas as pd
import numpy as np
import pydicom
import nibabel as nib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MRI_ROOT = os.path.join(BASE_DIR, "data", "original", "MRI")
LABELS_CSV = os.path.join(BASE_DIR, "data", "curated", "labels.csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "curated", "nifti")

os.makedirs(OUT_DIR, exist_ok=True)


def find_all_dicom_files(root_dir):
    dicom_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(path, force=True)
                if hasattr(ds, "pixel_array"):
                    dicom_files.append(path)
            except Exception:
                continue
    return dicom_files



def load_volume(dicom_files):
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, "InstanceNumber"):
                slices.append(ds)
        except Exception:
            continue

    if len(slices) == 0:
        return None

    slices.sort(key=lambda x: int(x.InstanceNumber))
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    return volume



def convert_subject(subject_id):
    subject_path = os.path.join(MRI_ROOT, subject_id)

    if not os.path.isdir(subject_path):
        print(f"[SKIP] Missing folder: {subject_id}")
        return False

    dicom_files = find_all_dicom_files(subject_path)

    if len(dicom_files) == 0:
        print(f"[SKIP] Empty DICOM series: {subject_id}")
        return False

    volume = load_volume(dicom_files)
    if volume is None:
        print(f"[SKIP] No valid slices: {subject_id}")
        return False

    nifti = nib.Nifti1Image(volume, affine=np.eye(4))
    out_path = os.path.join(OUT_DIR, f"{subject_id}.nii.gz")
    nib.save(nifti, out_path)

    print(f"[OK] Converted {subject_id}")
    return True


def main():
    df = pd.read_csv(LABELS_CSV)

    ok, fail = 0, 0
    for subject in df["subject_id"]:
        if convert_subject(subject):
            ok += 1
        else:
            fail += 1

    print("\nConversion Summary")
    print("------------------")
    print("Successful:", ok)
    print("Failed:", fail)
    print("Output:", OUT_DIR)


if __name__ == "__main__":
    main()
