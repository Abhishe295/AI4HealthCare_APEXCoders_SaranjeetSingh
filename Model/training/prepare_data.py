import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW = os.path.join(BASE, "data", "raw", "Alzheimer_s Dataset")
OUT = os.path.join(BASE, "data", "processed")

mapping = {
    "NonDemented": "CN",
    "VeryMildDemented": "MCI",
    "MildDemented": "MCI",
    "ModerateDemented": "AD"
}

# clean old
if os.path.exists(OUT):
    shutil.rmtree(OUT)

for split in ["train", "test"]:
    for cls in ["CN", "MCI", "AD"]:
        os.makedirs(os.path.join(OUT, split, cls), exist_ok=True)

for split in ["train", "test"]:
    for src_cls, dst_cls in mapping.items():
        src_folder = os.path.join(RAW, split, src_cls)
        dst_folder = os.path.join(OUT, split, dst_cls)

        for f in os.listdir(src_folder):
            shutil.copy(
                os.path.join(src_folder, f),
                os.path.join(dst_folder, f"{src_cls}_{f}")
            )

print("✅ Data prepared (train/test preserved)")