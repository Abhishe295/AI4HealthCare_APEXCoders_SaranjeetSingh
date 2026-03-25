import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRC = os.path.join(BASE, "data", "processed")
DST = os.path.join(BASE, "data", "binary")

# clean
if os.path.exists(DST):
    shutil.rmtree(DST)

for split in ["train", "test"]:
    for cls in ["CN", "AD"]:
        os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

for split in ["train", "test"]:
    for cls in ["CN", "AD"]:
        src_folder = os.path.join(SRC, split, cls)
        dst_folder = os.path.join(DST, split, cls)

        for f in os.listdir(src_folder):
            shutil.copy(
                os.path.join(src_folder, f),
                os.path.join(dst_folder, f)
            )

print("✅ Binary dataset ready")