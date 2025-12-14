# fetch_ids.py
from datasets import load_dataset
ds = load_dataset("lmms-lab/GQA", "val_balanced_instructions", split="val")
ids = sorted({row["imageId"] for row in ds})
with open("val_image_ids.txt", "w") as fh:
    for iid in ids:
        fh.write(f"{iid}.jpg\n")