import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tensorflow as tf

# Suppress TF logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_PATH  = "nas_model_robust.h5"
DATA_DIR    = "dataset/train"
IMG_SIZE    = 224

class_names = [
    "Brahmi", "Kallurukki", "Karisalankanni", "Keezhanelli",
    "Kuthiraiveli", "Mudakathan", "Nilavembu", "Pirandai",
    "Sirukurinjan", "Vallarai"
]

model = load_model(MODEL_PATH)

def preprocess(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    return preprocess_input(arr)

# Build confusion matrix
n = len(class_names)
matrix = np.zeros((n, n), dtype=int)
folder_to_idx = {name.lower(): i for i, name in enumerate(class_names)}

for folder in sorted(os.listdir(DATA_DIR)):
    folder_path = os.path.join(DATA_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    # Extract the first word from the folder name (e.g., "Brahmi (Bacopa monnieri)" -> "Brahmi")
    base_name = folder.split(" ")[0].lower()
    true_idx = folder_to_idx.get(base_name)
    if true_idx is None:
        print(f"[WARN] Folder '{folder}' (base: '{base_name}') doesn't match any class name — skipping")
        continue

    for fname in os.listdir(folder_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        try:
            img = preprocess(os.path.join(folder_path, fname))
            pred_idx = int(np.argmax(model(img, training=False).numpy()[0]))
            matrix[true_idx][pred_idx] += 1
        except Exception as e:
            print(f"[SKIP] {fname}: {e}")

# Print per-class accuracy
print("\n" + "="*55)
print(f"{'CLASS':<22} {'CORRECT':>7} {'TOTAL':>6} {'ACCURACY':>9}")
print("="*55)
for i, name in enumerate(class_names):
    total   = matrix[i].sum()
    correct = matrix[i][i]
    acc     = (correct / total * 100) if total > 0 else 0
    flag    = " !!!" if acc < 70 else ""
    print(f"{name:<22} {correct:>7} {total:>6} {acc:>8.1f}%{flag}")

# Print confusion details for weak classes
print("\n" + "="*55)
print("CONFUSION DETAILS (wrong predictions only)")
print("="*55)
for i, name in enumerate(class_names):
    total   = matrix[i].sum()
    correct = matrix[i][i]
    if total == 0 or correct == total:
        continue
    wrong = [(class_names[j], matrix[i][j]) for j in range(n) if j != i and matrix[i][j] > 0]
    wrong.sort(key=lambda x: -x[1])
    if wrong:
        print(f"\n{name} - confused with:")
        for wname, count in wrong:
            pct = count / total * 100
            print(f"   -> {wname:<20} {count} times  ({pct:.0f}%)")

print("\n[TIP] Classes below 70% need more diverse images.")
print("[TIP] If two classes consistently confuse each other,")
print("      their images may look too similar — collect photos")
print("      that emphasise the visual differences between them.")
