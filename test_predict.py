import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _preprocess
from PIL import Image

def preprocess_image(path):
    img = Image.open(path)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = _preprocess(img_array)
    return img_array

model = load_model("nas_model_robust.h5")
CONFIDENCE_THRESHOLD = 0.25
ENTROPY_THRESHOLD = 2.1

def test_image(path):
    img = preprocess_image(path)
    preds = model.predict(img, verbose=0)[0]
    top_class = int(np.argmax(preds))
    top_confidence = float(np.max(preds))
    entropy = float(-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))))
    ood = False
    reason = "OK"
    if top_confidence < CONFIDENCE_THRESHOLD:
        ood = True
        reason = f"Low confidence ({round(top_confidence*100, 2)}%)"
    elif entropy > ENTROPY_THRESHOLD:
        ood = True
        reason = f"High entropy ({round(entropy, 4)})"
    return ood, reason, top_class, top_confidence

test_dir = r"dataset/train"
for d in os.listdir(test_dir):
    d_path = os.path.join(test_dir, d)
    if os.path.isdir(d_path):
        files = os.listdir(d_path)
        if files:
            img_path = os.path.join(d_path, files[0])
            ood, reason, cls, conf = test_image(img_path)
            print(f"Class Folder: {d}")
            print(f" -> Predicted Class ID: {cls}, Conf: {conf*100:.2f}%")
            print(f" -> OOD: {ood}, Reason: {reason}")
            print("-" * 30)
