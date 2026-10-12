from flask import Flask, render_template, request, url_for
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageSequence
import os
import signal
from werkzeug.utils import secure_filename

# ── Fast shutdown on Windows ──────────────────────────────────────────────────
def _force_exit(sig, frame):
    print("\n[INFO] Shutting down HerbalNet AI...")
    os._exit(0)

signal.signal(signal.SIGINT,  _force_exit)
signal.signal(signal.SIGTERM, _force_exit)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_folder_path = os.path.join(BASE_DIR, "templates")
static_folder_path = os.path.join(BASE_DIR, "static")
print(f"DEBUG: BASE_DIR = {BASE_DIR}")
print(f"DEBUG: template_folder = {template_folder_path}")
print(f"DEBUG: static_folder = {static_folder_path}")
print(f"DEBUG: template folder exists = {os.path.exists(template_folder_path)}")
print(f"DEBUG: predict.html exists = {os.path.exists(os.path.join(template_folder_path, 'predict.html'))}")

app = Flask(
    __name__,
    template_folder=template_folder_path,
    static_folder=static_folder_path,
)

# ===============================
# CONFIGURATION
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Model
if os.path.exists("nas_model_robust.keras"):
    model_path = "nas_model_robust.keras"
elif os.path.exists("nas_model_robust.h5"):
    model_path = "nas_model_robust.h5"
else:
    model_path = "nas_model.h5"
print(f"DEBUG: Loading model: {model_path}")

# FIX: Keras 3.13+ adds 'quantization_config' which breaks Keras 3.12-
from tensorflow.keras.layers import Dense

_original_dense_from_config = Dense.from_config

@classmethod
def _custom_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)

Dense.from_config = _custom_dense_from_config

model = load_model(model_path)
IMG_SIZE = model.input_shape[1]

# ── PREPROCESSING FIX ────────────────────────────────────────────────────────
# We have finally re-trained the model correctly. Both `augment_and_train.py`
# and this inference file now properly scale pixels into `[-1, 1]` using
# MobileNetV2's official native preprocessor!
# ─────────────────────────────────────────────────────────────────────────────
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as _preprocess

# ===============================
# OUT-OF-DISTRIBUTION SETTINGS
# ===============================
# These are intentionally lenient because the current model was trained on a
# small dataset and produces lower-confidence outputs overall.
# Once you retrain with 50+ images, tighten these back to 0.70 / 1.5
CONFIDENCE_THRESHOLD = 0.25
ENTROPY_THRESHOLD = 2.1

# ===============================
# PLANT DATA
# ===============================
class_names = {
    0: "Brahmi (Bacopa monnieri)",
    1: "Kallurukki (Aerva lanata)",
    2: "Karisalankanni (Eclipta prostrata)",
    3: "Keezhanelli (Phyllanthus niruri)",
    4: "Kuthiraiveli (Cleome gynandra)",
    5: "Mudakathan (Cardiospermum halicacabum)",
    6: "Nilavembu (Andrographis paniculata)",
    7: "Pirandai (Cissus quadrangularis)",
    8: "Sirukurinjan (Gymnema sylvestre)",
    9: "Vallarai (Centella asiatica)"
}

clinical_en = {
    0: "Improves memory, reduces anxiety, and enhances cognitive functions.",
    1: "Acts as a diuretic and supports kidney stone removal.",
    2: "Protects liver function and promotes hair growth.",
    3: "Supports liver health and helps in jaundice treatment.",
    4: "Relieves joint pain and reduces inflammation.",
    5: "Effective in treating arthritis and joint stiffness.",
    6: "Used for viral fevers and boosts immunity.",
    7: "Accelerates bone fracture healing.",
    8: "Helps regulate blood sugar levels.",
    9: "Improves brain function and wound healing."
}

clinical_ta = {
    0: "நினைவாற்றலை மேம்படுத்தி மன அழுத்தத்தை குறைக்க உதவுகிறது.",
    1: "சிறுநீரக கற்களை கரைக்கவும் சிறுநீரக ஆரோக்கியத்தையும் மேம்படுத்தும்.",
    2: "கல்லீரல் பாதுகாப்பு மற்றும் முடி வளர்ச்சிக்கு உதவும்.",
    3: "மஞ்சள் காமாலை மற்றும் கல்லீரல் நோய்களுக்கு பயன்படும்.",
    4: "மூட்டு வலி மற்றும் அழற்சியை குறைக்கும்.",
    5: "மூட்டு வாதம் மற்றும் வீக்கத்தை குறைக்கும்.",
    6: "வைரஸ் காய்ச்சல் மற்றும் நோய் எதிர்ப்பு சக்தி மேம்பாடு.",
    7: "எலும்பு முறிவு விரைவாக குணமாக உதவும்.",
    8: "ரத்த சர்க்கரை அளவைக் கட்டுப்படுத்தும்.",
    9: "நினைவாற்றல் மற்றும் காயம் ஆற உதவும்."
}

ayurveda_en = {
    0: "Medhya Rasayana for memory and neurological health.",
    1: "Used in Mutrakrichra and Ashmari (urinary disorders).",
    2: "Known as Bhringraj; used for liver and hair disorders.",
    3: "Used in Kamala (jaundice) and liver diseases.",
    4: "Used in Vata disorders and pain management.",
    5: "Used for Sandhivata (arthritis).",
    6: "Used in Jwara (fever) management.",
    7: "Known as Asthisamhara; used for bone healing.",
    8: "Known as Meshashringi; used in Madhumeha (diabetes).",
    9: "Medhya Rasayana for brain and skin health."
}

ayurveda_ta = {
    0: "மெத்ய ரஸாயனமாக நினைவாற்றலுக்குப் பயன்படும்.",
    1: "மூத்திர நோய்கள் மற்றும் சிறுநீரக கற்களுக்கு பயன்படும்.",
    2: "பிருங்கராஜம் என அழைக்கப்படுகிறது; கல்லீரல் நோய்களுக்கு பயன்படும்.",
    3: "காமலா (மஞ்சள் காமாலை) சிகிச்சைக்கு பயன்படும்.",
    4: "வாத நோய்கள் மற்றும் வலிக்கு பயன்படும்.",
    5: "சந்திவாதம் (மூட்டு வாதம்) சிகிச்சைக்கு பயன்படும்.",
    6: "ஜ்வரம் (காய்ச்சல்) சிகிச்சைக்கு பயன்படும்.",
    7: "அஸ்திசம்ஹாரா – எலும்பு வலிமைக்கு பயன்படும்.",
    8: "மதுமேகம் (சர்க்கரை நோய்) சிகிச்சைக்கு பயன்படும்.",
    9: "நினைவாற்றல் மற்றும் தோல் நோய்களுக்கு பயன்படும்."
}

# ===============================
# IMAGE PREPROCESS
# ===============================
def preprocess_image(path):
    img = Image.open(path)

    if getattr(img, "is_animated", False):
        img = next(ImageSequence.Iterator(img))

    img = img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = _preprocess(img_array)

    return img_array

# ===============================
# OUT-OF-DISTRIBUTION DETECTION
# ===============================
def is_out_of_distribution(preds):
    top_confidence = float(np.max(preds))
    entropy = float(-np.sum(preds * np.log(np.clip(preds, 1e-10, 1.0))))

    details = {
        "confidence": round(top_confidence * 100, 2),
        "entropy": round(entropy, 4),
        "confidence_threshold": round(CONFIDENCE_THRESHOLD * 100, 2),
        "entropy_threshold": ENTROPY_THRESHOLD,
    }

    if top_confidence < CONFIDENCE_THRESHOLD:
        return True, f"Low confidence ({details['confidence']}% < {details['confidence_threshold']}% required)", details

    if entropy > ENTROPY_THRESHOLD:
        return True, f"High uncertainty (entropy {details['entropy']} > threshold {ENTROPY_THRESHOLD})", details

    return False, "OK", details

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    img = preprocess_image(file_path)

    # Calling model() directly is faster for single images and perfectly avoids
    # the Keras progress bar's OSError: [Errno 22] Invalid argument in sys.stdout.flush
    preds = model(img, training=False).numpy()[0]
    idx = int(np.argmax(preds))

    image_url = url_for('static', filename=f'uploads/{filename}')

    ood, reason, details = is_out_of_distribution(preds)

    if ood:
        return render_template(
            "unknown.html",
            reason=reason,
            confidence=details["confidence"],
            entropy=details["entropy"],
            image_path=image_url,
            top_guess=class_names[idx],
        )

    return render_template(
        "result.html",
        plant=class_names[idx],
        confidence=round(float(preds[idx]) * 100, 2),
        clinical_en=clinical_en[idx],
        clinical_ta=clinical_ta[idx],
        ayurveda_en=ayurveda_en[idx],
        ayurveda_ta=ayurveda_ta[idx],
        image_path=image_url
    )

if __name__ == "__main__":
    print(f"Flask template_folder: {app.template_folder}")
    print(f"Flask jinja_loader: {app.jinja_loader}")
    # ── FIX: debug=False in production — debug=True causes double model load
    #         and interferes with the signal handler on Windows
    app.run(debug=False, use_reloader=False)