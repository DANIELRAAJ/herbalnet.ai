import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import albumentations as A

IMG_SIZE    = 224          # MobileNetV2 native size
BATCH_SIZE  = 8            # Keep small for tiny datasets
NUM_CLASSES = 10

# ===============================
# ARGUMENT PARSING
# ===============================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="./dataset",
                    help="Root folder. Must contain one subfolder per class.")
parser.add_argument("--epochs",   default=50, type=int)
parser.add_argument("--output",   default="nas_model.h5")
args = parser.parse_args()

# ===============================
# AUGMENTATION PIPELINES
# ===============================
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomShadow(p=0.3),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.CLAHE(p=0.3),
    A.RandomBrightnessContrast(p=0.4)
])

def train_preprocess(image):
    image_uint8 = image.astype(np.uint8)
    augmented = transform(image=image_uint8)["image"]
    return preprocess_input(augmented.astype(np.float32))

def val_preprocess(image):
    return preprocess_input(image.astype(np.float32))

train_datagen = ImageDataGenerator(
    preprocessing_function=train_preprocess,
    validation_split=0.2,
)

val_datagen = ImageDataGenerator(
    preprocessing_function=val_preprocess,
    validation_split=0.2,
)

# ===============================
# DATA LOADERS
# ===============================
print(f"\n[INFO] Loading data from: {args.data_dir}")

train_gen = train_datagen.flow_from_directory(
    args.data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42,
)

val_gen = val_datagen.flow_from_directory(
    args.data_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=42,
)

print(f"[INFO] Classes found: {list(train_gen.class_indices.keys())}")
print(f"[INFO] Training samples : {train_gen.samples}")
print(f"[INFO] Validation samples: {val_gen.samples}")

# ===============================
# MODEL — TRANSFER LEARNING
# ===============================
# MobileNetV2 pre-trained on ImageNet
base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,          # remove ImageNet classifier head
    weights="imagenet",
    pooling="avg",
)

# Phase 1: freeze the base — only train our new head
base.trainable = False

model = models.Sequential([
    base,
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),            # moderate dropout to combat overfitting
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ===============================
# CALLBACKS
# ===============================
cb_list = [
    # Stop early if val_loss stops improving
    callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    # Reduce LR when stuck
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1
    ),
    # Save best checkpoint
    callbacks.ModelCheckpoint(
        "best_checkpoint.h5", monitor="val_accuracy",
        save_best_only=True, verbose=1
    ),
]

# ===============================
# PHASE 1: TRAIN HEAD ONLY
# ===============================
print("\n[PHASE 1] Training classifier head (base frozen)...")
history1 = model.fit(
    train_gen,
    epochs=args.epochs // 2,
    validation_data=val_gen,
    callbacks=cb_list,
)

# ===============================
# PHASE 2: FINE-TUNE TOP LAYERS
# ===============================
# Unfreeze the top 30 layers of MobileNetV2 and fine-tune with a very
# low learning rate — this adapts the network to your leaf images.

print("\n[PHASE 2] Fine-tuning top layers of MobileNetV2...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5),   # much lower LR for fine-tuning
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history2 = model.fit(
    train_gen,
    epochs=args.epochs // 2,
    validation_data=val_gen,
    callbacks=cb_list,
)

# ===============================
# SAVE
# ===============================
model.save(args.output)
print(f"\n[INFO] Model saved to: {args.output}")

# ===============================
# EVALUATION SUMMARY
# ===============================
val_loss, val_acc = model.evaluate(val_gen, verbose=0)
print(f"\n{'='*40}")
print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%")
print(f"  Final Validation Loss     : {val_loss:.4f}")
print(f"{'='*40}")
print("\n[TIP] If accuracy is below 80%, collect more real images (see collection_guide.md)")
