import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
IMAGE_SIZE = (224, 224) 
BATCH_SIZE = 32
NUM_CLASSES = 10

train_data_dir = 'dataset/train/'

# 1. IMPROVED DATA AUGMENTATION & PROPER PREPROCESSING
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Proper NASNet scaling [-1, 1]
    rotation_range=40,                       # Leaves can be in any orientation
    width_shift_range=0.2,                   # Translation robustness
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,                          # Zoom robustenss
    horizontal_flip=True,
    vertical_flip=True,                      # Vertical flip for leaves is valid
    brightness_range=[0.8, 1.2],             # Lighting variations
    fill_mode='nearest',
    validation_split=0.2                     # 20% validation split instead of missing test folder
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Must match training preprocessing
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=42 # Make sure split is identical
)

validation_generator = validation_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=42 # Make sure split is identical
)

# 2. PROPER TRANSFER LEARNING - STAGE 1 (Train only the new head)
input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

# Load NASNetMobile without the top classification layer
base_model = NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_tensor)

# Freeze the base external model entirely so we don't destroy its ImageNet-learned features
base_model.trainable = False 

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x) # Better than flattening for spatial data
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output_tensor)

# Compile with a standard learning rate for the top layers
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Train ONLY the top classification layers for a few epochs
print("STAGE 1: Training only the new dense layers...")
history_stage1 = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator,
    verbose=1
)

# 3. PROPER TRANSFER LEARNING - STAGE 2 (Fine-tuning top blocks)
print("STAGE 2: Fine-tuning a few top layers of the base model...")

# Unfreeze the base model
base_model.trainable = True

# We only want to fine-tune the top layers, freeze the bottom generic feature extractors
# NASNetMobile has many layers, leaving the last 50 trainable is standard.
for layer in model.layers[:-50]: 
    layer.trainable = False

# Recompile with a MUCH LOWER learning rate (e.g., 1e-5) because we are modifying delicate weights
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

early_stopping_cb = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-7)

history_stage2 = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping_cb, reduce_lr_cb]
)

# Save the properly trained robust model
model.save('nas_model_robust.h5')
print("Model saved as nas_model_robust.h5")
