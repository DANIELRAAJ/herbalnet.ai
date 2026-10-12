import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
# Constants
IMAGE_SIZE = (224, 224)  # Target image size
BATCH_SIZE = 32  # Batch size
EPOCHS = 10  # Number of epochs

# Paths to the dataset
train_data_dir = 'dataset/train/'
validation_data_dir = 'dataset/test/'
# Data Augmentation for training set and rescaling for both train and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Data generators for training and validation
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Automatically determine the number of classes from the training generator
NUM_CLASSES = len(train_generator.class_indices)
print(f"Detected {NUM_CLASSES} classes: {train_generator.class_indices}")

# Function to display some training images with class names
def show_train_images(generator, num_images=9):
    images, labels = next(generator)
    plt.figure(figsize=(12, 12))
    for i in range(num_images):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        label_index = np.argmax(labels[i])
        class_name = list(generator.class_indices.keys())[label_index]
        plt.title(f'Class: {class_name}')
        plt.axis('off')
    plt.show()

# Display training images with their class names
show_train_images(train_generator)
# Model definition
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks to avoid overfitting
early_stopping_cb = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6)

# Train the model
cnn_history = cnn_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    callbacks=[early_stopping_cb, reduce_lr_cb]
)

cnn_model.predict(validation_generator)
# Save the trained CNN model
cnn_model.save('cnn_model.h5')
# Print final accuracy for CNN
train_accuracy_cnn = cnn_history.history['accuracy'][-1]*100
val_accuracy_cnn = cnn_history.history['val_accuracy'][-1]*100
print(f"Final Training Accuracy: {train_accuracy_cnn:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy_cnn:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Class labels from training generator
class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

y_true = []
y_pred = []

for i in range(len(validation_generator)):
    images, labels = validation_generator[i]
    y_true.extend(np.argmax(labels, axis=1))
    predictions = cnn_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))

# Explicit labels (0 to num_classes-1)
labels = list(range(num_classes))

# Classification report
print("Classification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=class_labels,
    zero_division=0   # avoids warning if class not predicted
))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Plotting training and validation metrics
def plot_metrics(history):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy plot
    ax1.plot(train_accuracy, label='Training Accuracy', color='blue')
    ax1.plot(val_accuracy, label='Validation Accuracy', color='green')
    ax1.set_title('Training vs Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(train_loss, label='Training Loss', color='red')
    ax2.plot(val_loss, label='Validation Loss', color='orange')
    ax2.set_title('Training vs Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# Plot metrics
plot_metrics(cnn_history)
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflow as tf

# Define input shape and model
input_tensor = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
base_model = NASNetMobile(include_top=False, weights='imagenet', input_tensor=input_tensor, pooling='avg')

# Add Dense layers on top
x = base_model.output
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_tensor = Dense(NUM_CLASSES, activation='softmax')(x)

nas_model = Model(inputs=base_model.input, outputs=output_tensor)

# Fine-tune the base model layers for better accuracy
for layer in base_model.layers:
    layer.trainable = True

# Compile the NAS model
nas_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the NAS model
nas_history = nas_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    verbose=1,
    callbacks=[early_stopping_cb, reduce_lr_cb]
)

# Save the trained NAS model
nas_model.save('nas_model.h5')
# Print final accuracy for NAS model
train_accuracy_nas = nas_history.history['accuracy'][-1]*100
val_accuracy_nas = nas_history.history['val_accuracy'][-1]*100
print(f"NAS Model - Final Training Accuracy: {train_accuracy_nas:.2f}%")
print(f"NAS Model - Final Validation Accuracy: {val_accuracy_nas:.2f}%")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Get class labels
class_labels = list(train_generator.class_indices.keys())
num_classes = len(class_labels)

# Collect true and predicted labels
y_true = []
y_pred = []

validation_generator.reset()  # IMPORTANT

for i in range(len(validation_generator)):
    images, labels = validation_generator[i]
    y_true.extend(np.argmax(labels, axis=1))
    preds = nas_model.predict(images, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ---- FIX HERE ----
labels = list(range(num_classes))

print("Classification Report:\n")
print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_labels,
        zero_division=0
    )
)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

import matplotlib.pyplot as plt

# Extract the training and validation metrics from the NAS model history
train_accuracy_nas = nas_history.history['accuracy']
val_accuracy_nas = nas_history.history['val_accuracy']
train_loss_nas = nas_history.history['loss']
val_loss_nas = nas_history.history['val_loss']

# Plotting accuracy and loss in separate subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy plot
ax1.plot(train_accuracy_nas, label='Training Accuracy', color='blue')
ax1.plot(val_accuracy_nas, label='Validation Accuracy', color='green')
ax1.set_title('NAS Model Training vs Validation Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Loss plot
ax2.plot(train_loss_nas, label='Training Loss', color='red')
ax2.plot(val_loss_nas, label='Validation Loss', color='orange')
ax2.set_title('NAS Model Training vs Validation Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Show both plots
plt.tight_layout()
plt.show()
# Extract final validation accuracy for both models
final_val_accuracy_cnn = cnn_history.history['val_accuracy'][-1]
final_val_accuracy_nas = nas_history.history['val_accuracy'][-1]

# Compare the two models based on the final validation accuracy
if final_val_accuracy_nas > final_val_accuracy_cnn:
    print("NAS-based model has higher accuracy.")
else:
    print("CNN-based model has higher accuracy.")

import matplotlib.pyplot as plt

# Extract final validation accuracy
final_val_accuracy_cnn = cnn_history.history['val_accuracy'][-1]
final_val_accuracy_nas = nas_history.history['val_accuracy'][-1]

models = ['CNN Model', 'NAS Model']
accuracies = [final_val_accuracy_cnn, final_val_accuracy_nas]

# Assign different colors
colors = ['#1f77b4', '#ff7f0e']  # Blue and Orange (you can pick any colors)

plt.figure(figsize=(8, 6))

bars = plt.bar(models, accuracies, color=colors)

# Title and labels
plt.title('Final Validation Accuracy Comparison', fontsize=14, pad=15)
plt.ylabel('Validation Accuracy (%)')

# Give extra headroom
plt.ylim(0, 1.1)

# Display percentage on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.03,
        f"{height * 100:.2f}%",
        ha='center',
        fontsize=12,
        fontweight='bold'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()




