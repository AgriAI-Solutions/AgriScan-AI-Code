"""
AgriScan AI - Plant Foliar Disease Diagnostic System
Train: python AgriScanAI.py --train-class-weighted
Predict: python AgriScanAI.py --predict "path/to/image.jpg"
"""
import os
import json
import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from keras.models import Sequential
from keras.utils import image_dataset_from_directory
from keras import layers, metrics
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from disease_knowledge_base import get_disease_information

SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 64

#Function to load the dataset
def load_datasets():
    """Load and preprocess train/val/test datasets. Returns datasets and class names."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    train_dir, test_dir = os.path.join(base_dir, "train"), os.path.join(base_dir, "test")
    
    for d in [base_dir, train_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    
    class_names = sorted([f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))])
    if len(class_names) == 0 or len(class_names) != len(set(class_names)):
        raise ValueError("Invalid or duplicate class names in training directory")
    
    print(f"Found {len(class_names)} disease classes")
    
    train_ds = image_dataset_from_directory(train_dir, labels="inferred", label_mode="categorical",
        class_names=class_names, color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=True, seed=SEED, validation_split=0.2, subset="training")
    
    val_ds = image_dataset_from_directory(train_dir, labels="inferred", label_mode="categorical",
        class_names=class_names, color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, seed=SEED, validation_split=0.2, subset="validation")
    
    test_ds = image_dataset_from_directory(test_dir, labels="inferred", label_mode="categorical",
        class_names=class_names, color_mode="rgb", image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, shuffle=False)
    
    test_class_names = sorted([f for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))])
    if set(test_class_names) != set(class_names):
        raise ValueError(f"Class mismatch between train and test directories")
    
    rescale = layers.Rescaling(1.0 / 255.0)
    augment = Sequential([layers.RandomFlip("horizontal"), layers.RandomZoom(0.1), layers.RandomContrast(0.1)], name="augment")
    
    train_ds = train_ds.map(lambda x, y: (rescale(augment(x, training=True)), y), num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names
    
    #Function to build the CNN model
def build_model(class_names):
    """Build and compile CNN model."""
    model = Sequential(name="agriscan_cnn")
    model.add(layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)))
    
    filters = 32
    for _ in range(3):
        model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        filters *= 2
    model.add(Conv2D(filters, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_names), activation="softmax"))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy",
        metrics=["accuracy", metrics.Precision(name="precision"), metrics.Recall(name="recall")])
    model.summary()
    return model
    
#Function to plot training history
def plot_training_history(history_dict, title_suffix, output_path):
    """Generate and save accuracy/loss plots."""
    epochs_range = range(1, len(history_dict["loss"]) + 1)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_dict["loss"], "b-o", label="Training Loss")
    if "val_loss" in history_dict:
        plt.plot(epochs_range, history_dict["val_loss"], "r-o", label="Validation Loss")
    plt.title(f"Loss Over Epochs {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_dict["accuracy"], "b-o", label="Training Accuracy")
    if "val_accuracy" in history_dict:
        plt.plot(epochs_range, history_dict["val_accuracy"], "r-o", label="Validation Accuracy")
    plt.title(f"Accuracy Over Epochs {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


#Function to evaluate the model
def evaluate_model(model, test_ds, class_names, title_suffix, output_suffix):
    """Evaluate model, generate confusion matrix and classification report."""
    loss, acc, prec, rec = model.evaluate(test_ds, verbose=1)[:4]
    print(f"Test - Loss: {loss:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
    
    predictions = model.predict(test_ds, verbose=1)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model predictions contain non-finite values (NaN or Inf)")
    
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_ds])
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(16, 14))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=90)
    plt.title(f"AgriScan AI - 38 Class Plant Disease Confusion Matrix {title_suffix}")
    plt.tight_layout()
    plt.savefig(f"results/confusion_matrix{output_suffix}.png", dpi=300)
    plt.close(fig)
    print("Confusion matrix saved")
    
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    print(f"\nClassification Report {title_suffix}:")
    print(report)
    
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True, zero_division=0)
    with open(f"results/classification_report{output_suffix}.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    print("Classification report saved")
