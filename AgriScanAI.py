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
