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


