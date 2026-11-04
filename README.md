# AgriScan AI - Plant Foliar Disease Detection System

## Overview

AgriScan AI is an intelligent agricultural diagnostic system that leverages deep learning and computer vision to automatically detect, classify, and provide management guidance for crop foliar diseases. Built using a custom Convolutional Neural Network (CNN) trained on the PlantVillage dataset, the system can identify **38 distinct plant disease conditions** across **14 agricultural crop species** with high accuracy.

## Documentation

This repository contains comprehensive documentation for the AgriScan AI project:

- **AgriScanAI_full_documentation.txt** - Complete project documentation including AI solution, business objectives, requirements, constraints, risks, tools and problem definition
- **PROJECT_DESCRIPTION.md** - Technical overview with architecture, features and implementation details

## Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.21.0
- Keras 3.15.1
- NumPy 2.4.6
- Matplotlib 3.11.2
- scikit-learn 1.9.1
- Pillow 12.3.0

### Installation

```bash
# Install dependencies
pip install tensorflow==2.21.0 keras==3.15.1 numpy==2.4.6 matplotlib==3.11.2 scikit-learn==1.9.1 pillow==12.3.0
```

### Usage

#### Train the Model

```bash
python AgriScanAI.py --train
```

#### Predict Disease from Image

```bash
python AgriScanAI.py --predict "path/to/image.jpg"
```

#### Interactive Mode

```bash
python AgriScanAI.py
```

## Features

### Multi-Class Disease Detection
- Detects 38 unique plant disease conditions
- Covers 14 major agricultural crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry and Tomato
- Distinguishes between healthy and diseased foliage

### Intelligent Classification
- Top-3 predictions with confidence scores for ambiguous cases
- Reliability assessment contextualizing model confidence
- Disease-specific knowledge base with symptoms, prevention and treatment recommendations

### Robust CNN Architecture
- Custom-built CNN trained from scratch (no transfer learning)
- 4 convolutional blocks with BatchNormalization for stability
- Dropout regularization to prevent overfitting
- Optimized for 128x128 RGB images (efficient computation)

### Comprehensive Evaluation
- Confusion matrix visualization
- Per-class precision, recall, and F1-scores
- Training history tracking (loss and accuracy curves)
- Validation-based early stopping and model checkpointing

## Supported Crops and Diseases

| Crop | Diseases Detected |
|------|-------------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| Orange (Citrus) | Citrus Greening / Huanglongbing |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy |

## Project Structure

```
AgriScan AI/
├── AgriScanAI.py                 # Main application script
├── disease_knowledge_base.py     # Disease information database
├── dataset/                      # PlantVillage dataset
│   ├── train/                   # Training images (43,596)
│   └── test/                    # Test images (10,709)
├── models/                      # Saved model files
│   ├── best_model.keras         # Trained CNN model
│   └── class_names.json         # Class label mappings
├── results/                     # Training outputs
│   ├── confusion_matrix.png     # Confusion matrix visualization
│   ├── accuracy_loss_curves.png # Training performance plots
│   ├── classification_report.json # Per-class metrics
│   └── training_history.json    # Training metrics over epochs
├── AgriScanAI_full_documentation.txt  # Complete project documentation
├── PROJECT_DESCRIPTION.md        # Technical overview
└── README.md                    # This file
```

## Technical Architecture

### CNN Model
```
Input: (128, 128, 3) RGB image
├─ Conv2D(32, 3x3, ReLU) + BatchNorm + MaxPool
├─ Conv2D(64, 3x3, ReLU) + BatchNorm + MaxPool
├─ Conv2D(128, 3x3, ReLU) + BatchNorm + MaxPool
├─ Conv2D(128, 3x3, ReLU) + BatchNorm + MaxPool
├─ Flatten
├─ Dense(256, ReLU) + Dropout(0.5)
└─ Dense(38, Softmax) ← Output: 38 disease classes
```

### Training Strategy
- **Optimizer**: Adam (learning rate = 1e-4)
- **Loss**: Categorical cross-entropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**:
  - EarlyStopping (patience=3, monitor val_loss)
  - ModelCheckpoint (save best by val_accuracy)
  - ReduceLROnPlateau (factor=0.5, patience=2)
- **Data Augmentation**: Horizontal flip, rotation (±18°), zoom (±10%), contrast (±10%)

## Performance

- **Training time**: ~2-4 hours for 6 epochs (GPU recommended)
- **Inference time**: <1 second per image (CPU)
- **Model size**: ~6MB (saved as .keras)
- **Accuracy**: Varies by class (see classification report in results/)

## Use Cases

### 1. Smallholder Farmers
- Quick diagnosis without waiting for extension services
- Early detection to prevent crop loss
- Actionable treatment recommendations

### 2. Agricultural Extension Officers
- Scalable diagnostic tool for field visits
- Consistent disease identification across agents
- Educational resource for farmer training

### 3. Research Institutions
- Baseline model for disease detection research
- Dataset and evaluation metrics for comparison
- Foundation for more advanced systems

### 4. Agricultural Technology Companies
- Integration into farm management platforms
- API for mobile applications
- Edge deployment for offline use

## Impact and Benefits

### Economic Impact
- **Reduced crop losses** through early detection
- **Lower input costs** by avoiding unnecessary chemical treatments
- **Increased yield** through timely intervention
- **Scalable solution** that can serve millions of farmers

### Social Impact
- **Democratizes agricultural expertise** - accessible to all farmers
- **Reduces dependency on scarce agricultural extension services**
- **Empowers smallholder farmers** with modern technology
- **Supports food security** by protecting crop yields

### Environmental Impact
- **Targeted treatment** reduces pesticide overuse
- **Precision agriculture** minimizes chemical runoff
- **Sustainable farming practices** through accurate diagnosis

## Limitations

- **Image quality dependency** - Requires clear, well-lit leaf images
- **Single-disease focus** - Cannot detect multiple diseases simultaneously
- **Field conditions** - Trained on controlled dataset images
- **Environmental factors** - Does not account for weather, soil conditions

## Disclaimer

All diagnoses generated by AgriScan AI are produced through computer vision pattern matching for advisory and educational purposes. When dealing with commercial crops or severe outbreaks, field confirmation by certified plant pathologists or local agricultural extension officers is strongly recommended before applying chemical treatments.

## Dataset Attribution

This project uses the PlantVillage dataset for training. The dataset is publicly available for research purposes. Please cite the original PlantVillage paper if using this system in academic work.

## License

Please refer to the PlantVillage dataset license for usage terms and conditions.

## Project Status

✅ **Core Development**: Complete
✅ **CNN Model**: Trained and evaluated
✅ **Prediction Interface**: Functional
✅ **Disease Knowledge Base**: Integrated
⚠️ **Streamlit Web App**: Optional (not implemented in core)
✅ **Documentation**: Comprehensive

## Contact & Support

For questions, issues, or contributions related to AgriScan AI, please refer to the project documentation in `AgriScanAI_full_documentation.txt` 

---

**Version**: 1.0
**Last Updated**: 2026
**Repository**: AgriScan AI

---

**Note**: The dataset folder is too large to upload to GitHub. Please download the PlantVillage dataset separately and place it in the `dataset/` directory with `train/` and `test/` subdirectories.
