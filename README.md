# 🎓 Student Engagement Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)
![Transformers](https://img.shields.io/badge/🤗_Transformers-4.44.2-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Real-time student engagement detection using BEiT Vision Transformer with webcam and screen capture support**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Model](#-model-architecture) • [Results](#-results)

</div>

---

## 📋 Overview

This project implements a **real-time student engagement detection system** for online classes using state-of-the-art Vision Transformers. The system can analyze student emotions through webcam feeds or screen captures (Zoom/Google Meet) to detect four engagement states:

- 😴 **Bored** - Student shows disinterest or fatigue
- 🤔 **Confused** - Student appears uncertain or puzzled  
- ✨ **Engaged** - Student actively participates and focuses
- 😐 **Neutral** - Baseline emotional state

### 🎯 Key Innovations

- **Two-Stage Transfer Learning**: Fine-tuned BEiT (pre-trained on FER2013/RAF-DB/AffectNet) on custom engagement-specific dataset
- **Minimal Data Requirements**: Achieves high accuracy with <200 images per class through strategic data augmentation
- **Explainable AI**: Integrated Grad-CAM visualization for model interpretability
- **Multiple Input Sources**: Supports webcam, screen capture, and video file analysis
- **Privacy-Conscious**: Can analyze screen captures without storing facial data

---

## 🚀 Features

### Core Capabilities
- ✅ **Real-time Webcam Detection** - Live emotion analysis from local camera
- ✅ **Screen Capture Mode** - Analyze students in Zoom/Meet without recording
- ✅ **Face Detection & Alignment** - MTCNN-based facial landmark detection
- ✅ **Temporal Smoothing** - Exponential moving average for stable predictions
- ✅ **Grad-CAM Visualization** - Understand which facial regions influence predictions
- ✅ **Multi-face Support** - Simultaneously analyze multiple students
- ✅ **Confidence Scoring** - Per-prediction confidence metrics

### Technical Features
- 🔥 GPU-accelerated inference (CUDA support)
- 📊 Comprehensive probability distributions for all emotions
- 🎨 Interactive visualizations with matplotlib/seaborn
- 🖼️ Batch image processing support
- 📹 Video file analysis capability

---

## 🛠️ Installation

### Prerequisites
- **Python**: 3.11
- **CUDA**: 11.8+ (optional, for GPU acceleration)
- **Git**: For cloning the repository
- **Webcam**: For live detection features

### Step 1: Clone Repository
```bash
git clone https://github.com/nihar245/Student-Engagement-Detection.git
cd Student-Engagement-Detection
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### (Optional) Register a Jupyter kernel for the venv

If you want the notebook to use the project's virtual environment as a kernel in Jupyter, register an IPython kernel that points to the venv Python. Run this after activating the venv:

```bash
python -m ipykernel install --user --name "project-venv" --display-name "Engagement-Detection (venv)"
```

After running the command above, select the kernel named "Engagement-Detection (venv)" inside Jupyter or VS Code when opening `Model_Live_Demo.ipynb`.

### Step 4: Download Trained Model

**The trained model is too large for GitHub (327 MB). Download it separately:**

#### Option 1: Google Drive (Recommended)
1. Download from: [**Student Engagement Model - Google Drive**](https://drive.google.com/drive/folders/1nz2Tlj6ShseltkWYHtM0fqRTzXOexGCg?usp=sharing)
2. Extract the contents to the project root directory
3. Ensure folder structure is: `./final model 4/` containing:
   - `config.json`
   - `model.safetensors`
   - `preprocessor_config.json`

#### Option 2: Command Line Download (Windows)
```bash
# Install gdown
pip install gdown

# Download model (replace FILE_ID with actual ID from the Drive link)
gdown --folder https://drive.google.com/drive/folders/1nz2Tlj6ShseltkWYHtM0fqRTzXOexGCg
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 2.4.1
CUDA Available: True  # Or False if CPU-only
```

---

## 📖 Usage

### Quick Start - Webcam Detection

Open `Model_Live_Demo.ipynb` in Jupyter Notebook or VS Code and run:

```python
# Single capture with full analysis
live_demo_single_capture()
```

**Features:**
- Captures image after 2-second countdown
- Detects and aligns all faces
- Predicts engagement for each face
- Shows Grad-CAM attention maps
- Displays probability distributions

### Real-Time Streaming

```python
# Continuous webcam analysis
real_time_emotion_detection()
```

**Controls:**
- Press `q` to quit
- Automatically tracks multiple faces
- Shows smoothed predictions with temporal filtering

### Screen Capture Mode (Zoom/Meet Analysis)

```python
# Capture full screen
screen_emotion_detection()

# Capture specific region (recommended)
region = select_screen_region()  # Interactive selection
screen_emotion_detection(region=region, fps=5)
```

**Use Cases:**
- Monitor student engagement in Zoom classes
- Analyze recorded lecture videos
- Privacy-preserving classroom observation

### Analyze Image Files

```python
# Analyze single image with Grad-CAM
results = analyze_image_file('path/to/image.jpg', use_gradcam=True)

# Batch processing
for img_path in image_list:
    results = analyze_image_file(img_path)
```

---

## 🧠 Model Architecture

### Base Model
- **Architecture**: [BEiT-Large (Microsoft)](https://huggingface.co/microsoft/beit-base-patch16-224-pt22k-ft22k)
- **Pre-training**: ImageNet-22k (14M images, 21,841 classes)
- **Parameters**: 86M trainable parameters
- **Input**: 224×224 RGB images
- **Backbone**: Vision Transformer with 12 layers

### Transfer Learning Pipeline

```
Stage 1: Base BEiT Model (ImageNet-22k pre-trained)
   ↓
Stage 2: Fine-tuned by Tanneru on emotion datasets
         (FER2013 + RAF-DB + AffectNet - 7 emotion classes)
         🔗 [Tanneru's HuggingFace Profile](https://huggingface.co/Tanneru)
   ↓
Stage 3: Our Fine-tuning (Custom engagement dataset)
         - 4 engagement classes
         - 150 samples per class (after augmentation)
         - 7 epochs training
```

### Data Augmentation Strategy

To overcome limited training data (<50 images per class), we applied:

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(10),
])
```

Plus **oversampling** to balance classes at 150 samples each.

### Face Detection Pipeline

```
Input Image → MTCNN Face Detection → Facial Landmarks
                      ↓
         Face Alignment (Eye-based rotation)
                      ↓
         BEiT Feature Extraction (224×224)
                      ↓
         Classification Head (4 classes)
                      ↓
         Softmax → Engagement Prediction
```

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 94.2% |
| **Validation F1-Score** | 0.91 (weighted) |
| **Inference Time (GPU)** | ~45ms per face |
| **Inference Time (CPU)** | ~180ms per face |

### Class-wise Performance

| Engagement State | Precision | Recall | F1-Score |
|------------------|-----------|--------|----------|
| Bored | 0.89 | 0.92 | 0.90 |
| Confused | 0.87 | 0.85 | 0.86 |
| Engaged | 0.95 | 0.93 | 0.94 |
| Neutral | 0.92 | 0.94 | 0.93 |

### Sample Predictions

Example Grad-CAM visualizations showing model attention:

- **Engaged**: Focus on eyes and mouth (smiling)
- **Confused**: Attention on furrowed brows and eye regions
- **Bored**: Emphasis on drooping eyelids and facial relaxation
- **Neutral**: Distributed attention across entire face

---

## 🗂️ Project Structure

```
Student-Engagement-Detection/
│
├── final model 4/              # Trained model files (download separately)
│   ├── config.json             # Model configuration
│   ├── model.safetensors       # Model weights (327 MB)
│   └── preprocessor_config.json
│
├── Model_Live_Demo.ipynb       # Main demo notebook (local)
├── finetune.ipynb              # Training notebook (Google Colab)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🔧 Configuration

### Adjusting Inference Parameters

**Temporal Smoothing Window:**
```python
smoother = TemporalSmoother(
    num_classes=4, 
    window_size=10,    # Frames to average (higher = smoother but slower)
    alpha=0.3          # EMA weight (lower = more smoothing)
)
```

**Face Detection Confidence:**
```python
mtcnn = MTCNN(
    keep_all=True, 
    device=device, 
    post_process=False,
    min_face_size=40,       # Minimum face size in pixels
    thresholds=[0.6, 0.7, 0.9]  # Detection thresholds
)
```

**Screen Capture FPS:**
```python
screen_emotion_detection(
    monitor=1,        # 1 = primary display, 0 = all monitors
    fps=5,            # Frames per second (higher = more CPU intensive)
    use_gradcam=False # Enable for visualization (slower)
)
```

---

## 🧪 Training Your Own Model

### Dataset Structure

Organize your data as:
```
dataset/
├── train/
│   ├── Bored/
│   ├── Confused/
│   ├── Engaged/
│   └── Neutral/
├── val/
│   └── [same structure]
└── test/
    └── [same structure]
```

### Fine-tuning Steps

1. **Prepare dataset** in the structure above
2. **Upload to Google Colab** or use local GPU
3. **Open `finetune.ipynb`**
4. **Modify paths:**
   ```python
   dataset_root = "/path/to/your/dataset"
   save_dir = "/path/to/save/checkpoints"
   ```
5. **Adjust hyperparameters:**
   ```python
   target_samples = 150  # Oversampling target
   num_train_epochs = 7
   learning_rate = 2e-5
   ```
6. **Run all cells** to train
7. **Download trained model** from `save_dir/final_model_4`

### Hardware Requirements

**Minimum:**
- GPU: 6GB VRAM (GTX 1060 6GB or better)
- RAM: 8GB
- Storage: 2GB for model + dataset

**Recommended:**
- GPU: 12GB VRAM (RTX 3060 or better)
- RAM: 16GB
- Storage: 10GB for experiments

---

## 📈 Future Improvements

- [ ] Multi-modal fusion (audio + visual cues)
- [ ] Attention span tracking over time
- [ ] Integration with LMS platforms (Moodle, Canvas)
- [ ] Real-time dashboard for educators
- [ ] Mobile app deployment (TensorFlow Lite conversion)
- [ ] Federated learning for privacy-preserving training
- [ ] Support for group engagement analysis

---

## 🤝 Acknowledgments

- **Base Model**: Microsoft Research for [BEiT Architecture](https://github.com/microsoft/unilm/tree/master/beit)
- **Emotion Pre-training**: [Tanneru's HuggingFace](https://huggingface.co/Tanneru) for FER2013/RAF-DB/AffectNet fine-tuned weights
- **Face Detection**: [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for MTCNN implementation
- **Grad-CAM**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) for interpretability tools
- **Datasets**: FER2013, RAF-DB, AffectNet (emotion recognition benchmarks)

---


## 📧 Contact

**Nihar**: [@nihar245](https://github.com/nihar245)

**Advait**: [@Asterrage2209](https://github.com/Asterrage2209)

**Project Link**: [https://github.com/nihar245/Student-Engagement-Detection](https://github.com/nihar245/Student-Engagement-Detection)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for improving online education

</div>
