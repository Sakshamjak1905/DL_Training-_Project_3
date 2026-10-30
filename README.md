# 🧠 DeepVision AI: CNN Image Classifier & Explainable AI Studio

An end-to-end Computer Vision & Deep Learning project built with **PyTorch** and **Streamlit**.

## 🚀 Quick Start (1-Click Run)

### Option 1: Double-Click Launcher
Double-click **`run_app.bat`** on Windows.

### Option 2: Command Line
```powershell
python -m streamlit run app2.py
```

---

## 🌟 Key Features

1. **Modular CNN Architecture (`cnn_model.py`)**:
   - Conv2D (32) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool
   - Conv2D (64) $\to$ BatchNorm $\to$ ReLU $\to$ MaxPool
   - Adaptive Pooling $\to$ Flatten $\to$ Linear(128) $\to$ Dropout(0.3) $\to$ Output
2. **Dual Dataset Pipelines**:
   - Built-in **CIFAR-10** (10 classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)
   - **Custom ZIP dataset upload** with automatic folder parsing (comes with `sample_cats_vs_dogs.zip`)
3. **Real-Time Data Augmentation Playground**:
   - Interactive sliders for rotation, horizontal flipping, brightness jitter, and zoom.
4. **Live Training & Convergence Tracking**:
   - Live epoch-by-epoch accuracy and loss curves with overfitting diagnostics.
   - One-click model weights export (`.pth`).
5. **Inference with Top-5 Confidence Bar Charts**:
   - Test on uploaded images or included benchmark sample images.
6. **Explainable AI (XAI)**:
   - **Grad-CAM (Gradient-weighted Class Activation Mapping)** attention heatmaps.
   - **Intermediate Feature Map Visualizer** (inspect what all 16/32 filters detect).
7. **Model Evaluation**:
   - Quantitative Confusion Matrix heatmap with Precision, Recall, and F1-Scores.
8. **Academic Viva / Oral Exam Cheat Sheet**:
   - Detailed answers to the top 8 questions asked by professors during evaluations.

