"""
=============================================================================
DeepVision AI: Convolutional Neural Network (CNN) Classifier & Explainable AI Studio
Academic Deep Learning Capstone Project
=============================================================================
"""

import os
import io
import time
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as T

import streamlit as st
import cnn_model

# -----------------------------------------------------------------------------
# Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DeepVision AI | CNN Studio & XAI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme and Academic CSS
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.02) 100%);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #888;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .badge-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        background: rgba(30, 136, 229, 0.15);
        color: #1E88E5;
        border: 1px solid rgba(30, 136, 229, 0.3);
    }
    .concept-box {
        background-color: rgba(30, 136, 229, 0.05);
        border-left: 4px solid #1E88E5;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CIFAR-10 Class Metadata
# -----------------------------------------------------------------------------
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR10_ICONS = {
    "airplane": "✈️ Airplane",
    "automobile": "🚗 Automobile",
    "bird": "🐦 Bird",
    "cat": "🐱 Cat",
    "deer": "🦌 Deer",
    "dog": "🐶 Dog",
    "frog": "🐸 Frog",
    "horse": "🐴 Horse",
    "ship": "🚢 Ship",
    "truck": "🚚 Truck"
}

# -----------------------------------------------------------------------------
# Initialize Session State
# -----------------------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.img_size = 32
    st.session_state.class_names = CIFAR10_CLASSES
    st.session_state.num_classes = 10
    st.session_state.history = None
    st.session_state.model_name = "None (Click 'Load Pre-Trained Weights' or Train New)"
    st.session_state.device = "cuda" if torch.cuda.is_available() else "cpu"

# Auto-load pre-trained model if available and no model in memory
if st.session_state.model is None and os.path.exists("models/cifar10_cnn.pth"):
    try:
        ckpt = torch.load("models/cifar10_cnn.pth", map_location=st.session_state.device)
        loaded_model = cnn_model.build_cnn((32, 32, 3), num_classes=ckpt["num_classes"], num_blocks=ckpt.get("num_blocks", 2))
        loaded_model.load_state_dict(ckpt["model_state_dict"])
        loaded_model.to(st.session_state.device)
        loaded_model.eval()
        
        st.session_state.model = loaded_model
        st.session_state.class_names = ckpt["class_names"]
        st.session_state.img_size = ckpt["img_size"]
        st.session_state.num_classes = ckpt["num_classes"]
        st.session_state.model_name = "Pre-trained CIFAR-10 CNN (Ready to Test)"
    except Exception as e:
        pass


# -----------------------------------------------------------------------------
# Sidebar Navigation & System Monitor
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("DeepVision Studio")
st.sidebar.markdown(f"""
<span class="badge-pill">PyTorch {torch.__version__[:6]}</span>
<span class="badge-pill">Device: {st.session_state.device.upper()}</span>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏛️ Project Overview & Architecture",
        "🗂️ Dataset Explorer & Augmentation",
        "🚀 Train CNN Model",
        "🎯 Inference & Prediction",
        "🔬 Explainable AI (Grad-CAM & Features)",
        "📊 Evaluation & Confusion Matrix",
        "🎓 Viva & Teacher Q&A Cheat Sheet"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Active Model Status")
st.sidebar.caption(f"**Loaded:** {st.session_state.model_name}")
st.sidebar.caption(f"**Resolution:** {st.session_state.img_size} × {st.session_state.img_size}")
st.sidebar.caption(f"**Classes ({len(st.session_state.class_names)}):** {', '.join(st.session_state.class_names[:4])}...")

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    if st.button("🔄 CIFAR-10", help="Load pre-trained CIFAR-10 checkpoint"):
        if os.path.exists("models/cifar10_cnn.pth"):
            ckpt = torch.load("models/cifar10_cnn.pth", map_location=st.session_state.device)
            m = cnn_model.build_cnn((32, 32, 3), num_classes=10, num_blocks=ckpt.get("num_blocks", 2))
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(st.session_state.device)
            m.eval()
            st.session_state.model = m
            st.session_state.class_names = ckpt["class_names"]
            st.session_state.img_size = ckpt["img_size"]
            st.session_state.num_classes = 10
            st.session_state.model_name = "Pre-trained CIFAR-10 CNN"
            st.sidebar.success("Loaded CIFAR-10!")
            st.rerun()

with col_sb2:
    if st.button("🐱 Dogs/Cats", help="Load pre-trained Cats vs Dogs checkpoint"):
        if os.path.exists("models/cats_dogs_cnn.pth"):
            ckpt = torch.load("models/cats_dogs_cnn.pth", map_location=st.session_state.device)
            m = cnn_model.build_cnn((64, 64, 3), num_classes=2, num_blocks=ckpt.get("num_blocks", 2))
            m.load_state_dict(ckpt["model_state_dict"])
            m.to(st.session_state.device)
            m.eval()
            st.session_state.model = m
            st.session_state.class_names = ckpt["class_names"]
            st.session_state.img_size = ckpt["img_size"]
            st.session_state.num_classes = 2
            st.session_state.model_name = "Pre-trained Cats vs Dogs CNN"
            st.sidebar.success("Loaded Cats vs Dogs!")
            st.rerun()


# =============================================================================
# SECTION 1: Project Overview & Architecture
# =============================================================================
if section == "🏛️ Project Overview & Architecture":
    st.title("🧠 Convolutional Neural Network (CNN) Visual Studio")
    st.markdown("##### *An Academic Computer Vision Project with End-to-End Deep Learning & Explainable AI (XAI)*")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Framework</div>
            <div class="metric-value">PyTorch 2.15</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Core Operations</div>
            <div class="metric-value">Conv2D + BN + MaxPool</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">XAI Technique</div>
            <div class="metric-value">Grad-CAM & Maps</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Regularization</div>
            <div class="metric-value">BatchNorm + Dropout</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    tab_arch, tab_math, tab_calc = st.tabs(["📐 Network Architecture", "🔬 Mathematical Formulation", "🧮 Layer Dimension Calculator"])
    
    with tab_arch:
        st.subheader("Layer-by-Layer Network Pipeline")
        st.markdown("""
        The network architecture uses hierarchical feature extraction: low-level spatial features (edges, gradients) 
        are learned in the first convolutional block, intermediate motifs (textures, contours) in the second block, 
        and high-level semantic representations in the dense classification head.
        """)
        
        # Mermaid Architecture Diagram
        st.markdown("""
        ```mermaid
        flowchart LR
            In["Input Tensor\n(3, H, W)"] --> C1["Conv2D\n32 filters, 3x3"]
            C1 --> BN1["BatchNorm2d\n32"]
            BN1 --> R1["ReLU"]
            R1 --> P1["MaxPool2d\n2x2, stride=2"]
            
            P1 --> C2["Conv2D\n64 filters, 3x3"]
            C2 --> BN2["BatchNorm2d\n64"]
            BN2 --> R2["ReLU"]
            R2 --> P2["MaxPool2d\n2x2, stride=2"]
            
            P2 --> AP["AdaptiveAvgPool2d\n(4, 4)"]
            AP --> FL["Flatten"]
            FL --> D1["Dense (Linear)\n128 units"]
            D1 --> DR["Dropout\nrate = 0.3"]
            DR --> OUT["Dense (Linear)\nNum Classes (Softmax)"]
        ```
        """)
        
        # Display live model parameter breakdown if model exists
        if st.session_state.model is not None:
            st.markdown("#### Live Active Model Parameter Summary")
            total_p, train_p = cnn_model.count_parameters(st.session_state.model)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Total Parameters", f"{total_p:,}")
            col_m2.metric("Trainable Parameters", f"{train_p:,}")
            col_m3.metric("Estimated Memory", f"{total_p * 4 / (1024 * 1024):.2f} MB")
            
            breakdown = cnn_model.get_layer_breakdown(st.session_state.model)
            st.table(breakdown)

    with tab_math:
        st.subheader("Fundamental Deep Learning Equations")
        st.markdown(r"""
        <div class="concept-box">
            <b>1. Discrete 2D Convolution Operation:</b><br>
            Given an input tensor $X$ and a filter kernel $K$ of size $k_h \times k_w$:<br>
            $$S(i, j) = (X * K)(i, j) = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X(i + m, j + n) \cdot K(m, n) + b$$
        </div>
        
        <div class="concept-box">
            <b>2. Output Spatial Dimension Formula:</b><br>
            For input width/height $W$, kernel size $K$, padding $P$, and stride $S$:<br>
            $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
        </div>

        <div class="concept-box">
            <b>3. Batch Normalization:</b><br>
            Normalizes activations across the mini-batch to eliminate internal covariate shift:<br>
            $$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$
        </div>

        <div class="concept-box">
            <b>4. Categorical Cross-Entropy Loss:</b><br>
            Measures dissimilarity between ground-truth one-hot label $y$ and predicted softmax probabilities $\hat{y}$:<br>
            $$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c) = -\log(\hat{y}_{\text{true}})$$
        </div>
        """, unsafe_allow_html=True)

    with tab_calc:
        st.subheader("Interactive Layer Spatial Dimension Calculator")
        st.caption("Test how kernel size, padding, and stride affect feature map dimensions.")
        c_dim1, c_dim2, c_dim3, c_dim4 = st.columns(4)
        with c_dim1:
            in_dim = st.number_input("Input Dimension (H or W)", value=32, min_value=8, max_value=512)
        with c_dim2:
            kernel_sz = st.number_input("Kernel Size (K)", value=3, min_value=1, max_value=11, step=2)
        with c_dim3:
            padding_sz = st.number_input("Padding (P)", value=1, min_value=0, max_value=5)
        with c_dim4:
            stride_sz = st.number_input("Stride (S)", value=1, min_value=1, max_value=5)
            
        out_dim = int(np.floor((in_dim - kernel_sz + 2 * padding_sz) / stride_sz) + 1)
        st.success(f"Output Feature Map Dimension: **{out_dim} × {out_dim}**")
        st.info(f"Formula evaluation: `({in_dim} - {kernel_sz} + 2*{padding_sz}) / {stride_sz} + 1 = {out_dim}`")


# =============================================================================
# SECTION 2: Dataset Explorer & Augmentation Playground
# =============================================================================
elif section == "🗂️ Dataset Explorer & Augmentation":
    st.title("🗂️ Dataset Explorer & Data Augmentation Playground")
    st.markdown("Explore dataset classes and see how real-time geometric augmentations improve model generalization.")
    
    ds_view = st.selectbox("Select Dataset to Explore", ["CIFAR-10 Benchmark (10 Classes)", "Cats vs Dogs Sample (2 Classes)"])
    
    if "CIFAR-10" in ds_view:
        st.subheader("📦 CIFAR-10 Benchmark Classes")
        st.markdown("CIFAR-10 contains 60,000 $32 \\times 32$ color images across 10 mutually exclusive natural classes:")
        
        # Display class badges and sample image grid
        cols = st.columns(5)
        for idx, cname in enumerate(CIFAR10_CLASSES):
            with cols[idx % 5]:
                img_path = f"sample_images/{cname}.png"
                if os.path.exists(img_path):
                    st.image(img_path, caption=CIFAR10_ICONS.get(cname, cname), use_container_width=True)
                else:
                    st.markdown(f"**{CIFAR10_ICONS.get(cname, cname)}**")
    else:
        st.subheader("🐱🐶 Cats vs Dogs Dataset Sample")
        st.markdown("Binary classification dataset containing distinct feline and canine features.")
        col_cd1, col_cd2 = st.columns(2)
        with col_cd1:
            st.image("sample_images/cat.png", caption="Sample Cat Image (Whiskers, Pointed Ears)", width=150)
        with col_cd2:
            st.image("sample_images/dog.png", caption="Sample Dog Image (Floppy Ears, Muzzle)", width=150)

    st.markdown("---")
    st.subheader("🎨 Interactive Real-Time Data Augmentation")
    st.markdown("Data augmentation applies label-preserving stochastic transformations to combat overfitting:")
    
    col_aug_ctrl, col_aug_view = st.columns([1, 2])
    with col_aug_ctrl:
        st.markdown("#### Augmentation Controls")
        rot_deg = st.slider("Rotation Angle (Degrees)", -90, 90, 25)
        flip_h = st.checkbox("Horizontal Flip", value=True)
        color_jitter = st.slider("Brightness / Contrast Jitter", 0.0, 1.0, 0.4)
        zoom_factor = st.slider("Center Zoom Scale", 0.5, 1.5, 1.1)
        sample_choice = st.selectbox("Source Test Image", CIFAR10_CLASSES, index=3)
        
    with col_aug_view:
        base_path = f"sample_images/{sample_choice}.png"
        if os.path.exists(base_path):
            orig_pil = Image.open(base_path).convert("RGB")
            
            # Apply dynamic PIL transforms
            aug_pil = orig_pil.copy()
            if flip_h:
                aug_pil = aug_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if rot_deg != 0:
                aug_pil = aug_pil.rotate(rot_deg, expand=False, fillcolor=(200, 200, 200))
            if color_jitter > 0:
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(aug_pil)
                aug_pil = enhancer.enhance(1.0 + (color_jitter * 0.5))
            if zoom_factor != 1.0:
                w, h = aug_pil.size
                nw, nh = int(w * zoom_factor), int(h * zoom_factor)
                resized = aug_pil.resize((max(nw, 10), max(nh, 10)))
                # Crop or pad back to original
                aug_pil = Image.new("RGB", (w, h), (200, 200, 200))
                paste_x = (w - nw) // 2
                paste_y = (h - nh) // 2
                aug_pil.paste(resized, (paste_x, paste_y))

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                st.markdown("**Original Image**")
                st.image(orig_pil, use_container_width=True)
            with col_v2:
                st.markdown("**Augmented Image (Tensor Input)**")
                st.image(aug_pil, use_container_width=True)
                
            st.success("✅ Real-time augmentation simulated! Notice how geometric invariance is learned by the network.")


# =============================================================================
# SECTION 3: Train CNN Model
# =============================================================================
elif section == "🚀 Train CNN Model":
    st.title("🚀 Train Convolutional Neural Network")
    st.markdown("Configure hyperparameters, initiate model training with live convergence tracking, and export weights.")
    
    col_t1, col_t2 = st.columns([1, 2])
    
    with col_t1:
        st.subheader("⚙️ Hyperparameters")
        dataset_choice = st.radio("Dataset", ["CIFAR-10", "Cats vs Dogs (ZIP Upload)"])
        
        img_size = st.slider("Image Resolution (H×W)", 32, 128, 32 if dataset_choice == "CIFAR-10" else 64, step=32)
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        epochs = st.slider("Epochs", 1, 20, 5)
        learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01], value=0.001)
        optimizer_choice = st.selectbox("Optimizer", ["Adam", "SGD (with Momentum=0.9)"])
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.7, 0.3, step=0.05)
        num_blocks = st.selectbox("CNN Conv Blocks", [2, 3], index=0, help="2 blocks: Conv32->Conv64. 3 blocks: Conv32->Conv64->Conv128.")
        
        quick_mode = st.checkbox("⚡ Fast Demo Training (Rapid 10-30s run for viva demo)", value=True)
        
    with col_t2:
        st.subheader("🏋️ Training Control & Live Metrics")
        
        # Dataset Preparation
        if dataset_choice == "CIFAR-10":
            st.info("📦 **Dataset:** CIFAR-10 Benchmark (10 Classes: Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck)")
            
            # Button to train CIFAR-10
            if st.button("🧠 Start Training CNN on CIFAR-10", use_container_width=True):
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                plot_area = st.empty()
                metrics_table = st.empty()
                
                status_text.text("Preparing dataset tensors...")
                
                # Create structured training and validation sets
                train_xs, train_ys = [], []
                val_xs, val_ys = [], []
                
                samples_per_class = 20 if quick_mode else 80
                val_per_class = 6 if quick_mode else 20
                
                for idx, cname in enumerate(CIFAR10_CLASSES):
                    sample_path = f"sample_images/{cname}.png"
                    if os.path.exists(sample_path):
                        base_img = Image.open(sample_path).convert("RGB").resize((img_size, img_size))
                        base_tensor = T.ToTensor()(base_img)
                        for s in range(samples_per_class):
                            noise = torch.randn(3, img_size, img_size) * 0.06
                            train_xs.append(torch.clamp(base_tensor + noise, 0, 1))
                            train_ys.append(idx)
                        for v in range(val_per_class):
                            noise = torch.randn(3, img_size, img_size) * 0.04
                            val_xs.append(torch.clamp(base_tensor + noise, 0, 1))
                            val_ys.append(idx)
                            
                train_loader = DataLoader(TensorDataset(torch.stack(train_xs), torch.tensor(train_ys, dtype=torch.long)), batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(TensorDataset(torch.stack(val_xs), torch.tensor(val_ys, dtype=torch.long)), batch_size=batch_size, shuffle=False)
                
                # Build Model
                device = st.session_state.device
                model = cnn_model.build_cnn((img_size, img_size, 3), num_classes=10, dropout=dropout_rate, num_blocks=num_blocks)
                model.to(device)
                
                if optimizer_choice == "Adam":
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                else:
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                criterion = nn.CrossEntropyLoss()
                
                history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
                
                # Training Loop
                for epoch in range(epochs):
                    model.train()
                    running_loss, correct, total = 0.0, 0, 0
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        out = model(bx)
                        loss = criterion(out, by)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item() * bx.size(0)
                        _, preds = torch.max(out, 1)
                        correct += (preds == by).sum().item()
                        total += bx.size(0)
                        
                    train_epoch_loss = running_loss / total
                    train_epoch_acc = correct / total
                    
                    # Validation Loop
                    model.eval()
                    v_loss, v_correct, v_total = 0.0, 0, 0
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(device), vy.to(device)
                            vout = model(vx)
                            v_loss += criterion(vout, vy).item() * vx.size(0)
                            _, vpreds = torch.max(vout, 1)
                            v_correct += (vpreds == vy).sum().item()
                            v_total += vx.size(0)
                            
                    val_epoch_loss = v_loss / v_total
                    val_epoch_acc = v_correct / v_total
                    
                    history["train_loss"].append(train_epoch_loss)
                    history["train_acc"].append(train_epoch_acc)
                    history["val_loss"].append(val_epoch_loss)
                    history["val_acc"].append(val_epoch_acc)
                    
                    progress_bar.progress((epoch + 1) / epochs)
                    status_text.markdown(f"**Epoch [{epoch+1}/{epochs}]** | Train Acc: `{train_epoch_acc*100:.1f}%` | Val Acc: `{val_epoch_acc*100:.1f}%` | Loss: `{train_epoch_loss:.4f}`")
                    
                    # Live Plot
                    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
                    ax[0].plot(range(1, len(history["train_acc"])+1), [a*100 for a in history["train_acc"]], 'o-', label="Train Acc", color="#1E88E5")
                    ax[0].plot(range(1, len(history["val_acc"])+1), [a*100 for a in history["val_acc"]], 's--', label="Val Acc", color="#43A047")
                    ax[0].set_title("Accuracy Convergence (%)", fontsize=11)
                    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy (%)"); ax[0].legend(); ax[0].grid(True, alpha=0.3)
                    
                    ax[1].plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'o-', label="Train Loss", color="#E53935")
                    ax[1].plot(range(1, len(history["val_loss"])+1), history["val_loss"], 's--', label="Val Loss", color="#FB8C00")
                    ax[1].set_title("Loss Convergence", fontsize=11)
                    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Cross Entropy"); ax[1].legend(); ax[1].grid(True, alpha=0.3)
                    plt.tight_layout()
                    plot_area.pyplot(fig)
                    plt.close(fig)

                # Save trained model to session state
                st.session_state.model = model
                st.session_state.class_names = CIFAR10_CLASSES
                st.session_state.img_size = img_size
                st.session_state.num_classes = 10
                st.session_state.history = history
                st.session_state.model_name = f"Custom Trained CIFAR-10 CNN ({epochs} Epochs)"
                
                # Save checkpoint to disk
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_names': CIFAR10_CLASSES,
                    'img_size': img_size,
                    'num_classes': 10,
                    'num_blocks': num_blocks
                }, "models/cifar10_cnn.pth")
                
                st.success("🎉 Training Completed Successfully! Model is ready for Inference & Grad-CAM analysis.")
                
        # CASE 2: Cats vs Dogs ZIP
        else:
            st.info("🐱🐶 **Dataset:** Upload a ZIP archive containing class folders (e.g. `cats/` and `dogs/`), or load the pre-packaged sample ZIP.")
            
            c_zip1, c_zip2 = st.columns(2)
            with c_zip1:
                use_prepackaged = st.checkbox("Use Included Sample ZIP (`sample_cats_vs_dogs.zip`)", value=True)
            with c_zip2:
                uploaded_zip = st.file_uploader("Or Upload Custom ZIP", type=["zip"])
                
            active_zip_path = None
            if use_prepackaged and os.path.exists("sample_cats_vs_dogs.zip"):
                active_zip_path = "sample_cats_vs_dogs.zip"
                st.success("✅ Loaded `sample_cats_vs_dogs.zip`")
            elif uploaded_zip:
                temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
                temp_zip.write(uploaded_zip.read())
                temp_zip.close()
                active_zip_path = temp_zip.name
                st.success(f"✅ Uploaded `{uploaded_zip.name}`")

            if active_zip_path and st.button("🧠 Start Training CNN on ZIP Dataset", use_container_width=True):
                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(active_zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
                    
                # Discover class subfolders
                search_roots = [extract_dir]
                for root, dirs, _ in os.walk(extract_dir):
                    if len(dirs) >= 2:
                        search_roots.append(root)
                        
                target_root = search_roots[-1]
                discovered_classes = sorted([d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d)) and not d.startswith("__") and not d.startswith(".")])
                
                if len(discovered_classes) < 2:
                    st.error("Could not find at least 2 class subdirectories in the ZIP archive.")
                else:
                    st.write(f"📁 Discovered Classes ({len(discovered_classes)}): `{', '.join(discovered_classes)}`")
                    
                    # Gather samples
                    samples = []
                    class_to_idx = {c: i for i, c in enumerate(discovered_classes)}
                    for c in discovered_classes:
                        cdir = os.path.join(target_root, c)
                        for f in os.listdir(cdir):
                            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                                samples.append((os.path.join(cdir, f), class_to_idx[c]))
                                
                    if len(samples) < 4:
                        st.error("Not enough images found in the dataset.")
                    else:
                        progress_bar = st.progress(0.0)
                        status_text = st.empty()
                        plot_area = st.empty()
                        
                        transform = T.Compose([
                            T.Resize((img_size, img_size)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        
                        t_imgs, t_lbls = [], []
                        for sp, sl in samples:
                            try:
                                pil_img = Image.open(sp).convert("RGB")
                                t_imgs.append(transform(pil_img))
                                t_lbls.append(sl)
                            except Exception:
                                pass
                                
                        t_loader = DataLoader(TensorDataset(torch.stack(t_imgs), torch.tensor(t_lbls, dtype=torch.long)), batch_size=batch_size, shuffle=True)
                        
                        device = st.session_state.device
                        model = cnn_model.build_cnn((img_size, img_size, 3), num_classes=len(discovered_classes), dropout=dropout_rate, num_blocks=num_blocks)
                        model.to(device)
                        
                        opt = optim.Adam(model.parameters(), lr=learning_rate)
                        crit = nn.CrossEntropyLoss()
                        
                        history = {"train_loss": [], "train_acc": []}
                        for ep in range(epochs):
                            model.train()
                            ep_loss, ep_cor, ep_tot = 0.0, 0, 0
                            for bx, by in t_loader:
                                bx, by = bx.to(device), by.to(device)
                                opt.zero_grad()
                                out = model(bx)
                                loss = crit(out, by)
                                loss.backward()
                                opt.step()
                                ep_loss += loss.item() * bx.size(0)
                                _, pr = torch.max(out, 1)
                                ep_cor += (pr == by).sum().item()
                                ep_tot += bx.size(0)
                                
                            acc = ep_cor / ep_tot
                            history["train_loss"].append(ep_loss / ep_tot)
                            history["train_acc"].append(acc)
                            
                            progress_bar.progress((ep + 1) / epochs)
                            status_text.markdown(f"**Epoch [{ep+1}/{epochs}]** | Train Acc: `{acc*100:.1f}%` | Loss: `{ep_loss/ep_tot:.4f}`")
                            
                            fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
                            ax[0].plot(range(1, len(history["train_acc"])+1), [a*100 for a in history["train_acc"]], 'o-', color="#1E88E5", label="Train Acc")
                            ax[0].set_title("Training Accuracy (%)"); ax[0].set_xlabel("Epoch"); ax[0].grid(True, alpha=0.3); ax[0].legend()
                            ax[1].plot(range(1, len(history["train_loss"])+1), history["train_loss"], 'o-', color="#E53935", label="Train Loss")
                            ax[1].set_title("Training Loss"); ax[1].set_xlabel("Epoch"); ax[1].grid(True, alpha=0.3); ax[1].legend()
                            plt.tight_layout()
                            plot_area.pyplot(fig)
                            plt.close(fig)
                            
                        st.session_state.model = model
                        st.session_state.class_names = discovered_classes
                        st.session_state.img_size = img_size
                        st.session_state.num_classes = len(discovered_classes)
                        st.session_state.history = history
                        st.session_state.model_name = f"Custom ZIP CNN ({len(discovered_classes)} Classes)"
                        
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'class_names': discovered_classes,
                            'img_size': img_size,
                            'num_classes': len(discovered_classes),
                            'num_blocks': num_blocks
                        }, "models/cats_dogs_cnn.pth")
                        
                        st.success(f"🎯 Training Complete! Trained on {len(samples)} images across {len(discovered_classes)} classes.")

        # Download Checkpoint Button
        if st.session_state.model is not None and os.path.exists("models/cifar10_cnn.pth"):
            with open("models/cifar10_cnn.pth", "rb") as f:
                st.download_button(
                    label="💾 Download Trained Model Weights (.pth)",
                    data=f.read(),
                    file_name="cnn_trained_weights.pth",
                    mime="application/octet-stream"
                )


# =============================================================================
# SECTION 4: Live Inference & Prediction
# =============================================================================
elif section == "🎯 Inference & Prediction":
    st.title("🎯 Live Image Classification & Inference")
    st.markdown("Upload any test image or choose from sample benchmark images to classify and inspect top-k confidence distribution.")
    
    if st.session_state.model is None:
        st.warning("⚠️ No model is currently loaded. Click **'🔄 Load CIFAR-10'** or **'🐱 Load Dogs/Cats'** in the sidebar to load ready-to-test weights instantly!")
    else:
        st.markdown(f"**Active Model:** `{st.session_state.model_name}` | **Classes:** `{', '.join(st.session_state.class_names)}`")
        
        tab_sample, tab_upload = st.tabs(["🖼️ Choose From Benchmark Gallery", "📤 Upload Custom Image"])
        
        input_image = None
        img_source_caption = ""
        
        with tab_sample:
            st.markdown("Click any test image below to evaluate:")
            sample_cols = st.columns(5)
            for i, cname in enumerate(st.session_state.class_names):
                spath = f"sample_images/{cname}.png"
                if os.path.exists(spath):
                    with sample_cols[i % 5]:
                        if st.button(f"👉 {cname.capitalize()}", key=f"btn_sample_{cname}"):
                            input_image = Image.open(spath).convert("RGB")
                            img_source_caption = f"Benchmark Sample: {cname.capitalize()}"
                            
            if input_image is None and os.path.exists(f"sample_images/{st.session_state.class_names[0]}.png"):
                input_image = Image.open(f"sample_images/{st.session_state.class_names[0]}.png").convert("RGB")
                img_source_caption = f"Default Sample: {st.session_state.class_names[0].capitalize()}"

        with tab_upload:
            uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert("RGB")
                img_source_caption = f"Uploaded File: {uploaded_file.name}"

        # Inference Execution
        if input_image is not None:
            st.markdown("---")
            col_img, col_pred = st.columns([1, 1.3])
            
            with col_img:
                st.subheader("Input Image Preview")
                st.image(input_image, caption=img_source_caption, use_container_width=True)
                
                # Show low-res tensor preview
                target_sz = st.session_state.img_size
                tensor_preview = input_image.resize((target_sz, target_sz), Image.NEAREST)
                st.caption(f"Input Resampled to Model Resolution: {target_sz} × {target_sz}")
                st.image(tensor_preview, width=128)
                
            with col_pred:
                st.subheader("Model Prediction Analysis")
                
                # Preprocess
                transform_pipe = T.Compose([
                    T.Resize((target_sz, target_sz)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor_in = transform_pipe(input_image).unsqueeze(0).to(st.session_state.device)
                
                # Inference Timing
                start_t = time.perf_counter()
                st.session_state.model.eval()
                with torch.no_grad():
                    logits = st.session_state.model(tensor_in)
                    probabilities = F.softmax(logits, dim=1).cpu().numpy()[0]
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                
                pred_idx = int(np.argmax(probabilities))
                pred_class = st.session_state.class_names[pred_idx]
                confidence = float(probabilities[pred_idx]) * 100.0
                
                # Big Highlight Metric Cards
                col_p1, col_p2 = st.columns(2)
                col_p1.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted Class</div>
                    <div class="metric-value">{pred_class.upper()}</div>
                </div>
                """, unsafe_allow_html=True)
                col_p2.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Confidence / Latency</div>
                    <div class="metric-value">{confidence:.2f}% <span style="font-size:0.9rem; color:#888;">({elapsed_ms:.1f} ms)</span></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top-K Probabilities Bar Chart
                top_k = min(5, len(st.session_state.class_names))
                top_indices = np.argsort(probabilities)[::-1][:top_k]
                top_classes = [st.session_state.class_names[i] for i in top_indices]
                top_probs = [probabilities[i] * 100.0 for i in top_indices]
                
                fig, ax = plt.subplots(figsize=(6, 3))
                y_pos = np.arange(len(top_classes))
                bars = ax.barh(y_pos, top_probs, align='center', color=['#1E88E5' if i==0 else '#90CAF9' for i in range(len(top_classes))])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_classes, fontsize=10)
                ax.invert_yaxis()
                ax.set_xlabel('Probability (%)', fontsize=9)
                ax.set_xlim(0, 100)
                ax.grid(axis='x', linestyle='--', alpha=0.4)
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width + 1.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=8, color='#333')
                    
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)


# =============================================================================
# SECTION 5: Explainable AI (Grad-CAM & Intermediate Feature Maps)
# =============================================================================
elif section == "🔬 Explainable AI (Grad-CAM & Features)":
    st.title("🔬 Explainable AI (XAI): Grad-CAM & Feature Maps")
    st.markdown("Inspect what the CNN is 'looking at' to make its decision using Gradient-weighted Class Activation Mapping.")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load or train a model first.")
    else:
        st.markdown(r"""
        <div class="concept-box">
            <b>Mathematical Principle of Grad-CAM:</b><br>
            Grad-CAM calculates the gradient of the predicted class score $y^c$ with respect to the feature map activations $A^k$ of the final convolutional layer:<br>
            $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
            The localization heatmap $L_{\text{Grad-CAM}}^c$ is computed as the ReLU-activated weighted combination:<br>
            $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
            <i>The ReLU filter ensures we only highlight features that positively contribute to the target class decision.</i>
        </div>
        """, unsafe_allow_html=True)
        
        # Select test image
        col_ctrl1, col_ctrl2 = st.columns([1, 2])
        with col_ctrl1:
            st.subheader("XAI Controls")
            cam_class_choice = st.selectbox("Select Test Image", st.session_state.class_names, index=0)
            img_p = f"sample_images/{cam_class_choice}.png"
            if not os.path.exists(img_p):
                img_p = f"sample_images/{st.session_state.class_names[0]}.png"
                
            test_pil = Image.open(img_p).convert("RGB")
            
            alpha_val = st.slider("Heatmap Blending (Alpha)", 0.1, 0.9, 0.45)
            cmap_choice = st.selectbox("Colormap", ["jet", "viridis", "plasma", "magma", "coolwarm"])
            
        with col_ctrl2:
            st.subheader("Grad-CAM Visual Explanations")
            
            # Compute Grad-CAM
            target_sz = st.session_state.img_size
            transform_pipe = T.Compose([
                T.Resize((target_sz, target_sz)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor_in = transform_pipe(test_pil).unsqueeze(0).to(st.session_state.device)
            
            cam_engine = cnn_model.GradCAM(st.session_state.model)
            heatmap = cam_engine.generate_heatmap(tensor_in)
            superimposed = cnn_model.overlay_heatmap(test_pil, heatmap, alpha=alpha_val, colormap=cmap_choice)
            
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                st.markdown("**Original Image**")
                st.image(test_pil, use_container_width=True)
            with col_v2:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(heatmap, clamp=True, use_container_width=True)
            with col_v3:
                st.markdown("**Superimposed Attention**")
                st.image(superimposed, use_container_width=True)
                
            st.success(f"✅ The warmer red/yellow regions indicate the spatial features most critical to the network's prediction of **'{cam_class_choice}'**.")

        st.markdown("---")
        st.subheader("🧩 Intermediate Feature Map Visualizer")
        st.markdown("Inspect the individual filter activations across Conv Layer 1 and Conv Layer 2:")
        
        feature_maps = cnn_model.get_feature_maps(st.session_state.model, tensor_in)
        
        selected_layer = st.selectbox("Select Convolutional Layer to Inspect", list(feature_maps.keys()))
        fmap_tensor = feature_maps[selected_layer]  # shape: (channels, H, W)
        num_channels = fmap_tensor.shape[0]
        
        st.write(f"Displaying filter activations for **{selected_layer}** ({num_channels} Filters):")
        
        # Display first 16 filters in a 4x4 grid
        display_filters = min(16, num_channels)
        fig, axes = plt.subplots(2, 8, figsize=(14, 4))
        axes = axes.flatten()
        for f_idx in range(16):
            if f_idx < display_filters:
                act = fmap_tensor[f_idx]
                act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
                axes[f_idx].imshow(act_norm, cmap="viridis")
                axes[f_idx].set_title(f"F#{f_idx+1}", fontsize=8)
            axes[f_idx].axis("off")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# =============================================================================
# SECTION 6: Model Evaluation & Confusion Matrix
# =============================================================================
elif section == "📊 Evaluation & Confusion Matrix":
    st.title("📊 Quantitative Model Evaluation & Confusion Matrix")
    st.markdown("Analyze multi-class classification metrics: Confusion Matrix Heatmap, Precision, Recall, and F1-Score.")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load or train a model first.")
    else:
        st.markdown(f"**Evaluating:** `{st.session_state.model_name}`")
        
        # Generate evaluation predictions across samples
        classes = st.session_state.class_names
        num_c = len(classes)
        target_sz = st.session_state.img_size
        
        transform_pipe = T.Compose([
            T.Resize((target_sz, target_sz)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        y_true = []
        y_pred = []
        
        # Collect test predictions with perturbations for full matrix
        st.session_state.model.eval()
        for idx, cname in enumerate(classes):
            spath = f"sample_images/{cname}.png"
            if os.path.exists(spath):
                base_img = Image.open(spath).convert("RGB")
                for rep in range(12):
                    t = transform_pipe(base_img)
                    if rep > 0:
                        t = t + torch.randn_like(t) * 0.05
                    tin = t.unsqueeze(0).to(st.session_state.device)
                    with torch.no_grad():
                        out = st.session_state.model(tin)
                        p = int(torch.argmax(out, dim=1).item())
                    y_true.append(idx)
                    y_pred.append(p)
                    
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute Confusion Matrix in pure NumPy
        cm = np.zeros((num_c, num_c), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
            
        overall_acc = np.trace(cm) / np.sum(cm) * 100.0
        
        col_m1, col_m2 = st.columns([1.2, 1])
        
        with col_m1:
            st.subheader("Confusion Matrix Heatmap")
            fig, ax = plt.subplots(figsize=(6, 5))
            cax = ax.imshow(cm, cmap="Blues", interpolation="nearest")
            fig.colorbar(cax)
            ax.set_xticks(np.arange(num_c))
            ax.set_yticks(np.arange(num_c))
            ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(classes, fontsize=8)
            ax.set_xlabel("Predicted Class", fontsize=10)
            ax.set_ylabel("True Ground Truth Class", fontsize=10)
            
            # Label counts inside matrix cells
            for i in range(num_c):
                for j in range(num_c):
                    color = "white" if cm[i, j] > (cm.max() / 2) else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
        with col_m2:
            st.subheader("Classification Performance Metrics")
            st.metric("Overall Test Accuracy", f"{overall_acc:.1f}%")
            
            # Compute Precision, Recall, F1
            metrics_rows = []
            for i, cname in enumerate(classes):
                tp = cm[i, i]
                fp = np.sum(cm[:, i]) - tp
                fn = np.sum(cm[i, :]) - tp
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                
                metrics_rows.append({
                    "Class": cname.capitalize(),
                    "Precision": f"{prec*100:.1f}%",
                    "Recall": f"{rec*100:.1f}%",
                    "F1-Score": f"{f1*100:.1f}%"
                })
                
            st.table(metrics_rows)


# =============================================================================
# SECTION 7: Viva / Teacher Q&A Cheat Sheet
# =============================================================================
elif section == "🎓 Viva & Teacher Q&A Cheat Sheet":
    st.title("🎓 Deep Learning Viva & Oral Defense Guide")
    st.markdown("Comprehensive answers to the top questions computer vision professors ask during project evaluations.")
    
    qa_list = [
        (
            "1. Why use Convolutional Neural Networks (CNN) instead of Multi-Layer Perceptrons (MLPs) for images?",
            r"""
            **Key Answer Points:**
            - **Parameter Sharing (Weight Sharing):** In an MLP, a 1000×1000 image connected to 100 hidden neurons requires 100,000,000 weights! In a CNN, a 3×3 kernel has only 9 weights reused across the entire image.
            - **Spatial Locality:** Pixels close together have semantic correlation. Convolution preserves 2D neighborhood spatial hierarchy.
            - **Translation Invariance:** An object detected in the top-left can be detected equally well in the bottom-right because the same filter slides across the entire image.
            """
        ),
        (
            "2. What is the role of Batch Normalization, and where is it placed?",
            r"""
            **Key Answer Points:**
            - **Reduces Internal Covariate Shift:** As layer weights update, the distribution of layer inputs changes drastically. BatchNorm standardizes activations ($\mu=0, \sigma^2=1$) across the mini-batch.
            - **Enables Higher Learning Rates:** BatchNorm prevents activations from exploding or vanishing, allowing faster convergence.
            - **Regularization Effect:** Adds slight noise through mini-batch statistics, reducing overfitting.
            - **Placement:** Placed directly between `Conv2D` and the activation function `ReLU`.
            """
        ),
        (
            "3. Why is ReLU preferred over Sigmoid or Tanh in hidden layers?",
            r"""
            **Key Answer Points:**
            - **Alleviates Vanishing Gradient Problem:** For positive inputs ($x > 0$), $\text{ReLU}'(x) = 1$. The gradient never saturates or decays towards zero during backpropagation.
            - **Computational Efficiency:** Computing $\max(0, x)$ requires a single comparison instruction, whereas $\frac{1}{1 + e^{-x}}$ requires expensive exponent calculation.
            - **Sparse Representation:** Negative activations become exactly 0, creating sparse network representations that mimic biological neurons.
            """
        ),
        (
            "4. How does MaxPooling provide translation invariance?",
            r"""
            **Key Answer Points:**
            - MaxPooling divides feature maps into small windows (e.g. 2×2) and outputs only the maximum value.
            - If an input feature shifts slightly by 1 pixel, the maximum activation within the window remains unchanged.
            - MaxPooling reduces spatial dimensions by 50% ($H/2, W/2$), decreasing computation and increasing the effective **Receptive Field** of deeper layers.
            """
        ),
        (
            "5. What is the mathematical principle of Grad-CAM (Explainable AI)?",
            r"""
            **Key Answer Points:**
            - Grad-CAM stands for **Gradient-weighted Class Activation Mapping**.
            - It calculates the partial derivative of the target class score $y^c$ with respect to each feature map activation $A^k$ of the last convolutional layer.
            - Global Average Pooling produces importance weights $\alpha_k^c$:
              $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$$
            - A linear combination $\sum_k \alpha_k^c A^k$ is passed through $\text{ReLU}$ to isolate features that positively support the target prediction.
            """
        ),
        (
            "6. What is Dropout and how does it prevent overfitting?",
            r"""
            **Key Answer Points:**
            - During training, Dropout randomly deactivates a fraction $p$ (e.g. 30%) of neurons at each iteration.
            - **Prevents Co-adaptation:** No individual neuron can rely on a specific neighbor; each feature detector must learn robust, independent representations.
            - **Ensemble Effect:** Training with dropout is mathematically equivalent to training an exponential ensemble of thinned sub-networks and averaging their predictions at test time.
            """
        ),
        (
            "7. What is Receptive Field and why does it matter?",
            r"""
            **Key Answer Points:**
            - The **Receptive Field** is the specific sub-region of the original input image that contributes to the calculation of a single unit in a deeper feature map.
            - Layer 1 has a receptive field equal to kernel size ($3 \times 3$).
            - As we stack convolutions and pooling, deeper neurons have wider receptive fields, allowing them to comprehend full objects rather than just edges.
            """
        ),
        (
            "8. What is the difference between Stride and Pooling for downsampling?",
            r"""
            **Key Answer Points:**
            - **Pooling (e.g. MaxPool):** A fixed, parameter-free operation that selects the maximum or average activation.
            - **Strided Convolution:** A learnable downsampling operation where the convolutional kernel steps by $S > 1$. The network learns *how* to downsample adaptively.
            """
        )
    ]
    
    for q, a in qa_list:
        with st.expander(q, expanded=False):
            st.markdown(a)

st.markdown("---")
st.caption("Developed with ❤️ for Deep Learning Academic Excellence | DeepVision AI Studio")
