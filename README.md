
# 🎥 Grad-CAM Visualization of Multi-Scale Vision Transformer (MViT V2)

This repository provides a tool to **overlay Grad-CAM heatmaps on video frames** using a **Multi-Scale Vision Transformer V2 (MViT V2)** model and save the visualization as an animated `.gif`.

## 🔍 Overview

This tool visualizes the spatio-temporal attention of MViT by generating Grad-CAM maps over **sliding temporal windows** from a 64-frame input video. The output is an animated `.gif` showing where the model focuses its attention during classification.

## 🧠 Model

- **Backbone**: [`torchvision.models.video.mvit_v2_s`](https://pytorch.org/vision/stable/models/generated/torchvision.models.video.mvit_v2_s.html)
- **Architecture**: Replaces classification head with identity and adds a custom linear classifier over 4 sliding windows.
- **Input**: 64-frame RGB video tensor of shape `(1, 3, 64, H, W)`
- **Temporal Splitting**: 4 overlapping 16-frame clips

## 📦 Features

- Extracts Grad-CAM maps from the final transformer block (`norm1`) for each temporal clip
- Overlays heatmaps on original video frames
- Combines overlays into a smooth `.gif`
- Supports attention visualization for the predicted or specified class

## 🖼️ Output

- Overlaid activation heatmaps as `.gif`
- Optionally, individual annotated frames can be saved as `.png`

## 🚀 Usage

```python
from model import MViTSlidingClassifier
from visualize import generate_gradcam_gif

# Load pretrained model
model = MViTSlidingClassifier()
model.load_state_dict(torch.load("path_to_model.pth"))
model.eval()

# Input video tensor of shape (1, 3, 64, H, W)
generate_gradcam_gif(video_tensor, model, save_path="output.gif")
