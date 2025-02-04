# YOLOv10 Helmet Safety Detection

## Overview

This project aims to enhance safety in construction sites by ensuring that workers wear helmets. It fine-tunes the YOLOv10 object detection model to detect workers and check if they are wearing helmets.

## Dataset

The dataset consists of labeled images of construction workers with and without helmets.

Download the dataset from Google Drive.

No preprocessing is required before training.

## Model Details

- **Base Model**: Pre-trained YOLOv10.
- **Modifications**: The model is fine-tuned on a custom dataset specific to helmet detection.
- **Training Platform**: Google Colab (GPU: T4).

## Training Configuration

Here are the key hyperparameters used in fine-tuning:

- **Batch Size**: Adjust based on available GPU memory.
- **Learning Rate**: Use a learning rate scheduler for optimal convergence.
- **Epochs**: Typically between 50-100.
- **Optimizer**: AdamW or SGD (based on performance).
- **Loss Function**: CIoU Loss for bounding boxes.

## Installation & Setup

First, install all required dependencies using:

```sh
pip install -r requirements.txt
```

## Usage Instructions

### 1. Train the Model

Run the training script in Google Colab:

```sh
!python train.py --data dataset.yaml --cfg yolov10.yaml --weights yolov10_pretrained.pt --epochs 100 --batch-size 16
```

### 2. Inference on New Images

To test the model on new images:

```sh
!python detect.py --weights runs/train/exp/weights/best.pt --source images/
```

### 3. Inference on a Video

```sh
!python detect.py --weights runs/train/exp/weights/best.pt --source video.mp4
```

### 4. Visualize Training Results

```python
from utils.plots import plot_results
plot_results()
```

## Results

- Model performance metrics (mAP, FPS, etc.) will be added soon.
- Sample output images will be included in a future update.

## License

This project is for research and educational purposes only.
