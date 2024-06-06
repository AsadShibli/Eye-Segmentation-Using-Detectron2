# Custom Instance Segmentation with Detectron2

This repository contains a project for training a custom instance segmentation model using Detectron2 on your own dataset. The images were collected and annotated using [makesense.ai](https://www.makesense.ai/).

It's worth noting that a limited number of images were employed in this training process. Incorporating a larger dataset could potentially enhance the model's performance. Therefore, expanding the dataset with additional images may lead to improved segmentation accuracy and robustness.
## demo

![git](https://github.com/AsadShibli/Eye-Segmentation-Using-Detectron2/assets/119102237/577f6eee-bee5-4114-b18a-f713a9f3aff2)


## Dataset Overview
- **Training Set:** 14 images
- **Validation Set:** 9 images
- **Test Set:** 3 images

## Data Augmentation
- On-the-fly data augmentation was performed during training to enhance the robustness of the model.

## Project Structure
- `data/`
  - `train/`: Directory containing training images and corresponding annotation files in JSON format.
  - `val/`: Directory containing validation images and corresponding annotation files in JSON format.
  - `test/`: Directory containing test images.
## Getting Started
### 1. Data Preparation
- Collect and annotate images using [makesense.ai](https://www.makesense.ai/).
- Split the dataset into training, validation, and test sets, ensuring that annotation files in JSON format are placed in their respective directories.


## Installation

To install the necessary dependencies, follow the steps below:

### Install Detectron2

```sh
# Install necessary dependencies
python -m pip install pyyaml==5.1

# Clone the Detectron2 repository
git clone https://github.com/facebookresearch/detectron2

# Install Detectron2
cd detectron2
python -m pip install -e .

# Verify installation
python -c "import detectron2; print(detectron2.__version__)"
```
## Setup

Ensure you have the correct versions of CUDA and PyTorch installed. This project was tested with CUDA 12.2 and PyTorch 2.3.
```python
import torch
import detectron2

# Check versions
torch_version = ".".join(torch.__version__.split(".")[:2])
cuda_version = torch.__version__.split("+")[-1]

print("torch: ", torch_version, "; cuda: ", cuda_version)
print("detectron2:", detectron2.__version__)
```
## Usage
### Basic Setup
1.Setup Detectron2 Logger
```python
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
```
2.Import Libraries
```python
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```
## Training the Model
#### 1.Prepare Your Dataset 
Annotate your images using makesense.ai and export them in the appropriate format for Detectron2.
#### 2.Register the Dataset
```python
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "path/to/annotations.json", "path/to/images")
```
#### 3.Configure the Model
```python
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only has one class (eye)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
```
#### 4.Train the Model
```python
from detectron2.engine import DefaultTrainer

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```
#### 5.Make Predictions
```python
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

# Test on a single image
im = cv2.imread("path/to/test/image.jpg")
outputs = predictor(im)
```
## Acknowledgements

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [makesense.ai](https://www.makesense.ai/)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
