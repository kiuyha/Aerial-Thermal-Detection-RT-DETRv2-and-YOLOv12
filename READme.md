# Models for Human Detection in Thermal UAV Imagery

You can try out using this demo [here](https://huggingface.co/spaces/Kiuyha/Aerial-Thermal-SAR-Detection-Demo).

This repository provides two object detection models, **RT-DETRv2** and **YOLOv12**, for identifying people in thermal images captured by Unmanned Aerial Vehicles (UAVs). These models are optimized for Search and Rescue (SAR) applications where accuracy and speed are critical.

The models were trained on a custom-processed version of the HIT-UAV dataset.

## Dataset Information 📚

The models were trained on a dataset derived from the **HIT-UAV infrared thermal dataset**. You can check the preprocessed dataset [here](https://huggingface.co/datasets/Kiuyha/hit-uav-thermal-human-detection).

  * **Source Dataset**: Contained over 2,800 thermal images from a UAV perspective.
  * **Preprocessing & Augmentation**: The raw data was standardized through a rigorous preprocessing pipeline:
      * **Normalization**: Thermal pixel values were normalized from a 0-255 range to a 0-1 range.
      * **Resolution**: All images were resized to 640x640 pixels.
      * **Augmentation**: To create a more robust model, the following augmentations were applied: rotation (±90°), horizontal/vertical flipping, shear (±15°), brightness (±15%), saturation (±25%), hue (±15%), exposure (±15%), and noise injection (≥ 2% of pixels).
  * **Annotation**: Annotations were created manually using Roboflow. A single class, `person`, was used to label humans in the images.
  * **Final Dataset Split**: After processing, the final dataset consisted of **4,874 annotated images**, split as follows:
      * **Training**: 4,016 images (82%)
      * **Validation**: 570 images (12%)
      * **Testing**: 288 images (6%)

-----

## Model Performance 📊
Example Results:
<img src="https://raw.githubusercontent.com/kiuyha/Aerial-Thermal-Detection-RT-DETRv2-and-YOLOv12/refs/heads/main/Example%20result.png">

Both models were trained in the same environment (**Google Colab with a Tesla T4 GPU**). Below is a summary of their performance on the test set.

| Model       | mAP50-95 (%)     | Latency (ms)     | FLOPs (G)        | Recommendation                               |
| :---------- | :--------------- | :--------------- | :--------------- | :------------------------------------------- |
| **RT-DETRv2** | **52.37** | **35.07** | 103.43            | **Best for performance, accuracy, and speed.** |
| YOLOv12     | 51.88            | 52.24            | **88.55** | Best for resource-constrained environments.  |

Comparison Graph:
<img src="https://raw.githubusercontent.com/kiuyha/Aerial-Thermal-Detection-RT-DETRv2-and-YOLOv12/refs/heads/main/Graph%20comparison.png">

For SAR missions where every detection and every second counts, **RT-DETRv2 is the highly recommended model** due to its superior accuracy and lower real-world latency.

-----

## Getting Started 🚀

You can download the trained model weights (`.pt` files) from this repository.

The pre-trained models are available for download from the Hugging Face Hub. This keeps the repository lightweight and easy to clone.

### Download Models

*  **[RT-DETRv2 (`best.pt`)](https://huggingface.co/Kiuyha/rtdetrv2-human-detection-thermal-uav/resolve/main/best.pt)**
* **[YOLOv12 (`best.pt`)](https://huggingface.co/Kiuyha/yolov12-human-detection-thermal-uav/resolve/main/best.pt)**

### Prerequisites

  * Python 3.8+
  * ultralytics >= 8.3.167
  * cv2 >= 4.6.0

### Usage Example

Here is a simple example of how to load a model and run inference on an image using 

-----

**RT-DETRv2**

```python
from ultralytics import RTDETR

from ultralytics import RTDETR
import cv2
from google.colab.patches import cv2_imshow

modelRTDETR = RTDETR('/RT-DETRv2/weights/best.pt')

# Load an image
image_path = '/path/to/image.jpg'
image = cv2.imread(image_path)

# Perform inference
results = modelRTDETR(image)

# Visualize the results
annotated_frame = results[0].plot()

# Display or save the output
cv2_imshow(annotated_frame)
```

**YOLOv12**

```python
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow

modelRTDETR = YOLO('/YOLOv12/weights/best.pt')

# Load an image
image_path = '/path/to/image.jpg'
image = cv2.imread(image_path)

# Perform inference
results = modelRTDETR(image)

# Visualize the results
annotated_frame = results[0].plot()

# Display or save the output
cv2_imshow(annotated_frame)
```