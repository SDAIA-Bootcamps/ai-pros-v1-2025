# %% [markdown]
# # ML System Architecture: Driver Safety Monitoring System
#
# Welcome to this lab on designing and implementing a multi-component ML system!
#
# ## Learning Objectives
#
# In this lab, you will learn how to:
# 1. **Design ML System Architecture**: Understand how to break down a complex problem into modular components
# 2. **Use Pretrained Models**: Leverage existing models (YOLO) for car detection
# 3. **Fine-tune Models**: Transfer learning to adapt YOLO for custom driver detection
# 4. **Integrate Multiple Models**: Chain models together into a cohesive pipeline
# 5. **Apply OCR**: Use Tesseract for license plate text extraction
#
# ## Problem Statement
#
# You are building a **Driver Safety Monitoring System** for traffic enforcement. The system receives images from street cameras and needs to:
#
# 1. **Detect cars** in the image
# 2. **Detect drivers** and identify if they are:
#    - Using their phone while driving
#    - Wearing their seatbelt
# 3. **Detect license plates** on vehicles with violations
# 4. **Read license plate numbers** using OCR
#
# This is a real-world example of a multi-component ML system that requires careful architecture decisions.

# %% [markdown]
# ## System Architecture Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                    Driver Safety Monitoring System                       │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  INPUT: Street Camera Image                                              │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    │
#                                    ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPONENT 1: Car Detection (Pretrained YOLO)                           │
# │  - Input: Full image                                                     │
# │  - Output: Bounding boxes of detected cars                               │
# └─────────────────────────────────────────────────────────────────────────┘
#                                    │
#                     ┌──────────────┴──────────────┐
#                     ▼                              ▼
# ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
# │  COMPONENT 2: Driver Detection  │  │  COMPONENT 3: License Plate     │
# │  (Fine-tuned YOLO)              │  │  Detection (Pretrained Model)   │
# │  - Input: Cropped car region    │  │  - Input: Cropped car region    │
# │  - Output: Driver status        │  │  - Output: Plate bounding box   │
# │    • phone_usage: Yes/No        │  └─────────────────────────────────┘
# │    • seatbelt: Yes/No           │                    │
# └─────────────────────────────────┘                    ▼
#                                      ┌─────────────────────────────────┐
#                                      │  COMPONENT 4: OCR (Tesseract)   │
#                                      │  - Input: Cropped plate image   │
#                                      │  - Output: License plate text   │
#                                      └─────────────────────────────────┘
#                                                        │
#                                                        ▼
# ┌─────────────────────────────────────────────────────────────────────────┐
# │  OUTPUT: Detection Results                                               │
# │  - Car locations                                                         │
# │  - Violation status (phone/seatbelt)                                    │
# │  - License plate numbers                                                 │
# └─────────────────────────────────────────────────────────────────────────┘
# ```

# %% [markdown]
# ## Setup and Installation
#
# First, let's install all the required dependencies.

# %%
# Install required packages
%pip install ultralytics opencv-python pytesseract pillow matplotlib numpy pyyaml tqdm -qqq

# %%
# Import libraries
import os
import sys
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# For YOLO
from ultralytics import YOLO

# For OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not available. OCR functionality will be limited.")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

print("All imports successful!")
print(f"Tesseract available: {TESSERACT_AVAILABLE}")

# %% [markdown]
# ## Part 1: Car Detection with Pretrained YOLO
#
# The first component of our system is **car detection**. We'll use a pretrained YOLOv8 model
# which has been trained on the COCO dataset and can detect 80 different object classes,
# including cars, trucks, and buses.
#
# ### Why Use Pretrained Models?
#
# - **Time-efficient**: Training from scratch requires massive datasets and compute
# - **Performance**: Pretrained models have learned rich feature representations
# - **Transfer Learning**: These features generalize well to related tasks

# %%
class CarDetector:
    """
    Component 1: Car Detection using Pretrained YOLO

    This class uses a pretrained YOLOv8 model to detect vehicles in images.
    The model is trained on COCO dataset which includes car, truck, and bus classes.
    """

    # COCO class IDs for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        5: 'bus',
        7: 'truck'
    }

    def __init__(self, model_size: str = 'n'):
        """
        Initialize the car detector.

        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
                       'n' = nano (fastest, least accurate)
                       'x' = extra large (slowest, most accurate)
        """
        print(f"Loading YOLOv8{model_size} pretrained model...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        print("Car detector initialized!")

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect vehicles in an image.

        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for detection

        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score
                - class_name: Type of vehicle (car, bus, truck)
        """
        # Run inference
        results = self.model(image, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Filter for vehicle classes and confidence threshold
                if class_id in self.VEHICLE_CLASSES and confidence >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_name': self.VEHICLE_CLASSES[class_id]
                    })

        return detections

    def visualize(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on the image."""
        img_copy = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']

            # Draw rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return img_copy

# %% [markdown]
# ### Testing the Car Detector
#
# Let's create a synthetic test image to verify our car detector works.

# %%
# Create a synthetic test image for demonstration
def create_synthetic_car_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a simple synthetic image for testing."""
    # Create a road-like background
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky (blue gradient)
    for y in range(height // 2):
        blue_val = 200 - int(50 * y / (height // 2))
        image[y, :] = [blue_val + 55, blue_val + 30, blue_val]

    # Road (gray)
    image[height // 2:, :] = [80, 80, 80]

    # Road lines (white dashed)
    for x in range(0, width, 40):
        cv2.line(image, (x, height * 3 // 4), (x + 20, height * 3 // 4),
                (255, 255, 255), 2)

    return image

# Initialize the car detector
print("Initializing Car Detector...")
car_detector = CarDetector(model_size='n')

# Create and test with synthetic image
test_image = create_synthetic_car_image()

print("\nRunning car detection on synthetic image...")
detections = car_detector.detect(test_image)
print(f"Found {len(detections)} vehicles")

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
plt.title("Input Image (Synthetic)")
plt.axis('off')

plt.subplot(1, 2, 2)
result_img = car_detector.visualize(test_image, detections)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f"Detection Results ({len(detections)} vehicles)")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nCar Detector Component: Ready!")

# %% [markdown]
# ## Part 2: Driver Detection with Fine-tuned YOLO
#
# The second component detects drivers and classifies their behavior:
# - **Phone usage**: Is the driver using their phone?
# - **Seatbelt**: Is the driver wearing their seatbelt?
#
# For this component, we need to **fine-tune** (transfer learning) a YOLO model on our custom dataset.
#
# ### Why Fine-tuning?
#
# The pretrained YOLO model doesn't know about phone usage or seatbelts. We need to:
# 1. Prepare a custom dataset with these annotations
# 2. Fine-tune the model to recognize these specific classes
#
# ### Dataset Format (COCO-style to YOLO conversion)
#
# Our dataset has 4000 images with 4 classes:
# - `driver_phone`: Driver using phone
# - `driver_no_phone`: Driver not using phone
# - `seatbelt_on`: Seatbelt worn
# - `seatbelt_off`: Seatbelt not worn

# %%
# Define the dataset structure and conversion utilities

def create_mock_coco_dataset(output_dir: str, num_images: int = 100) -> Dict:
    """
    Create a mock COCO-style dataset for demonstration.

    In a real scenario, you would have actual annotated images.
    This function creates synthetic data for testing the pipeline.

    Args:
        output_dir: Directory to save the mock dataset
        num_images: Number of mock images to create

    Returns:
        COCO-format annotations dictionary
    """
    output_path = Path(output_dir)
    images_path = output_path / "images"
    images_path.mkdir(parents=True, exist_ok=True)

    # Define categories for driver detection
    categories = [
        {"id": 0, "name": "driver_phone", "supercategory": "driver"},
        {"id": 1, "name": "driver_no_phone", "supercategory": "driver"},
        {"id": 2, "name": "seatbelt_on", "supercategory": "seatbelt"},
        {"id": 3, "name": "seatbelt_off", "supercategory": "seatbelt"}
    ]

    images = []
    annotations = []
    annotation_id = 0

    for img_id in range(num_images):
        # Create synthetic image (simulating car interior view)
        width, height = 640, 480
        image = np.random.randint(50, 150, (height, width, 3), dtype=np.uint8)

        # Add some structure to make it look like a car interior
        # Dashboard area (darker)
        image[height*2//3:, :] = image[height*2//3:, :] // 2 + 30

        # Save image
        img_filename = f"img_{img_id:04d}.jpg"
        cv2.imwrite(str(images_path / img_filename), image)

        images.append({
            "id": img_id,
            "file_name": img_filename,
            "width": width,
            "height": height
        })

        # Create random annotations
        # Driver region (upper left area typically)
        driver_x = random.randint(50, 200)
        driver_y = random.randint(50, 150)
        driver_w = random.randint(100, 200)
        driver_h = random.randint(150, 250)

        # Phone status (50% using phone)
        phone_class = 0 if random.random() < 0.5 else 1
        annotations.append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": phone_class,
            "bbox": [driver_x, driver_y, driver_w, driver_h],
            "area": driver_w * driver_h,
            "iscrowd": 0
        })
        annotation_id += 1

        # Seatbelt status (70% wearing seatbelt)
        seatbelt_class = 2 if random.random() < 0.7 else 3
        # Seatbelt region overlaps with driver
        sb_x = driver_x + driver_w // 4
        sb_y = driver_y + driver_h // 3
        sb_w = driver_w // 2
        sb_h = driver_h // 2

        annotations.append({
            "id": annotation_id,
            "image_id": img_id,
            "category_id": seatbelt_class,
            "bbox": [sb_x, sb_y, sb_w, sb_h],
            "area": sb_w * sb_h,
            "iscrowd": 0
        })
        annotation_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    # Save COCO annotations
    with open(output_path / "annotations.json", 'w') as f:
        json.dump(coco_data, f, indent=2)

    return coco_data


def convert_coco_to_yolo(coco_data: Dict, images_dir: str, output_dir: str) -> None:
    """
    Convert COCO-format annotations to YOLO format.

    YOLO format: Each image has a corresponding .txt file with:
    <class_id> <x_center> <y_center> <width> <height>

    All coordinates are normalized to [0, 1]

    Args:
        coco_data: COCO format annotations dictionary
        images_dir: Directory containing images
        output_dir: Directory to save YOLO format labels
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create image id to info mapping
    images_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    img_annotations = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Convert each image's annotations
    for img_id, anns in img_annotations.items():
        img_info = images_info[img_id]
        img_width = img_info['width']
        img_height = img_info['height']

        # Create label file
        label_filename = Path(img_info['file_name']).stem + '.txt'

        yolo_annotations = []
        for ann in anns:
            # COCO bbox is [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to YOLO format (center coordinates, normalized)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            width_norm = w / img_width
            height_norm = h / img_height

            # Clip to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width_norm = max(0, min(1, width_norm))
            height_norm = max(0, min(1, height_norm))

            class_id = ann['category_id']
            yolo_annotations.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
            )

        # Write label file
        with open(output_path / label_filename, 'w') as f:
            f.write('\n'.join(yolo_annotations))

    print(f"Converted {len(img_annotations)} images to YOLO format")

# %% [markdown]
# ### Creating the Dataset Structure for YOLO Training
#
# YOLO requires a specific directory structure:
# ```
# dataset/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── val/
# │   ├── images/
# │   └── labels/
# └── data.yaml
# ```

# %%
def prepare_yolo_dataset(base_dir: str, num_train: int = 80, num_val: int = 20) -> str:
    """
    Prepare a complete YOLO dataset structure.

    Args:
        base_dir: Base directory for the dataset
        num_train: Number of training images
        num_val: Number of validation images

    Returns:
        Path to the data.yaml configuration file
    """
    base_path = Path(base_dir)

    # Create directories
    train_images = base_path / "train" / "images"
    train_labels = base_path / "train" / "labels"
    val_images = base_path / "val" / "images"
    val_labels = base_path / "val" / "labels"

    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)

    # Create training data
    print("Creating training dataset...")
    train_coco = create_mock_coco_dataset(str(base_path / "temp_train"), num_train)
    convert_coco_to_yolo(train_coco, str(base_path / "temp_train" / "images"), str(train_labels))

    # Move images
    for img_file in (base_path / "temp_train" / "images").glob("*.jpg"):
        shutil.move(str(img_file), str(train_images / img_file.name))

    # Create validation data
    print("Creating validation dataset...")
    val_coco = create_mock_coco_dataset(str(base_path / "temp_val"), num_val)
    convert_coco_to_yolo(val_coco, str(base_path / "temp_val" / "images"), str(val_labels))

    # Move images
    for img_file in (base_path / "temp_val" / "images").glob("*.jpg"):
        shutil.move(str(img_file), str(val_images / img_file.name))

    # Cleanup temp directories
    shutil.rmtree(base_path / "temp_train", ignore_errors=True)
    shutil.rmtree(base_path / "temp_val", ignore_errors=True)

    # Create data.yaml configuration
    data_yaml = {
        'path': str(base_path.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'driver_phone',
            1: 'driver_no_phone',
            2: 'seatbelt_on',
            3: 'seatbelt_off'
        }
    }

    yaml_path = base_path / "data.yaml"
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(data_yaml, f, default_flow_style=False)

    print(f"\nDataset created at: {base_path}")
    print(f"Training images: {num_train}")
    print(f"Validation images: {num_val}")
    print(f"Classes: {list(data_yaml['names'].values())}")

    return str(yaml_path)

# Create the dataset
DATASET_DIR = "./driver_detection_dataset"
yaml_path = prepare_yolo_dataset(DATASET_DIR, num_train=80, num_val=20)
print(f"\nDataset configuration: {yaml_path}")

# %% [markdown]
# ### Fine-tuning YOLO for Driver Detection
#
# Now we'll fine-tune a YOLOv8 model on our custom dataset. This is **transfer learning**:
#
# 1. Start with pretrained weights (trained on COCO)
# 2. Replace the output layer for our 4 classes
# 3. Train on our custom dataset
#
# The pretrained weights give us:
# - Feature extraction capabilities (edges, shapes, objects)
# - Faster convergence
# - Better performance with limited data

# %%
class DriverDetector:
    """
    Component 2: Driver Detection using Fine-tuned YOLO

    This class handles training and inference for driver behavior detection.
    Detects: phone usage and seatbelt status.
    """

    CLASS_NAMES = {
        0: 'driver_phone',
        1: 'driver_no_phone',
        2: 'seatbelt_on',
        3: 'seatbelt_off'
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the driver detector.

        Args:
            model_path: Path to fine-tuned model weights, or None to use base model
        """
        if model_path and Path(model_path).exists():
            print(f"Loading fine-tuned model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Initializing base YOLOv8n model for fine-tuning")
            self.model = YOLO('yolov8n.pt')

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640,
              batch_size: int = 16, save_dir: str = './runs/driver_detection') -> str:
        """
        Fine-tune the model on custom dataset.

        Args:
            data_yaml: Path to data.yaml configuration
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size for training
            save_dir: Directory to save training results

        Returns:
            Path to the best trained model weights
        """
        print(f"\n{'='*60}")
        print("Starting Fine-tuning for Driver Detection")
        print(f"{'='*60}")
        print(f"Dataset config: {data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Image size: {imgsz}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")

        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=save_dir,
            name='train',
            exist_ok=True,
            verbose=True,
            device='cpu',  # Use CPU for compatibility
            workers=0,     # Avoid multiprocessing issues
            amp=False      # Disable mixed precision on CPU
        )

        # Get the best model path
        best_model_path = Path(save_dir) / 'train' / 'weights' / 'best.pt'

        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best model saved at: {best_model_path}")
        print(f"{'='*60}\n")

        # Load the best model
        self.model = YOLO(str(best_model_path))

        return str(best_model_path)

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.3) -> Dict:
        """
        Detect driver status in an image.

        Args:
            image: Input image (cropped car interior region)
            confidence_threshold: Minimum confidence for detection

        Returns:
            Dictionary with detection results:
                - phone_usage: True if using phone, False otherwise
                - seatbelt: True if wearing seatbelt, False otherwise
                - detections: List of all detections with bboxes
        """
        results = self.model(image, verbose=False)

        phone_usage = None
        seatbelt = None
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence >= confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_name': self.CLASS_NAMES[class_id],
                        'class_id': class_id
                    }
                    detections.append(detection)

                    # Determine phone usage
                    if class_id == 0:  # driver_phone
                        if phone_usage is None or confidence > phone_usage[1]:
                            phone_usage = (True, confidence)
                    elif class_id == 1:  # driver_no_phone
                        if phone_usage is None or confidence > phone_usage[1]:
                            phone_usage = (False, confidence)

                    # Determine seatbelt status
                    if class_id == 2:  # seatbelt_on
                        if seatbelt is None or confidence > seatbelt[1]:
                            seatbelt = (True, confidence)
                    elif class_id == 3:  # seatbelt_off
                        if seatbelt is None or confidence > seatbelt[1]:
                            seatbelt = (False, confidence)

        return {
            'phone_usage': phone_usage[0] if phone_usage else None,
            'seatbelt': seatbelt[0] if seatbelt else None,
            'detections': detections
        }

    def visualize(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """Visualize detection results on the image."""
        img_copy = image.copy()

        colors = {
            'driver_phone': (0, 0, 255),      # Red
            'driver_no_phone': (0, 255, 0),   # Green
            'seatbelt_on': (0, 255, 0),       # Green
            'seatbelt_off': (0, 0, 255)       # Red
        }

        for det in result['detections']:
            x1, y1, x2, y2 = det['bbox']
            color = colors.get(det['class_name'], (255, 255, 0))

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            label = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Add status text
        status_text = []
        if result['phone_usage'] is not None:
            phone_status = "PHONE IN USE" if result['phone_usage'] else "No phone"
            status_text.append(phone_status)
        if result['seatbelt'] is not None:
            belt_status = "Seatbelt ON" if result['seatbelt'] else "NO SEATBELT"
            status_text.append(belt_status)

        for i, text in enumerate(status_text):
            color = (0, 0, 255) if "PHONE IN USE" in text or "NO SEATBELT" in text else (0, 255, 0)
            cv2.putText(img_copy, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img_copy

# %% [markdown]
# ### Train the Driver Detection Model
#
# Now let's train our driver detection model. We'll use a small number of epochs
# for demonstration purposes (in production, you'd use 50-100+ epochs).

# %%
# Initialize and train the driver detector
print("Initializing Driver Detector for training...")
driver_detector = DriverDetector()

# Train with minimal epochs for demonstration (use more epochs in production)
# NOTE: Training will use CPU - in production, use GPU for faster training
print("\nStarting training (this may take a few minutes on CPU)...")
best_model_path = driver_detector.train(
    data_yaml=yaml_path,
    epochs=1,  # Use 1 epoch for quick testing, use 50+ in production
    imgsz=320, # Smaller image size for faster training
    batch_size=8,
    save_dir='./runs/driver_detection'
)

print(f"\nTrained model saved at: {best_model_path}")

# %% [markdown]
# ### Test the Driver Detector
#
# Let's test our fine-tuned model on a sample image.

# %%
# Create a test image and run detection
test_car_interior = np.random.randint(50, 150, (480, 640, 3), dtype=np.uint8)
test_car_interior[320:, :] = test_car_interior[320:, :] // 2 + 30  # Dashboard area

print("Running driver detection on test image...")
result = driver_detector.detect(test_car_interior)

print(f"\nDetection Results:")
print(f"  Phone usage: {result['phone_usage']}")
print(f"  Seatbelt: {result['seatbelt']}")
print(f"  Number of detections: {len(result['detections'])}")

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_car_interior, cv2.COLOR_BGR2RGB))
plt.title("Input: Car Interior")
plt.axis('off')

plt.subplot(1, 2, 2)
result_img = driver_detector.visualize(test_car_interior, result)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title("Detection Results")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nDriver Detector Component: Ready!")

# %% [markdown]
# ## Part 3: License Plate Detection
#
# The third component detects license plates on vehicles. We'll use a pretrained model
# that's specifically trained for license plate detection.
#
# For this lab, we'll use a simple approach with the standard YOLO model and look for
# rectangular regions, or you could use a specialized license plate detection model.

# %%
class LicensePlateDetector:
    """
    Component 3: License Plate Detection

    This class detects license plates in vehicle images.
    Uses YOLO or can be extended to use specialized plate detection models.
    """

    def __init__(self):
        """Initialize the license plate detector."""
        # For simplicity, we'll use a pretrained YOLO model
        # In production, you'd use a specialized plate detection model
        print("Initializing License Plate Detector...")
        self.model = YOLO('yolov8n.pt')
        print("License Plate Detector initialized!")

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.3) -> List[Dict]:
        """
        Detect license plates in an image.

        This is a simplified implementation. In production, you would use
        a model specifically trained for license plate detection.

        Args:
            image: Input image (cropped vehicle region)
            confidence_threshold: Minimum confidence threshold

        Returns:
            List of detected plates with bounding boxes
        """
        # For demo purposes, we'll simulate plate detection
        # In a real system, you'd use a specialized model

        height, width = image.shape[:2]

        # Simulate detecting a plate in the lower portion of the vehicle
        # Real systems use trained models for accurate detection
        plates = []

        # Heuristic: Look for rectangular regions in lower third of image
        # This is a placeholder - real systems use trained detectors
        plate_region = {
            'bbox': [
                int(width * 0.3),  # x1
                int(height * 0.7),  # y1
                int(width * 0.7),  # x2
                int(height * 0.9)   # y2
            ],
            'confidence': 0.85,
            'class_name': 'license_plate'
        }
        plates.append(plate_region)

        return plates

    def crop_plate(self, image: np.ndarray, bbox: List[int], padding: int = 5) -> np.ndarray:
        """
        Crop the license plate region from the image.

        Args:
            image: Full image
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Extra padding around the plate

        Returns:
            Cropped plate image
        """
        x1, y1, x2, y2 = bbox
        height, width = image.shape[:2]

        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)

        return image[y1:y2, x1:x2].copy()

    def visualize(self, image: np.ndarray, plates: List[Dict]) -> np.ndarray:
        """Draw plate detections on the image."""
        img_copy = image.copy()

        for plate in plates:
            x1, y1, x2, y2 = plate['bbox']
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Plate: {plate['confidence']:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return img_copy

# %%
# Test the license plate detector
print("Testing License Plate Detector...")
plate_detector = LicensePlateDetector()

# Create a test image
test_vehicle = np.random.randint(100, 180, (300, 400, 3), dtype=np.uint8)

# Add a simulated plate region (white rectangle)
test_vehicle[210:270, 120:280] = [255, 255, 255]

# Detect plates
plates = plate_detector.detect(test_vehicle)
print(f"Detected {len(plates)} license plate(s)")

# Visualize
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_vehicle, cv2.COLOR_BGR2RGB))
plt.title("Input: Vehicle Region")
plt.axis('off')

plt.subplot(1, 2, 2)
result_img = plate_detector.visualize(test_vehicle, plates)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f"Detected Plates: {len(plates)}")
plt.axis('off')

plt.tight_layout()
plt.show()

print("\nLicense Plate Detector Component: Ready!")

# %% [markdown]
# ## Part 4: OCR for License Plate Reading
#
# The fourth component uses OCR (Optical Character Recognition) to read the text
# from detected license plates. We'll use Tesseract, an open-source OCR engine.
#
# ### OCR Pipeline:
# 1. Receive cropped plate image
# 2. Preprocess (grayscale, threshold, denoise)
# 3. Run Tesseract OCR
# 4. Post-process text (clean up)

# %%
class LicensePlateOCR:
    """
    Component 4: License Plate OCR using Tesseract

    This class reads text from license plate images using OCR.
    """

    def __init__(self):
        """Initialize the OCR engine."""
        print("Initializing License Plate OCR...")
        self.tesseract_available = TESSERACT_AVAILABLE

        if self.tesseract_available:
            # Configure Tesseract for license plates
            # --psm 7: Treat image as a single text line
            # --oem 3: Default OCR engine mode
            self.config = '--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            print("Tesseract OCR initialized!")
        else:
            print("Warning: Tesseract not available. Using mock OCR for demonstration.")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess plate image for better OCR results.

        Steps:
        1. Convert to grayscale
        2. Resize if too small
        3. Apply thresholding
        4. Denoise

        Args:
            image: Input plate image (BGR)

        Returns:
            Preprocessed image ready for OCR
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small (OCR works better on larger images)
        height, width = gray.shape
        if height < 50:
            scale = 50 / height
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def read_plate(self, image: np.ndarray) -> Dict:
        """
        Read text from a license plate image.

        Args:
            image: Cropped plate image

        Returns:
            Dictionary with:
                - text: Detected plate number
                - confidence: OCR confidence score
                - preprocessed_image: Image after preprocessing
        """
        # Preprocess the image
        preprocessed = self.preprocess(image)

        if self.tesseract_available:
            try:
                # Run Tesseract OCR
                text = pytesseract.image_to_string(preprocessed, config=self.config)

                # Clean up the text
                text = ''.join(c for c in text if c.isalnum()).upper()

                # Get confidence (using image_to_data for detailed output)
                data = pytesseract.image_to_data(preprocessed, output_type=pytesseract.Output.DICT)
                confidences = [int(c) for c in data['conf'] if int(c) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            except Exception as e:
                print(f"OCR Error: {e}")
                text = ""
                avg_confidence = 0
        else:
            # Mock OCR for demonstration when Tesseract isn't available
            # Generate a random plate number
            import string
            letters = ''.join(random.choices(string.ascii_uppercase, k=3))
            numbers = ''.join(random.choices(string.digits, k=4))
            text = f"{letters}{numbers}"
            avg_confidence = 75.0
            print(f"(Mock OCR) Generated plate: {text}")

        return {
            'text': text,
            'confidence': avg_confidence,
            'preprocessed_image': preprocessed
        }

    def visualize_preprocessing(self, original: np.ndarray, preprocessed: np.ndarray) -> None:
        """Visualize the preprocessing steps."""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        if len(original.shape) == 3:
            plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original, cmap='gray')
        plt.title("Original Plate Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(preprocessed, cmap='gray')
        plt.title("Preprocessed for OCR")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # Show thresholding effect
        plt.imshow(preprocessed > 127, cmap='gray')
        plt.title("Binary Threshold")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

# %%
# Test the OCR component
print("Testing License Plate OCR...")
ocr = LicensePlateOCR()

# Create a synthetic plate image
plate_img = np.ones((60, 200, 3), dtype=np.uint8) * 255

# Add text to the plate (simulated)
cv2.putText(plate_img, "ABC1234", (20, 45),
           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

# Read the plate
result = ocr.read_plate(plate_img)

print(f"\nOCR Results:")
print(f"  Detected Text: {result['text']}")
print(f"  Confidence: {result['confidence']:.1f}%")

# Visualize preprocessing
ocr.visualize_preprocessing(plate_img, result['preprocessed_image'])

print("\nLicense Plate OCR Component: Ready!")

# %% [markdown]
# ## Part 5: Complete Pipeline Integration
#
# Now let's integrate all four components into a unified pipeline that processes
# images from street cameras and produces complete detection results.

# %%
class DriverSafetySystem:
    """
    Complete Driver Safety Monitoring System

    This class integrates all four components:
    1. Car Detection (pretrained YOLO)
    2. Driver Detection (fine-tuned YOLO)
    3. License Plate Detection
    4. License Plate OCR

    The system processes street camera images and identifies traffic violations.
    """

    def __init__(self, driver_model_path: Optional[str] = None):
        """
        Initialize the complete system.

        Args:
            driver_model_path: Path to fine-tuned driver detection model
        """
        print("\n" + "="*60)
        print("Initializing Driver Safety Monitoring System")
        print("="*60 + "\n")

        # Initialize all components
        print("Loading Component 1: Car Detector...")
        self.car_detector = CarDetector(model_size='n')

        print("\nLoading Component 2: Driver Detector...")
        self.driver_detector = DriverDetector(model_path=driver_model_path)

        print("\nLoading Component 3: License Plate Detector...")
        self.plate_detector = LicensePlateDetector()

        print("\nLoading Component 4: License Plate OCR...")
        self.ocr = LicensePlateOCR()

        print("\n" + "="*60)
        print("System Initialization Complete!")
        print("="*60 + "\n")

    def process_image(self, image: np.ndarray,
                     car_confidence: float = 0.5,
                     driver_confidence: float = 0.3) -> Dict:
        """
        Process a street camera image through the complete pipeline.

        Args:
            image: Input image from street camera (BGR format)
            car_confidence: Confidence threshold for car detection
            driver_confidence: Confidence threshold for driver detection

        Returns:
            Dictionary containing all detection results
        """
        results = {
            'cars': [],
            'violations': [],
            'processing_steps': []
        }

        # Step 1: Detect cars
        results['processing_steps'].append("Step 1: Detecting vehicles...")
        car_detections = self.car_detector.detect(image, car_confidence)
        results['processing_steps'].append(f"  Found {len(car_detections)} vehicles")

        # Process each detected car
        for i, car in enumerate(car_detections):
            car_result = {
                'car_id': i,
                'car_bbox': car['bbox'],
                'car_confidence': car['confidence'],
                'car_type': car['class_name'],
                'driver_status': None,
                'plate_info': None
            }

            # Crop the car region
            x1, y1, x2, y2 = car['bbox']
            car_crop = image[y1:y2, x1:x2]

            if car_crop.size == 0:
                continue

            # Step 2: Detect driver status
            results['processing_steps'].append(f"Step 2: Analyzing driver in car {i}...")
            driver_result = self.driver_detector.detect(car_crop, driver_confidence)
            car_result['driver_status'] = {
                'phone_usage': driver_result['phone_usage'],
                'seatbelt': driver_result['seatbelt'],
                'detections': driver_result['detections']
            }

            # Check for violations
            is_violation = False
            violation_types = []

            if driver_result['phone_usage'] == True:
                is_violation = True
                violation_types.append('phone_usage')

            if driver_result['seatbelt'] == False:
                is_violation = True
                violation_types.append('no_seatbelt')

            # Step 3: If violation detected, get license plate
            if is_violation:
                results['processing_steps'].append(f"  Violation detected! Getting license plate...")

                # Detect license plate
                plates = self.plate_detector.detect(car_crop)

                if plates:
                    plate = plates[0]  # Take the first detected plate
                    plate_crop = self.plate_detector.crop_plate(car_crop, plate['bbox'])

                    # Step 4: Read license plate
                    results['processing_steps'].append(f"Step 4: Reading license plate...")
                    ocr_result = self.ocr.read_plate(plate_crop)

                    car_result['plate_info'] = {
                        'plate_bbox': plate['bbox'],
                        'plate_text': ocr_result['text'],
                        'ocr_confidence': ocr_result['confidence']
                    }

                    results['processing_steps'].append(f"  Plate number: {ocr_result['text']}")

                # Record violation
                results['violations'].append({
                    'car_id': i,
                    'violation_types': violation_types,
                    'plate_number': car_result['plate_info']['plate_text'] if car_result['plate_info'] else 'Unknown'
                })

            results['cars'].append(car_result)

        return results

    def visualize_results(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """
        Create a visualization of all detection results.

        Args:
            image: Original input image
            results: Processing results from process_image()

        Returns:
            Annotated image with all detections
        """
        img_copy = image.copy()

        for car in results['cars']:
            x1, y1, x2, y2 = car['car_bbox']

            # Determine color based on violation status
            is_violation = any(v['car_id'] == car['car_id'] for v in results['violations'])
            color = (0, 0, 255) if is_violation else (0, 255, 0)

            # Draw car bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)

            # Add label
            label_parts = [f"Car {car['car_id']}"]

            if car['driver_status']:
                if car['driver_status']['phone_usage']:
                    label_parts.append("PHONE!")
                if car['driver_status']['seatbelt'] == False:
                    label_parts.append("NO BELT!")

            if car['plate_info']:
                label_parts.append(f"Plate: {car['plate_info']['plate_text']}")

            label = " | ".join(label_parts)

            # Draw label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Add violation summary
        if results['violations']:
            summary = f"VIOLATIONS DETECTED: {len(results['violations'])}"
            cv2.putText(img_copy, summary, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(img_copy, "No violations", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img_copy

    def generate_report(self, results: Dict) -> str:
        """Generate a text report of the detection results."""
        report = []
        report.append("=" * 60)
        report.append("DRIVER SAFETY MONITORING REPORT")
        report.append("=" * 60)
        report.append("")
        report.append("PROCESSING LOG:")
        for step in results['processing_steps']:
            report.append(f"  {step}")
        report.append("")
        report.append("-" * 60)
        report.append("VEHICLE SUMMARY:")
        report.append(f"  Total vehicles detected: {len(results['cars'])}")
        report.append(f"  Violations found: {len(results['violations'])}")
        report.append("")

        if results['violations']:
            report.append("-" * 60)
            report.append("VIOLATION DETAILS:")
            for v in results['violations']:
                report.append(f"  Car ID: {v['car_id']}")
                report.append(f"    Violations: {', '.join(v['violation_types'])}")
                report.append(f"    License Plate: {v['plate_number']}")
                report.append("")

        report.append("=" * 60)

        return "\n".join(report)

# %% [markdown]
# ### Running the Complete Pipeline
#
# Let's test our complete system with a synthetic street scene.

# %%
# Create a synthetic street scene for testing
def create_test_street_scene(width: int = 800, height: int = 600) -> np.ndarray:
    """Create a synthetic street scene for testing."""
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky
    for y in range(height // 2):
        blue_val = 200 - int(50 * y / (height // 2))
        image[y, :] = [blue_val + 55, blue_val + 30, blue_val]

    # Road
    image[height // 2:, :] = [80, 80, 80]

    # Road markings
    for x in range(0, width, 50):
        cv2.line(image, (x, height * 3 // 4), (x + 30, height * 3 // 4),
                (255, 255, 255), 3)

    # Add some "buildings" (rectangles) in background
    for i in range(3):
        x = i * 250 + 50
        cv2.rectangle(image, (x, 100), (x + 150, height // 2 - 10),
                     (150, 150, 150), -1)

    return image

# Initialize the complete system
print("Initializing the complete Driver Safety System...")

# Use the trained model if available
trained_model_path = './runs/driver_detection/train/weights/best.pt'
if not Path(trained_model_path).exists():
    trained_model_path = None

system = DriverSafetySystem(driver_model_path=trained_model_path)

# Create test scene
test_scene = create_test_street_scene()

# Process the image
print("\nProcessing street camera image...")
results = system.process_image(test_scene)

# Generate and print report
report = system.generate_report(results)
print(report)

# Visualize results
plt.figure(figsize=(14, 8))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_scene, cv2.COLOR_BGR2RGB))
plt.title("Input: Street Camera Image")
plt.axis('off')

plt.subplot(1, 2, 2)
result_img = system.visualize_results(test_scene, results)
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title("Output: Detection Results")
plt.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Part 6: System Architecture Best Practices
#
# ### Key Takeaways
#
# 1. **Modular Design**: Each component has a single responsibility
#    - Car Detector: Find vehicles
#    - Driver Detector: Analyze driver behavior
#    - Plate Detector: Locate license plates
#    - OCR: Read plate text
#
# 2. **Use Pretrained Models When Possible**:
#    - Car detection uses COCO-pretrained YOLO
#    - Significant time savings vs training from scratch
#
# 3. **Fine-tune for Custom Tasks**:
#    - Driver detection required custom training
#    - Transfer learning preserves useful features
#
# 4. **Sequential Processing**:
#    - Components form a pipeline
#    - Output of one feeds into the next
#
# 5. **Error Handling**:
#    - Each component handles missing/invalid input
#    - System continues even if one detection fails

# %% [markdown]
# ### Architecture Decisions and Trade-offs
#
# | Decision | Trade-off |
# |----------|-----------|
# | YOLOv8n (nano) | Faster inference, lower accuracy |
# | YOLOv8x (extra) | Higher accuracy, slower inference |
# | Single model per task | Modular, easier to maintain |
# | Combined multi-task model | Faster, harder to update |
# | CPU inference | Portable, slower |
# | GPU inference | Faster, requires hardware |

# %% [markdown]
# ## Exercises
#
# 1. **Model Size Experiment**: Try different YOLO model sizes (n, s, m, l, x) and compare speed vs accuracy
#
# 2. **Data Augmentation**: Add augmentations to the training pipeline to improve model robustness
#
# 3. **Real Images**: Test the system with real street camera images
#
# 4. **Performance Optimization**: Profile the pipeline and identify bottlenecks
#
# 5. **Additional Violations**: Extend the system to detect other violations (running red lights, speeding indicators)

# %% [markdown]
# ## Summary
#
# In this lab, you learned how to:
#
# 1. **Design a multi-component ML system** for driver safety monitoring
# 2. **Use pretrained YOLO models** for vehicle detection
# 3. **Fine-tune models** using transfer learning for custom tasks
# 4. **Integrate OCR** for license plate reading
# 5. **Build a complete pipeline** that chains multiple models together
#
# This architecture pattern applies to many real-world ML systems:
# - Autonomous vehicles (perception pipeline)
# - Medical imaging (detection + classification + segmentation)
# - Document processing (detection + OCR + NLP)
# - Retail analytics (person detection + tracking + behavior analysis)
#
# The key principles are:
# - **Modularity**: Each component does one thing well
# - **Reusability**: Pretrained models save development time
# - **Flexibility**: Components can be updated independently
# - **Scalability**: Pipeline can be distributed across machines

# %%
print("\n" + "="*60)
print("Lab Complete!")
print("="*60)
print("""
You have successfully built a complete ML system for driver safety monitoring!

Key Components Implemented:
1. Car Detection (Pretrained YOLO)
2. Driver Detection (Fine-tuned YOLO)
3. License Plate Detection
4. License Plate OCR (Tesseract)

Next Steps:
- Experiment with real images
- Try different model sizes
- Add more violation types
- Deploy to production!
""")
