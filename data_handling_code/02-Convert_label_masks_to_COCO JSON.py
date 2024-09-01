# https://youtu.be/R-N-YXzvOmY
"""
With this code, we will convert our labeled mask image annotations to coco json 
format so they can be used in training Detectron2 (or Mask R-CNN). 

COCO JSON refers to the JSON format used by the Common Objects in Context (COCO) dataset, which is a large-scale object detection, segmentation, and captioning dataset.
This format is used to structure and store annotations for images in a way that is compatible with the COCO dataset's data schema.

Detectron2 is a powerful and flexible library for object detection and segmentation developed by Facebook AI Research (FAIR). 
It is the successor to Detectron and provides state-of-the-art models and tools for computer vision tasks. 
To train a custom model in Detectron2, you need to prepare your dataset in COCO format or other supported formats and configure Detectron2 to use it.

Requirements for Detectron2:
Hardware: 8 NVIDIA V100s with NVLink.
Software: Python 3.7, CUDA 10.1, cuDNN 7.6.5, PyTorch 1.5, TensorFlow 1.15.0rc2, Keras 2.2.5, MxNet 1.6.0b20190820.
Model: an end-to-end R-50-FPN Mask-RCNN model, using the same hyperparameter as the Detectron baseline config (it does not have scale augmentation).

"""


import os
import cv2
import numpy as np
import json
import shutil
from sklearn.model_selection import train_test_split

def get_image_mask_pairs(data_dir):
    """
    Preprocessing to fetch path of my image and mask pairs individually.
    """
    image_paths = []
    mask_paths = []
    
    for root, _, files in os.walk(data_dir):
        if 'tissue images' in root:
            for file in files:
                if file.endswith('.png'):
                    image_paths.append(os.path.join(root, file))
                    mask_paths.append(os.path.join(root.replace('tissue images', 'label masks modify'), file.replace('.png', '.tif')))
    
    return image_paths, mask_paths

def mask_to_polygons(mask, epsilon=1.0):
    """
    mask_to_polygons(mask,epsilon) converts mask to polygon. 
        step 1: 
            Find contours from binary mask. 
            Contours are curves that join all the continuous points along a boundary having the same color or intensity.
        Step 2:
            Convert Contours to Polygons. Polygon can be a n-sided figure. In our case, we consider a rectangle as the least acceptable polygon.

    Importance of Polygon Conversion

    Segmentation Representation:
        Mask to Polygons: In segmentation tasks, especially with the COCO dataset format, object masks are often represented as polygons rather than binary masks. Converting masks to polygons allows you to represent the shapes of objects in a more compact and standardized format.
        Efficient Representation: Polygons can be more memory-efficient than high-resolution binary masks, particularly for complex objects with detailed boundaries.

    Compatibility with COCO Format:
        Annotation Format: COCO dataset annotations often use polygonal representations for object segmentation. Detectron2 and other frameworks expect annotations in this format for tasks like instance segmentation.
        Polygonal Masks: Converting masks to polygons ensures compatibility with datasets and models that expect polygon-based annotations.

    Improved Performance:
        Processing Efficiency: Polygonal representations can simplify computations for certain tasks. For example, some algorithms or models may process polygons more efficiently than binary masks, especially for large datasets.
        Simplified Post-processing: Polygonal representations can simplify post-processing steps, such as object boundary adjustments and visualizations.    
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 2:
            poly = contour.reshape(-1).tolist()
            if len(poly) > 4:  # Ensure valid polygon
                polygons.append(poly)
    return polygons

def process_data(image_paths, mask_paths, output_dir):
    """
    process_data() creates the COCO Json format 
    """
    annotations = []
    images = []
    image_id = 0
    ann_id = 0
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        image_id += 1
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # Copy image to output directory
        shutil.copy(img_path, os.path.join(output_dir, os.path.basename(img_path)))
        
        images.append({
            "id": image_id,
            "file_name": os.path.basename(img_path),
            "height": img.shape[0],
            "width": img.shape[1]
        })
        
        unique_values = np.unique(mask)
        for value in unique_values:
            if value == 0:  # Ignore background
                continue
            
            object_mask = (mask == value).astype(np.uint8) * 255
            polygons = mask_to_polygons(object_mask)
            
            for poly in polygons:
                ann_id += 1
                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,  # Only one category: Nuclei
                    "segmentation": [poly],
                    "area": cv2.contourArea(np.array(poly).reshape(-1, 2)),
                    "bbox": list(cv2.boundingRect(np.array(poly).reshape(-1, 2))),
                    "iscrowd": 0
                })
    
    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "Nuclei"}]
    }
    
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_output, f)

def main():
    
    data_dir = 'data/test'
    output_dir = 'COCO_output'
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    
    """
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    image_paths, mask_paths = get_image_mask_pairs(data_dir)
    
    # Split data into train and val
    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths, test_size=0.2, random_state=42)
    
    # Process train and val data
    process_data(train_img_paths, train_mask_paths, train_dir)
    process_data(val_img_paths, val_mask_paths, val_dir)
    """

if __name__ == '__main__':
    main()
