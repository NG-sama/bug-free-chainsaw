# Nuclei Segmentation Project

This repository contains code for nuclei segmentation using two different approaches: Detectron2 and YOLO v8. It also includes data handling scripts for preprocessing and converting data between different formats.

## Repository Structure

```
.
├── 336a_detectron2_nuclei_segmentation.ipynb
├── 336b_yolov8_nuclei_segmentation.ipynb
├── data_handling_code/
│   ├── 01-remove_unwanted_data.py
│   ├── 02-Convert_label_masks_to_COCO JSON.py
│   ├── 03-Convert_COCO JSON_to_YOLOv8.py
│   ├── 04a-visualize-COCO labels.py
│   ├── 04b_visualize-COCO labels_filled.py
│   ├── 04c-visualize-YOLO labels.py
|   └── 04d-visualize-YOLO labels-filled.py
|    
└── README.md
```

## Segmentation Models

### 336a: Detectron2 Nuclei Segmentation

This script (`336a_Detectron2_Instance_Nuclei.ipynb`) implements nuclei segmentation using Facebook's Detectron2 framework. Key features include:

- Installation of Detectron2 and required dependencies
- Dataset registration using COCO format
- Model configuration and training
- Evaluation using COCO metrics
- (Optional) Inference on test images and saving results

### 336b: YOLO v8 Nuclei Segmentation

The script `336b_training_YOLO_V8_Nuclei.ipynb` implements nuclei segmentation using the YOLO v8 model. (Note: Details about this script are not provided in the given content.)

## Data Handling Code

The `data_handling_code` folder contains scripts for various data preprocessing and conversion tasks:

1. **01-remove_unwanted_data.py**: Scripts for initial data preparation and cleaning from Kaggle.
2. **02-Convert_label_masks_to_COCO JSON.py**: Tools to convert label masks to COCO JSON format.
3. **03-Convert_COCO JSON_to_YOLOv8.py**: Scripts to convert COCO JSON annotations to YOLO format.
4. **04(a-d)-Visualize**: Tools for visualizing the labeled data in different formats.

## Acknowledgments

- All credits go to Bhattiprolu S. who has worked on creating a wonderful Youtube channel explaing how to get kickstarted with machine vision and image segmentation for biologists. 

## References
1. Mahbod, A., Polak, C., Feldmann, K., Khan, R., Gelles, K., Dorffner, G., Woitek, R., Hatamikia, S., & Ellinger, I. (2023). NuInsSeg: A fully annotated dataset for nuclei instance segmentation in H&E-stained histological images. arXiv preprint arXiv:2308.01760. [https://arxiv.org/abs/2308.01760](https://arxiv.org/abs/2308.01760)
2. Bhattiprolu, S. (2023). python_for_microscopists. GitHub. [https://github.com/bnsreenu/python_for_microscopists/tree/master/336-Nuclei-Instance-Detectron2.0_YOLOv8_code](https://github.com/bnsreenu/python_for_microscopists/tree/master/336-Nuclei-Instance-Detectron2.0_YOLOv8_code)
