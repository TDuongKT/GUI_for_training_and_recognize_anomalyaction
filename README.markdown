# U2NETLite Segmentation and Analysis System

This project implements a lightweight image segmentation system using the U2NETLite model, designed for segmenting objects in images and videos based on the COCO dataset, combined with a feature extraction and analysis pipeline for action recognition and anomaly detection in videos. The system provides two user-friendly GUIs (built with Tkinter):

1. **Training and Testing GUI** (`gui_train_seg.py` in the training module): For training the U2NETLite model and testing it on videos.
2. **Feature Extraction and Analysis GUI** (`gui_anomaly.py` in the analysis module): For extracting features from segmented videos, optimizing features, clustering, anomaly detection, and inference.

## Project Structure

```
project/
├── config.yaml                 # Configuration file for training, testing, and analysis
├── u2netlite.py                # U2NETLite model definition
├── coco_segmentation_dataset.py # COCO dataset loader for training
├── train_u2netlite.py          # Training script for U2NETLite
├── test_u2netlite_clip.py      # Video testing script for segmentation
├── gui_train_seg.py            # GUI for training and testing (training module)
├── gui_anomaly.py              # GUI for feature extraction and analysis (analysis module)
├── feature_extraction.py       # Feature extraction from videos
├── model_utils.py              # Utility functions for model loading and frame preprocessing
├── shape_features.py           # Shape feature calculation
├── feature_optimization.py     # Feature optimization
├── clustering_analysis.py      # Clustering analysis
├── anomaly_detection.py        # Anomaly detection
├── inference.py                # Inference for action recognition
├── logs/                       # Directory for log files
│   ├── training.log
│   ├── test_video_clip.log
│   ├── gui.log (training)
│   └── gui.log (analysis)
├── data/                       # Directory for training/validation data
│   ├── train/
│   │   ├── images/
│   │   └── annotations/
│   ├── val/
│   │   ├── images/
│   │   └── annotations/
│   └── backgrounds/
├── videos/
│   └── train/                  # Directory for input videos (analysis)
├── models/                     # Directory for saved models
│   ├── best_model.pth
│   ├── last_model.pth
│   └── 70.pth
├── features/                   # Directory for extracted features
│   ├── video1_features.npy
│   ├── combination_results.npy
│   ├── clustering_results.npy
│   └── svm_model.npy
├── output/                     # Directory for segmented videos and inference results
│   ├── output_segmented.mp4
│   └── inference_results.json
```



## Installation Requirements

Install the required Python libraries for both training/testing and analysis:

```bash
pip install torch opencv-python albumentations pycocotools segmentation_models_pytorch pyyaml numpy scipy scikit-learn tkinter
```

Ensure a compatible GPU with CUDA support for faster training, testing, and inference (optional).

## GUI Overview

The system provides two GUIs:

1. **Training and Testing GUI**: Configures and runs training on COCO datasets and tests segmentation on videos. It has two tabs: **Training** and **Testing**.
2. **Feature Extraction and Analysis GUI**: Processes segmented videos to extract features, optimize them, perform clustering, detect anomalies, and run inference for action recognition. It has five tabs: **Feature Extraction**, **Feature Optimization**, **Clustering Analysis**, **Anomaly Detection**, and **Inference**.

Below are the configurable variables for each GUI.

### Training and Testing GUI Variables

#### Training Tab

| Variable | Description | Example Value | Notes |
|----------|-------------|---------------|-------|
| **Training Image Dir** | Directory containing training images (*.jpg). | `data/train/images` | Must contain image files. |
| **Training Annotation** | Path to COCO JSON annotation file for training. | `data/train/annotations/data.json` | Must be a valid COCO JSON file. |
| **Validation Image Dir** | Directory containing validation images (*.jpg). | `data/val/images` | Must contain image files. |
| **Validation Annotation** | Path to COCO JSON annotation file for validation. | `data/val/annotations/data.json` | Must be a valid COCO JSON file. |
| **Background Dir** | Directory containing background images for augmentation. | `data/backgrounds` | Optional; can be empty. |
| **Background Probability** | Probability of replacing the background (0.0 to 1.0). | `0.7` | Controls frequency of background replacement. |
| **Number of Classes** | Total number of classes (including background). | `3` | Must be `len(category_ids) + 1`. |
| **Category IDs** | Comma-separated list of COCO category IDs. | `1,2` | Must match IDs in COCO annotations. |
| **Batch Size** | Number of samples per training batch. | `8` | Adjust based on GPU memory. |
| **Learning Rate** | Learning rate for the Adam optimizer. | `0.001` | Typical range: 0.0001 to 0.01. |
| **Epochs** | Number of training epochs. | `50` | Higher values increase training time. |
| **Best Model Path** | Path to save the best model (based on validation IoU). | `models/best_model.pth` | Ensure directory exists. |
| **Last Model Path** | Path to save the final model after training. | `models/last_model.pth` | Ensure directory exists. |

#### Testing Tab

| Variable | Description | Example Value | Notes |
|----------|---------------|----------------|----------------|
| **Input Video** | Path to the input video for segmentation. | `videos/test/input_video.mp4` | Supports `.mp4`, `.avi`. |
| **Output Video** | Path to save the segmented video (*.mpeg.mp4). | `output/output_segmented.mp4` | Ensure directory exists. |
| **Model Path** | Path to the trained model file (*.pth). | `models/best_model.pt.pth` | Must match `num_classes` used in training. |

### Feature Extraction and Analysis GUI Variables

#### Feature Extraction Tab

| Variable | Description | Description | Example Value | Example Value | Notes |
| **Video Directory** | | Directory containing input videos (*.mph.mp4, *.avi). | `vivi.mp4`, `.avi`. | `videos/train` | Can contain any number of videos. |
| **Model Path** | | Path to the trained U2NETLite model (*.pth). | `m.models/70.pth` | Must match `num_classes`. |
| **Number of Classes** | | Total number of classes (including background). | `4` | Must match the number of class names and colors in `config.yaml`. |
| **Class Names** | | Comma-separated list of class names. | `b.background,class1,class2,class3` | Must match `num_classes`. |
| **Output Directory** | | Directory to save extracted features (*.npy). | `f.features` | Contains `video_name_features.npy`. |

#### Feature Optimization Tab

| Variable | Description | Example Value | Notes |
|----------|-------------|---------------|-------|
| **Number of Clusters** | Number of clusters for KMeans clustering. | `3` | Must be at least 2. |
| **Min Features** | Minimum number of features in each combination. | `5` | Must be at least 1. |
| **Max Features** | Maximum number of features in each combination. | `10` | Must be >= Min Features. |
| **Max Combinations** | Maximum number of feature combinations to evaluate. | `2000` | Must be at least 1. |
| **Features Directory** | Directory containing feature files (*.npy). | `features` | Must contain `video_name_features.npy`. |

#### Clustering Analysis Tab

| Variable | Description | Example Value | Notes |
|----------|-------------|---------------|-------|
| **Features Directory** | Directory containing feature files (*.npy) and `combination_results.npy`. | `features` | Must contain `video_name_features.npy` and `combination_results.npy`. |
| **Number of Clusters** | Number of clusters (from Feature Optimization). | `3` | Read-only, set in Feature Optimization tab. |

#### Anomaly Detection Tab

| Variable | Description | Example Value | Notes |
|----------|-------------|---------------|-------|
| **Features Directory** | Directory containing `clustering_results.npy`. | `features` | Must contain `clustering_results.npy`. |
| **SVM Nu** | Proportion of outliers for One-Class SVM (0 to 1). | `0.05` | Must be between 0 and 1. |
| **SVM Gamma** | Kernel coefficient for One-Class SVM (>0). | `0.1` | Must be greater than 0. |

#### Inference Tab

| Variable | Description | Example Value | Notes |
|----------|-------------|---------------|-------|
| **Input Video** | Path to input video (*.mp4, *.avi). | `videos/test/video.mp4` | Single video file. |
| **U2NETLite Model** | Path to U2NETLite model (*.pth). | `models/70.pth` | Must match `num_classes`. |
| **SVM Model** | Path to SVM model (*.npy). | `features/svm_model.npy` | Generated in Anomaly Detection tab. |
| **Output Video** | Path to save segmented video (*.mp4). | `output/output_segmented.mp4` | Overlaid with masks. |
| **JSON Results** | Path to save inference results (*.json). | `output/inference_results.json` | Contains video path, datetime, silhouette score, SVM score, and anomaly status. |
| **Number of Classes** | Total number of classes (from Feature Extraction). | `4` | Read-only, set in Feature Extraction tab. |
| **Class Names** | List of class names (from Feature Extraction). | `background,class1,class2,class3` | Read-only, set in Feature Extraction tab. |
| **Number of Clusters** | Number of clusters (from Feature Optimization). | `3` | Read-only, set in Feature Optimization tab. |

## Usage Guide

### 1. Setup

1. Clone or download the project repository.
2. Create the required directories: `logs/`, `models/`, `features/`, `videos/train/`, `output/`, `data/train/`, `data/val/`, `data/backgrounds/`.
3. Prepare your data:
   - For training: Place images and COCO JSON annotations in `data/train/` and `data/val/`.
   - For analysis: Place input videos in `videos/train/` or any directory for inference.
4. Ensure all required libraries are installed (see Installation Requirements).
5. Update `config.yaml` to match your dataset, especially `num_classes`, `class_names`, `category_ids`, and `testing.colors`.

### 2. Running the Training and Testing GUI

Run the training/testing GUI:

```bash
python gui_train_seg.py
```

#### Training Tab

1. **Configure Data Paths**:
   - Select directories/files for `Training Image Dir`, `Training Annotation`, `Validation Image Dir`, `Validation Annotation`, and `Background Dir` using "Browse".
   - Ensure paths are valid.

2. **Set Parameters**:
   - Enter `Background Probability` (e.g., `0.7`).
   - Set `Number of Classes` (e.g., `3` for background, bottle, cup).
   - Enter `Category IDs` (e.g., `1,2` for bottle and cup).
   - Configure `Batch Size` (e.g., `8`), `Learning Rate` (e.g., `0.001`), and `Epochs` (e.g., `50`).
   - Select `Best Model Path` and `Last Model Path` using "Browse".

3. **Start Training**:
   - Click "Start Training".
   - Monitor progress in the progress bar and log area (shows loss, IoU, Dice per epoch).
   - Logs are saved to `logs/training.log`.

4. **Notes**:
   - Ensure `Number of Classes` is `len(category_ids) + 1`.
   - Training time depends on `Epochs` and hardware.

#### Testing Tab

1. **Configure Paths**:
   - Select `Input Video`, `Output Video`, and `Model Path` using "Browse".
   - Ensure the model matches `num_classes` from training.

2. **Start Testing**:
   - Click "Start Testing".
   - Monitor progress in the progress bar and log area (shows frame processing status).
   - Logs are saved to `logs/test_video_clip.log`.

3. **View Results**:
   - The segmented video is saved at `Output Video`.
   - Masks are colored per `testing.colors` in `config.yaml`.

### 3. Running the Feature Extraction and Analysis GUI

Run the feature extraction and analysis GUI:

```bash
python gui_anomaly.py
```

#### Feature Extraction Tab

1. **Configure Paths**:
   - Select `Video Directory`, `Model Path`, and `Output Directory` using "Browse".
   - Set `Number of Classes` (e.g., `4`) and `Class Names` (e.g., `background,class1,class2,class3`).

2. **Start Extraction**:
   - Click "Extract Features".
   - Monitor progress in the progress bar and log area.
   - Features are saved as `video_name_features.npy` in `Output Directory`.

#### Feature Optimization Tab

1. **Set Parameters**:
   - Enter `Number of Clusters` (e.g., `3`), `Min Features` (e.g., `5`), `Max Features` (e.g., `10`), and `Max Combinations` (e.g., `2000`).
   - Select `Features Directory` containing `video_name_features.npy`.

2. **Start Optimization**:
   - Click "Find Optimized Features".
   - Monitor progress and results (best feature combination, silhouette score).
   - Results are saved as `combination_results.npy`.

#### Clustering Analysis Tab

1. **Configure**:
   - Select `Features Directory` containing `video_name_features.npy` and `combination_results.npy`.
   - Verify `Number of Clusters` (read-only).

2. **Start Clustering**:
   - Click "Run Clustering Analysis".
   - Results (silhouette score) are saved as `clustering_results.npy`.

#### Anomaly Detection Tab

1. **Configure**:
   - Select `Features Directory` containing `clustering_results.npy`.
   - Enter `SVM Nu` (e.g., `0.05`) and `SVM Gamma` (e.g., `0.1`).

2. **Start Detection**:
   - Click "Run Anomaly Detection".
   - Results (anomaly scores, threshold) are displayed and saved as `svm_model.npy`.

#### Inference Tab

1. **Configure**:
   - Select `Input Video`, `U2NETLite Model`, `SVM Model`, `Output Video`, and `JSON Results` using "Browse".
   - Verify `Number of Classes`, `Class Names`, and `Number of Clusters` (read-only).

2. **Start Inference**:
   - Click "Run Inference".
   - Monitor progress and results (silhouette score, SVM score, anomaly status).
   - Segmented video is saved at `Output Video`, and results are saved at `JSON Results`.

### 4. Testing with Different Configurations

#### Different `num_classes`

- **Training**:
  - Set `Number of Classes` to `4` and `Category IDs` to `1,2,3` in the Training tab.
  - Train a new model.
  - Update `testing.colors` in `config.yaml` to include 4 colors.
- **Testing**:
  - Use the new model in the Testing tab.
- **Feature Extraction and Analysis**:
  - Set `Number of Classes` to `4` and `Class Names` to `background,class1,class2,class3` in the Feature Extraction tab.
  - Ensure `testing.colors` has 4 entries.
  - Run the analysis pipeline.

#### Different Videos

- Place new videos in `videos/test/` and select them in the Testing or Inference tabs.
- Adjust output paths as needed.

## Notes and Troubleshooting

- **Log Files**:
  - Training/Testing: Check `logs/training.log`, `logs/test_video_clip.log`, `logs/gui.log` (training).
  - Analysis: Check `logs/gui.log` (analysis).
- **GUI Freezing**:
  - Reduce `Batch Size`, `image_size` (in `config.yaml`), or number of feature combinations.
  - Ensure sufficient RAM/GPU memory.
- **Invalid Inputs**:
  - Training: Ensure `Number of Classes` is `len(category_ids) + 1`, and paths are valid.
  - Analysis: Verify `num_classes` matches `class_names` and `testing.colors`. Ensure `Features Directory` contains required files (`video_name_features.npy`, `combination_results.npy`, `clustering_results.npy`, `svm_model.npy`).
- **Video Output Issues**:
  - If output videos don’t play, try changing the codec in `test_u2netlite_clip.py` or `inference.py` (e.g., `mp4v` to `XVID`, save as `.avi`).
- **Feature Shape**:
  - Features should have shape `(100, 1, N)` where `N = (num_classes-1)*10 + (num_classes-1)*(num_classes-2)/2`.
  - Example: For `num_classes=4`, `N = 3*10 + 3*2/2 = 33`.
- **Inference Issues**:
  - Ensure `svm_model.npy` and `combination_results.npy` exist.
  - Verify the U2NETLite model matches `num_classes`.
  - Adjust `SVM Nu` or `SVM Gamma` if no anomalies are detected.
- **Customization**:
  - Edit `testing.colors` in `config.yaml` to change mask colors (must match `num_classes`).
  - Modify `image_size` in `config.yaml` to adjust resolution.
  - Update `class_names` in the Feature Extraction tab to match your dataset.

## Contact

For issues or feature requests, please contact the project maintainers or open an issue on the repository.