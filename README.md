# STYLESKETCH

This repository implements a Generative Adversarial Network (GAN) for two tasks:
- **Color-to-Sketch**: Convert a color image to a grayscale sketch.
- **Sketch-to-Color**: Colorize a sketch image using a color palette extracted from a reference color image.

The project uses PyTorch and includes scripts for data preparation, training, and inference.

## Table of Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [File Structure](#file-structure)
- [Notes](#notes)

## Requirements

### Software
- Python 3.8 or higher
- PyTorch 1.9 or higher
- CUDA (optional, for GPU training)
- Other dependencies:
  ```bash
  pip install torch torchvision numpy opencv-python pillow matplotlib tqdm scikit-learn streamlit
  ```

### Hardware
- **CPU**: Minimum 8 cores for data loading and training.
- **GPU**: NVIDIA GPU with at least 8GB VRAM (recommended for faster training).
- **RAM**: At least 16GB.
- **Storage**: At least 10GB for dataset and checkpoints.

## Data Preparation

### 1. Dataset
#### 1.1. Download Dataset

The dataset is hosted on Google Drive, containing color images organized by style folders (e.g., style1, style2).

1. Install gdown:
``` bash
pip install gdown
```

2. Download and unzip:
``` bash
gdown <https://drive.google.com/drive/folders/1jYsOWDKwuW77aeh9eDjlmZPAPvjnuk75?usp=drive_link> -O images.zip
unzip images.zip -d images
```

### 2. Directory Structure
After unzipping, the dataset should have the following structure:
```
images/
├── style1/
│   ├── image1.png
│   ├── image2.png
│   ├── image4.png
│   └── ...
├── style2/
│   ├── image1.png
│   ├── image2.png
│   ├── image3.png
│   └── ...
├── ...
```
Each style{i} folder contains color images (PNG or JPEG) for a specific style.

The PairImageFolder class in dataloader.py will load images from these folders, generate sketches using Color2Sketch, and extract color palettes using color_cluster.

### 3. Notes
- Ensure all images are in PNG or JPEG format.
- The color_cluster function extracts up to nclusters colors from each image.
- No separate sketch images are needed, as Color2Sketch generates them during training.

## Training

### 1. Setup
- Clone the repository:
  ```bash
  git clone <https://github.com/Tien-Dung-03/STYLESKETCH.git>
  cd <STYLESKETCH>
  ```

### 2. Configure Training
Edit `train.py` or pass arguments via command line to configure:
- `--data_dir`: Path to dataset (e.g., "images/")
- `--batch_size`: Default=4
- `--epochs`: Default=100
- `--lr`: Default=0.0002
- `--nclusters`: Default=9
- `--num_workers`: Default=4
- `--save_interval`: Default=10
- `--eval_interval`: Default=5, Number of epochs between evaluations
- `--resume`: Path to checkpoint to resume from (e.g., "checkpoints/checkpoint_epoch_20.pth")
- `--train_ratio`: Default=0.8, Ratio of training data
- `--random_style`: action='store_true', Use random style codes
- `--seed`: Default=42, Random seed for reproducibility
- `--pin_memory`: Use pin memory in DataLoader

### 3. Run Training
Run the training script:
```bash
python train.py --data_dir "images" --resume "checkpoints/checkpoint_epoch_20.pth" --batch_size 4 --epochs 50 --lr 0.0002 --nclusters 9 --num_workers 2 --save_interval 5 --eval_interval 5 --random_style --pin_memory
```

- **GPU Note**: If using GPU, make sure CUDA is installed. And if not using GPU comment out the following line in train.py:
```python
  # mp.set_start_method('spawn', force=True)
  # Rest of train.py code
```
- **Output**: Checkpoints are saved in `checkpoints/` (e.g., `checkpoint_epoch_5.pth`, `checkpoint_epoch_10.pth`).

### 4. Training Details
- **Models**:
  - `Color2Sketch`: Converts color images to sketches.
  - `Sketch2Color`: Colorizes sketches using a style code and color palette.
  - `Discriminator`: Distinguishes real and fake image pairs.
- **Losses** (from `losses.py`):
  - Adversarial loss (`adv_loss`): Ensures generated images are realistic.
  - Feature matching loss (`feat_loss`): Matches VGG features.
  - Perceptual loss (`perc_loss`): Matches high-level VGG features.
  - Style loss (`style_loss`): Matches style via Gram matrices.
  - Color loss (`color_loss`): Ensures colors match the palette.
- **Training Log**:
  - Logs are printed per epoch, showing `G_loss`, `D_loss`, and `Style` (style loss).
  - Test losses are evaluated at epochs 5 and 10.

## Inference

### 1. Prepare Inference Script
The inference.py script provides a Streamlit interface with two functions:

- color_to_sketch_and_recolor: Extracts palette from a color image, generates a sketch, and recolors it.
- sketch_to_color: Colorizes a provided sketch using a palette from a color image.

### 2. Run Inference
1. Run the Streamlit app:
```bash
streamlit run inference.py
```
2. Open the browser at http://localhost:8501 (or the URL shown).

3. Use the interface to:

  - Select function (color_to_sketch_and_recolor or sketch_to_color).
  - Upload checkpoint file (.pth).
  - Upload color image (PNG/JPEG).
  - Upload sketch image (for test_image).
  - Choose style (0-3) and number of colors (1-15).
  - Click "Run Inference" to view results.

- Output:

  - Displays: Original color image, sketch (generated/input), recolored image, color palette.
  - Images are saved as output_color_to_sketch_style{i}.png or output_test_image_style{i}.png.

### 3.  Example Usage
- For color_to_sketch_and_recolor:
  - Upload dataset/style1/image1.png as color image.
  - Select style=0, nclusters=9.
  - View: Color image, generated sketch, recolored image, palette.

- For sketch_to_color:
  - Upload dataset/style1/image1.png (color) and dataset/style1/sketch_image1.png (sketch).
  - Select style=0, nclusters=9.
  - View: Color image, input sketch, colored image, palette.

## File Structure
```
├── checkpoints/                # Model checkpoints
├── images/                    # Dataset with style subfolders
│   ├── style1/
│   │   ├── image1.png
│   │   ├── image2.png
│   │   ├── ...
│   ├── style2/
│   │   ├── image1.png
│   │   ├── ...
|   ├── ...
├── dataloader.py               # Data loading and palette extraction
├── losses.py                   # Loss functions
├── mymodels.py                 # Model architectures
├── train.py                    # Training script
├── inference.py                # Streamlit inference script
├── README.md                   # This file
```

## Notes
- Training Time: ~10-16 minutes/epoch on an NVIDIA RTX 3080 for 577 batches (batch size 4). 10 epochs take ~2 hours.
- Inference Quality: Recoloring may not perfectly match the palette due to low lambda_color=1.0. Increase to 5.0 in losses.py and retrain.
- Sketch Images: For test_image, provide sketch images or generate them using Color2Sketch.
- GPU Errors: Use mp.set_start_method('spawn') in train.py for num_workers > 0 on GPU.
- Streamlit: Ensure streamlit is installed. Run on a local machine or server with GPU for CUDA support.
- Dataset: Ensure sufficient images per style folder. Add more styles or images for better generalization.
- Customization:
  - Adjust nclusters for palette size.
  - Try different selected_style values.
  - Modify transform for other image sizes (ensure model compatibility).
  - For issues, open an issue or submit a pull request.
