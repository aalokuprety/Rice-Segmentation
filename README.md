# Rice Seeds Segmentation Project

This project implements computer vision techniques for segmenting individual rice seeds from images.

## Project Structure

```
Rice segmentation/
├── data/
│   ├── raw/          # Original rice seed images
│   └── processed/    # Preprocessed and segmented images
├── notebooks/        # Jupyter notebooks for analysis and experimentation
├── scripts/          # Python scripts for segmentation algorithms
├── models/           # Trained models (if using ML approaches)
├── results/          # Output results and evaluation metrics
└── README.md         # This file
```

## Setup

1. Install required dependencies:
   ```bash
   pip install opencv-python scikit-image numpy matplotlib pillow jupyter
   ```

2. **Capture your rice seed images** (see [IMAGE_ACQUISITION_GUIDE.md](IMAGE_ACQUISITION_GUIDE.md) for detailed instructions)

3. Place your rice seed images in the `data/raw/` directory

3. Run the notebooks in order:
   - `01_data_exploration.ipynb` - Explore and visualize your data
   - `02_preprocessing.ipynb` - Image preprocessing and enhancement
   - `03_segmentation.ipynb` - Rice seed segmentation algorithms

## Segmentation Approaches

This project implements several segmentation techniques:

1. **Traditional Computer Vision Methods:**
   - Thresholding (Otsu, adaptive)
   - Edge detection (Canny, Sobel)
   - Watershed algorithm
   - Region growing
   - Morphological operations

2. **Evaluation Metrics:**
   - Intersection over Union (IoU)
   - Dice coefficient
   - Precision and Recall
   - Visual comparison tools

## Usage

See individual notebooks and scripts for detailed usage instructions.

## Results

Results and evaluation metrics will be saved in the `results/` directory.