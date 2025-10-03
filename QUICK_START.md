# Quick Start Guide for Rice Seeds Segmentation

## Project Overview
This project provides a complete computer vision solution for segmenting individual rice seeds from images. It includes multiple segmentation algorithms, evaluation metrics, and a user-friendly Jupyter notebook interface.

## Getting Started

### 1. Add Your Rice Seed Images
- Place your rice seed images in the `data/raw/` directory
- Supported formats: JPG, JPEG, PNG, BMP, TIFF
- For best results, use high-contrast images with good lighting

### 2. Start with the Jupyter Notebook
Open and run the notebook: `notebooks/01_rice_segmentation_analysis.ipynb`

This notebook will guide you through:
- Loading and visualizing your images
- Applying different preprocessing techniques
- Trying various segmentation methods
- Extracting morphological features
- Comparing results

### 3. Use the Segmentation Methods Script
For programmatic access, use: `scripts/rice_segmentation_methods.py`

```python
from rice_segmentation_methods import RiceSeedSegmenter

# Create segmenter
segmenter = RiceSeedSegmenter()

# Segment an image
results = segmenter.segment_image('path/to/image.jpg')
print(f"Found {results['seed_count']} seeds")

# Compare different methods
from rice_segmentation_methods import compare_segmentation_methods
comparison = compare_segmentation_methods('path/to/image.jpg')
```

### 4. Evaluate Your Results
Use the evaluation metrics: `scripts/evaluation_metrics.py`

```python
from evaluation_metrics import SegmentationEvaluator

evaluator = SegmentationEvaluator()
metrics = evaluator.evaluate_segmentation(predicted_mask, ground_truth_mask)
print(f"IoU: {metrics['iou']:.3f}")
```

### 5. Process Multiple Images
Use the main pipeline: `scripts/main_pipeline.py`

```python
from main_pipeline import RiceSegmentationPipeline

pipeline = RiceSegmentationPipeline()

# Process single image
result = pipeline.process_single_image('image.jpg')

# Process batch of images
results_df = pipeline.process_batch('data/raw/')
```

## Available Segmentation Methods
1. **Otsu Thresholding** - Simple, fast, good for high-contrast images
2. **Adaptive Thresholding** - Better for varying lighting conditions
3. **Watershed Algorithm** - Excellent for separating touching seeds
4. **Connected Components** - Good for well-separated objects
5. **Region Growing** - Precise but slower method
6. **Canny Edge Detection** - Edge-based approach

## Key Features
- **Preprocessing**: Noise reduction, contrast enhancement, filtering
- **Segmentation**: Multiple algorithms for different scenarios
- **Feature Extraction**: Area, perimeter, aspect ratio, circularity, etc.
- **Evaluation**: IoU, Dice coefficient, precision, recall, F1 score
- **Visualization**: Comprehensive plots and overlays
- **Batch Processing**: Handle multiple images efficiently

## Example Workflow
1. Load your rice seed images into `data/raw/`
2. Open the Jupyter notebook and run all cells
3. Examine the results and choose the best segmentation method
4. Use the scripts for batch processing or integration into other projects
5. Evaluate results if you have ground truth data

## Troubleshooting
- **No seeds detected**: Try adjusting the area thresholds in the segmenter configuration
- **Too many false positives**: Increase minimum area or adjust aspect ratio filters
- **Seeds not separated**: Use watershed algorithm with appropriate min_distance parameter
- **Poor image quality**: Improve preprocessing or try different thresholding methods

## Next Steps
- Fine-tune parameters for your specific rice variety
- Create ground truth annotations for quantitative evaluation
- Experiment with deep learning approaches for more complex scenarios
- Integrate with automated counting or quality assessment systems

Happy segmenting! 🌾