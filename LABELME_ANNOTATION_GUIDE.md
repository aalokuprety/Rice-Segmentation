# Manual Annotation Guide with LabelMe

## Why Use LabelMe for Rice Seed Annotation?

When your images contain:
- ✅ Rice seeds (the objects you want to count)
- ❌ Trash/debris/foreign material
- ❌ Broken seeds
- ❌ Empty husks (that look like seeds)
- ❌ Background artifacts

**LabelMe** helps you create **ground truth annotations** to:
1. Distinguish real seeds from trash
2. Mark filled seeds vs empty husks
3. Create training data for better segmentation
4. Validate automated segmentation results
5. Calculate accurate quality metrics

---

## Installing LabelMe

### Option 1: Using pip (Recommended)
```bash
pip install labelme
```

### Option 2: Using conda
```bash
conda install -c conda-forge labelme
```

### Verify Installation
```bash
labelme --version
```

---

## Quick Start with LabelMe

### 1. Launch LabelMe
```bash
# In your terminal/PowerShell
cd "c:\Fall 2025\ABE Work\Rice segmentation"
labelme
```

Or open a specific image directory:
```bash
labelme data/raw/variety_A
```

### 2. Basic LabelMe Workflow
```
1. Open Image (File → Open)
2. Create Polygon (Right-click → Create Polygons)
3. Click around seed boundary
4. Press Enter to complete polygon
5. Label it (e.g., "seed", "trash", "empty_husk")
6. Repeat for all objects
7. Save annotations (automatically saves as JSON)
```

---

## Recommended Annotation Strategy

### Label Categories for Your Project

#### **Option 1: Simple (Just Seeds)**
```
- seed          (any rice seed - filled or empty)
- trash         (debris, foreign material)
- background    (ignore)
```

#### **Option 2: Quality Classification (Recommended)** ⭐
```
- filled_seed      (viable rice seed with grain)
- empty_husk       (husk without grain inside)
- broken_seed      (damaged/broken seeds)
- trash            (debris, foreign material)
- background       (ignore this area)
```

#### **Option 3: Detailed Analysis**
```
- filled_seed_good     (perfect viable seed)
- filled_seed_small    (undersized but filled)
- empty_husk           (no grain inside)
- partially_filled     (some grain but not full)
- broken_seed          (cracked/damaged)
- trash                (non-seed material)
```

---

## Annotation Best Practices

### 1. **Consistency is Key**
- ✅ Use the same label names across all images
- ✅ Label ALL objects in each image (don't skip any seeds)
- ✅ Be consistent about what counts as "trash" vs "broken seed"

### 2. **Boundary Accuracy**
- Draw polygons close to actual seed boundaries
- Don't need to be pixel-perfect, but reasonably accurate
- Use 8-12 points per seed polygon (enough for good shape)

### 3. **Work Smart**
- Start with 5-10 images per variety (not all 15 immediately)
- This gives you enough data to test segmentation improvement
- Add more annotations if needed

### 4. **Organization**
```
Annotate in this order:
1. Variety A: 5-10 images
2. Test improved segmentation
3. If working well → annotate more
4. Repeat for other varieties
```

---

## LabelMe Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Create polygon | Ctrl + Click |
| Complete polygon | Enter |
| Undo last point | Backspace |
| Delete shape | Delete |
| Save | Ctrl + S |
| Next image | D or → |
| Previous image | A or ← |
| Zoom in | Ctrl + + |
| Zoom out | Ctrl + - |

---

## Annotation Output

LabelMe creates JSON files alongside your images:

```
data/raw/variety_A/
├── image_001.jpg
├── image_001.json      ← Annotation file
├── image_002.jpg
├── image_002.json      ← Annotation file
└── ...
```

### JSON Structure:
```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "filled_seed",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    },
    {
      "label": "trash",
      "points": [[x1, y1], [x2, y2], ...],
      "shape_type": "polygon"
    }
  ],
  "imagePath": "image_001.jpg",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

---

## Using Annotations in Your Analysis

### 1. **Ground Truth for Evaluation**
Compare automated segmentation against your manual annotations:
```python
from evaluation_metrics import SegmentationEvaluator
import json

# Load your LabelMe annotation
with open('data/raw/variety_A/image_001.json') as f:
    annotation = json.load(f)

# Create ground truth mask from annotations
# Compare with automated segmentation
# Calculate IoU, Dice, precision, recall
```

### 2. **Filter Out Trash**
```python
# Extract only "filled_seed" labels
# Ignore "trash" and "empty_husk" in counting
# Get accurate seed counts
```

### 3. **Quality Metrics**
```python
# Calculate fill rate per variety:
filled_seeds = count_labels("filled_seed")
empty_husks = count_labels("empty_husk")
trash_count = count_labels("trash")

fill_rate = filled_seeds / (filled_seeds + empty_husks) * 100
cleanliness = (filled_seeds + empty_husks) / (filled_seeds + empty_husks + trash_count) * 100
```

---

## Recommended Workflow

### **Phase 1: Initial Annotation (Week 1)**
```
Day 1: Install LabelMe, learn interface (30 min)
Day 2: Annotate 5 images from Variety A (1-2 hours)
Day 3: Test automated segmentation with annotations
Day 4: Refine if needed, annotate 5 more images
```

### **Phase 2: Systematic Annotation (Week 2)**
```
- Annotate 10 images per variety
- Total: 40 images annotated (manageable)
- Provides good ground truth dataset
```

### **Phase 3: Analysis (Week 3)**
```
- Compare automated vs manual counting
- Calculate quality metrics per variety
- Identify best performing variety
- Generate final report
```

---

## Time Investment

### Per Image:
- **Simple seeds (no trash)**: 3-5 minutes
- **With trash/debris**: 5-10 minutes
- **Complex/overlapping**: 10-15 minutes

### Total Time Estimate:
```
5 images per variety × 5 min each × 4 varieties = ~100 minutes (1.5 hours)
10 images per variety × 5 min each × 4 varieties = ~200 minutes (3.5 hours)
```

**Recommendation**: Start with 5 images per variety (40 min per variety, ~2.5 hours total)

---

## Tips for Efficient Annotation

### 1. **Start with Cleanest Images**
- Pick images with fewer trash pieces
- Get comfortable with the tool first
- Build confidence before tackling messy images

### 2. **Use Consistent Criteria**
Write down your decision rules:
```
Filled seed = Feels heavy, looks opaque, normal size
Empty husk = Feels light, may look translucent, might be flatter
Trash = Anything that's clearly not rice (chaff, sticks, dust)
Broken seed = Cracked or fragmented rice
```

### 3. **Take Breaks**
- Annotate 5 images, take a break
- Prevents fatigue and improves accuracy
- Better to do 5 well than 15 poorly

### 4. **Batch by Variety**
- Complete one variety before moving to next
- Maintains consistency within variety
- Easier to remember what you're looking at

---

## Advanced: Creating Masks from LabelMe

### Convert JSON to Binary Masks
```python
import json
import numpy as np
from PIL import Image, ImageDraw

def labelme_to_mask(json_file, image_shape):
    """Convert LabelMe JSON to segmentation mask"""
    with open(json_file) as f:
        data = json.load(f)
    
    mask = Image.new('L', (image_shape[1], image_shape[0]), 0)
    draw = ImageDraw.Draw(mask)
    
    for shape in data['shapes']:
        if shape['label'] in ['filled_seed', 'empty_husk']:  # Include these
            points = [tuple(pt) for pt in shape['points']]
            draw.polygon(points, fill=255)
    
    return np.array(mask)

# Use this mask for evaluation
ground_truth_mask = labelme_to_mask('image_001.json', image.shape)
```

---

## Quality Control Checklist

Before finishing annotations:
- [ ] All seeds labeled (not just some)
- [ ] Trash/debris marked separately
- [ ] Consistent label names across all images
- [ ] Polygons follow seed boundaries reasonably well
- [ ] JSON files saved properly
- [ ] At least 5-10 images per variety annotated

---

## Alternative: Faster Annotation Methods

If LabelMe is too slow, consider:

### **1. Bounding Boxes (Faster)**
- Use rectangle tool instead of polygons
- Less accurate but much faster
- Good enough for counting

### **2. Points (Fastest)**
- Just mark seed centers with points
- Very fast (~1 min per image)
- Good for counting, not for shape analysis

### **3. Semi-Automated**
- Run automated segmentation first
- Use LabelMe to correct mistakes
- Faster than starting from scratch

---

## Integration with Your Pipeline

### Updated Workflow:
```
1. Capture images → data/raw/variety_X/
2. Annotate subset with LabelMe → creates .json files
3. Run automated segmentation → compare with annotations
4. Calculate metrics:
   - Automated count vs manual count
   - IoU/Dice for segmentation quality
   - Fill rate (filled vs empty)
   - Cleanliness (seeds vs trash)
5. Generate comparison report
```

---

## Example Analysis with Annotations

```python
# For each variety:
results = {
    'variety_A': {
        'total_objects': 450,
        'filled_seeds': 380,
        'empty_husks': 55,
        'trash': 15,
        'fill_rate': 87.4%,
        'cleanliness': 96.7%
    },
    'variety_B': {
        'total_objects': 425,
        'filled_seeds': 360,
        'empty_husks': 50,
        'trash': 15,
        'fill_rate': 87.8%,
        'cleanliness': 96.5%
    },
    # ... etc
}

# Winner: Variety with highest fill_rate and cleanliness
```

---

## Getting Started

### Right Now:
1. Install LabelMe: `pip install labelme`
2. Open one test image: `labelme data/raw/variety_A/image_001.jpg`
3. Practice annotating 1 image (10 minutes)
4. If comfortable, annotate 5 more images
5. Use annotations to improve your analysis

### This Week:
- Annotate 5-10 images per variety
- Test automated segmentation with ground truth
- Calculate accurate quality metrics

---

## Need Help?

LabelMe Documentation: https://github.com/wkentaro/labelme
LabelMe Tutorial: https://www.youtube.com/results?search_query=labelme+tutorial

Good luck with your annotations! This will significantly improve the quality and accuracy of your analysis! 🏷️🌾