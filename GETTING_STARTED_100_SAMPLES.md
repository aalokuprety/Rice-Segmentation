# Getting Started: Processing 100 Rice Samples

## Overview
With 100 samples across 3-4 varieties, you need a **hybrid approach**: automated segmentation for efficiency + selective manual annotation for validation.

---

## Phase 1: Organization (Day 1 - 30 minutes)

### Step 1: Organize Your Images

Distribute your 100 samples into variety folders:

```
data/raw/
├── variety_A/    # ~25 images of Variety A
├── variety_B/    # ~25 images of Variety B
├── variety_C/    # ~25 images of Variety C
└── variety_D/    # ~25 images of Variety D (if you have 4 varieties)
```

**Naming Convention:**
- Use clear, consistent names: `varietyA_001.jpg`, `varietyA_002.jpg`, etc.
- Include capture date if relevant: `varietyA_20251016_001.jpg`
- Keep names short and consistent

**PowerShell Commands to Organize:**
```powershell
# Example: Move images to variety folders
Move-Item "image_001.jpg" "data/raw/variety_A/varietyA_001.jpg"
Move-Item "image_002.jpg" "data/raw/variety_A/varietyA_002.jpg"
# ... etc
```

---

## Phase 2: Strategic Annotation Plan (Day 1-2)

**DON'T annotate all 100 images!** That would take 10+ hours.

### Annotation Strategy:

**Annotate only 15-20 images total** (representative subset):
- **4-5 images per variety** for ground truth
- Choose images with:
  - ✅ Good representation of seed quality
  - ✅ Mix of filled seeds and empty husks
  - ✅ Some trash/debris present
  - ✅ Various lighting conditions
  - ✅ Different seed densities

**Time Investment:** 2-3 hours total for 15-20 images

### Create Annotation Subset:

Copy representative images to annotation folder:
```powershell
# Create annotation folder
New-Item -ItemType Directory -Path "data/annotations"

# Copy representative images
Copy-Item "data/raw/variety_A/varietyA_001.jpg" "data/annotations/"
Copy-Item "data/raw/variety_A/varietyA_008.jpg" "data/annotations/"
Copy-Item "data/raw/variety_A/varietyA_015.jpg" "data/annotations/"
# ... select 4-5 per variety
```

---

## Phase 3: Manual Annotation (Day 2-3)

### Install LabelMe:
```powershell
pip install labelme
```

### Launch LabelMe:
```powershell
labelme data/annotations
```

### Annotation Labels (Use Quality Classification):
1. **filled_seed** - Viable seeds with grain inside
2. **empty_husk** - Husks without grain
3. **broken_seed** - Damaged/broken seeds
4. **trash** - Debris, plant material, etc.

### Workflow:
1. Open first image
2. Use polygon tool (Ctrl+N) to outline each object
3. Label appropriately
4. Save (Ctrl+S) - creates `.json` file
5. Next image (D key)
6. Repeat for 15-20 selected images

**Time estimate:** 5-10 minutes per image = 2-3 hours total

---

## Phase 4: Automated Segmentation (Day 3-4)

### Test on Single Image:
```python
from scripts.rice_segmentation_methods import RiceSeedSegmenter
import cv2

# Load test image
segmenter = RiceSeedSegmenter()
image = cv2.imread('data/raw/variety_A/varietyA_001.jpg')

# Try different methods
results_otsu = segmenter.otsu_segmentation(image)
results_adaptive = segmenter.adaptive_segmentation(image)
results_watershed = segmenter.watershed_segmentation(image)

# Compare visually
print(f"Otsu found: {results_otsu['seed_count']} seeds")
print(f"Adaptive found: {results_adaptive['seed_count']} seeds")
print(f"Watershed found: {results_watershed['seed_count']} seeds")
```

### Run Batch Processing:

Use the new batch processing script:
```powershell
# Process all images in variety_A
python scripts/batch_process_samples.py --variety variety_A --method adaptive

# Process all varieties
python scripts/batch_process_samples.py --all-varieties --method adaptive

# With visualization
python scripts/batch_process_samples.py --all-varieties --method adaptive --visualize
```

**Time estimate:** 5-15 minutes for 100 images (depends on hardware)

---

## Phase 5: Validation (Day 4-5)

### Compare Automated vs Manual:
```python
from scripts.labelme_helper import LabelMeAnnotationProcessor
from scripts.rice_segmentation_methods import RiceSeedSegmenter

processor = LabelMeAnnotationProcessor()
segmenter = RiceSeedSegmenter()

# For each annotated image
for json_file in Path('data/annotations').glob('*.json'):
    # Get manual count
    manual_metrics = processor.calculate_quality_metrics(str(json_file))
    
    # Get automated count
    image_path = json_file.with_suffix('.jpg')
    image = cv2.imread(str(image_path))
    auto_results = segmenter.adaptive_segmentation(image)
    
    # Compare
    print(f"\nImage: {json_file.stem}")
    print(f"Manual count: {manual_metrics['total_seeds']}")
    print(f"Automated count: {auto_results['seed_count']}")
    print(f"Difference: {abs(manual_metrics['total_seeds'] - auto_results['seed_count'])}")
```

### Validation Metrics:
- **Count accuracy:** Should be within ±5-10%
- **IoU:** Should be > 0.7 for good segmentation
- **Visual inspection:** Check a few overlay images

**If accuracy is poor:**
- Adjust segmentation parameters (min/max seed area, aspect ratio)
- Try different preprocessing (more blur, CLAHE adjustment)
- Use watershed instead of simple thresholding

---

## Phase 6: Quality Analysis (Day 5-6)

### Calculate Metrics Per Variety:

```python
from scripts.labelme_helper import compare_varieties

# Compare all varieties based on annotations
comparison = compare_varieties(
    'data/raw',
    ['variety_A', 'variety_B', 'variety_C', 'variety_D']
)

print(comparison)
```

### For Automated Results:

```python
import pandas as pd
from pathlib import Path

# Collect all automated results
all_results = []
for variety in ['variety_A', 'variety_B', 'variety_C', 'variety_D']:
    results_file = f'results/{variety}_automated_results.csv'
    if Path(results_file).exists():
        df = pd.read_csv(results_file)
        df['variety'] = variety
        all_results.append(df)

# Combine and analyze
combined = pd.concat(all_results, ignore_index=True)

# Summary statistics
summary = combined.groupby('variety').agg({
    'seed_count': ['mean', 'std', 'min', 'max'],
    'mean_area': 'mean',
    'mean_circularity': 'mean'
}).round(2)

print(summary)
```

---

## Phase 7: Statistical Comparison (Day 6-7)

### Variety Comparison:

```python
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ANOVA test
varieties = ['variety_A', 'variety_B', 'variety_C', 'variety_D']
groups = [combined[combined['variety'] == v]['seed_count'].values 
          for v in varieties]

f_stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✓ Significant differences found between varieties!")
    
    # Pairwise comparisons (Tukey HSD)
    from scipy.stats import tukey_hsd
    result = tukey_hsd(*groups)
    print("\nPairwise comparisons:")
    print(result)

# Visualization
plt.figure(figsize=(12, 6))

# Box plot
plt.subplot(1, 2, 1)
sns.boxplot(data=combined, x='variety', y='seed_count')
plt.title('Seed Count Distribution by Variety')
plt.ylabel('Seed Count')

# Bar plot with error bars
plt.subplot(1, 2, 2)
variety_means = combined.groupby('variety')['seed_count'].agg(['mean', 'std'])
plt.bar(variety_means.index, variety_means['mean'], 
        yerr=variety_means['std'], capsize=5)
plt.title('Mean Seed Count by Variety')
plt.ylabel('Mean Seed Count')

plt.tight_layout()
plt.savefig('results/variety_comparison.png', dpi=300)
plt.show()
```

---

## Recommended Timeline

| Day | Task | Time | Status |
|-----|------|------|--------|
| **Day 1** | Organize 100 images into folders | 30 min | ⬜ |
| **Day 1** | Select 15-20 representative images | 30 min | ⬜ |
| **Day 2-3** | Annotate 15-20 images with LabelMe | 2-3 hrs | ⬜ |
| **Day 3** | Test segmentation methods on samples | 1 hr | ⬜ |
| **Day 3-4** | Run batch processing on all 100 images | 1 hr | ⬜ |
| **Day 4-5** | Validate automated vs manual results | 2 hrs | ⬜ |
| **Day 5-6** | Calculate quality metrics | 2 hrs | ⬜ |
| **Day 6-7** | Statistical analysis & reporting | 2-3 hrs | ⬜ |

**Total Time Investment:** ~12-15 hours over 1 week

---

## Key Decisions You Need to Make

### 1. **How many varieties?**
- [ ] 3 varieties (33-34 samples each)
- [ ] 4 varieties (25 samples each)

### 2. **What's your primary metric?**
- [ ] **Fill rate** (filled seeds / total seeds) - requires distinguishing filled vs empty
- [ ] **Seed count** (total viable seeds per image)
- [ ] **Seed size** (mean seed area)
- [ ] **Cleanliness** (seeds vs trash ratio)

### 3. **Segmentation approach?**
- [ ] **Automated only** - Fast, may need parameter tuning
- [ ] **Hybrid** (recommended) - Automated + manual validation subset
- [ ] **Manual only** - Most accurate but very time-consuming (10+ hours)

---

## Quick Start Commands

### 1. Install LabelMe:
```powershell
pip install labelme
```

### 2. Test segmentation on one image:
```powershell
python -c "from scripts.rice_segmentation_methods import RiceSeedSegmenter; import cv2; s=RiceSeedSegmenter(); img=cv2.imread('data/raw/variety_A/varietyA_001.jpg'); r=s.adaptive_segmentation(img); print(f'Found {r[\"seed_count\"]} seeds')"
```

### 3. Process all samples:
```powershell
python scripts/batch_process_samples.py --all-varieties --method adaptive
```

### 4. Compare varieties:
```powershell
python -c "from scripts.labelme_helper import compare_varieties; compare_varieties('data/raw', ['variety_A', 'variety_B', 'variety_C', 'variety_D'])"
```

---

## Need Help?

- **Segmentation not working well?** Adjust parameters in `rice_segmentation_methods.py`
- **Too much trash?** Use manual annotation on subset, then create trash filter
- **Seeds touching?** Use watershed segmentation with adjusted `min_distance`
- **Inconsistent lighting?** Apply CLAHE preprocessing more aggressively

---

## Expected Outcomes

By the end of this workflow, you'll have:

✅ 100 images processed with automated segmentation  
✅ 15-20 images with ground truth annotations  
✅ Seed counts, sizes, and shapes for all samples  
✅ Quality metrics per variety (fill rate, cleanliness)  
✅ Statistical comparison identifying best variety  
✅ Publication-ready figures and tables  

**Result:** Clear answer to "Which rice variety has the best seed set?"