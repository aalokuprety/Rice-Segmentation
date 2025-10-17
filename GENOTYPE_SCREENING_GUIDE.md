# Large-Scale Genotype Screening Guide

## Project: 100 Rice Genotypes Comparison

You're conducting a **genotype screening experiment** to identify rice lines with the best seed set/fill rate among 100 different genetic lines.

---

## 📸 Photography Protocol

### Recommended Setup

**Photos per genotype:** 2-3 replicates
- **Minimum:** 2 photos × 100 genotypes = 200 photos
- **Recommended:** 3 photos × 100 genotypes = 300 photos ⭐

### Per Photo Guidelines

1. **Seed count:** 20-30 seeds per photo
2. **Background:** White paper or light surface
3. **Spacing:** Seeds should not touch (or minimal touching)
4. **Lighting:** Consistent overhead lighting
5. **Distance:** ~30 cm from camera to seeds
6. **Focus:** Ensure all seeds are in focus

### Capture Workflow

For each genotype:
1. Label a small container/bag: "Genotype 001"
2. Pour ~30 seeds onto white background
3. Spread them out evenly
4. Take photo 1 (rep1)
5. Rearrange seeds slightly
6. Take photo 2 (rep2)
7. Optionally take photo 3 (rep3)
8. Move to next genotype

**Time estimate:** 
- 2 photos/genotype: ~2 minutes each = ~3-4 hours total
- 3 photos/genotype: ~3 minutes each = ~5-6 hours total

---

## 📁 Organization Strategy

### Option A: Flat Structure (Simpler)

```
data/raw/
├── gen001_rep1.jpg
├── gen001_rep2.jpg
├── gen001_rep3.jpg
├── gen002_rep1.jpg
├── gen002_rep2.jpg
...
├── gen100_rep1.jpg
├── gen100_rep2.jpg
└── gen100_rep3.jpg
```

**Advantages:**
- Simple to manage
- Easy batch processing
- Extract genotype from filename

**Create with PowerShell:**
```powershell
# All files in one directory
cd "c:\Fall 2025\ABE Work\Rice segmentation\data\raw"
# Just place your photos here with proper naming
```

### Option B: Individual Folders (Better for Many Photos)

```
data/raw/
├── genotype_001/
│   ├── rep1.jpg
│   ├── rep2.jpg
│   └── rep3.jpg
├── genotype_002/
│   ├── rep1.jpg
│   ├── rep2.jpg
│   └── rep3.jpg
...
└── genotype_100/
    ├── rep1.jpg
    ├── rep2.jpg
    └── rep3.jpg
```

**Create with PowerShell:**
```powershell
# Create 100 genotype folders
cd "c:\Fall 2025\ABE Work\Rice segmentation\data\raw"
1..100 | ForEach-Object { 
    New-Item -ItemType Directory -Name ("genotype_{0:D3}" -f $_) -Force 
}
```

---

## 🔬 Analysis Approach

### Goals

1. **Rank genotypes** by seed quality metrics
2. **Identify top performers** (e.g., top 10-20 genotypes)
3. **Characterize variation** within and between genotypes
4. **Statistical comparison** to find significant differences

### Key Metrics Per Genotype

**From automated segmentation:**
- Mean seed count per photo
- Mean seed size (area)
- Seed size variability
- Mean seed shape (circularity, aspect ratio)
- Fill rate (if distinguishable)

**Statistical measures:**
- Mean ± standard deviation (from 2-3 replicates)
- Coefficient of variation (CV%)
- Confidence intervals

---

## 📊 Processing Workflow

### Step 1: Test Processing (10 minutes)

Test on first 10 genotypes:

```powershell
# Process first genotype as test
python scripts/batch_process_samples.py --test-mode --max-images 3
```

Check if segmentation quality is good.

### Step 2: Batch Process All (30-60 minutes)

```powershell
# Process all photos
python scripts/genotype_screening.py --input data/raw --output results/genotype_screening
```

This will:
- Detect all genotypes automatically
- Process all replicates per genotype
- Calculate mean statistics per genotype
- Generate ranking table
- Create visualization plots

### Step 3: Analyze Results (30 minutes)

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/genotype_screening/genotype_summary.csv')

# Sort by fill rate or seed count
top_20 = df.nlargest(20, 'mean_seed_count')

print("Top 20 Genotypes:")
print(top_20[['genotype', 'mean_seed_count', 'std_seed_count', 'cv_percent']])

# Visualize
plt.figure(figsize=(15, 6))
plt.bar(range(20), top_20['mean_seed_count'], yerr=top_20['std_seed_count'])
plt.xlabel('Genotype Rank')
plt.ylabel('Mean Seed Count')
plt.title('Top 20 Rice Genotypes by Seed Count')
plt.xticks(range(20), top_20['genotype'], rotation=45)
plt.tight_layout()
plt.savefig('results/top_20_genotypes.png')
plt.show()
```

---

## 📈 Expected Outputs

### Summary Table (genotype_summary.csv)

| genotype | n_photos | mean_seed_count | std_seed_count | cv% | mean_seed_area | mean_circularity | fill_rate | rank |
|----------|----------|-----------------|----------------|-----|----------------|------------------|-----------|------|
| gen047 | 3 | 45.3 | 2.1 | 4.6 | 1250 | 0.82 | 85% | 1 |
| gen082 | 3 | 43.8 | 3.5 | 8.0 | 1180 | 0.79 | 82% | 2 |
| gen015 | 3 | 42.1 | 1.8 | 4.3 | 1220 | 0.81 | 81% | 3 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Visualizations

1. **Ranking plot** - Bar chart of all 100 genotypes sorted by performance
2. **Distribution plot** - Histogram showing spread of seed counts
3. **Top performers** - Detailed view of top 10-20 genotypes
4. **Quality metrics** - Scatter plots (size vs count, circularity vs count)
5. **Heatmap** - All genotypes × all metrics

---

## 🎯 Statistical Analysis

### Identify Significant Differences

```python
import scipy.stats as stats

# Load all individual photo results
df_all = pd.read_csv('results/genotype_screening/all_photos.csv')

# ANOVA across all genotypes
genotypes = df_all['genotype'].unique()
groups = [df_all[df_all['genotype'] == g]['seed_count'].values for g in genotypes]

f_stat, p_value = stats.f_oneway(*groups)
print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4e}")

if p_value < 0.05:
    print("✓ Significant differences exist among genotypes!")
    
# Identify top performers (e.g., top 10%)
threshold = df_summary['mean_seed_count'].quantile(0.90)
top_genotypes = df_summary[df_summary['mean_seed_count'] >= threshold]
print(f"\nTop 10% genotypes (n={len(top_genotypes)}):")
print(top_genotypes['genotype'].tolist())
```

---

## ⏱️ Timeline Estimate

| Phase | Task | Time | Cumulative |
|-------|------|------|------------|
| **1** | Photograph 100 genotypes (3 photos each) | 5-6 hrs | 6 hrs |
| **2** | Organize and name files | 1 hr | 7 hrs |
| **3** | Test processing (10 genotypes) | 30 min | 7.5 hrs |
| **4** | Batch process all 300 photos | 30-60 min | 8.5 hrs |
| **5** | Initial analysis and ranking | 1 hr | 9.5 hrs |
| **6** | Statistical analysis | 1 hr | 10.5 hrs |
| **7** | Generate figures and report | 2 hrs | 12.5 hrs |
| | **TOTAL** | **~12-13 hours** | |

**Spread over 3-4 days:** ~3-4 hours per day

---

## 💡 Strategic Decisions

### Annotation Strategy

With 100 genotypes, **don't manually annotate**! Instead:

**Option 1: No Manual Annotation** (Recommended)
- Use automated segmentation only
- Visually spot-check 10-20 photos
- Faster, still reliable for relative comparison

**Option 2: Minimal Annotation** (If concerned about accuracy)
- Manually annotate 1 photo from 10 representative genotypes
- Validate automated segmentation accuracy
- Adjust parameters if needed
- Time: ~1 hour

### Quality Control

**Automated QC flags:**
- Images with unusually high/low seed counts (outliers)
- Images with poor segmentation (review manually)
- High within-genotype variation (CV > 20%)

---

## 🚀 Next Steps

### Immediate Actions

1. **Decide on replication:**
   - [ ] 2 photos per genotype (faster, good)
   - [ ] 3 photos per genotype (recommended)

2. **Choose organization:**
   - [ ] Flat structure (data/raw/gen001_rep1.jpg)
   - [ ] Folder structure (data/raw/genotype_001/rep1.jpg)

3. **Start photography:**
   - [ ] Set up consistent photo station
   - [ ] Photograph all 100 genotypes
   - [ ] Use consistent naming

4. **Run analysis:**
   - [ ] Test on first 10 genotypes
   - [ ] Process all 300 photos
   - [ ] Generate ranking

---

## 📋 Deliverables

At the end, you'll have:

✅ **Ranking table** - All 100 genotypes sorted by performance  
✅ **Top performers list** - Best 10-20 genotypes identified  
✅ **Statistical report** - ANOVA results, confidence intervals  
✅ **Visualizations** - Bar charts, distributions, heatmaps  
✅ **Raw data** - Seed counts, sizes, shapes for all photos  
✅ **Recommendations** - Which genotypes to advance  

**Answer to your question:** "Which of these 100 rice lines should we focus on for breeding/further testing?"

---

## ❓ Questions to Clarify

1. Do you have genotype names/IDs already? (Or should we use gen001-gen100?)
2. What's your primary selection criterion?
   - Highest seed count?
   - Largest seeds?
   - Best fill rate?
   - Combination?
3. Have you started photographing yet?
4. Do you have access to the seeds now?

Let me know and I'll help you get started! 🌾