# Data Organization Guide

## Folder Structure for Variety Comparison

```
data/
├── raw/
│   ├── variety_A/          # Your first rice variety/genotype
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   ├── image_003.jpg
│   │   └── ... (10-20 images)
│   │
│   ├── variety_B/          # Second variety
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   │
│   ├── variety_C/          # Third variety
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   │
│   └── variety_D/          # Fourth variety (if applicable)
│       ├── image_001.jpg
│       ├── image_002.jpg
│       └── ...
│
└── processed/
    ├── variety_A/          # Processed results for variety A
    ├── variety_B/
    ├── variety_C/
    └── variety_D/
```

---

## Naming Your Folders

### Option 1: Generic Names (if you want to keep varieties anonymous)
```
variety_A/
variety_B/
variety_C/
variety_D/
```

### Option 2: Descriptive Names (recommended for easier tracking)
```
control/              # Your control/reference variety
genotype_1/          # Experimental genotype 1
genotype_2/          # Experimental genotype 2
hybrid_cross/        # Hybrid variety
```

### Option 3: Actual Variety Names
```
IR64/                # Common rice variety name
basmati/
jasmine/
local_variety/
```

---

## File Naming Within Each Folder

Keep it **simple and consistent** within each folder:

```
variety_A/
├── 001.jpg
├── 002.jpg
├── 003.jpg
├── 004.jpg
└── ...

variety_B/
├── 001.jpg
├── 002.jpg
├── 003.jpg
└── ...
```

Or slightly more descriptive:
```
variety_A/
├── varietyA_001.jpg
├── varietyA_002.jpg
├── varietyA_003.jpg
└── ...
```

---

## Benefits of Separate Folders

✅ **Easy to organize** - Know which images belong to which variety
✅ **Easy to process** - Can process one variety at a time
✅ **Easy to compare** - Can compare folder-to-folder results
✅ **Easy to add/remove** - Can easily add more varieties later
✅ **Clear documentation** - Anyone can understand your organization
✅ **Batch processing** - Can point scripts to specific folders

---

## How to Use in Your Analysis

### Process One Variety at a Time:
```python
# In the Jupyter notebook
variety_name = "variety_A"
input_dir = f"../data/raw/{variety_name}"

# Load all images from this folder
# Process and analyze
# Save results to processed/{variety_name}/
```

### Process All Varieties in a Loop:
```python
varieties = ['variety_A', 'variety_B', 'variety_C', 'variety_D']

for variety in varieties:
    input_dir = f"../data/raw/{variety}"
    output_dir = f"../data/processed/{variety}"
    
    # Process all images
    # Extract features
    # Save results
```

---

## Recommended Workflow

### Step 1: Prepare Physical Samples
- [ ] Separate your 3-4 rice varieties into labeled containers
- [ ] Decide on naming convention (variety_A, variety_B, etc.)

### Step 2: Set Up Photo Station
- [ ] White background
- [ ] Consistent lighting
- [ ] Camera/phone on stand

### Step 3: Photograph One Variety at a Time
- [ ] **Variety A**: Take 10-20 photos → Save to `data/raw/variety_A/`
- [ ] **Variety B**: Take 10-20 photos → Save to `data/raw/variety_B/`
- [ ] **Variety C**: Take 10-20 photos → Save to `data/raw/variety_C/`
- [ ] **Variety D**: Take 10-20 photos → Save to `data/raw/variety_D/`

### Step 4: Create a Sample Metadata File (Optional but Recommended)
Create `data/sample_metadata.txt`:
```
variety_A: IR64 genotype, collected 2025-10-03
variety_B: Local variety, collected 2025-10-03
variety_C: Hybrid cross A×B, collected 2025-10-03
variety_D: Control sample, collected 2025-10-03
```

---

## Tips for Success

1. **Label physical samples clearly** before starting
2. **Photograph one variety completely** before moving to the next
3. **Use consistent setup** - same background, lighting, camera height
4. **Keep notes** - which folder corresponds to which actual sample
5. **Back up images** before processing (copy to another location)

---

## Example Session

```
Day 1: Setup and Variety A
- Set up photo station
- Photograph variety A (15 images)
- Transfer to data/raw/variety_A/
- Test run one image through notebook

Day 2: Varieties B & C
- Photograph variety B (15 images)
- Photograph variety C (15 images)
- Transfer to respective folders

Day 3: Variety D & Processing
- Photograph variety D (15 images)
- Begin batch processing all varieties

Day 4: Analysis
- Compare results across varieties
- Generate comparison plots
- Identify best performing variety
```

---

## Current Folder Structure (Ready to Use!)

I've already created these folders for you:
```
✅ data/raw/variety_A/
✅ data/raw/variety_B/
✅ data/raw/variety_C/
✅ data/raw/variety_D/
```

Just start adding your images to these folders!

---

## Quick Reference

| Folder | Purpose | Content |
|--------|---------|---------|
| `variety_A/` | First rice variety | 10-20 images of seeds |
| `variety_B/` | Second variety | 10-20 images of seeds |
| `variety_C/` | Third variety | 10-20 images of seeds |
| `variety_D/` | Fourth variety | 10-20 images of seeds |

**Total images needed: 40-80** (10-20 per variety × 4 varieties)

Good luck organizing your data! 📁🌾