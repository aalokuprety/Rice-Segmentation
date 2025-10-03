# Image Acquisition Guidelines for Rice Seed Segmentation

## Photography Setup

### 1. **Background**
- ✅ **Use a contrasting, uniform background**
  - Dark background (black/dark blue) works best for white/light rice
  - White/light background works best for dark/colored rice varieties
  - Matte finish preferred (avoid glossy surfaces that create reflections)
  - Consider using a lightbox or photography backdrop

### 2. **Lighting**
- ✅ **Diffused, even lighting** is critical
  - Use soft, indirect lighting (avoid harsh shadows)
  - Position lights at 45° angles from both sides
  - Consider using a ring light or LED panel for even illumination
  - Avoid direct sunlight (too harsh and creates strong shadows)
  - Natural daylight through a window with diffuser is acceptable
  
- ❌ **Avoid:**
  - Harsh shadows
  - Specular reflections/glare on seeds
  - Uneven lighting across the image

### 3. **Camera Setup**
- **Distance**: 30-50 cm above the seeds (for smartphone) or adjust for your camera
- **Angle**: Directly overhead (90° angle, perpendicular to surface)
- **Focus**: Ensure all seeds are in sharp focus
- **Resolution**: Minimum 1920x1080 pixels (higher is better)
- **Format**: JPEG or PNG (avoid heavy compression)

### 4. **Rice Seed Arrangement**
- ✅ **Recommended layout:**
  - Spread seeds evenly, avoid overlapping
  - Leave some space between seeds (easier for segmentation)
  - Don't overcrowd the image (10-50 seeds per image is ideal)
  - Can include touching seeds to test watershed algorithm
  
- 📏 **Spacing:**
  - Minimum 2-3mm between seeds when possible
  - Mix of separated and slightly touching seeds is good for testing

### 5. **Image Quality Checklist**
- [ ] High contrast between seeds and background
- [ ] All seeds in sharp focus
- [ ] Even lighting (no dark corners or bright spots)
- [ ] No shadows obscuring seed boundaries
- [ ] Seeds clearly visible (not blurry)
- [ ] Consistent scale/distance across all images

---

## How Many Images to Capture?

### **Minimum Dataset: 10-20 images**
Good for initial testing and method development

### **Recommended Dataset: 50-100 images**
Better for:
- Testing robustness across different conditions
- Statistical analysis of results
- Method comparison and validation
- Training if you plan to use machine learning later

### **Comprehensive Dataset: 200+ images**
Ideal for:
- Publication-quality research
- Machine learning/deep learning approaches
- Multiple rice varieties
- Various environmental conditions

---

## Dataset Diversity (Important!)

### **Vary these conditions across your images:**

1. **Rice Varieties** (if applicable)
   - Different sizes
   - Different colors
   - Different shapes

2. **Seed Count per Image**
   - Some with 10-15 seeds
   - Some with 30-50 seeds
   - Some with 50-100 seeds

3. **Arrangement**
   - Well-separated seeds
   - Some touching seeds (to test separation)
   - Different orientations

4. **Image Conditions**
   - Different lighting angles (slight variations)
   - Morning vs afternoon light
   - Different background shades (if testing robustness)

---

## Sample Image Capture Protocol

### **Session 1: Baseline Images (15-20 images)**
- Optimal lighting
- Dark background
- 20-30 seeds per image
- Well-separated seeds

### **Session 2: Challenging Images (10-15 images)**
- Some seeds touching
- Higher density (40-60 seeds)
- Test different orientations

### **Session 3: Variety Testing (10-15 images)**
- Different rice varieties (if available)
- Different seed conditions (whole, broken, etc.)

---

## Equipment Recommendations

### **Budget Option (Smartphone)**
- Modern smartphone with good camera (12MP+)
- Tripod or phone stand
- White poster board or black fabric for background
- Desk lamp with diffuser or natural window light
- Total cost: $0-50 (if you have a phone)

### **Better Quality Setup**
- DSLR or mirrorless camera
- Copy stand or overhead tripod
- LED ring light or softbox lights
- Colored backdrop paper
- Total cost: $200-500

### **Professional Setup**
- High-resolution camera with macro lens
- Lightbox or professional lighting rig
- Motorized scanning platform
- Calibrated color standards
- Total cost: $1000+

---

## Quick Start Photography Tips

1. **Start Simple**: Use your smartphone on a black piece of paper under good light
2. **Test First**: Take 5-10 test images and run them through the segmentation notebook
3. **Adjust**: Based on results, improve lighting, background, or arrangement
4. **Batch Capture**: Once you have good settings, capture all images in one session for consistency
5. **Organize**: Name files systematically (e.g., `rice_variety_001.jpg`, `rice_variety_002.jpg`)

---

## Example File Naming Convention

```
rice_white_well_separated_001.jpg
rice_white_well_separated_002.jpg
rice_white_touching_001.jpg
rice_white_high_density_001.jpg
rice_brown_well_separated_001.jpg
```

Or simpler:
```
sample_001.jpg
sample_002.jpg
sample_003.jpg
```

---

## Post-Capture Checklist

Before running segmentation:
- [ ] Check all images are in focus
- [ ] Verify consistent lighting across dataset
- [ ] Ensure good contrast in all images
- [ ] Remove any blurry or poor quality images
- [ ] Resize if necessary (2-4 megapixels is usually sufficient)
- [ ] Organize in `data/raw/` directory

---

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Seeds look washed out | Reduce exposure or move lights further away |
| Too many shadows | Add fill light from opposite side |
| Reflections on seeds | Use diffused lighting or polarizing filter |
| Background too similar to seeds | Change background color/material |
| Blurry images | Use tripod, increase shutter speed, or improve focus |
| Inconsistent colors | Use manual white balance or color calibration card |

---

## Ready to Start?

**Recommended Starting Point:**
1. Gather 30-50 rice seeds
2. Place on dark construction paper or fabric
3. Set up your smartphone on a stack of books or simple stand
4. Use desk lamp with white paper as diffuser
5. Take 10-15 test images
6. Run through the segmentation notebook
7. Adjust and capture more images as needed

**Good luck with your image acquisition! 📷🌾**