#!/usr/bin/env python3
"""
LabelMe Annotation Helper

Utility functions for working with LabelMe annotations in the rice segmentation project.
Converts LabelMe JSON files to masks, counts objects, and calculates quality metrics.
"""

import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt


class LabelMeAnnotationProcessor:
    """Process LabelMe annotations for rice seed analysis."""
    
    def __init__(self):
        """Initialize the annotation processor."""
        self.valid_labels = ['filled_seed', 'empty_husk', 'broken_seed', 'trash', 'seed']
    
    def load_annotation(self, json_path: str) -> Dict:
        """
        Load LabelMe JSON annotation file.
        
        Args:
            json_path: Path to LabelMe JSON file
            
        Returns:
            Dictionary with annotation data
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def annotation_to_mask(self, json_path: str, label_filter: Optional[List[str]] = None) -> np.ndarray:
        """
        Convert LabelMe annotation to binary mask.
        
        Args:
            json_path: Path to LabelMe JSON file
            label_filter: List of labels to include (None = all except trash)
            
        Returns:
            Binary mask as numpy array
        """
        annotation = self.load_annotation(json_path)
        
        # Get image dimensions
        height = annotation['imageHeight']
        width = annotation['imageWidth']
        
        # Create blank mask
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Default filter: include seeds but not trash
        if label_filter is None:
            label_filter = ['filled_seed', 'empty_husk', 'seed', 'broken_seed']
        
        # Draw polygons
        for shape in annotation['shapes']:
            if shape['label'] in label_filter:
                points = [tuple(pt) for pt in shape['points']]
                draw.polygon(points, fill=255)
        
        return np.array(mask)
    
    def annotation_to_labeled_mask(self, json_path: str) -> np.ndarray:
        """
        Convert LabelMe annotation to labeled mask (each object has unique ID).
        
        Args:
            json_path: Path to LabelMe JSON file
            
        Returns:
            Labeled mask with unique ID for each object
        """
        annotation = self.load_annotation(json_path)
        
        height = annotation['imageHeight']
        width = annotation['imageWidth']
        
        # Create blank mask
        labeled_mask = np.zeros((height, width), dtype=np.int32)
        
        # Draw each shape with unique label
        label_id = 1
        for shape in annotation['shapes']:
            if shape['label'] != 'trash' and shape['label'] != 'background':
                mask = Image.new('L', (width, height), 0)
                draw = ImageDraw.Draw(mask)
                points = [tuple(pt) for pt in shape['points']]
                draw.polygon(points, fill=255)
                
                # Add to labeled mask
                temp_mask = np.array(mask)
                labeled_mask[temp_mask > 0] = label_id
                label_id += 1
        
        return labeled_mask
    
    def count_objects_by_label(self, json_path: str) -> Dict[str, int]:
        """
        Count objects in annotation by label type.
        
        Args:
            json_path: Path to LabelMe JSON file
            
        Returns:
            Dictionary with counts for each label
        """
        annotation = self.load_annotation(json_path)
        
        counts = {}
        for shape in annotation['shapes']:
            label = shape['label']
            counts[label] = counts.get(label, 0) + 1
        
        return counts
    
    def calculate_quality_metrics(self, json_path: str) -> Dict[str, float]:
        """
        Calculate quality metrics from annotations.
        
        Args:
            json_path: Path to LabelMe JSON file
            
        Returns:
            Dictionary with quality metrics
        """
        counts = self.count_objects_by_label(json_path)
        
        filled = counts.get('filled_seed', 0) + counts.get('seed', 0)
        empty = counts.get('empty_husk', 0)
        broken = counts.get('broken_seed', 0)
        trash = counts.get('trash', 0)
        
        total_seeds = filled + empty + broken
        total_objects = total_seeds + trash
        
        metrics = {
            'total_objects': total_objects,
            'filled_seeds': filled,
            'empty_husks': empty,
            'broken_seeds': broken,
            'trash_count': trash,
            'total_seeds': total_seeds,
            'fill_rate': (filled / total_seeds * 100) if total_seeds > 0 else 0,
            'cleanliness': (total_seeds / total_objects * 100) if total_objects > 0 else 0,
            'viability': (filled / (filled + empty) * 100) if (filled + empty) > 0 else 0
        }
        
        return metrics
    
    def process_directory(self, directory: str) -> pd.DataFrame:
        """
        Process all LabelMe annotations in a directory.
        
        Args:
            directory: Path to directory with images and JSON files
            
        Returns:
            DataFrame with metrics for each image
        """
        directory = Path(directory)
        json_files = list(directory.glob('*.json'))
        
        results = []
        for json_file in json_files:
            try:
                metrics = self.calculate_quality_metrics(str(json_file))
                metrics['image_name'] = json_file.stem
                metrics['json_path'] = str(json_file)
                results.append(metrics)
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        return pd.DataFrame(results)
    
    def compare_with_automated(self, json_path: str, automated_labels: np.ndarray) -> Dict[str, float]:
        """
        Compare manual annotations with automated segmentation.
        
        Args:
            json_path: Path to LabelMe annotation
            automated_labels: Labeled mask from automated segmentation
            
        Returns:
            Dictionary with comparison metrics
        """
        # Get ground truth mask
        gt_mask = self.annotation_to_mask(json_path)
        
        # Convert automated labels to binary mask
        pred_mask = (automated_labels > 0).astype(np.uint8) * 255
        
        # Calculate metrics
        intersection = np.logical_and(gt_mask > 0, pred_mask > 0).sum()
        union = np.logical_or(gt_mask > 0, pred_mask > 0).sum()
        
        iou = intersection / union if union > 0 else 0
        
        # Dice coefficient
        dice = 2 * intersection / (gt_mask.sum() + pred_mask.sum()) if (gt_mask.sum() + pred_mask.sum()) > 0 else 0
        
        # Count comparison
        gt_counts = self.count_objects_by_label(json_path)
        gt_seed_count = sum(v for k, v in gt_counts.items() if k != 'trash')
        
        from skimage import measure
        pred_seed_count = len(measure.regionprops(automated_labels))
        
        count_error = abs(gt_seed_count - pred_seed_count)
        count_accuracy = (1 - count_error / gt_seed_count) * 100 if gt_seed_count > 0 else 0
        
        return {
            'iou': iou,
            'dice_coefficient': dice,
            'manual_count': gt_seed_count,
            'automated_count': pred_seed_count,
            'count_error': count_error,
            'count_accuracy': count_accuracy
        }
    
    def visualize_annotation(self, image_path: str, json_path: str, save_path: Optional[str] = None):
        """
        Visualize image with annotation overlay.
        
        Args:
            image_path: Path to original image
            json_path: Path to LabelMe annotation
            save_path: Optional path to save visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotation
        annotation = self.load_annotation(json_path)
        
        # Create overlay
        overlay = image.copy()
        
        # Color map for different labels
        color_map = {
            'filled_seed': (0, 255, 0),      # Green
            'empty_husk': (255, 255, 0),     # Yellow
            'broken_seed': (255, 165, 0),    # Orange
            'trash': (255, 0, 0),            # Red
            'seed': (0, 255, 0)              # Green
        }
        
        # Draw each shape
        for shape in annotation['shapes']:
            label = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            color = color_map.get(label, (128, 128, 128))
            
            # Draw filled polygon
            cv2.fillPoly(overlay, [points], color)
            
            # Draw outline
            cv2.polylines(overlay, [points], True, (255, 255, 255), 2)
        
        # Blend overlay with original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        # Add legend
        y_offset = 30
        for label, color in color_map.items():
            cv2.rectangle(result, (10, y_offset-20), (30, y_offset), color, -1)
            cv2.putText(result, label, (40, y_offset-5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_offset += 30
        
        # Display or save
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.title(f'Annotation: {Path(image_path).name}')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def create_summary_report(self, directory: str, variety_name: str) -> pd.DataFrame:
        """
        Create summary report for all annotations in a directory.
        
        Args:
            directory: Path to directory with annotations
            variety_name: Name of the rice variety
            
        Returns:
            DataFrame with summary statistics
        """
        results_df = self.process_directory(directory)
        
        if results_df.empty:
            print(f"No annotations found in {directory}")
            return pd.DataFrame()
        
        # Calculate summary statistics
        summary = {
            'variety': variety_name,
            'images_annotated': len(results_df),
            'total_objects': results_df['total_objects'].sum(),
            'total_filled_seeds': results_df['filled_seeds'].sum(),
            'total_empty_husks': results_df['empty_husks'].sum(),
            'total_trash': results_df['trash_count'].sum(),
            'avg_fill_rate': results_df['fill_rate'].mean(),
            'std_fill_rate': results_df['fill_rate'].std(),
            'avg_cleanliness': results_df['cleanliness'].mean(),
            'avg_seeds_per_image': results_df['total_seeds'].mean()
        }
        
        print(f"\n{'='*50}")
        print(f"Summary Report: {variety_name}")
        print(f"{'='*50}")
        print(f"Images annotated: {summary['images_annotated']}")
        print(f"Total seeds: {summary['total_filled_seeds'] + summary['total_empty_husks']}")
        print(f"Filled seeds: {summary['total_filled_seeds']}")
        print(f"Empty husks: {summary['total_empty_husks']}")
        print(f"Trash pieces: {summary['total_trash']}")
        print(f"Average fill rate: {summary['avg_fill_rate']:.2f}%")
        print(f"Average cleanliness: {summary['avg_cleanliness']:.2f}%")
        print(f"{'='*50}\n")
        
        return pd.DataFrame([summary])


def compare_varieties(base_dir: str, varieties: List[str]) -> pd.DataFrame:
    """
    Compare multiple varieties based on their annotations.
    
    Args:
        base_dir: Base directory containing variety folders
        varieties: List of variety folder names
        
    Returns:
        DataFrame with comparison results
    """
    processor = LabelMeAnnotationProcessor()
    base_path = Path(base_dir)
    
    all_summaries = []
    for variety in varieties:
        variety_dir = base_path / variety
        if variety_dir.exists():
            summary = processor.create_summary_report(str(variety_dir), variety)
            if not summary.empty:
                all_summaries.append(summary)
        else:
            print(f"Warning: Directory not found: {variety_dir}")
    
    if all_summaries:
        comparison_df = pd.concat(all_summaries, ignore_index=True)
        
        # Sort by fill rate
        comparison_df = comparison_df.sort_values('avg_fill_rate', ascending=False)
        
        print("\n" + "="*70)
        print("VARIETY COMPARISON REPORT")
        print("="*70)
        print(comparison_df.to_string(index=False))
        print("="*70 + "\n")
        
        return comparison_df
    else:
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    print("LabelMe Annotation Helper for Rice Segmentation")
    print("="*50)
    print("\nExample usage:")
    print("\n1. Process single annotation:")
    print("   processor = LabelMeAnnotationProcessor()")
    print("   metrics = processor.calculate_quality_metrics('image_001.json')")
    print("\n2. Process entire variety:")
    print("   df = processor.process_directory('data/raw/variety_A')")
    print("\n3. Compare varieties:")
    print("   comparison = compare_varieties('data/raw', ['variety_A', 'variety_B', 'variety_C'])")
    print("\n4. Visualize annotation:")
    print("   processor.visualize_annotation('image.jpg', 'image.json')")
    print("="*50)