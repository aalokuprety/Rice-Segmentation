#!/usr/bin/env python3
"""
Rice Segmentation Evaluation Metrics

This module provides comprehensive evaluation metrics for rice seed segmentation,
including IoU, Dice coefficient, precision, recall, and visualization tools.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
import seaborn as sns


class SegmentationEvaluator:
    """
    Comprehensive evaluation class for rice seed segmentation results.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_iou(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Intersection over Union (IoU) between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            IoU score (0-1)
        """
        # Ensure masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        # Calculate intersection and union
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        union = np.logical_or(pred_binary, gt_binary).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def calculate_dice_coefficient(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Dice coefficient between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Dice coefficient (0-1)
        """
        # Ensure masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        intersection = np.logical_and(pred_binary, gt_binary).sum()
        total_pixels = pred_binary.sum() + gt_binary.sum()
        
        if total_pixels == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return 2.0 * intersection / total_pixels
    
    def calculate_precision_recall(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
        """
        Calculate precision and recall between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Tuple of (precision, recall)
        """
        # Ensure masks are binary
        pred_binary = (pred_mask > 0).astype(np.uint8)
        gt_binary = (gt_mask > 0).astype(np.uint8)
        
        true_positive = np.logical_and(pred_binary, gt_binary).sum()
        false_positive = np.logical_and(pred_binary, ~gt_binary.astype(bool)).sum()
        false_negative = np.logical_and(~pred_binary.astype(bool), gt_binary).sum()
        
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        
        return precision, recall
    
    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_hausdorff_distance(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """
        Calculate Hausdorff distance between predicted and ground truth contours.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        # Find contours
        pred_contours, _ = cv2.findContours(pred_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not pred_contours or not gt_contours:
            return float('inf')
        
        # Get all contour points
        pred_points = np.vstack([contour.reshape(-1, 2) for contour in pred_contours])
        gt_points = np.vstack([contour.reshape(-1, 2) for contour in gt_contours])
        
        # Calculate distances
        def directed_hausdorff(points1, points2):
            distances = []
            for p1 in points1:
                min_dist = min(np.linalg.norm(p1 - p2) for p2 in points2)
                distances.append(min_dist)
            return max(distances) if distances else 0
        
        d1 = directed_hausdorff(pred_points, gt_points)
        d2 = directed_hausdorff(gt_points, pred_points)
        
        return max(d1, d2)
    
    def evaluate_segmentation(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation of segmentation results.
        
        Args:
            pred_mask: Predicted binary mask
            gt_mask: Ground truth binary mask
            
        Returns:
            Dictionary with all evaluation metrics
        """
        iou = self.calculate_iou(pred_mask, gt_mask)
        dice = self.calculate_dice_coefficient(pred_mask, gt_mask)
        precision, recall = self.calculate_precision_recall(pred_mask, gt_mask)
        f1 = self.calculate_f1_score(precision, recall)
        
        try:
            hausdorff = self.calculate_hausdorff_distance(pred_mask, gt_mask)
        except:
            hausdorff = float('inf')
        
        return {
            'iou': iou,
            'dice_coefficient': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'hausdorff_distance': hausdorff
        }
    
    def evaluate_object_detection(self, pred_labels: np.ndarray, gt_labels: np.ndarray, 
                                iou_threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate object detection performance (individual seed detection).
        
        Args:
            pred_labels: Predicted labeled image
            gt_labels: Ground truth labeled image
            iou_threshold: IoU threshold for considering a detection as correct
            
        Returns:
            Dictionary with detection metrics
        """
        pred_props = measure.regionprops(pred_labels)
        gt_props = measure.regionprops(gt_labels)
        
        # Create binary masks for each object
        pred_objects = []
        for prop in pred_props:
            mask = np.zeros_like(pred_labels, dtype=np.uint8)
            mask[pred_labels == prop.label] = 1
            pred_objects.append(mask)
        
        gt_objects = []
        for prop in gt_props:
            mask = np.zeros_like(gt_labels, dtype=np.uint8)
            mask[gt_labels == prop.label] = 1
            gt_objects.append(mask)
        
        # Match predicted objects to ground truth
        matched_pred = set()
        matched_gt = set()
        
        for i, pred_obj in enumerate(pred_objects):
            best_iou = 0
            best_match = -1
            
            for j, gt_obj in enumerate(gt_objects):
                if j in matched_gt:
                    continue
                
                iou = self.calculate_iou(pred_obj, gt_obj)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_match = j
            
            if best_match >= 0:
                matched_pred.add(i)
                matched_gt.add(best_match)
        
        # Calculate detection metrics
        true_positives = len(matched_pred)
        false_positives = len(pred_objects) - true_positives
        false_negatives = len(gt_objects) - len(matched_gt)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = self.calculate_f1_score(precision, recall)
        
        return {
            'detection_precision': precision,
            'detection_recall': recall,
            'detection_f1': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'predicted_count': len(pred_objects),
            'ground_truth_count': len(gt_objects)
        }
    
    def create_comparison_visualization(self, original_image: np.ndarray, 
                                      pred_mask: np.ndarray, gt_mask: np.ndarray,
                                      metrics: Dict[str, float], title: str = "Segmentation Comparison"):
        """
        Create a visualization comparing predicted and ground truth segmentations.
        
        Args:
            original_image: Original input image
            pred_mask: Predicted segmentation mask
            gt_mask: Ground truth mask
            metrics: Evaluation metrics dictionary
            title: Title for the visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray' if len(original_image.shape) == 2 else None)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('Prediction')
        axes[0, 2].axis('off')
        
        # Overlay comparison
        overlay = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
        overlay[gt_mask > 0] = [0, 255, 0]  # Ground truth in green
        overlay[pred_mask > 0] = [255, 0, 0]  # Prediction in red
        overlay[np.logical_and(gt_mask > 0, pred_mask > 0)] = [255, 255, 0]  # Overlap in yellow
        
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay (GT: Green, Pred: Red, Overlap: Yellow)')
        axes[1, 0].axis('off')
        
        # Difference visualization
        difference = np.zeros_like(gt_mask, dtype=np.int16)
        difference[np.logical_and(gt_mask > 0, pred_mask == 0)] = 1  # False negative
        difference[np.logical_and(gt_mask == 0, pred_mask > 0)] = -1  # False positive
        
        axes[1, 1].imshow(difference, cmap='RdBu', vmin=-1, vmax=1)
        axes[1, 1].set_title('Errors (Blue: FN, Red: FP)')
        axes[1, 1].axis('off')
        
        # Metrics text
        axes[1, 2].axis('off')
        metrics_text = f"""
        Evaluation Metrics:
        
        IoU: {metrics.get('iou', 0):.3f}
        Dice: {metrics.get('dice_coefficient', 0):.3f}
        Precision: {metrics.get('precision', 0):.3f}
        Recall: {metrics.get('recall', 0):.3f}
        F1 Score: {metrics.get('f1_score', 0):.3f}
        """
        axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def create_metrics_comparison_plot(self, results_dict: Dict[str, Dict[str, float]], 
                                     title: str = "Method Comparison"):
        """
        Create a bar plot comparing metrics across different methods.
        
        Args:
            results_dict: Dictionary where keys are method names and values are metrics dictionaries
            title: Title for the plot
        """
        # Convert to DataFrame for easier plotting
        df_data = []
        for method, metrics in results_dict.items():
            for metric, value in metrics.items():
                df_data.append({'Method': method, 'Metric': metric, 'Value': value})
        
        df = pd.DataFrame(df_data)
        
        # Create subplot for each metric
        metrics_to_plot = ['iou', 'dice_coefficient', 'precision', 'recall', 'f1_score']
        available_metrics = [m for m in metrics_to_plot if m in df['Metric'].values]
        
        if not available_metrics:
            print("No common metrics found for comparison")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            metric_data = df[df['Metric'] == metric]
            
            axes[i].bar(metric_data['Method'], metric_data['Value'])
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, (method, value) in enumerate(zip(metric_data['Method'], metric_data['Value'])):
                axes[i].text(j, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def evaluate_batch_results(self, predictions_dir: Union[str, Path], 
                             ground_truth_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Evaluate a batch of segmentation results.
        
        Args:
            predictions_dir: Directory containing predicted segmentation masks
            ground_truth_dir: Directory containing ground truth masks
            
        Returns:
            DataFrame with evaluation results for each image
        """
        predictions_dir = Path(predictions_dir)
        ground_truth_dir = Path(ground_truth_dir)
        
        results = []
        
        # Find matching files
        pred_files = list(predictions_dir.glob("*.png")) + list(predictions_dir.glob("*.jpg"))
        
        for pred_file in pred_files:
            gt_file = ground_truth_dir / pred_file.name
            
            if not gt_file.exists():
                print(f"Ground truth not found for {pred_file.name}")
                continue
            
            # Load masks
            pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            
            if pred_mask is None or gt_mask is None:
                print(f"Could not load masks for {pred_file.name}")
                continue
            
            # Evaluate
            metrics = self.evaluate_segmentation(pred_mask, gt_mask)
            metrics['image_name'] = pred_file.name
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def create_batch_evaluation_report(self, results_df: pd.DataFrame, 
                                     save_path: Optional[Union[str, Path]] = None):
        """
        Create a comprehensive evaluation report for batch results.
        
        Args:
            results_df: DataFrame with evaluation results
            save_path: Optional path to save the report
        """
        if results_df.empty:
            print("No results to evaluate")
            return
        
        # Calculate summary statistics
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        summary_stats = results_df[numeric_columns].describe()
        
        print("Batch Evaluation Report")
        print("=" * 50)
        print(f"Total images evaluated: {len(results_df)}")
        print(f"Mean IoU: {results_df['iou'].mean():.3f} ± {results_df['iou'].std():.3f}")
        print(f"Mean Dice: {results_df['dice_coefficient'].mean():.3f} ± {results_df['dice_coefficient'].std():.3f}")
        print(f"Mean F1: {results_df['f1_score'].mean():.3f} ± {results_df['f1_score'].std():.3f}")
        
        print("\nDetailed Statistics:")
        print(summary_stats.round(3))
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IoU distribution
        axes[0, 0].hist(results_df['iou'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('IoU Distribution')
        axes[0, 0].set_xlabel('IoU')
        axes[0, 0].set_ylabel('Count')
        
        # Dice coefficient distribution
        axes[0, 1].hist(results_df['dice_coefficient'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Dice Coefficient Distribution')
        axes[0, 1].set_xlabel('Dice Coefficient')
        axes[0, 1].set_ylabel('Count')
        
        # Precision vs Recall scatter plot
        axes[1, 0].scatter(results_df['precision'], results_df['recall'], alpha=0.6)
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Correlation heatmap
        corr_matrix = results_df[numeric_columns].corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        axes[1, 1].set_yticklabels(corr_matrix.columns)
        axes[1, 1].set_title('Metric Correlations')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1])
        
        # Add correlation values
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = axes[1, 1].text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Save report if requested
        if save_path is not None:
            save_path = Path(save_path)
            results_df.to_csv(save_path / 'evaluation_results.csv', index=False)
            summary_stats.to_csv(save_path / 'summary_statistics.csv')
            plt.savefig(save_path / 'evaluation_plots.png', dpi=300, bbox_inches='tight')
            print(f"Report saved to {save_path}")


# Utility functions for creating synthetic ground truth
def create_synthetic_ground_truth(original_image: np.ndarray, 
                                manual_annotations: Optional[List[Dict]] = None) -> np.ndarray:
    """
    Create synthetic ground truth for testing evaluation metrics.
    
    Args:
        original_image: Original input image
        manual_annotations: List of manual annotations (optional)
        
    Returns:
        Synthetic ground truth mask
    """
    h, w = original_image.shape[:2]
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    
    if manual_annotations is None:
        # Create some synthetic rice-like shapes
        num_seeds = np.random.randint(5, 15)
        
        for i in range(num_seeds):
            # Random center
            center_x = np.random.randint(50, w - 50)
            center_y = np.random.randint(50, h - 50)
            
            # Random size (rice-like proportions)
            width = np.random.randint(20, 40)
            height = np.random.randint(40, 80)
            
            # Random rotation
            angle = np.random.randint(0, 180)
            
            # Create ellipse
            axes_length = (width // 2, height // 2)
            cv2.ellipse(gt_mask, (center_x, center_y), axes_length, angle, 0, 360, 255, -1)
    else:
        # Use manual annotations
        for annotation in manual_annotations:
            if annotation['type'] == 'ellipse':
                cv2.ellipse(gt_mask, 
                           (annotation['center_x'], annotation['center_y']),
                           (annotation['width'] // 2, annotation['height'] // 2),
                           annotation['angle'], 0, 360, 255, -1)
            elif annotation['type'] == 'polygon':
                points = np.array(annotation['points'], dtype=np.int32)
                cv2.fillPoly(gt_mask, [points], 255)
    
    return gt_mask


if __name__ == "__main__":
    print("Rice Segmentation Evaluation Metrics")
    print("=" * 40)
    print("This module provides comprehensive evaluation tools for rice seed segmentation:")
    print("- IoU (Intersection over Union)")
    print("- Dice coefficient")
    print("- Precision and Recall")
    print("- F1 score")
    print("- Hausdorff distance")
    print("- Object detection metrics")
    print("- Batch evaluation capabilities")
    print("- Visualization tools")
    print("\nExample usage:")
    print("evaluator = SegmentationEvaluator()")
    print("metrics = evaluator.evaluate_segmentation(predicted_mask, ground_truth_mask)")
    print("evaluator.create_comparison_visualization(image, pred, gt, metrics)")