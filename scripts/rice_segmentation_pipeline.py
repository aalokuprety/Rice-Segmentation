#!/usr/bin/env python3
"""
Rice Seeds Segmentation Pipeline

Complete end-to-end pipeline for processing rice seed images, from loading to 
final segmented output with comprehensive analysis and evaluation.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime
import traceback

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent))

from rice_segmentation_methods import RiceSeedSegmenter
from evaluation_metrics import SegmentationEvaluator


class RiceSegmentationPipeline:
    """
    Complete pipeline for rice seed segmentation and analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self._load_config(config_path)
        self.segmenter = RiceSeedSegmenter(
            min_seed_area=self.config['segmentation']['min_seed_area'],
            max_seed_area=self.config['segmentation']['max_seed_area'],
            min_aspect_ratio=self.config['segmentation']['min_aspect_ratio'],
            max_aspect_ratio=self.config['segmentation']['max_aspect_ratio']
        )
        self.evaluator = SegmentationEvaluator()
        self._setup_logging()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "segmentation": {
                "min_seed_area": 50,
                "max_seed_area": 5000,
                "min_aspect_ratio": 1.5,
                "max_aspect_ratio": 6.0,
                "default_method": "watershed"
            },
            "evaluation": {
                "iou_threshold": 0.5,
                "enable_evaluation": False,
                "ground_truth_dir": None
            },
            "output": {
                "save_intermediate": True,
                "save_visualizations": True,
                "save_features": True,
                "output_format": "png"
            },
            "processing": {
                "batch_size": 10,
                "parallel": False,
                "show_progress": True
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge configurations
                for section, values in user_config.items():
                    if section in default_config:
                        default_config[section].update(values)
                    else:
                        default_config[section] = values
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def process_single_image(self, image_path: Union[str, Path], 
                           method: str = None, output_dir: Optional[Path] = None) -> Dict:
        """
        Process a single image through the complete pipeline.
        
        Args:
            image_path: Path to input image
            method: Segmentation method to use
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with processing results
        """
        image_path = Path(image_path)
        method = method or self.config['segmentation']['default_method']
        
        if output_dir is None:
            output_dir = Path("results") / image_path.stem
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger.info(f"Processing {image_path.name} with {method} method")
            
            # Run segmentation
            results = self.segmenter.segment_image(
                image_path, 
                method=method, 
                save_results=self.config['output']['save_intermediate'],
                output_dir=output_dir
            )
            
            # Add metadata
            results['image_path'] = str(image_path)
            results['processing_time'] = datetime.now().isoformat()
            results['config'] = self.config
            
            # Save comprehensive results
            if self.config['output']['save_features'] and not results['features'].empty:
                features_path = output_dir / 'detailed_features.csv'
                results['features'].to_csv(features_path, index=False)
                self.logger.info(f"Features saved to {features_path}")
            
            # Create visualizations
            if self.config['output']['save_visualizations']:
                self._create_result_visualization(results, output_dir)
            
            # Evaluation if ground truth is available
            if self.config['evaluation']['enable_evaluation']:
                evaluation_results = self._evaluate_result(results, image_path)
                results['evaluation'] = evaluation_results
            
            # Save processing report
            self._save_processing_report(results, output_dir)
            
            self.logger.info(f"Successfully processed {image_path.name}: {results['seed_count']} seeds detected")
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                'image_path': str(image_path),
                'error': str(e),
                'seed_count': 0,
                'success': False
            }
    
    def process_batch(self, input_dir: Union[str, Path], 
                     output_dir: Union[str, Path] = None,
                     methods: List[str] = None) -> pd.DataFrame:
        """
        Process multiple images in batch.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            methods: List of methods to compare (optional)
            
        Returns:
            DataFrame with batch processing results
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir) if output_dir else Path("batch_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
            image_files.extend(input_dir.glob(ext.upper()))
        
        if not image_files:
            self.logger.warning(f"No images found in {input_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(image_files)} images for batch processing")
        
        # Default method if none specified
        if methods is None:
            methods = [self.config['segmentation']['default_method']]
        
        all_results = []
        method_comparison = {}
        
        # Process with each method
        for method in methods:
            self.logger.info(f"Processing batch with {method} method")
            method_results = []
            
            for i, image_path in enumerate(image_files):
                if self.config['processing']['show_progress']:
                    self.logger.info(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
                
                # Create method-specific output directory
                image_output_dir = output_dir / method / image_path.stem
                
                result = self.process_single_image(image_path, method, image_output_dir)
                result['method'] = method
                method_results.append(result)
            
            method_comparison[method] = method_results
            all_results.extend(method_results)
        
        # Convert to DataFrame
        results_df = self._compile_batch_results(all_results)
        
        # Save batch summary
        batch_summary_path = output_dir / 'batch_summary.csv'
        results_df.to_csv(batch_summary_path, index=False)
        
        # Create comparison report if multiple methods
        if len(methods) > 1:
            self._create_method_comparison_report(method_comparison, output_dir)
        
        # Create overall batch report
        self._create_batch_report(results_df, output_dir)
        
        self.logger.info(f"Batch processing complete. Results saved to {output_dir}")
        return results_df
    
    def _create_result_visualization(self, results: Dict, output_dir: Path):
        """Create comprehensive visualization of segmentation results."""\n        try:\n            fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n            \n            # Original image\n            axes[0, 0].imshow(results['original_image'])\n            axes[0, 0].set_title('Original Image')\n            axes[0, 0].axis('off')\n            \n            # Preprocessed image\n            axes[0, 1].imshow(results['preprocessed_image'], cmap='gray')\n            axes[0, 1].set_title('Preprocessed')\n            axes[0, 1].axis('off')\n            \n            # Binary image (if available)\n            if results['binary_image'] is not None:\n                axes[0, 2].imshow(results['binary_image'], cmap='gray')\n                axes[0, 2].set_title('Binary Segmentation')\n            else:\n                axes[0, 2].axis('off')\n            axes[0, 2].axis('off')\n            \n            # Labeled segmentation\n            from skimage import measure\n            colored_labels = measure.label2rgb(results['labels'], bg_label=0)\n            axes[1, 0].imshow(colored_labels)\n            axes[1, 0].set_title(f'Segmented Seeds: {results[\"seed_count\"]}')\n            axes[1, 0].axis('off')\n            \n            # Feature plots if available\n            if not results['features'].empty:\n                # Area distribution\n                axes[1, 1].hist(results['features']['area'], bins=20, alpha=0.7, edgecolor='black')\n                axes[1, 1].set_title('Seed Area Distribution')\n                axes[1, 1].set_xlabel('Area (pixels)')\n                axes[1, 1].set_ylabel('Count')\n                \n                # Aspect ratio vs area\n                if 'aspect_ratio' in results['features'].columns:\n                    scatter = axes[1, 2].scatter(results['features']['area'], \n                                               results['features']['aspect_ratio'], \n                                               alpha=0.6, c=results['features'].index, cmap='viridis')\n                    axes[1, 2].set_xlabel('Area')\n                    axes[1, 2].set_ylabel('Aspect Ratio')\n                    axes[1, 2].set_title('Area vs Aspect Ratio')\n                    axes[1, 2].grid(True, alpha=0.3)\n                else:\n                    axes[1, 2].axis('off')\n            else:\n                axes[1, 1].axis('off')\n                axes[1, 2].axis('off')\n            \n            plt.suptitle(f'Rice Segmentation Results - {results[\"method_used\"].title()} Method', \n                        fontsize=16)\n            plt.tight_layout()\n            \n            # Save visualization\n            viz_path = output_dir / f'segmentation_visualization.{self.config[\"output\"][\"output_format\"]}'\n            plt.savefig(viz_path, dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            self.logger.info(f\"Visualization saved to {viz_path}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Error creating visualization: {e}\")\n    \n    def _evaluate_result(self, results: Dict, image_path: Path) -> Dict:\n        \"\"\"Evaluate segmentation result against ground truth if available.\"\"\"\n        try:\n            gt_dir = Path(self.config['evaluation']['ground_truth_dir'])\n            gt_path = gt_dir / f\"{image_path.stem}_gt.png\"\n            \n            if not gt_path.exists():\n                return {'error': 'Ground truth not found'}\n            \n            # Load ground truth\n            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)\n            pred_mask = (results['labels'] > 0).astype(np.uint8) * 255\n            \n            # Evaluate segmentation\n            seg_metrics = self.evaluator.evaluate_segmentation(pred_mask, gt_mask)\n            \n            # Evaluate object detection\n            det_metrics = self.evaluator.evaluate_object_detection(\n                results['labels'], \n                cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE),\n                iou_threshold=self.config['evaluation']['iou_threshold']\n            )\n            \n            return {**seg_metrics, **det_metrics}\n            \n        except Exception as e:\n            return {'error': f'Evaluation failed: {str(e)}'}\n    \n    def _save_processing_report(self, results: Dict, output_dir: Path):\n        \"\"\"Save detailed processing report.\"\"\"\n        try:\n            report = {\n                'image_info': {\n                    'path': results['image_path'],\n                    'processing_time': results['processing_time'],\n                    'method_used': results['method_used']\n                },\n                'segmentation_results': {\n                    'seed_count': results['seed_count'],\n                    'total_seed_area': results['features']['area'].sum() if not results['features'].empty else 0,\n                    'mean_seed_area': results['features']['area'].mean() if not results['features'].empty else 0,\n                    'area_std': results['features']['area'].std() if not results['features'].empty else 0\n                },\n                'feature_summary': {},\n                'evaluation_results': results.get('evaluation', {}),\n                'configuration': results['config']\n            }\n            \n            # Add feature summary if features exist\n            if not results['features'].empty:\n                numeric_features = results['features'].select_dtypes(include=[np.number])\n                for col in numeric_features.columns:\n                    report['feature_summary'][col] = {\n                        'mean': float(numeric_features[col].mean()),\n                        'std': float(numeric_features[col].std()),\n                        'min': float(numeric_features[col].min()),\n                        'max': float(numeric_features[col].max())\n                    }\n            \n            # Save report\n            report_path = output_dir / 'processing_report.json'\n            with open(report_path, 'w') as f:\n                json.dump(report, f, indent=2, default=str)\n            \n            self.logger.info(f\"Processing report saved to {report_path}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Error saving processing report: {e}\")\n    \n    def _compile_batch_results(self, all_results: List[Dict]) -> pd.DataFrame:\n        \"\"\"Compile batch results into a summary DataFrame.\"\"\"\n        batch_data = []\n        \n        for result in all_results:\n            row = {\n                'image_name': Path(result['image_path']).name,\n                'method': result.get('method', 'unknown'),\n                'seed_count': result.get('seed_count', 0),\n                'success': result.get('success', True)\n            }\n            \n            # Add feature summaries\n            if 'features' in result and not result['features'].empty:\n                features = result['features']\n                row.update({\n                    'total_area': features['area'].sum(),\n                    'mean_area': features['area'].mean(),\n                    'area_std': features['area'].std(),\n                    'mean_aspect_ratio': features.get('aspect_ratio', pd.Series()).mean(),\n                    'mean_circularity': features.get('circularity', pd.Series()).mean()\n                })\n            \n            # Add evaluation metrics if available\n            if 'evaluation' in result and isinstance(result['evaluation'], dict):\n                for metric, value in result['evaluation'].items():\n                    if isinstance(value, (int, float)):\n                        row[f'eval_{metric}'] = value\n            \n            batch_data.append(row)\n        \n        return pd.DataFrame(batch_data)\n    \n    def _create_method_comparison_report(self, method_comparison: Dict, output_dir: Path):\n        \"\"\"Create comparison report for different methods.\"\"\"\n        try:\n            comparison_data = []\n            \n            for method, results in method_comparison.items():\n                successful_results = [r for r in results if r.get('success', True)]\n                \n                if not successful_results:\n                    continue\n                \n                seed_counts = [r.get('seed_count', 0) for r in successful_results]\n                \n                method_summary = {\n                    'method': method,\n                    'images_processed': len(results),\n                    'successful_processes': len(successful_results),\n                    'success_rate': len(successful_results) / len(results),\n                    'mean_seed_count': np.mean(seed_counts),\n                    'std_seed_count': np.std(seed_counts),\n                    'total_seeds_detected': sum(seed_counts)\n                }\n                \n                comparison_data.append(method_summary)\n            \n            comparison_df = pd.DataFrame(comparison_data)\n            comparison_path = output_dir / 'method_comparison.csv'\n            comparison_df.to_csv(comparison_path, index=False)\n            \n            # Create comparison visualization\n            if len(comparison_data) > 1:\n                fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n                \n                methods = comparison_df['method']\n                \n                # Success rate\n                axes[0, 0].bar(methods, comparison_df['success_rate'])\n                axes[0, 0].set_title('Success Rate by Method')\n                axes[0, 0].set_ylabel('Success Rate')\n                axes[0, 0].set_ylim(0, 1)\n                \n                # Mean seed count\n                axes[0, 1].bar(methods, comparison_df['mean_seed_count'])\n                axes[0, 1].set_title('Mean Seeds Detected')\n                axes[0, 1].set_ylabel('Mean Seed Count')\n                \n                # Total seeds\n                axes[1, 0].bar(methods, comparison_df['total_seeds_detected'])\n                axes[1, 0].set_title('Total Seeds Detected')\n                axes[1, 0].set_ylabel('Total Seeds')\n                \n                # Processing efficiency\n                axes[1, 1].bar(methods, comparison_df['images_processed'])\n                axes[1, 1].set_title('Images Processed')\n                axes[1, 1].set_ylabel('Number of Images')\n                \n                plt.tight_layout()\n                plt.savefig(output_dir / 'method_comparison.png', dpi=300, bbox_inches='tight')\n                plt.close()\n            \n            self.logger.info(f\"Method comparison saved to {comparison_path}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Error creating method comparison: {e}\")\n    \n    def _create_batch_report(self, results_df: pd.DataFrame, output_dir: Path):\n        \"\"\"Create comprehensive batch processing report.\"\"\"\n        try:\n            # Summary statistics\n            summary = {\n                'total_images': len(results_df),\n                'successful_processes': len(results_df[results_df['success'] == True]),\n                'total_seeds_detected': results_df['seed_count'].sum(),\n                'mean_seeds_per_image': results_df['seed_count'].mean(),\n                'processing_timestamp': datetime.now().isoformat()\n            }\n            \n            # Save summary\n            with open(output_dir / 'batch_summary.json', 'w') as f:\n                json.dump(summary, f, indent=2, default=str)\n            \n            # Create visualizations\n            fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n            \n            # Seed count distribution\n            axes[0, 0].hist(results_df['seed_count'], bins=20, alpha=0.7, edgecolor='black')\n            axes[0, 0].set_title('Seed Count Distribution')\n            axes[0, 0].set_xlabel('Seeds per Image')\n            axes[0, 0].set_ylabel('Number of Images')\n            \n            # Success rate pie chart\n            success_counts = results_df['success'].value_counts()\n            axes[0, 1].pie(success_counts.values, labels=['Success', 'Failed'], autopct='%1.1f%%')\n            axes[0, 1].set_title('Processing Success Rate')\n            \n            # Area statistics (if available)\n            if 'mean_area' in results_df.columns:\n                axes[1, 0].hist(results_df['mean_area'].dropna(), bins=20, alpha=0.7, edgecolor='black')\n                axes[1, 0].set_title('Mean Seed Area Distribution')\n                axes[1, 0].set_xlabel('Mean Area per Image')\n                axes[1, 0].set_ylabel('Number of Images')\n            else:\n                axes[1, 0].axis('off')\n            \n            # Method performance (if multiple methods)\n            if 'method' in results_df.columns and results_df['method'].nunique() > 1:\n                method_performance = results_df.groupby('method')['seed_count'].mean()\n                axes[1, 1].bar(method_performance.index, method_performance.values)\n                axes[1, 1].set_title('Average Seeds Detected by Method')\n                axes[1, 1].set_ylabel('Mean Seed Count')\n                axes[1, 1].tick_params(axis='x', rotation=45)\n            else:\n                axes[1, 1].axis('off')\n            \n            plt.tight_layout()\n            plt.savefig(output_dir / 'batch_analysis.png', dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            # Print summary\n            print(\"\\nBatch Processing Summary:\")\n            print(\"=\" * 30)\n            for key, value in summary.items():\n                if isinstance(value, float):\n                    print(f\"{key}: {value:.2f}\")\n                else:\n                    print(f\"{key}: {value}\")\n            \n            self.logger.info(f\"Batch report created in {output_dir}\")\n            \n        except Exception as e:\n            self.logger.error(f\"Error creating batch report: {e}\")\n    \n    def create_config_template(self, output_path: str = \"config_template.json\"):\n        \"\"\"Create a configuration template file.\"\"\"\n        template_config = {\n            \"segmentation\": {\n                \"min_seed_area\": 50,\n                \"max_seed_area\": 5000,\n                \"min_aspect_ratio\": 1.5,\n                \"max_aspect_ratio\": 6.0,\n                \"default_method\": \"watershed\"\n            },\n            \"evaluation\": {\n                \"iou_threshold\": 0.5,\n                \"enable_evaluation\": False,\n                \"ground_truth_dir\": \"path/to/ground_truth\"\n            },\n            \"output\": {\n                \"save_intermediate\": True,\n                \"save_visualizations\": True,\n                \"save_features\": True,\n                \"output_format\": \"png\"\n            },\n            \"processing\": {\n                \"batch_size\": 10,\n                \"parallel\": False,\n                \"show_progress\": True\n            }\n        }\n        \n        with open(output_path, 'w') as f:\n            json.dump(template_config, f, indent=2)\n        \n        print(f\"Configuration template saved to {output_path}\")\n        print(\"Edit this file to customize processing parameters.\")\n\n\ndef main():\n    \"\"\"Command line interface for the rice segmentation pipeline.\"\"\"\n    parser = argparse.ArgumentParser(description='Rice Seeds Segmentation Pipeline')\n    parser.add_argument('input', help='Input image file or directory')\n    parser.add_argument('-o', '--output', help='Output directory')\n    parser.add_argument('-m', '--method', \n                       choices=['otsu', 'adaptive', 'watershed', 'connected_components', 'region_growing', 'canny'],\n                       default='watershed', help='Segmentation method')\n    parser.add_argument('-c', '--config', help='Configuration file path')\n    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')\n    parser.add_argument('--compare-methods', action='store_true', \n                       help='Compare multiple segmentation methods')\n    parser.add_argument('--create-config', action='store_true',\n                       help='Create configuration template')\n    \n    args = parser.parse_args()\n    \n    if args.create_config:\n        pipeline = RiceSegmentationPipeline()\n        pipeline.create_config_template()\n        return\n    \n    # Initialize pipeline\n    pipeline = RiceSegmentationPipeline(args.config)\n    \n    input_path = Path(args.input)\n    output_path = Path(args.output) if args.output else Path(\"results\")\n    \n    if args.batch or input_path.is_dir():\n        # Batch processing\n        methods = ['otsu', 'adaptive', 'watershed', 'connected_components', 'canny'] if args.compare_methods else [args.method]\n        results_df = pipeline.process_batch(input_path, output_path, methods)\n        print(f\"\\nProcessed {len(results_df)} images. Results saved to {output_path}\")\n    else:\n        # Single image processing\n        if not input_path.exists():\n            print(f\"Error: Input file {input_path} does not exist\")\n            return\n        \n        result = pipeline.process_single_image(input_path, args.method, output_path)\n        if result.get('success', True):\n            print(f\"\\nSuccessfully processed {input_path.name}\")\n            print(f\"Seeds detected: {result['seed_count']}\")\n            print(f\"Results saved to: {output_path}\")\n        else:\n            print(f\"\\nFailed to process {input_path.name}: {result.get('error', 'Unknown error')}\")\n\n\nif __name__ == \"__main__\":\n    main()"