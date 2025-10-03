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


def main():
    """Command line interface for the rice segmentation pipeline."""
    parser = argparse.ArgumentParser(description='Rice Seeds Segmentation Pipeline')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('-o', '--output', help='Output directory')
    parser.add_argument('-m', '--method', 
                       choices=['otsu', 'adaptive', 'watershed', 'connected_components', 'region_growing', 'canny'],
                       default='watershed', help='Segmentation method')
    parser.add_argument('-c', '--config', help='Configuration file path')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    parser.add_argument('--compare-methods', action='store_true', 
                       help='Compare multiple segmentation methods')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RiceSegmentationPipeline(args.config)
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else Path("results")
    
    if args.batch or input_path.is_dir():
        # Batch processing
        methods = ['otsu', 'adaptive', 'watershed', 'connected_components', 'canny'] if args.compare_methods else [args.method]
        results_df = pipeline.process_batch(input_path, output_path, methods)
        print(f"\\nProcessed {len(results_df)} images. Results saved to {output_path}")
    else:
        # Single image processing
        if not input_path.exists():
            print(f"Error: Input file {input_path} does not exist")
            return
        
        result = pipeline.process_single_image(input_path, args.method, output_path)
        if result.get('success', True):
            print(f"\\nSuccessfully processed {input_path.name}")
            print(f"Seeds detected: {result['seed_count']}")
            print(f"Results saved to: {output_path}")
        else:
            print(f"\\nFailed to process {input_path.name}: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    print("Rice Seeds Segmentation Pipeline")
    print("=" * 40)
    print("Usage:")
    print("1. Single image: python rice_segmentation_pipeline.py image.jpg")
    print("2. Batch processing: python rice_segmentation_pipeline.py input_dir --batch")
    print("3. Method comparison: python rice_segmentation_pipeline.py input_dir --compare-methods")
    print("\\nFor command line usage, uncomment the main() call below")
    # main()  # Uncomment this line for command line usage