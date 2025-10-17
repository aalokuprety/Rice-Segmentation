#!/usr/bin/env python3
"""
Genotype Screening Processor

Process and analyze 100+ rice genotypes with multiple photo replicates per genotype.
Generates ranking tables and identifies top-performing genotypes.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import argparse
import json
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from rice_segmentation_methods import RiceSeedSegmenter


class GenotypeScreeningProcessor:
    """Process multiple genotypes with replicates for large-scale screening."""
    
    def __init__(self, output_dir: str = '../results/genotype_screening'):
        """Initialize genotype screening processor."""
        self.segmenter = RiceSeedSegmenter()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'visualizations').mkdir(exist_ok=True)
        (self.output_dir / 'individual_results').mkdir(exist_ok=True)
        (self.output_dir / 'summary_plots').mkdir(exist_ok=True)
    
    def extract_genotype_info(self, image_path: Path) -> Tuple[str, int]:
        """
        Extract genotype ID and replicate number from filename.
        
        Supports patterns:
        - gen001_rep1.jpg → (gen001, 1)
        - genotype_001/rep1.jpg → (genotype_001, 1)
        - gen001_r1.jpg → (gen001, 1)
        - 001_1.jpg → (001, 1)
        """
        filename = image_path.stem
        parent = image_path.parent.name
        
        # Pattern 1: genXXX_repY or genXXX_rY
        match = re.match(r'(gen(?:otype)?_?\d+)_r(?:ep)?(\d+)', filename, re.IGNORECASE)
        if match:
            return match.group(1), int(match.group(2))
        
        # Pattern 2: Parent folder is genotype, filename is rep
        if 'gen' in parent.lower():
            match = re.match(r'r(?:ep)?(\d+)', filename, re.IGNORECASE)
            if match:
                return parent, int(match.group(1))
        
        # Pattern 3: Simple number pattern XXX_Y
        match = re.match(r'(\d+)_(\d+)', filename)
        if match:
            return f"gen{match.group(1)}", int(match.group(2))
        
        # Default: use full filename as genotype, replicate 1
        return filename, 1
    
    def process_single_image(self, image_path: Path, method: str = 'adaptive') -> Dict:
        """Process a single image and return results."""
        try:
            # Extract genotype info
            genotype_id, replicate = self.extract_genotype_info(image_path)
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {'error': 'Failed to load image', 'success': False}
            
            # Segment
            if method == 'otsu':
                results = self.segmenter.otsu_segmentation(image)
            elif method == 'adaptive':
                results = self.segmenter.adaptive_segmentation(image)
            elif method == 'watershed':
                results = self.segmenter.watershed_segmentation(image)
            else:
                results = self.segmenter.adaptive_segmentation(image)
            
            # Extract features
            features = self.segmenter.extract_features(results['labeled_mask'])
            
            # Compile results
            output = {
                'genotype': genotype_id,
                'replicate': replicate,
                'image_name': image_path.name,
                'image_path': str(image_path),
                'seed_count': results['seed_count'],
                'mean_area': features['area'].mean() if len(features) > 0 else 0,
                'std_area': features['area'].std() if len(features) > 0 else 0,
                'min_area': features['area'].min() if len(features) > 0 else 0,
                'max_area': features['area'].max() if len(features) > 0 else 0,
                'mean_perimeter': features['perimeter'].mean() if len(features) > 0 else 0,
                'mean_aspect_ratio': features['aspect_ratio'].mean() if len(features) > 0 else 0,
                'mean_circularity': features['circularity'].mean() if len(features) > 0 else 0,
                'mean_solidity': features['solidity'].mean() if len(features) > 0 else 0,
                'total_seed_area': features['area'].sum() if len(features) > 0 else 0,
                'coverage_percent': (features['area'].sum() / (image.shape[0] * image.shape[1]) * 100) if len(features) > 0 else 0,
                'success': True
            }
            
            return output
            
        except Exception as e:
            return {
                'genotype': 'unknown',
                'replicate': 0,
                'image_name': image_path.name,
                'error': str(e),
                'success': False
            }
    
    def process_all_images(self, input_dir: Path, method: str = 'adaptive',
                          file_pattern: str = '*.jpg') -> pd.DataFrame:
        """
        Process all images in directory.
        
        Args:
            input_dir: Directory with images (flat or nested)
            method: Segmentation method
            file_pattern: File pattern to match
            
        Returns:
            DataFrame with all results
        """
        # Find all images (recursive search)
        image_files = list(input_dir.rglob(file_pattern))
        image_files.extend(input_dir.rglob(file_pattern.replace('.jpg', '.JPG')))
        image_files.extend(input_dir.rglob(file_pattern.replace('.jpg', '.png')))
        image_files.extend(input_dir.rglob(file_pattern.replace('.jpg', '.PNG')))
        
        # Remove duplicates
        image_files = list(set(image_files))
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return pd.DataFrame()
        
        print(f"\nFound {len(image_files)} images")
        print(f"Processing with {method} segmentation...")
        print("="*70)
        
        results = []
        for img_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single_image(img_path, method)
            if result.get('success', False):
                results.append(result)
            else:
                print(f"\nWarning: Failed to process {img_path.name}")
        
        if not results:
            print("No images processed successfully!")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save individual results
        csv_path = self.output_dir / 'individual_results' / 'all_photos_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Individual results saved to {csv_path}")
        
        return df
    
    def calculate_genotype_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate summary statistics per genotype.
        
        Args:
            df: DataFrame with individual photo results
            
        Returns:
            DataFrame with genotype-level statistics
        """
        # Group by genotype
        grouped = df.groupby('genotype')
        
        summary_stats = []
        for genotype, group in grouped:
            stats_dict = {
                'genotype': genotype,
                'n_replicates': len(group),
                
                # Seed count statistics
                'mean_seed_count': group['seed_count'].mean(),
                'std_seed_count': group['seed_count'].std(),
                'min_seed_count': group['seed_count'].min(),
                'max_seed_count': group['seed_count'].max(),
                'cv_seed_count': (group['seed_count'].std() / group['seed_count'].mean() * 100) if group['seed_count'].mean() > 0 else 0,
                
                # Seed area statistics
                'mean_seed_area': group['mean_area'].mean(),
                'std_seed_area': group['mean_area'].std(),
                
                # Seed shape statistics
                'mean_aspect_ratio': group['mean_aspect_ratio'].mean(),
                'mean_circularity': group['mean_circularity'].mean(),
                'mean_solidity': group['mean_solidity'].mean(),
                
                # Coverage
                'mean_coverage': group['coverage_percent'].mean(),
            }
            
            # Calculate confidence interval for seed count (95% CI)
            if len(group) > 1:
                se = stats.sem(group['seed_count'])
                ci = se * stats.t.ppf((1 + 0.95) / 2, len(group) - 1)
                stats_dict['ci_95_lower'] = stats_dict['mean_seed_count'] - ci
                stats_dict['ci_95_upper'] = stats_dict['mean_seed_count'] + ci
            else:
                stats_dict['ci_95_lower'] = stats_dict['mean_seed_count']
                stats_dict['ci_95_upper'] = stats_dict['mean_seed_count']
            
            summary_stats.append(stats_dict)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        # Sort by mean seed count (descending)
        summary_df = summary_df.sort_values('mean_seed_count', ascending=False).reset_index(drop=True)
        
        # Add ranking
        summary_df['rank'] = range(1, len(summary_df) + 1)
        
        # Save summary
        csv_path = self.output_dir / 'genotype_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"✓ Genotype summary saved to {csv_path}")
        
        return summary_df
    
    def create_ranking_plot(self, summary_df: pd.DataFrame, top_n: int = 20):
        """Create ranking plot for top N genotypes."""
        top_genotypes = summary_df.head(top_n)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Top N with error bars
        ax = axes[0]
        x = range(len(top_genotypes))
        ax.bar(x, top_genotypes['mean_seed_count'], 
               yerr=top_genotypes['std_seed_count'],
               capsize=5, alpha=0.7, color='#2ecc71')
        ax.set_xlabel('Genotype Rank', fontsize=12)
        ax.set_ylabel('Mean Seed Count', fontsize=12)
        ax.set_title(f'Top {top_n} Rice Genotypes by Seed Count', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_genotypes['genotype'], rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: All genotypes overview
        ax = axes[1]
        all_x = range(len(summary_df))
        colors = ['#2ecc71' if i < top_n else '#95a5a6' for i in range(len(summary_df))]
        ax.bar(all_x, summary_df['mean_seed_count'], color=colors, alpha=0.7)
        ax.set_xlabel('Genotype Rank', fontsize=12)
        ax.set_ylabel('Mean Seed Count', fontsize=12)
        ax.set_title(f'All {len(summary_df)} Genotypes (Top {top_n} Highlighted)', 
                    fontsize=14, fontweight='bold')
        ax.axhline(y=summary_df['mean_seed_count'].mean(), color='r', 
                  linestyle='--', label='Overall Mean', linewidth=2)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'summary_plots' / f'top_{top_n}_genotypes.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Ranking plot saved to {save_path}")
        plt.close()
    
    def create_distribution_plots(self, df: pd.DataFrame, summary_df: pd.DataFrame):
        """Create distribution and comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Genotype Screening Analysis', fontsize=16, fontweight='bold')
        
        # 1. Seed count distribution
        ax = axes[0, 0]
        ax.hist(summary_df['mean_seed_count'], bins=30, alpha=0.7, color='#3498db', edgecolor='black')
        ax.axvline(summary_df['mean_seed_count'].mean(), color='r', linestyle='--', 
                  linewidth=2, label=f'Mean: {summary_df["mean_seed_count"].mean():.1f}')
        ax.axvline(summary_df['mean_seed_count'].median(), color='g', linestyle='--', 
                  linewidth=2, label=f'Median: {summary_df["mean_seed_count"].median():.1f}')
        ax.set_xlabel('Mean Seed Count', fontsize=11)
        ax.set_ylabel('Number of Genotypes', fontsize=11)
        ax.set_title('Distribution of Seed Counts Across Genotypes')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Seed count vs seed area
        ax = axes[0, 1]
        scatter = ax.scatter(summary_df['mean_seed_area'], summary_df['mean_seed_count'],
                           c=summary_df['rank'], cmap='viridis_r', alpha=0.6, s=100)
        ax.set_xlabel('Mean Seed Area (pixels²)', fontsize=11)
        ax.set_ylabel('Mean Seed Count', fontsize=11)
        ax.set_title('Seed Count vs Seed Area')
        plt.colorbar(scatter, ax=ax, label='Rank')
        ax.grid(alpha=0.3)
        
        # 3. Coefficient of variation
        ax = axes[1, 0]
        cv_data = summary_df[summary_df['n_replicates'] > 1]['cv_seed_count']
        ax.hist(cv_data, bins=20, alpha=0.7, color='#e74c3c', edgecolor='black')
        ax.axvline(cv_data.mean(), color='k', linestyle='--', 
                  linewidth=2, label=f'Mean CV: {cv_data.mean():.1f}%')
        ax.set_xlabel('Coefficient of Variation (%)', fontsize=11)
        ax.set_ylabel('Number of Genotypes', fontsize=11)
        ax.set_title('Within-Genotype Variability (CV%)')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Seed shape metrics
        ax = axes[1, 1]
        ax.scatter(summary_df['mean_circularity'], summary_df['mean_aspect_ratio'],
                  c=summary_df['rank'], cmap='plasma_r', alpha=0.6, s=100)
        ax.set_xlabel('Mean Circularity', fontsize=11)
        ax.set_ylabel('Mean Aspect Ratio', fontsize=11)
        ax.set_title('Seed Shape Characteristics')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / 'summary_plots' / 'distribution_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plots saved to {save_path}")
        plt.close()
    
    def print_summary_report(self, summary_df: pd.DataFrame, top_n: int = 20):
        """Print text summary report."""
        print("\n" + "="*70)
        print("GENOTYPE SCREENING SUMMARY REPORT")
        print("="*70)
        print(f"Total genotypes analyzed: {len(summary_df)}")
        print(f"Mean replicates per genotype: {summary_df['n_replicates'].mean():.1f}")
        print(f"\nOverall Statistics:")
        print(f"  Mean seed count: {summary_df['mean_seed_count'].mean():.1f} ± {summary_df['mean_seed_count'].std():.1f}")
        print(f"  Range: {summary_df['mean_seed_count'].min():.1f} - {summary_df['mean_seed_count'].max():.1f}")
        print(f"  Median: {summary_df['mean_seed_count'].median():.1f}")
        
        print(f"\n{'-'*70}")
        print(f"TOP {top_n} PERFORMING GENOTYPES")
        print(f"{'-'*70}")
        print(f"{'Rank':<6} {'Genotype':<15} {'Mean±SD':<20} {'Range':<20} {'CV%':<8}")
        print(f"{'-'*70}")
        
        for idx, row in summary_df.head(top_n).iterrows():
            rank = row['rank']
            genotype = row['genotype']
            mean_sd = f"{row['mean_seed_count']:.1f}±{row['std_seed_count']:.1f}"
            range_str = f"{row['min_seed_count']:.0f}-{row['max_seed_count']:.0f}"
            cv = f"{row['cv_seed_count']:.1f}"
            print(f"{rank:<6} {genotype:<15} {mean_sd:<20} {range_str:<20} {cv:<8}")
        
        # Identify outliers (top 10%)
        threshold_90 = summary_df['mean_seed_count'].quantile(0.90)
        top_10_percent = summary_df[summary_df['mean_seed_count'] >= threshold_90]
        
        print(f"\n{'-'*70}")
        print(f"TOP 10% THRESHOLD: {threshold_90:.1f} seeds")
        print(f"Genotypes in top 10%: {len(top_10_percent)}")
        print(f"Genotypes: {', '.join(top_10_percent['genotype'].tolist())}")
        print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Process rice genotype screening experiment')
    parser.add_argument('--input', type=str, default='../data/raw',
                       help='Input directory with images')
    parser.add_argument('--output', type=str, default='../results/genotype_screening',
                       help='Output directory')
    parser.add_argument('--method', type=str, default='adaptive',
                       choices=['otsu', 'adaptive', 'watershed'],
                       help='Segmentation method')
    parser.add_argument('--pattern', type=str, default='*.jpg',
                       help='File pattern to match')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top genotypes to highlight')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = GenotypeScreeningProcessor(output_dir=args.output)
    
    print("\n" + "="*70)
    print("RICE GENOTYPE SCREENING PROCESSOR")
    print("="*70)
    print(f"Input directory: {args.input}")
    print(f"Segmentation method: {args.method}")
    print("="*70)
    
    start_time = datetime.now()
    
    # Process all images
    df_individual = processor.process_all_images(
        Path(args.input),
        method=args.method,
        file_pattern=args.pattern
    )
    
    if df_individual.empty:
        print("No data to analyze!")
        return
    
    # Calculate genotype statistics
    print("\nCalculating genotype statistics...")
    df_summary = processor.calculate_genotype_statistics(df_individual)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    processor.create_ranking_plot(df_summary, top_n=args.top_n)
    processor.create_distribution_plots(df_individual, df_summary)
    
    # Print summary report
    processor.print_summary_report(df_summary, top_n=args.top_n)
    
    # Processing time
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Total processing time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Average time per image: {duration/len(df_individual):.2f} seconds\n")


if __name__ == '__main__':
    main()