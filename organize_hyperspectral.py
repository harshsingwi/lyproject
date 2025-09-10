#!/usr/bin/env python3
"""
Hyperspectral Dataset Organizer
===============================

Organizes raw hyperspectral files into structured dataset
for training and evaluation.
"""

import os
import shutil
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List
import logging
import re

logger = logging.getLogger(__name__)

class HyperspectralDatasetOrganizer:
    """Organizes hyperspectral dataset for grapevine disease detection"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.class_dirs = {
            'healthy': self.output_dir / 'healthy',
            'diseased': self.output_dir / 'diseased',
            'abiotic': self.output_dir / 'abiotic',
            'test': self.output_dir / 'test'
        }
        
        for class_dir in self.class_dirs.values():
            class_dir.mkdir(parents=True, exist_ok=True)
    
    def discover_hyperspectral_files(self) -> List[Path]:
        """Discover all hyperspectral files with various extensions"""
        extensions = ['.raw', '.hdr', '.dat', '.img', '.h5', '.npy']
        
        hyperspectral_files = []
        for ext in extensions:
            hyperspectral_files.extend(list(self.source_dir.rglob(f'*{ext}')))
        
        # Group files by base name (without extension)
        file_groups = {}
        for file_path in hyperspectral_files:
            base_name = file_path.stem
            if base_name not in file_groups:
                file_groups[base_name] = []
            file_groups[base_name].append(file_path)
        
        return file_groups
    
    def classify_by_directory(self, file_path: Path) -> str:
        """Classify files based on directory structure"""
        path_str = str(file_path).lower()
        
        if 'healthy' in path_str or 'normal' in path_str or 'control' in path_str:
            return 'healthy'
        elif 'disease' in path_str or 'biotic' in path_str or 'infected' in path_str:
            return 'diseased'
        elif 'stress' in path_str or 'abiotic' in path_str or 'deficiency' in path_str:
            return 'abiotic'
        else:
            return 'test'
    
    def organize_files(self) -> Dict[str, int]:
        """Organize hyperspectral files into class directories"""
        results = {class_name: 0 for class_name in self.class_dirs.keys()}
        results['errors'] = 0
        
        file_groups = self.discover_hyperspectral_files()
        
        for base_name, files in file_groups.items():
            try:
                # Determine classification from first file
                classification = self.classify_by_directory(files[0])
                target_dir = self.class_dirs[classification]
                
                # Copy all related files
                for file_path in files:
                    target_path = target_dir / file_path.name
                    shutil.copy2(file_path, target_path)
                
                results[classification] += 1
                logger.debug(f"Organized {base_name} -> {classification}")
                
            except Exception as e:
                logger.error(f"Error organizing {base_name}: {e}")
                results['errors'] += 1
        
        return results
    
    def generate_metadata_csv(self):
        """Generate metadata CSV file for the organized dataset"""
        metadata = []
        
        for class_name, class_dir in self.class_dirs.items():
            # Look for raw files in each class directory
            raw_files = list(class_dir.glob('*.raw'))
            
            for raw_file in raw_files:
                metadata.append({
                    'filename': raw_file.name,
                    'class': class_name,
                    'filepath': str(raw_file),
                    'hdr_file': str(raw_file.with_suffix('.hdr')) if raw_file.with_suffix('.hdr').exists() else ''
                })
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(metadata)
        
        # Save to CSV
        metadata_path = self.output_dir / 'metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")
        
        return metadata_df
    
    def run_organization(self):
        """Main organization workflow"""
        logger.info("Starting hyperspectral dataset organization...")
        
        # Organize files
        results = self.organize_files()
        
        # Generate metadata
        metadata_df = self.generate_metadata_csv()
        
        # Print summary
        print("\n" + "="*60)
        print("HYPERSPECTRAL DATASET ORGANIZATION REPORT")
        print("="*60)
        print(f"Source directory: {self.source_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"\nOrganized files:")
        for class_name, count in results.items():
            if class_name != 'errors':
                print(f"  {class_name}: {count}")
        print(f"Errors: {results['errors']}")
        print(f"Total files: {len(metadata_df)}")
        print("="*60)

def main():
    """Main function for hyperspectral dataset organization"""
    parser = argparse.ArgumentParser(
        description="Organize hyperspectral dataset for grapevine disease detection"
    )
    
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Directory containing raw hyperspectral files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for organized dataset')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        organizer = HyperspectralDatasetOrganizer(args.source_dir, args.output_dir)
        organizer.run_organization()
    except Exception as e:
        logger.error(f"Error during organization: {e}")
        raise

if __name__ == "__main__":
    main()