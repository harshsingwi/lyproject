#!/usr/bin/env python3
"""
Grapevine Disease Dataset Organizer
===================================

Organizes hyperspectral PNG images into training dataset structure
for grapevine disease detection.

Usage:
    python organize_dataset.py --png-dir ./raw_png_images --output-dir ./organized_dataset --metadata description-2.csv
"""

import os
import shutil
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GrapevineDatasetOrganizer:
    """Organizes grapevine disease detection dataset from PNG images"""
    
    def __init__(self, png_source_dir: str, output_dir: str, metadata_path: str):
        self.png_source_dir = Path(png_source_dir)
        self.output_dir = Path(output_dir)
        self.metadata_path = Path(metadata_path)
        
        # Create output directory structure
        self.class_dirs = {
            'healthy': self.output_dir / 'healthy',
            'diseased': self.output_dir / 'diseased', 
            'test': self.output_dir / 'test'  # For unmatched files
        }
        
        for class_dir in self.class_dirs.values():
            class_dir.mkdir(parents=True, exist_ok=True)
    
    def load_metadata(self) -> pd.DataFrame:
        """Load and validate metadata from CSV file"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        logger.info(f"Loading metadata from {self.metadata_path}")
        
        # Try different delimiters and encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1']
        delimiters = [';', ',', '\t']
        
        for encoding in encodings:
            for delimiter in delimiters:
                try:
                    metadata = pd.read_csv(self.metadata_path, delimiter=delimiter, encoding=encoding)
                    logger.info(f"Successfully loaded metadata with {delimiter} delimiter and {encoding} encoding")
                    
                    # Validate required columns
                    required_columns = ['imageID', 'directoryName', 'symptom']
                    missing_columns = [col for col in required_columns if col not in metadata.columns]
                    
                    if missing_columns:
                        logger.warning(f"Missing columns in metadata: {missing_columns}")
                        # Try case-insensitive matching
                        for col in required_columns:
                            col_lower = col.lower()
                            matching_cols = [c for c in metadata.columns if c.lower() == col_lower]
                            if matching_cols:
                                metadata.rename(columns={matching_cols[0]: col}, inplace=True)
                    
                    return metadata
                except Exception as e:
                    continue
        
        raise ValueError("Could not parse metadata file with any delimiter or encoding")

    def classify_disease_status(self, symptom: str) -> str:
        """Classify symptom into disease categories for grapevines"""
        if pd.isna(symptom) or not isinstance(symptom, str):
            return 'test'
            
        symptom_lower = symptom.lower().strip()
        
        # Handle common encoding issues
        symptom_lower = (symptom_lower
                        .replace('@', 'é')
                        .replace('?', 'é')
                        .replace('é', 'é')  # Different é encoding
                        .replace('Ã©', 'é'))  # UTF-8 mishandling
        
        # Healthy classifications
        healthy_patterns = [
            'healthy', 'ok', 'normal', 'sain', 'saine', 
            'bon état', 'good condition', 'no symptoms'
        ]
        
        # Disease patterns (common grapevine diseases)
        disease_patterns = [
            'flavescence', 'dorée', 'doree', 'fd', 'golden flavescence',
            'treehopper', 'leafhopper', 'cicadelle',
            'mildew', 'downy mildew', 'powdery mildew',
            'esca', 'black measles', 'black dead arm',
            'botrytis', 'grey rot', 'bunch rot',
            'eutypa', 'dieback', 'dead arm',
            'phomopsis', 'cane and leaf spot',
            'black rot', 'anthracnose',
            'virus', 'leafroll', 'fanleaf',
            'deficiency', 'chlorosis', 'nutritional',
            'stress', 'water stress', 'drought',
            'damaged', 'injury', 'wound',
            'wood diseases', 'trunk diseases',
            'senescence',  
            'discoloration' 
        ]
        
        if any(pattern in symptom_lower for pattern in healthy_patterns):
            return 'healthy'
        elif any(pattern in symptom_lower for pattern in disease_patterns):
            return 'diseased'
        else:
            logger.warning(f"Unknown symptom pattern: '{symptom}' -> classifying as 'test'")
            return 'test'

    def discover_png_files(self) -> List[Path]:
        """Discover all PNG image files with comprehensive pattern matching"""
        png_patterns = [
            '*.png', '*.PNG',
            'REFLECTANCE_*.png', 'REFLECTANCE_*.PNG',
            'reflectance_*.png', 'reflectance_*.PNG',
            'RGB_*.png', 'RGB_*.PNG',
            '2020-*.png', '2020-*.PNG',
            '2021-*.png', '2021-*.PNG',
            '**/*.png', '**/*.PNG'  # Recursive search
        ]
        
        png_files = []
        for pattern in png_patterns:
            try:
                png_files.extend(list(self.png_source_dir.rglob(pattern)))
            except Exception as e:
                logger.debug(f"Pattern {pattern} failed: {e}")
        
        # Remove duplicates and sort
        png_files = sorted(list(set(png_files)))
        
        logger.info(f"Discovered {len(png_files)} PNG files in {self.png_source_dir}")
        
        # Log first few files for verification
        for i, file_path in enumerate(png_files[:5]):
            logger.debug(f"Sample file {i+1}: {file_path.name}")
        
        return png_files
    
    def extract_identifier(self, filename: str) -> str:
        """Extract unique identifier from filename patterns"""
        patterns = [
            r'REFLECTANCE_(\d{4}-\d{2}-\d{2}_\d+)',
            r'reflectance_(\d{4}-\d{2}-\d{2}_\d+)',
            r'RGB_(\d{4}-\d{2}-\d{2}_\d+)',
            r'(\d{4}-\d{2}-\d{2}_\d+)',
            r'(\d{4}-\d{2}-\d{2}_\d{3})',
            r'image_?(\d+)',
            r'sample_?(\d+)',
            r'(\d+)_reflectance',
            r'reflectance_(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def match_images_to_metadata(self, metadata: pd.DataFrame, png_files: List[Path]) -> Dict:
        """Match PNG files to metadata entries using multiple strategies"""
        matched_files = []
        unmatched_files = []
        
        # Create comprehensive lookup dictionary
        metadata_lookup = {}
        for _, row in metadata.iterrows():
            # Use directoryName as primary key
            if 'directoryName' in row and pd.notna(row['directoryName']):
                dir_name = str(row['directoryName']).strip()
                metadata_lookup[dir_name] = row
            
            # Use imageID as secondary key
            if 'imageID' in row and pd.notna(row['imageID']):
                image_id = str(row['imageID']).strip()
                metadata_lookup[image_id] = row
            
            # Create variations for flexible matching
            if 'directoryName' in row and pd.notna(row['directoryName']):
                dir_var = str(row['directoryName']).replace('-', '_').strip()
                if dir_var not in metadata_lookup:
                    metadata_lookup[dir_var] = row
        
        logger.info(f"Created metadata lookup with {len(metadata_lookup)} keys")
        
        # Match files using multiple strategies
        for png_file in png_files:
            filename = png_file.name
            file_id = self.extract_identifier(filename)
            
            matched = False
            
            # Strategy 1: Direct ID match
            if file_id and file_id in metadata_lookup:
                metadata_row = metadata_lookup[file_id]
                matched_files.append({
                    'image_path': png_file,
                    'metadata': metadata_row,
                    'class': self.classify_disease_status(metadata_row.get('symptom', ''))
                })
                matched = True
            
            # Strategy 2: Partial filename matching
            if not matched:
                for key in metadata_lookup.keys():
                    if (key in filename or filename in key or 
                        key.replace('-', '_') in filename.replace('-', '_')):
                        metadata_row = metadata_lookup[key]
                        matched_files.append({
                            'image_path': png_file,
                            'metadata': metadata_row,
                            'class': self.classify_disease_status(metadata_row.get('symptom', ''))
                        })
                        matched = True
                        break
            
            # Strategy 3: Stem matching (filename without extension)
            if not matched:
                file_stem = png_file.stem
                for key in metadata_lookup.keys():
                    if key in file_stem or file_stem in key:
                        metadata_row = metadata_lookup[key]
                        matched_files.append({
                            'image_path': png_file,
                            'metadata': metadata_row,
                            'class': self.classify_disease_status(metadata_row.get('symptom', ''))
                        })
                        matched = True
                        break
            
            if not matched:
                unmatched_files.append(png_file)
                logger.debug(f"Unmatched file: {filename}")
        
        logger.info(f"Successfully matched {len(matched_files)} images")
        logger.info(f"Unmatched files: {len(unmatched_files)}")
        
        return {
            'matched': matched_files,
            'unmatched': unmatched_files
        }
    
    def organize_training_data(self, matched_data: Dict) -> Dict:
        """Organize matched files into training dataset structure"""
        results = {
            'healthy': 0,
            'diseased': 0,
            'test': 0,
            'errors': []
        }
        
        # Process matched files
        for file_info in matched_data['matched']:
            try:
                target_class = file_info['class']
                target_dir = self.class_dirs[target_class]
                
                # Generate clean filename
                if 'imageID' in file_info['metadata'] and pd.notna(file_info['metadata']['imageID']):
                    image_id = str(file_info['metadata']['imageID']).strip()
                    clean_name = re.sub(r'[^\w\-_]', '_', image_id)  # Sanitize filename
                    target_path = target_dir / f"{clean_name}.png"
                else:
                    # Fallback to original filename
                    target_path = target_dir / file_info['image_path'].name
                
                # Copy image file
                shutil.copy2(file_info['image_path'], target_path)
                
                results[target_class] += 1
                logger.debug(f"Organized {target_path.name} -> {target_class}")
                
            except Exception as e:
                error_msg = f"Error organizing {file_info['image_path'].name}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        # Process unmatched files (put in test folder)
        for png_file in matched_data['unmatched']:
            try:
                target_path = self.class_dirs['test'] / png_file.name
                shutil.copy2(png_file, target_path)
                results['test'] += 1
            except Exception as e:
                error_msg = f"Error copying unmatched file {png_file.name}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results

    def generate_organization_report(self, metadata: pd.DataFrame, results: Dict):
        """Generate comprehensive organization report"""
        print("\n" + "="*60)
        print("GRAPEVINE DISEASE DATASET ORGANIZATION REPORT")
        print("="*60)
        
        print(f"\n📁 Source PNG Directory: {self.png_source_dir}")
        print(f"📁 Output Dataset Directory: {self.output_dir}")
        print(f"📄 Metadata File: {self.metadata_path}")
        
        print(f"\n📊 Metadata Statistics:")
        print(f"   Total metadata entries: {len(metadata)}")
        
        if 'symptom' in metadata.columns:
            symptom_counts = metadata['symptom'].value_counts()
            print(f"   Symptom distribution:")
            for symptom, count in symptom_counts.items():
                classification = self.classify_disease_status(symptom)
                print(f"     {symptom}: {count} -> {classification}")
        
        print(f"\n✅ Organization Results:")
        print(f"   Healthy images: {results['healthy']}")
        print(f"   Diseased images: {results['diseased']}")
        print(f"   Test images (unmatched): {results['test']}")
        
        # FIXED: Calculate total organized correctly
        total_organized = results['healthy'] + results['diseased'] + results['test']
        print(f"   Total organized: {total_organized}")
        
        if results['errors']:
            print(f"\n❌ Errors: {len(results['errors'])}")
            for error in results['errors'][:3]:
                print(f"   {error}")
            if len(results['errors']) > 3:
                print(f"   ... and {len(results['errors']) - 3} more errors")
        
        # Dataset balance analysis
        print(f"\n⚖️  Dataset Balance:")
        total_labeled = results['healthy'] + results['diseased']
        if total_labeled > 0:
            healthy_percent = (results['healthy'] / total_labeled) * 100
            diseased_percent = (results['diseased'] / total_labeled) * 100
            print(f"   Healthy: {healthy_percent:.1f}% ({results['healthy']})")
            print(f"   Diseased: {diseased_percent:.1f}% ({results['diseased']})")
        
        print(f"\n📂 Dataset Structure:")
        for class_name, class_dir in self.class_dirs.items():
            class_count = len(list(class_dir.glob('*.png')))
            print(f"   {class_dir.name}: {class_count} images")
        
        print(f"\n🎯 Next steps:")
        print("   1. Review the dataset structure")
        print("   2. Verify image classifications")
        print("   3. Run training: python cli.py train --config config.yaml")
        print("="*60)

    def run_organization(self):
        """Main organization workflow"""
        logger.info("Starting dataset organization...")
        
        # Load metadata
        metadata = self.load_metadata()
        
        # Discover PNG files
        png_files = self.discover_png_files()
        
        if not png_files:
            logger.error("No PNG files found in source directory")
            return
        
        # Match files to metadata
        matched_data = self.match_images_to_metadata(metadata, png_files)
        
        # Organize files
        results = self.organize_training_data(matched_data)
        
        # Generate report
        self.generate_organization_report(metadata, results)
        
        logger.info("Dataset organization completed!")


def main():
    """Main function for dataset organization"""
    parser = argparse.ArgumentParser(
        description="Organize grapevine disease detection dataset from PNG images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default metadata
  python organize_dataset.py --png-dir ./raw_png_images --output-dir ./organized_dataset
  
  # Specify custom metadata file
  python organize_dataset.py --png-dir ./raw_png_images --output-dir ./organized_dataset --metadata description-2.csv
  
  # Verbose output for debugging
  python organize_dataset.py --png-dir ./raw_png_images --output-dir ./organized_dataset --verbose
        """
    )
    
    parser.add_argument('--png-dir', type=str, required=True,
                       help='Directory containing PNG image files')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for organized dataset')
    parser.add_argument('--metadata', type=str, default='description-2.csv',
                       help='Path to metadata CSV file (default: description-2.csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging for debugging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize organizer
        organizer = GrapevineDatasetOrganizer(
            png_source_dir=args.png_dir,
            output_dir=args.output_dir,
            metadata_path=args.metadata
        )
        
        # Run organization
        organizer.run_organization()
        
    except Exception as e:
        logger.error(f"Error during dataset organization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()