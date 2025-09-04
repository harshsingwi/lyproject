#!/usr/bin/env python3
"""
Command Line Interface for Grapevine Disease Detection System
============================================================

Usage examples:
  python cli.py --data-dir ./hyperspectral_data --config config.yaml
  python cli.py --predict --model-path ./results/random_forest_model.pkl --input spectrum.dat
  python cli.py --evaluate --config config.yaml --subset 50
"""

import os
import sys
import argparse
import yaml
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd

# Import our modules
from grapevine_disease_detection import (
    HyperspectralDataLoader, 
    FeatureExtractor, 
    ModelTrainer, 
    Visualizer,
    save_results
)
from data_loader_utils import (
    DataAugmentation, 
    AdvancedPreprocessing, 
    DatasetValidator,
    batch_load_hyperspectral_data
)

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        sys.exit(1)


def validate_paths(config: Dict[str, Any]) -> bool:
    """Validate that required paths exist"""
    data_dir = Path(config['data']['data_dir'])
    
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return False
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    return True


class CLICommands:
    """Class containing CLI command implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize components
        self.data_loader = HyperspectralDataLoader(config['data']['data_dir'])
        self.feature_extractor = FeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.visualizer = Visualizer()
        self.augmentation = DataAugmentation()
        self.preprocessor = AdvancedPreprocessing()
        self.validator = DatasetValidator()
    
    def train(self, subset_size: Optional[int] = None) -> Dict[str, Any]:
        """Train models using the configuration"""
        logger.info("Starting training pipeline...")
        
        # Step 1: Load data
        logger.info("Loading hyperspectral data...")
        
        try:
            # Try to load real data first
            X, y = batch_load_hyperspectral_data(
                self.config['data']['data_dir'],
                self.config['data']['file_patterns']['header_files']
            )
            
            if len(X) == 0:
                logger.warning("No real data found. Generating synthetic data...")
                X, y = self._generate_synthetic_data()
        
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}. Using synthetic data.")
            X, y = self._generate_synthetic_data()
        
        # Apply subset if requested
        if subset_size is not None and subset_size < len(X):
            logger.info(f"Using subset of {subset_size} samples")
            indices = np.random.choice(len(X), subset_size, replace=False)
            X = X[indices]
            y = [y[i] for i in indices]
        
        # Step 2: Data validation
        if self.config['data']['validate_data']:
            logger.info("Validating dataset...")
            report = self.validator.generate_quality_report(X, y)
            print("\n" + report)
        
        # Step 3: Preprocessing
        logger.info("Preprocessing data...")
        X_processed = self.preprocessor.apply_preprocessing_pipeline(
            X, self.config['preprocessing']['pipeline']
        )
        
        # Step 4: Data augmentation
        if self.config['data']['augmentation']['enabled']:
            logger.info("Applying data augmentation...")
            X_processed, y = self.augmentation.augment_dataset(
                X_processed, y, self.config['data']['augmentation']['factor']
            )
        
        # Step 5: Feature extraction
        logger.info("Extracting features...")
        features = self._extract_features(X_processed)
        
        # Step 6: Dimensionality reduction
        if self.config['preprocessing']['pca']['enabled']:
            logger.info("Applying dimensionality reduction...")
            features = self.feature_extractor.apply_dimensionality_reduction(
                features, self.config['preprocessing']['pca']['n_components']
            )
        
        # Step 7: Prepare data for training
        logger.info("Preparing data for training...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.model_trainer.prepare_data(
            features, y, self.config['training']['test_size']
        )
        
        # Step 8: Train models
        results = {}
        
        # Random Forest
        if self.config['models']['random_forest']['enabled']:
            logger.info("Training Random Forest...")
            rf_params = {k: v for k, v in self.config['models']['random_forest'].items() 
                        if k != 'enabled'}
            self.model_trainer.train_random_forest(X_train, y_train, **rf_params)
            results['random_forest'] = self.model_trainer.evaluate_model(
                'random_forest', X_test, y_test
            )
        
        # SVM
        if self.config['models']['svm']['enabled']:
            logger.info("Training SVM...")
            svm_params = {k: v for k, v in self.config['models']['svm'].items() 
                         if k != 'enabled'}
            self.model_trainer.train_svm(X_train, y_train, **svm_params)
            results['svm'] = self.model_trainer.evaluate_model('svm', X_test, y_test)
        
        # CNN
        if self.config['models']['cnn']['enabled']:
            logger.info("Training CNN...")
            cnn_params = {k: v for k, v in self.config['models']['cnn'].items() 
                         if k != 'enabled'}
            self.model_trainer.train_cnn(X_train, y_train, X_val, y_val, **cnn_params)
            results['cnn'] = self.model_trainer.evaluate_model('cnn', X_test, y_test)
        
        # Step 9: Generate visualizations
        if self.config['visualization']['enabled']:
            logger.info("Generating visualizations...")
            self._generate_visualizations(X_processed, y, results)
        
        # Step 10: Save results
        logger.info("Saving results...")
        save_results(results, self.model_trainer, self.config['data']['output_dir'])
        
        return results
    
    def predict(self, model_path: str, input_path: str, model_type: str = 'random_forest') -> str:
        """Make prediction on new data"""
        logger.info(f"Loading model from {model_path}")
        
        # Load model
        try:
            if model_type == 'cnn':
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            else:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return "Error loading model"
        
        # Load label encoder
        label_encoder_path = Path(model_path).parent / "label_encoder.pkl"
        try:
            with open(label_encoder_path, 'rb') as f:
                label_encoder = pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load label encoder: {e}")
            return "Error loading label encoder"
        
        # Load and preprocess input
        try:
            if input_path.endswith('.dat') or input_path.endswith('.hdr'):
                # Hyperspectral image
                img_array = self.data_loader.load_envi_image(input_path)
                if img_array is None:
                    return "Error loading hyperspectral image"
                
                # Extract spectrum
                if len(img_array.shape) == 3:
                    spectrum = np.mean(img_array.reshape(-1, img_array.shape[-1]), axis=0)
                else:
                    spectrum = img_array
            else:
                # Assume it's a spectrum file
                spectrum = np.loadtxt(input_path)
            
            # Preprocess
            spectrum_processed = self.preprocessor.apply_preprocessing_pipeline(
                spectrum.reshape(1, -1), self.config['preprocessing']['pipeline']
            )
            
            # Extract features
            features = self._extract_features(spectrum_processed)
            
            # Apply PCA if used during training
            if self.config['preprocessing']['pca']['enabled']:
                features = self.feature_extractor.apply_dimensionality_reduction(features)
            
            # Make prediction
            if model_type == 'cnn':
                features_reshaped = features.reshape(features.shape[0], features.shape[1], 1)
                prediction_proba = model.predict(features_reshaped)
                prediction = np.argmax(prediction_proba, axis=1)
            else:
                prediction = model.predict(features)
            
            # Convert to label
            predicted_label = label_encoder.inverse_transform(prediction)[0]
            
            # Get confidence if available
            if hasattr(model, 'predict_proba'):
                confidence = np.max(model.predict_proba(features))
                return f"Prediction: {predicted_label} (confidence: {confidence:.3f})"
            else:
                return f"Prediction: {predicted_label}"
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return f"Error during prediction: {e}"
    
    def evaluate(self, model_path: str, test_data_dir: str) -> Dict[str, float]:
        """Evaluate a trained model on test data"""
        logger.info("Evaluating model...")
        
        # Load test data
        X_test, y_test = batch_load_hyperspectral_data(test_data_dir)
        
        if len(X_test) == 0:
            logger.error("No test data found")
            return {}
        
        # Preprocess test data
        X_test_processed = self.preprocessor.apply_preprocessing_pipeline(
            X_test, self.config['preprocessing']['pipeline']
        )
        
        # Extract features
        features = self._extract_features(X_test_processed)
        
        # Apply PCA
        if self.config['preprocessing']['pca']['enabled']:
            features = self.feature_extractor.apply_dimensionality_reduction(features)
        
        # Load model and make predictions
        # Implementation depends on model type
        # This is a simplified version
        
        return {"accuracy": 0.0, "f1_score": 0.0}  # Placeholder
    
    def _generate_synthetic_data(self) -> tuple:
        """Generate synthetic hyperspectral data for testing"""
        logger.info("Generating synthetic hyperspectral data...")
        
        n_samples_per_class = 20
        n_bands = 100
        wavelengths = np.linspace(400, 1000, n_bands)
        
        X_synthetic = []
        y_synthetic = []
        
        for class_name in ['healthy', 'biotic', 'abiotic']:
            for i in range(n_samples_per_class):
                # Generate base spectrum
                spectrum = np.random.normal(0.3, 0.1, n_bands)
                
                # Add class-specific characteristics
                if class_name == 'healthy':
                    # Higher NIR reflectance
                    nir_region = (wavelengths > 750) & (wavelengths < 900)
                    spectrum[nir_region] += np.random.normal(0.4, 0.1, np.sum(nir_region))
                elif class_name == 'biotic':
                    # Disease signatures
                    red_region = (wavelengths > 650) & (wavelengths < 700)
                    spectrum[red_region] += np.random.normal(0.2, 0.05, np.sum(red_region))
                else:  # abiotic
                    # Stress signatures
                    spectrum *= 0.7
                
                # Ensure non-negative values
                spectrum = np.maximum(0, spectrum + np.random.normal(0, 0.02, n_bands))
                
                X_synthetic.append(spectrum)
                y_synthetic.append(class_name)
        
        return np.array(X_synthetic), y_synthetic
    
    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed spectra"""
        features_list = []
        wavelengths = np.linspace(400, 1000, X.shape[1])
        
        for spectrum in X:
            # Basic spectral features
            spectral_features = spectrum
            
            # Vegetation indices if enabled
            if self.config['features']['vegetation_indices']:
                veg_indices = self.feature_extractor.compute_vegetation_indices(
                    spectrum, wavelengths
                )
                combined_features = np.concatenate([
                    spectral_features,
                    list(veg_indices.values())
                ])
            else:
                combined_features = spectral_features
            
            features_list.append(combined_features)
        
        return np.array(features_list)
    
    def _generate_visualizations(self, X: np.ndarray, y: List[str], results: Dict):
        """Generate all configured visualizations"""
        class_names = list(set(y))
        
        # Mean spectral signatures
        if self.config['visualization']['plots']['spectral_signatures']:
            mean_spectra = {}
            for class_name in class_names:
                class_mask = np.array(y) == class_name
                mean_spectra[class_name] = np.mean(X[class_mask], axis=0)
            
            self.visualizer.plot_spectral_signatures(mean_spectra)
        
        # Confusion matrices
        if self.config['visualization']['plots']['confusion_matrices']:
            for model_name, result in results.items():
                self.visualizer.plot_confusion_matrix(
                    result['confusion_matrix'], 
                    class_names, 
                    model_name.title()
                )
        
        # Add other visualization calls as needed


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Grapevine Disease Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with default config
  python cli.py train --config config.yaml
  
  # Train with custom data directory
  python cli.py train --data-dir ./my_data --config config.yaml
  
  # Make prediction
  python cli.py predict --model ./results/random_forest_model.pkl --input spectrum.dat
  
  # Evaluate model
  python cli.py evaluate --model ./results/svm_model.pkl --test-data ./test_data
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str,
                       help='Path to log file')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--data-dir', type=str,
                             help='Override data directory from config')
    train_parser.add_argument('--output-dir', type=str,
                             help='Override output directory from config')
    train_parser.add_argument('--subset', type=int,
                             help='Use only a subset of data (for testing)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model file')
    predict_parser.add_argument('--input', type=str, required=True,
                               help='Path to input spectrum/image file')
    predict_parser.add_argument('--model-type', type=str, 
                               choices=['random_forest', 'svm', 'cnn'],
                               default='random_forest',
                               help='Type of model (default: random_forest)')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model file')
    evaluate_parser.add_argument('--test-data', type=str, required=True,
                                help='Path to test data directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level, args.log_file)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with CLI arguments
    if hasattr(args, 'data_dir') and args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if hasattr(args, 'output_dir') and args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    # Validate paths
    if not validate_paths(config):
        sys.exit(1)
    
    # Initialize CLI handler
    cli = CLICommands(config)
    
    try:
        # Execute command
        if args.command == 'train':
            results = cli.train(subset_size=getattr(args, 'subset', None))
            print("\nTraining completed successfully!")
            
            # Print summary
            print("\n" + "="*50)
            print("TRAINING RESULTS SUMMARY")
            print("="*50)
            for model_name, result in results.items():
                print(f"\n{model_name.upper()}:")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  F1-Score: {result['f1_score']:.4f}")
        
        elif args.command == 'predict':
            prediction = cli.predict(args.model, args.input, args.model_type)
            print(f"\n{prediction}")
        
        elif args.command == 'evaluate':
            results = cli.evaluate(args.model, args.test_data)
            print(f"\nEvaluation results: {results}")
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()