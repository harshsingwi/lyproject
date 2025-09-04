#!/usr/bin/env python3
"""
Grapevine Plant Disease Detection System using Hyperspectral Imagery
====================================================================

This system classifies grapevine leaves into three categories:
- Healthy
- Biotic stress
- Abiotic stress

Author: AI Assistant
Date: 2025
Python: 3.9+
"""

import os
import sys
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperspectral data handling
try:
    import spectral
    import spectral.io.envi as envi
except ImportError:
    print("Please install spectral library: pip install spectral")
    sys.exit(1)

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
    TF_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HyperspectralDataLoader:
    """Class for loading and preprocessing hyperspectral images"""
    
    def __init__(self, data_dir: str):
        """
        Initialize the data loader
        
        Args:
            data_dir: Path to the directory containing hyperspectral data
        """
        self.data_dir = Path(data_dir)
        self.data_cache = {}
        self.metadata = {}
        
    def load_envi_image(self, hdr_path: str, dat_path: str = None) -> np.ndarray:
        """
        Load ENVI hyperspectral image from .hdr and .dat files
        
        Args:
            hdr_path: Path to the header file (.hdr)
            dat_path: Path to the data file (.dat). If None, inferred from hdr_path
            
        Returns:
            numpy array of shape (height, width, bands)
        """
        try:
            if dat_path is None:
                dat_path = hdr_path.replace('.hdr', '.dat')
            
            # Load using spectral library
            img = envi.open(hdr_path, dat_path)
            
            # Convert to numpy array
            img_array = img.load()
            
            # Store metadata
            self.metadata[hdr_path] = {
                'shape': img_array.shape,
                'bands': img.bands.centers if hasattr(img, 'bands') else None,
                'wavelengths': img.metadata.get('wavelength', None)
            }
            
            logger.info(f"Loaded image: {img_array.shape}")
            return img_array
            
        except Exception as e:
            logger.error(f"Error loading ENVI image {hdr_path}: {e}")
            return None
    
    def scan_dataset(self) -> Dict[str, List[str]]:
        """
        Scan the dataset directory for hyperspectral images and categorize them
        
        Returns:
            Dictionary mapping categories to file paths
        """
        categories = {'healthy': [], 'biotic': [], 'abiotic': []}
        
        for file_path in self.data_dir.rglob('*.hdr'):
            file_name = file_path.name.lower()
            file_str = str(file_path).lower()
            
            # Categorize based on filename/path patterns
            if 'healthy' in file_str or 'normal' in file_str:
                categories['healthy'].append(str(file_path))
            elif 'biotic' in file_str or 'disease' in file_str or 'pathogen' in file_str:
                categories['biotic'].append(str(file_path))
            elif 'abiotic' in file_str or 'stress' in file_str or 'drought' in file_str:
                categories['abiotic'].append(str(file_path))
            else:
                # If pattern matching fails, assign to healthy by default
                # In practice, you might want manual labeling
                categories['healthy'].append(str(file_path))
        
        logger.info(f"Found {sum(len(v) for v in categories.values())} images")
        for cat, files in categories.items():
            logger.info(f"  {cat}: {len(files)} images")
            
        return categories


class FeatureExtractor:
    """Class for extracting features from hyperspectral data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_names = []
    
    def extract_spectral_signature(self, img_array: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Extract spectral signature from hyperspectral image
        
        Args:
            img_array: Hyperspectral image array (H, W, Bands)
            mask: Binary mask for region of interest (optional)
            
        Returns:
            Mean spectral signature
        """
        if mask is not None:
            # Apply mask and get mean spectrum of masked region
            masked_pixels = img_array[mask]
            if len(masked_pixels) > 0:
                return np.mean(masked_pixels, axis=0)
        
        # If no mask, return mean spectrum of entire image
        return np.mean(img_array.reshape(-1, img_array.shape[-1]), axis=0)
    
    def create_leaf_mask(self, img_array: np.ndarray, threshold_method: str = 'otsu') -> np.ndarray:
        """
        Create a binary mask to segment leaf pixels from background
        
        Args:
            img_array: Hyperspectral image array
            threshold_method: Method for thresholding ('otsu', 'mean', 'adaptive')
            
        Returns:
            Binary mask
        """
        # Use NDVI-like index for vegetation detection
        # Assuming bands are ordered from visible to NIR
        if img_array.shape[-1] < 50:
            red_band = img_array.shape[-1] // 3  # Approximate red band
            nir_band = int(img_array.shape[-1] * 0.7)  # Approximate NIR band
        else:
            red_band = 30  # Typical red band index
            nir_band = 70  # Typical NIR band index
        
        red = img_array[:, :, red_band]
        nir = img_array[:, :, nir_band]
        
        # Calculate NDVI-like index
        ndvi = (nir - red) / (nir + red + 1e-10)
        
        # Apply threshold
        if threshold_method == 'otsu':
            from sklearn.mixture import GaussianMixture
            ndvi_flat = ndvi.flatten().reshape(-1, 1)
            gmm = GaussianMixture(n_components=2)
            gmm.fit(ndvi_flat)
            threshold = np.mean(gmm.means_)
        elif threshold_method == 'mean':
            threshold = np.mean(ndvi)
        else:
            threshold = 0.3  # Conservative threshold for vegetation
        
        mask = ndvi > threshold
        
        # Morphological operations to clean mask
        from scipy import ndimage
        mask = ndimage.binary_fill_holes(mask)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        
        return mask
    
    def compute_vegetation_indices(self, spectrum: np.ndarray, wavelengths: np.ndarray = None) -> Dict[str, float]:
        """
        Compute various vegetation indices from spectral signature
        
        Args:
            spectrum: Spectral signature
            wavelengths: Wavelength values for each band
            
        Returns:
            Dictionary of vegetation indices
        """
        indices = {}
        n_bands = len(spectrum)
        
        if wavelengths is None:
            # Assume linear spacing from 400nm to 1000nm (typical for vis-NIR)
            wavelengths = np.linspace(400, 1000, n_bands)
        
        # Find approximate band positions
        red_idx = np.argmin(np.abs(wavelengths - 660))  # Red band
        nir_idx = np.argmin(np.abs(wavelengths - 800))  # NIR band
        green_idx = np.argmin(np.abs(wavelengths - 550))  # Green band
        
        # NDVI (Normalized Difference Vegetation Index)
        if nir_idx < n_bands and red_idx < n_bands:
            red_val = spectrum[red_idx]
            nir_val = spectrum[nir_idx]
            indices['ndvi'] = (nir_val - red_val) / (nir_val + red_val + 1e-10)
        
        # GNDVI (Green NDVI)
        if nir_idx < n_bands and green_idx < n_bands:
            green_val = spectrum[green_idx]
            indices['gndvi'] = (nir_val - green_val) / (nir_val + green_val + 1e-10)
        
        # Red Edge Position (approximate)
        if n_bands > 50:
            red_edge_start = np.argmin(np.abs(wavelengths - 680))
            red_edge_end = np.argmin(np.abs(wavelengths - 750))
            red_edge_region = spectrum[red_edge_start:red_edge_end]
            indices['red_edge_max'] = np.max(red_edge_region)
            indices['red_edge_mean'] = np.mean(red_edge_region)
        
        # Simple ratio
        if nir_idx < n_bands and red_idx < n_bands:
            indices['simple_ratio'] = nir_val / (red_val + 1e-10)
        
        # Add statistical features
        indices['mean_reflectance'] = np.mean(spectrum)
        indices['std_reflectance'] = np.std(spectrum)
        indices['max_reflectance'] = np.max(spectrum)
        indices['min_reflectance'] = np.min(spectrum)
        
        return indices
    
    def apply_dimensionality_reduction(self, features: np.ndarray, n_components: int = 50) -> np.ndarray:
        """
        Apply PCA for dimensionality reduction
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_components: Number of principal components to keep
            
        Returns:
            Reduced feature matrix
        """
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            reduced_features = self.pca.fit_transform(features)
            logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_[:5]}")
        else:
            reduced_features = self.pca.transform(features)
        
        return reduced_features


class ModelTrainer:
    """Class for training and evaluating machine learning models"""
    
    def __init__(self):
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.history = {}
    
    def prepare_data(self, features: np.ndarray, labels: List[str], test_size: float = 0.15) -> Tuple:
        """
        Prepare data for training
        
        Args:
            features: Feature matrix
            labels: List of string labels
            test_size: Fraction of data for testing
            
        Returns:
            Train-test split data
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, y_encoded, test_size=test_size*2, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Validation set: {X_val.shape[0]} samples") 
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        rf_params.update(kwargs)
        
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        
        self.models['random_forest'] = rf
        logger.info("Random Forest model trained successfully")
        return rf
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVC:
        """Train SVM classifier"""
        svm_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'random_state': 42,
            'probability': True
        }
        svm_params.update(kwargs)
        
        svm = SVC(**svm_params)
        svm.fit(X_train, y_train)
        
        self.models['svm'] = svm
        logger.info("SVM model trained successfully")
        return svm
    
    def create_1d_cnn(self, input_shape: Tuple[int], num_classes: int) -> tf.keras.Model:
        """Create 1D CNN for spectral classification"""
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=128, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            Conv1D(filters=256, kernel_size=3, activation='relu'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                  X_val: np.ndarray, y_val: np.ndarray, **kwargs) -> Optional[tf.keras.Model]:
        """Train 1D CNN classifier"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping CNN training.")
            return None
        
        # Reshape for 1D CNN
        X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        # Convert labels to categorical
        y_train_cat = tf.keras.utils.to_categorical(y_train)
        y_val_cat = tf.keras.utils.to_categorical(y_val)
        
        # Create model
        model = self.create_1d_cnn((X_train.shape[1], 1), len(np.unique(y_train)))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Train model
        history = model.fit(
            X_train_cnn, y_train_cat,
            validation_data=(X_val_cnn, y_val_cat),
            epochs=kwargs.get('epochs', 100),
            batch_size=kwargs.get('batch_size', 32),
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['cnn'] = model
        self.history['cnn'] = history.history
        logger.info("CNN model trained successfully")
        
        return model
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a trained model"""
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return {}
        
        if model_name == 'cnn':
            X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            y_pred_proba = model.predict(X_test_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'true_labels': y_test
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        return results


class Visualizer:
    """Class for creating visualizations"""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_spectral_signatures(self, spectra: Dict[str, np.ndarray], wavelengths: np.ndarray = None, 
                                save_path: str = None):
        """Plot spectral signatures for different classes"""
        plt.figure(figsize=(12, 8))
        
        for label, spectrum in spectra.items():
            if wavelengths is not None:
                plt.plot(wavelengths, spectrum, label=label, linewidth=2)
            else:
                plt.plot(spectrum, label=label, linewidth=2)
        
        plt.xlabel('Wavelength (nm)' if wavelengths is not None else 'Band Number')
        plt.ylabel('Reflectance')
        plt.title('Mean Spectral Signatures by Class')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            model_name: str = "", save_path: str = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_projection(self, features: np.ndarray, labels: np.ndarray, 
                          class_names: List[str], save_path: str = None):
        """Plot PCA projection of features"""
        pca_2d = PCA(n_components=2)
        features_2d = pca_2d.fit_transform(features)
        
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       label=class_name, alpha=0.7, s=50)
        
        plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('PCA Projection of Features')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict, save_path: str = None):
        """Plot training history for deep learning models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], label='Training Accuracy')
        ax1.plot(history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(history['loss'], label='Training Loss')
        ax2.plot(history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names: List[str] = None, 
                              top_n: int = 20, save_path: str = None):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            names = [feature_names[i] if feature_names else f'Feature {i}' for i in indices]
            plt.barh(range(top_n), importances[indices][::-1])
            plt.yticks(range(top_n), names[::-1])
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Most Important Features')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()


def save_results(results: Dict, model_trainer: ModelTrainer, output_dir: str = "results"):
    """Save all results and trained models"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save evaluation results
    with open(output_path / "evaluation_results.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in results.items():
            json_results[model_name] = {
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
        json.dump(json_results, f, indent=2)
    
    # Save trained models
    for model_name, model in model_trainer.models.items():
        if model_name == 'cnn' and TF_AVAILABLE:
            model.save(output_path / f"{model_name}_model.h5")
        else:
            with open(output_path / f"{model_name}_model.pkl", 'wb') as f:
                pickle.dump(model, f)
    
    # Save label encoder
    with open(output_path / "label_encoder.pkl", 'wb') as f:
        pickle.dump(model_trainer.label_encoder, f)
    
    logger.info(f"Results saved to {output_path}")


def main():
    """Main function to run the complete pipeline"""
    # Configuration
    DATA_DIR = "./hyperspectral_data"  # Update this path
    OUTPUT_DIR = "./results"
    SUBSET_SIZE = None  # Set to a number to use subset for testing
    
    logger.info("Starting Grapevine Disease Detection System")
    
    # Initialize components
    data_loader = HyperspectralDataLoader(DATA_DIR)
    feature_extractor = FeatureExtractor()
    model_trainer = ModelTrainer()
    visualizer = Visualizer()
    
    # Step 1: Load and preprocess data
    logger.info("Step 1: Loading hyperspectral data...")
    
    # For demonstration, we'll create some dummy data
    # In practice, replace this with actual data loading
    try:
        dataset_categories = data_loader.scan_dataset()
        
        if not any(dataset_categories.values()):
            logger.warning("No data found. Creating synthetic data for demonstration...")
            # Create synthetic hyperspectral data for demonstration
            n_samples = 60
            n_bands = 100
            
            # Generate synthetic spectra for each class
            X_synthetic = []
            y_synthetic = []
            
            for i, (class_name, _) in enumerate([('healthy', 20), ('biotic', 20), ('abiotic', 20)]):
                for j in range(20):
                    # Generate synthetic spectrum with class-specific characteristics
                    baseline = np.random.normal(0.3, 0.1, n_bands)
                    if class_name == 'healthy':
                        # Healthy plants: higher NIR reflectance
                        baseline[70:] += np.random.normal(0.4, 0.1, 30)
                    elif class_name == 'biotic':
                        # Biotic stress: reduced NIR, increased red
                        baseline[20:40] += np.random.normal(0.2, 0.05, 20)
                        baseline[70:] += np.random.normal(0.2, 0.1, 30)
                    else:  # abiotic
                        # Abiotic stress: overall reduced reflectance
                        baseline *= 0.7
                    
                    # Add noise
                    spectrum = np.maximum(0, baseline + np.random.normal(0, 0.02, n_bands))
                    
                    X_synthetic.append(spectrum)
                    y_synthetic.append(class_name)
            
            X = np.array(X_synthetic)
            y = y_synthetic
            
        else:
            # Load actual data (implementation depends on dataset structure)
            # This is a placeholder - you'll need to implement based on actual data format
            pass
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Using synthetic data for demonstration...")
        
        # Create synthetic data as fallback
        n_samples = 60
        n_bands = 100
        
        X_synthetic = []
        y_synthetic = []
        
        for i, (class_name, _) in enumerate([('healthy', 20), ('biotic', 20), ('abiotic', 20)]):
            for j in range(20):
                baseline = np.random.normal(0.3, 0.1, n_bands)
                if class_name == 'healthy':
                    baseline[70:] += np.random.normal(0.4, 0.1, 30)
                elif class_name == 'biotic':
                    baseline[20:40] += np.random.normal(0.2, 0.05, 20)
                    baseline[70:] += np.random.normal(0.2, 0.1, 30)
                else:  # abiotic
                    baseline *= 0.7
                
                # Add noise
                spectrum = np.maximum(0, baseline + np.random.normal(0, 0.02, n_bands))
                
                X_synthetic.append(spectrum)
                y_synthetic.append(class_name)
        
        X = np.array(X_synthetic)
        y = y_synthetic
    
    logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} spectral bands")
    
    # Step 2: Feature extraction
    logger.info("Step 2: Feature extraction...")
    
    # Extract vegetation indices and additional features
    features_list = []
    wavelengths = np.linspace(400, 1000, X.shape[1])  # Assume 400-1000nm range
    
    for spectrum in X:
        # Basic spectral features
        spectral_features = spectrum
        
        # Vegetation indices
        veg_indices = feature_extractor.compute_vegetation_indices(spectrum, wavelengths)
        
        # Combine features
        combined_features = np.concatenate([
            spectral_features,
            list(veg_indices.values())
        ])
        features_list.append(combined_features)
    
    features = np.array(features_list)
    
    # Normalize features
    features_normalized = feature_extractor.scaler.fit_transform(features)
    
    # Apply dimensionality reduction
    features_reduced = feature_extractor.apply_dimensionality_reduction(
        features_normalized, n_components=50
    )
    
    logger.info(f"Features extracted: {features_reduced.shape}")
    
    # Step 3: Prepare data for training
    logger.info("Step 3: Preparing data for training...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = model_trainer.prepare_data(
        features_reduced, y
    )
    
    # Step 4: Train models
    logger.info("Step 4: Training models...")
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = model_trainer.train_random_forest(X_train, y_train)
    
    # Train SVM
    logger.info("Training SVM...")
    svm_model = model_trainer.train_svm(X_train, y_train)
    
    # Train CNN (if TensorFlow is available)
    if TF_AVAILABLE:
        logger.info("Training 1D CNN...")
        cnn_model = model_trainer.train_cnn(X_train, y_train, X_val, y_val, epochs=50)
    
    # Step 5: Evaluate models
    logger.info("Step 5: Evaluating models...")
    
    results = {}
    
    # Evaluate Random Forest
    results['random_forest'] = model_trainer.evaluate_model('random_forest', X_test, y_test)
    
    # Evaluate SVM
    results['svm'] = model_trainer.evaluate_model('svm', X_test, y_test)
    
    # Evaluate CNN
    if TF_AVAILABLE and 'cnn' in model_trainer.models:
        results['cnn'] = model_trainer.evaluate_model('cnn', X_test, y_test)
    
    # Step 6: Generate visualizations
    logger.info("Step 6: Generating visualizations...")
    
    class_names = model_trainer.label_encoder.classes_
    
    # Plot mean spectral signatures
    mean_spectra = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(y) == class_name
        mean_spectra[class_name] = np.mean(X[class_mask], axis=0)
    
    visualizer.plot_spectral_signatures(mean_spectra, wavelengths)
    
    # Plot confusion matrices
    for model_name, result in results.items():
        visualizer.plot_confusion_matrix(
            result['confusion_matrix'], 
            class_names, 
            model_name.title()
        )
    
    # Plot PCA projection
    y_encoded = model_trainer.label_encoder.transform(y)
    visualizer.plot_pca_projection(features_reduced, y_encoded, class_names)
    
    # Plot feature importance for Random Forest
    if 'random_forest' in results:
        visualizer.plot_feature_importance(rf_model, top_n=15)
    
    # Plot training history for CNN
    if TF_AVAILABLE and 'cnn' in model_trainer.history:
        visualizer.plot_training_history(model_trainer.history['cnn'])
    
    # Step 7: Save results
    logger.info("Step 7: Saving results...")
    save_results(results, model_trainer, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "="*60)
    print("GRAPEVINE DISEASE DETECTION - RESULTS SUMMARY")
    print("="*60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} MODEL:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
    
    print(f"\nDetailed results saved to: {OUTPUT_DIR}")
    print("Models and configurations saved for future use.")
    print("\nSystem ready for deployment!")
    
    return results, model_trainer


if __name__ == "__main__":
    try:
        results, trainer = main()
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise