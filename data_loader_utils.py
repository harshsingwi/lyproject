#!/usr/bin/env python3
"""
Data Loading and Preprocessing Utilities
========================================

Additional utilities for handling various hyperspectral data formats
and preprocessing operations.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import cv2

logger = logging.getLogger(__name__)


class DataAugmentation:
    """Class for hyperspectral data augmentation"""
    
    def __init__(self):
        self.augmentation_log = []
    
    def add_noise(self, spectrum: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Add Gaussian noise to spectrum"""
        noise = np.random.normal(0, noise_level, spectrum.shape)
        augmented = spectrum + noise
        return np.clip(augmented, 0, 1)
    
    def spectral_shift(self, spectrum: np.ndarray, shift_bands: int = 2) -> np.ndarray:
        """Simulate slight wavelength calibration shifts"""
        if shift_bands == 0:
            return spectrum
        
        shifted = np.roll(spectrum, shift_bands)
        
        # Handle edge effects
        if shift_bands > 0:
            shifted[:shift_bands] = spectrum[:shift_bands]
        else:
            shifted[shift_bands:] = spectrum[shift_bands:]
        
        return shifted
    
    def baseline_drift(self, spectrum: np.ndarray, drift_strength: float = 0.05) -> np.ndarray:
        """Simulate baseline drift"""
        n_bands = len(spectrum)
        # Create a smooth baseline drift
        drift = np.linspace(-drift_strength, drift_strength, n_bands)
        drift += np.random.normal(0, drift_strength/4, n_bands)
        
        # Apply Gaussian smoothing to make drift more realistic
        drift = ndimage.gaussian_filter1d(drift, sigma=n_bands/10)
        
        augmented = spectrum + drift
        return np.clip(augmented, 0, 1)
    
    def scale_variation(self, spectrum: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to simulate illumination changes"""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        augmented = spectrum * scale_factor
        return np.clip(augmented, 0, 1)
    
    def augment_dataset(self, X: np.ndarray, y: List[str], 
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, List[str]]:
        """
        Augment the entire dataset
        
        Args:
            X: Original spectra array
            y: Original labels
            augmentation_factor: Number of augmented samples per original sample
            
        Returns:
            Augmented X and y
        """
        X_aug = [X]  # Start with original data
        y_aug = [y]
        
        for i in range(augmentation_factor):
            X_temp = []
            for spectrum in X:
                # Apply random combination of augmentations
                aug_spectrum = spectrum.copy()
                
                if np.random.random() < 0.7:  # 70% chance
                    aug_spectrum = self.add_noise(aug_spectrum, 
                                                np.random.uniform(0.005, 0.02))
                
                if np.random.random() < 0.3:  # 30% chance
                    shift = np.random.randint(-3, 4)
                    aug_spectrum = self.spectral_shift(aug_spectrum, shift)
                
                if np.random.random() < 0.5:  # 50% chance
                    aug_spectrum = self.baseline_drift(aug_spectrum, 
                                                     np.random.uniform(0.01, 0.08))
                
                if np.random.random() < 0.6:  # 60% chance
                    aug_spectrum = self.scale_variation(aug_spectrum, (0.85, 1.15))
                
                X_temp.append(aug_spectrum)
            
            X_aug.append(np.array(X_temp))
            y_aug.append(y.copy())
        
        X_final = np.vstack(X_aug)
        y_final = []
        for y_list in y_aug:
            y_final.extend(y_list)
        
        logger.info(f"Dataset augmented from {len(y)} to {len(y_final)} samples")
        return X_final, y_final


class AdvancedPreprocessing:
    """Advanced preprocessing techniques for hyperspectral data"""
    
    def __init__(self):
        self.scalers = {}
    
    def savitzky_golay_filter(self, spectrum: np.ndarray, window_length: int = 11, 
                            polyorder: int = 2) -> np.ndarray:
        """
        Apply Savitzky-Golay smoothing filter
        
        Args:
            spectrum: Input spectrum
            window_length: Length of the filter window (must be odd)
            polyorder: Order of polynomial to fit
        """
        from scipy.signal import savgol_filter
        
        # Ensure window_length is odd and valid
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, len(spectrum))
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1
        
        return savgol_filter(spectrum, window_length, polyorder)
    
    def standard_normal_variate(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Apply Standard Normal Variate (SNV) normalization
        Removes multiplicative interferences of scatter and particle size
        """
        mean_spectrum = np.mean(spectrum)
        std_spectrum = np.std(spectrum)
        
        if std_spectrum == 0:
            return spectrum - mean_spectrum
        
        return (spectrum - mean_spectrum) / std_spectrum
    
    def multiplicative_scatter_correction(self, spectra: np.ndarray, reference: np.ndarray = None) -> np.ndarray:
        """
        Apply Multiplicative Scatter Correction (MSC)
        
        Args:
            spectra: Array of spectra (n_samples, n_bands)
            reference: Reference spectrum. If None, use mean spectrum
        """
        if reference is None:
            reference = np.mean(spectra, axis=0)
        
        corrected_spectra = []
        
        for spectrum in spectra:
            # Linear regression: spectrum = a + b * reference
            A = np.vstack([reference, np.ones(len(reference))]).T
            coeffs, _, _, _ = np.linalg.lstsq(A, spectrum, rcond=None)
            
            # Correct spectrum
            corrected = (spectrum - coeffs[1]) / coeffs[0]
            corrected_spectra.append(corrected)
        
        return np.array(corrected_spectra)
    
    def continuum_removal(self, spectrum: np.ndarray, wavelengths: np.ndarray = None) -> np.ndarray:
        """
        Apply continuum removal to emphasize absorption features
        
        Args:
            spectrum: Input spectrum
            wavelengths: Wavelength values (optional)
        """
        from scipy.spatial import ConvexHull
        
        if wavelengths is None:
            wavelengths = np.arange(len(spectrum))
        
        # Find convex hull
        points = np.column_stack((wavelengths, spectrum))
        hull = ConvexHull(points)
        hull_vertices = hull.vertices
        
        # Sort hull vertices by wavelength
        hull_vertices = hull_vertices[np.argsort(wavelengths[hull_vertices])]
        
        # Interpolate continuum
        continuum = np.interp(wavelengths, wavelengths[hull_vertices], spectrum[hull_vertices])
        
        # Remove continuum
        return spectrum / (continuum + 1e-10)
    
    def derivative_preprocessing(self, spectrum: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Apply derivative preprocessing to enhance spectral features
        
        Args:
            spectrum: Input spectrum
            order: Derivative order (1 or 2)
        """
        if order == 1:
            return np.gradient(spectrum)
        elif order == 2:
            first_derivative = np.gradient(spectrum)
            return np.gradient(first_derivative)
        else:
            raise ValueError("Only 1st and 2nd derivatives are supported")
    
    def apply_preprocessing_pipeline(self, spectra: np.ndarray, 
                                   pipeline: List[str]) -> np.ndarray:
        """
        Apply a pipeline of preprocessing steps
        
        Args:
            spectra: Input spectra array (n_samples, n_bands)
            pipeline: List of preprocessing step names
        
        Available steps:
        - 'savgol': Savitzky-Golay filtering
        - 'snv': Standard Normal Variate
        - 'msc': Multiplicative Scatter Correction
        - 'continuum': Continuum removal
        - 'derivative1': First derivative
        - 'derivative2': Second derivative
        - 'robust_scale': Robust scaling
        - 'minmax_scale': Min-Max scaling
        """
        processed = spectra.copy()
        
        for step in pipeline:
            logger.info(f"Applying preprocessing step: {step}")
            
            if step == 'savgol':
                processed = np.array([self.savitzky_golay_filter(spec) for spec in processed])
            
            elif step == 'snv':
                processed = np.array([self.standard_normal_variate(spec) for spec in processed])
            
            elif step == 'msc':
                processed = self.multiplicative_scatter_correction(processed)
            
            elif step == 'continuum':
                processed = np.array([self.continuum_removal(spec) for spec in processed])
            
            elif step == 'derivative1':
                processed = np.array([self.derivative_preprocessing(spec, 1) for spec in processed])
            
            elif step == 'derivative2':
                processed = np.array([self.derivative_preprocessing(spec, 2) for spec in processed])
            
            elif step == 'robust_scale':
                if 'robust_scaler' not in self.scalers:
                    self.scalers['robust_scaler'] = RobustScaler()
                processed = self.scalers['robust_scaler'].fit_transform(processed)
            
            elif step == 'minmax_scale':
                if 'minmax_scaler' not in self.scalers:
                    self.scalers['minmax_scaler'] = MinMaxScaler()
                processed = self.scalers['minmax_scaler'].fit_transform(processed)
            
            else:
                logger.warning(f"Unknown preprocessing step: {step}")
        
        return processed


class DatasetValidator:
    """Class for validating dataset quality and consistency"""
    
    def __init__(self):
        self.validation_results = {}
    
    def check_data_integrity(self, X: np.ndarray, y: List[str]) -> Dict:
        """Check basic data integrity"""
        results = {
            'n_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 0,
            'n_classes': len(set(y)),
            'class_distribution': {},
            'missing_values': 0,
            'infinite_values': 0,
            'negative_values': 0
        }
        
        # Class distribution
        for class_name in set(y):
            results['class_distribution'][class_name] = y.count(class_name)
        
        # Data quality checks
        results['missing_values'] = np.isnan(X).sum()
        results['infinite_values'] = np.isinf(X).sum()
        results['negative_values'] = (X < 0).sum()
        
        # Spectral range check
        results['min_reflectance'] = float(np.min(X))
        results['max_reflectance'] = float(np.max(X))
        results['mean_reflectance'] = float(np.mean(X))
        
        return results
    
    def check_spectral_consistency(self, X: np.ndarray) -> Dict:
        """Check spectral consistency across samples"""
        results = {
            'spectral_variability': {},
            'outlier_samples': [],
            'zero_variance_bands': []
        }
        
        # Check variance across bands
        band_variances = np.var(X, axis=0)
        results['zero_variance_bands'] = np.where(band_variances < 1e-10)[0].tolist()
        
        # Check for outlier samples using Mahalanobis distance
        try:
            from scipy.spatial.distance import mahalanobis
            
            mean_spectrum = np.mean(X, axis=0)
            cov_matrix = np.cov(X.T)
            
            # Handle singular covariance matrix
            try:
                inv_cov = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                inv_cov = np.linalg.pinv(cov_matrix)
            
            distances = []
            for i, spectrum in enumerate(X):
                try:
                    dist = mahalanobis(spectrum, mean_spectrum, inv_cov)
                    distances.append(dist)
                except:
                    distances.append(0)
            
            # Identify outliers (samples with distance > 3 standard deviations)
            threshold = np.mean(distances) + 3 * np.std(distances)
            results['outlier_samples'] = [i for i, d in enumerate(distances) if d > threshold]
            
        except Exception as e:
            logger.warning(f"Could not compute Mahalanobis distances: {e}")
        
        return results
    
    def generate_quality_report(self, X: np.ndarray, y: List[str]) -> str:
        """Generate a comprehensive data quality report"""
        integrity_results = self.check_data_integrity(X, y)
        consistency_results = self.check_spectral_consistency(X)
        
        report = []
        report.append("="*50)
        report.append("DATASET QUALITY REPORT")
        report.append("="*50)
        
        report.append(f"\nBasic Information:")
        report.append(f"  Samples: {integrity_results['n_samples']}")
        report.append(f"  Features (Bands): {integrity_results['n_features']}")
        report.append(f"  Classes: {integrity_results['n_classes']}")
        
        report.append(f"\nClass Distribution:")
        for class_name, count in integrity_results['class_distribution'].items():
            percentage = 100 * count / integrity_results['n_samples']
            report.append(f"  {class_name}: {count} ({percentage:.1f}%)")
        
        report.append(f"\nData Quality:")
        report.append(f"  Missing values: {integrity_results['missing_values']}")
        report.append(f"  Infinite values: {integrity_results['infinite_values']}")
        report.append(f"  Negative values: {integrity_results['negative_values']}")
        
        report.append(f"\nSpectral Range:")
        report.append(f"  Min reflectance: {integrity_results['min_reflectance']:.4f}")
        report.append(f"  Max reflectance: {integrity_results['max_reflectance']:.4f}")
        report.append(f"  Mean reflectance: {integrity_results['mean_reflectance']:.4f}")
        
        report.append(f"\nSpectral Consistency:")
        report.append(f"  Zero variance bands: {len(consistency_results['zero_variance_bands'])}")
        report.append(f"  Outlier samples: {len(consistency_results['outlier_samples'])}")
        
        if consistency_results['outlier_samples']:
            report.append(f"  Outlier sample indices: {consistency_results['outlier_samples'][:10]}")
            if len(consistency_results['outlier_samples']) > 10:
                report.append(f"    ... and {len(consistency_results['outlier_samples']) - 10} more")
        
        # Recommendations
        report.append(f"\nRecommendations:")
        
        if integrity_results['missing_values'] > 0:
            report.append("  - Handle missing values before training")
        
        if integrity_results['infinite_values'] > 0:
            report.append("  - Handle infinite values before training")
        
        if len(consistency_results['zero_variance_bands']) > 0:
            report.append("  - Consider removing zero-variance bands")
        
        if len(consistency_results['outlier_samples']) > 0:
            report.append("  - Investigate outlier samples for data quality issues")
        
        # Class balance check
        class_counts = list(integrity_results['class_distribution'].values())
        if max(class_counts) / min(class_counts) > 3:
            report.append("  - Dataset is imbalanced; consider using stratified sampling or class weights")
        
        if integrity_results['n_samples'] < 100:
            report.append("  - Small dataset size; consider data augmentation")
        
        return "\n".join(report)


