#!/usr/bin/env python3
"""
Hyperspectral Image Processor
=============================

Processes raw hyperspectral files (.raw, .hdr) and extracts features
for grapevine disease detection.
"""

import os
import numpy as np
import spectral
import h5py
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class HyperspectralProcessor:
    """Process hyperspectral images for grapevine disease detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.get('pca_components', 50))
        self.wavelengths = None
        
    def load_raw_hyperspectral(self, raw_path: str, hdr_path: str, 
                             image_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Load raw hyperspectral data with corresponding header file
        
        Args:
            raw_path: Path to .raw file
            hdr_path: Path to .hdr file
            image_shape: Expected shape (height, width, bands)
            
        Returns:
            Hyperspectral image cube
        """
        try:
            # Load ENVI header to get metadata
            if os.path.exists(hdr_path):
                hdr = spectral.envi.read_envi_header(hdr_path)
                self.wavelengths = np.array([float(w) for w in hdr.get('wavelength', [])])
                logger.info(f"Loaded wavelengths: {len(self.wavelengths)} bands")
            
            # Load raw data
            dtype = np.float32  # Adjust based on your data format
            img_data = np.fromfile(raw_path, dtype=dtype)
            
            # Reshape to image dimensions
            if img_data.size == np.prod(image_shape):
                img_cube = img_data.reshape(image_shape)
                logger.info(f"Loaded hyperspectral cube: {img_cube.shape}")
                return img_cube
            else:
                logger.error(f"Expected {np.prod(image_shape)} elements, got {img_data.size}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading raw hyperspectral data: {e}")
            return None
    
    def preprocess_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """Preprocess spectral data"""
        # Remove noise and artifacts
        spectra = np.clip(spectra, 0, 1)  # Assuming reflectance 0-1
        
        # Savitzky-Golay smoothing
        from scipy.signal import savgol_filter
        spectra = np.apply_along_axis(
            lambda x: savgol_filter(x, window_length=11, polyorder=3), 
            axis=1, arr=spectra
        )
        
        # Continuum removal
        spectra = self.continuum_removal(spectra)
        
        return spectra
    
    def continuum_removal(self, spectra: np.ndarray) -> np.ndarray:
        """Apply continuum removal to emphasize absorption features"""
        continuum_removed = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            spectrum = spectra[i]
            hull = np.maximum.accumulate(spectrum)
            continuum_removed[i] = spectrum / (hull + 1e-10)
            
        return continuum_removed
    
    def extract_spectral_features(self, img_cube: np.ndarray, 
                                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract spectral features from hyperspectral image
        
        Args:
            img_cube: Hyperspectral image (H, W, Bands)
            mask: Optional mask for ROI
            
        Returns:
            Spectral features array
        """
        if mask is not None:
            # Extract spectra from masked region
            masked_pixels = img_cube[mask]
            mean_spectrum = np.mean(masked_pixels, axis=0)
        else:
            # Use entire image
            mean_spectrum = np.mean(img_cube.reshape(-1, img_cube.shape[2]), axis=0)
        
        return mean_spectrum
    
    def compute_vegetation_indices(self, spectrum: np.ndarray) -> Dict[str, float]:
        """Compute vegetation indices from spectral signature"""
        indices = {}
        
        if self.wavelengths is None:
            logger.warning("No wavelength information available")
            return indices
        
        # Find band indices for key wavelengths
        red_idx = np.argmin(np.abs(self.wavelengths - 670))  # Red band ~670nm
        nir_idx = np.argmin(np.abs(self.wavelengths - 800))  # NIR band ~800nm
        green_idx = np.argmin(np.abs(self.wavelengths - 550))  # Green band ~550nm
        
        # NDVI
        if nir_idx < len(spectrum) and red_idx < len(spectrum):
            indices['ndvi'] = (spectrum[nir_idx] - spectrum[red_idx]) / (spectrum[nir_idx] + spectrum[red_idx] + 1e-10)
        
        # GNDVI
        if nir_idx < len(spectrum) and green_idx < len(spectrum):
            indices['gndvi'] = (spectrum[nir_idx] - spectrum[green_idx]) / (spectrum[nir_idx] + spectrum[green_idx] + 1e-10)
        
        # PRI (Photochemical Reflectance Index)
        pri_531_idx = np.argmin(np.abs(self.wavelengths - 531))
        pri_570_idx = np.argmin(np.abs(self.wavelengths - 570))
        if pri_531_idx < len(spectrum) and pri_570_idx < len(spectrum):
            indices['pri'] = (spectrum[pri_570_idx] - spectrum[pri_531_idx]) / (spectrum[pri_570_idx] + spectrum[pri_531_idx] + 1e-10)
        
        return indices
    
    def create_leaf_mask(self, img_cube: np.ndarray) -> np.ndarray:
        """Create mask to isolate leaf pixels"""
        # Use NDVI-like approach
        red_idx = np.argmin(np.abs(self.wavelengths - 670))
        nir_idx = np.argmin(np.abs(self.wavelengths - 800))
        
        red_band = img_cube[:, :, red_idx]
        nir_band = img_cube[:, :, nir_idx]
        
        # Calculate NDVI
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
        
        # Threshold to create mask
        mask = ndvi > 0.3  # Adjust threshold as needed
        
        # Clean up mask
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_fill_holes(mask)
        
        return mask
    
    def extract_all_features(self, img_cube: np.ndarray) -> np.ndarray:
        """Extract all features from hyperspectral image"""
        # Create leaf mask
        mask = self.create_leaf_mask(img_cube)
        
        # Extract mean spectrum
        mean_spectrum = self.extract_spectral_features(img_cube, mask)
        
        # Preprocess spectrum
        processed_spectrum = self.preprocess_spectra(mean_spectrum.reshape(1, -1))[0]
        
        # Compute vegetation indices
        indices = self.compute_vegetation_indices(processed_spectrum)
        
        # Combine all features
        features = np.concatenate([
            processed_spectrum,
            list(indices.values())
        ])
        
        return features
    
    def process_directory(self, data_dir: str) -> Tuple[np.ndarray, List[str]]:
        """Process all hyperspectral files in a directory"""
        features_list = []
        labels = []
        
        data_dir = Path(data_dir)
        
        # Look for raw files and corresponding hdr files
        raw_files = list(data_dir.rglob('*.raw'))
        
        for raw_file in raw_files:
            # Find corresponding hdr file
            hdr_file = raw_file.with_suffix('.hdr')
            if not hdr_file.exists():
                hdr_file = raw_file.parent / (raw_file.stem + '.hdr')
            
            # Determine image shape from metadata or config
            # This should be adjusted based on your specific data format
            img_shape = (512, 512, 224)  # Example shape, adjust as needed
            
            # Load image
            img_cube = self.load_raw_hyperspectral(str(raw_file), str(hdr_file), img_shape)
            if img_cube is None:
                continue
            
            # Extract features
            features = self.extract_all_features(img_cube)
            features_list.append(features)
            
            # Determine label from directory structure
            parent_dir = raw_file.parent.name.lower()
            if 'healthy' in parent_dir or 'normal' in parent_dir:
                labels.append('healthy')
            elif 'disease' in parent_dir or 'biotic' in parent_dir:
                labels.append('diseased')
            elif 'stress' in parent_dir or 'abiotic' in parent_dir:
                labels.append('abiotic')
            else:
                labels.append('unknown')
        
        return np.array(features_list), labels
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Apply scaling and dimensionality reduction"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        features_reduced = self.pca.fit_transform(features_scaled)
        
        return features_reduced
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform new features using fitted scaler and PCA"""
        features_scaled = self.scaler.transform(features)
        features_reduced = self.pca.transform(features_scaled)
        return features_reduced