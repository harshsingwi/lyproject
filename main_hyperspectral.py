#!/usr/bin/env python3
"""
Main Pipeline for Hyperspectral Grapevine Disease Detection
==========================================================

Complete workflow for processing hyperspectral data and training models.
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from hyperspectral_processor import HyperspectralProcessor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Try to import TensorFlow for CNN models
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. CNN models will be skipped.")

logger = logging.getLogger(__name__)

class HyperspectralModelTrainer:
    """Train models for hyperspectral grapevine disease detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.history = {}
        
    def create_1d_cnn(self, input_shape: Tuple, num_classes: int) -> tf.keras.Model:
        """Create 1D CNN model for spectral classification"""
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
    
    def train_1d_cnn(self, X_train: np.ndarray, y_train: np.ndarray, 
                    X_val: np.ndarray, y_val: np.ndarray) -> tf.keras.Model:
        """Train 1D CNN model"""
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
            optimizer=Adam(learning_rate=self.config['training']['learning_rate']),
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
            epochs=self.config['training']['epochs'],
            batch_size=self.config['training']['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['1d_cnn'] = model
        self.history['1d_cnn'] = history.history
        
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf
        
        return rf
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """Train SVM classifier"""
        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42,
            probability=True
        )
        
        svm.fit(X_train, y_train)
        self.models['svm'] = svm
        
        return svm
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a trained model"""
        model = self.models.get(model_name)
        if model is None:
            logger.error(f"Model {model_name} not found")
            return {}
        
        if model_name == '1d_cnn' and TF_AVAILABLE:
            X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
            y_pred_proba = model.predict(X_test_reshaped)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("Classification Report:\n" + report)
        
        return results

def main():
    """Main hyperspectral processing pipeline"""
    # Load configuration
    config_path = "config_hyperspectral.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Starting hyperspectral grapevine disease detection pipeline")
    
    # Initialize processor
    processor = HyperspectralProcessor(config)
    
    # Process data
    logger.info("Processing hyperspectral data...")
    features, labels = processor.process_directory(config['data']['data_dir'])
    
    if len(features) == 0:
        logger.error("No features extracted. Check your data directory and file formats.")
        return
    
    # Apply dimensionality reduction
    logger.info("Applying dimensionality reduction...")
    features_reduced = processor.fit_transform(features)
    
    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        features_reduced, y_encoded, 
        test_size=config['training']['validation_split'] + config['training']['test_split'],
        random_state=42, stratify=y_encoded
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=config['training']['test_split'] / (config['training']['validation_split'] + config['training']['test_split']),
        random_state=42, stratify=y_temp
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Validation set: {X_val.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    trainer = HyperspectralModelTrainer(config)
    
    # Train based on configured architecture
    model_architecture = config['model']['architecture']
    
    if model_architecture == '1d_cnn' and TF_AVAILABLE:
        logger.info("Training 1D CNN...")
        trainer.train_1d_cnn(X_train, y_train, X_val, y_val)
    elif model_architecture == 'random_forest':
        logger.info("Training Random Forest...")
        trainer.train_random_forest(X_train, y_train)
    elif model_architecture == 'svm':
        logger.info("Training SVM...")
        trainer.train_svm(X_train, y_train)
    else:
        logger.warning(f"Unsupported model architecture: {model_architecture}")
        return
    
    # Evaluate model
    results = trainer.evaluate_model(model_architecture, X_test, y_test)
    
    # Save results and model
    output_dir = Path(config['output']['result_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_df = pd.DataFrame({
        'accuracy': [results['accuracy']],
        'model': [model_architecture]
    })
    results_df.to_csv(output_dir / 'results.csv', index=False)
    
    # Save model
    model_dir = Path(config['output']['model_dir'])
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if model_architecture == '1d_cnn' and TF_AVAILABLE:
        trainer.models['1d_cnn'].save(model_dir / 'hyperspectral_model.h5')
    else:
        import joblib
        joblib.dump(trainer.models[model_architecture], model_dir / 'hyperspectral_model.joblib')
    
    # Save label encoder
    joblib.dump(label_encoder, model_dir / 'label_encoder.joblib')
    
    # Save processor for future use
    joblib.dump(processor, model_dir / 'hyperspectral_processor.joblib')
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Model saved to {model_dir}")
    
    # Generate plots
    generate_plots(results, label_encoder, output_dir)
    
    return results

def generate_plots(results: Dict, label_encoder, output_dir: Path):
    """Generate evaluation plots"""
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
               xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    plt.close()
    
    # Accuracy bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(['Accuracy'], [results['accuracy']])
    plt.ylim(0, 1)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    main()