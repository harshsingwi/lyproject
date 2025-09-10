#!/usr/bin/env python3
"""
Command Line Interface for Grapevine Disease Detection System
============================================================

Updated for PNG image dataset and simplified workflow.

Usage examples:
  python cli.py train --config config.yaml
  python cli.py predict --model ./models/grapevine_model.pth --image ./organized_dataset/test/[IMAGE_NAME]
  python cli.py evaluate --model ./models/grapevine_model.pth --data-dir ./organized_dataset
"""

import os
import sys
import argparse
import yaml
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

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
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(sys.stdout)
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
    
    # Create output directories if they don't exist
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_dir = Path(config['training']['model_dir'])
    model_dir.mkdir(exist_ok=True, parents=True)
    
    return True


class GrapevineDataset(Dataset):
    """PyTorch Dataset for grapevine disease detection"""
    
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir() and d.name in ['healthy', 'diseased', 'test']])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.is_dir():
                for img_file in class_dir.glob('*.png'):
                    samples.append((str(img_file), self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class GrapevineDiseaseModel:
    """CNN Model for grapevine disease detection"""
    
    def __init__(self, num_classes: int = 3, model_name: str = 'resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model(model_name, num_classes)
        self.model.to(self.device)
        
    def _create_model(self, model_name: str, num_classes: int) -> nn.Module:
        """Create CNN model"""
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:  # Simple CNN
            model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(128 * 28 * 28, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        return model
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              config: Dict[str, Any]) -> Dict[str, List[float]]:
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            logger.info(f'Epoch {epoch+1}/{config["epochs"]}: '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            scheduler.step()
        
        return history
    
    def predict(self, image: Image.Image) -> Tuple[str, float]:
        """Make prediction on a single image"""
        self.model.eval()
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Map prediction to class name
        class_names = ['diseased', 'healthy', 'test']
        class_name = class_names[predicted.item()]
        return class_name, confidence.item()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Get the actual number of classes from the model
        num_classes = self.model.fc.out_features if hasattr(self.model, 'fc') else self.model.classifier[1].out_features
        
        # Get class names from the test loader dataset
        if hasattr(test_loader.dataset, 'dataset'):
            # Handle the case where we have a Subset
            class_names = test_loader.dataset.dataset.classes
        else:
            class_names = test_loader.dataset.classes
        
        # If the model has fewer classes than the test dataset, truncate class names
        if len(class_names) > num_classes:
            class_names = class_names[:num_classes]
            # Also truncate labels if they exceed the model's class range
            all_labels = [min(label, num_classes - 1) for label in all_labels]
        
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds, target_names=class_names)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': all_preds,
            'labels': all_labels,
            'class_names': class_names
        }
    
    def save(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"Model loaded from {path}")


class CLICommands:
    """Class containing CLI command implementations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def train(self) -> Dict[str, Any]:
        """Train the grapevine disease detection model"""
        logger.info("Starting training pipeline...")
        
        # Data transformations
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = GrapevineDataset(
            self.config['data']['data_dir'], 
            transform=train_transform
        )
        
        # Debug: Check the classes found
        print(f"Classes found in dataset: {train_dataset.classes}")
        print(f"Number of classes: {len(train_dataset.classes)}")
        
        # Update config with actual number of classes
        self.config['model']['num_classes'] = len(train_dataset.classes)
        
        # Split into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_subset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        logger.info(f"Training on {len(train_subset)} samples, validating on {len(val_subset)} samples")
        logger.info(f"Classes: {train_dataset.classes}")
        
        # Initialize and train model
        model = GrapevineDiseaseModel(
            num_classes=len(train_dataset.classes),
            model_name=self.config['model']['architecture']
        )
        
        history = model.train(train_loader, val_loader, self.config['training'])
        
        # Save model
        model_path = Path(self.config['training']['model_dir']) / 'grapevine_model.pth'
        model.save(str(model_path))
        
        # Save class mapping
        class_mapping = {
            'classes': train_dataset.classes,
            'class_to_idx': train_dataset.class_to_idx
        }
        
        mapping_path = Path(self.config['training']['model_dir']) / 'class_mapping.pkl'
        with open(mapping_path, 'wb') as f:
            pickle.dump(class_mapping, f)
        
        # Generate training plots
        self._plot_training_history(history)
        
        return {
            'model_path': str(model_path),
            'mapping_path': str(mapping_path),
            'history': history,
            'classes': train_dataset.classes
        }
    
    def predict(self, model_path: str, image_path: str) -> str:
        """Make prediction on a single image"""
        logger.info(f"Making prediction on {image_path}")
        
        # Load class mapping
        mapping_path = Path(model_path).parent / 'class_mapping.pkl'
        with open(mapping_path, 'rb') as f:
            class_mapping = pickle.load(f)
        
        # Load model
        model = GrapevineDiseaseModel(num_classes=len(class_mapping['classes']))
        model.load(model_path)
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            prediction, confidence = model.predict(image)
            
            return (f"Prediction: {prediction} "
                   f"(confidence: {confidence:.3f})")
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return f"Error: {str(e)}"
    
    def evaluate(self, model_path: str, data_dir: str) -> Dict[str, Any]:
        """Evaluate model on test data"""
        logger.info(f"Evaluating model on {data_dir}")
        
        # Load class mapping
        mapping_path = Path(model_path).parent / 'class_mapping.pkl'
        with open(mapping_path, 'rb') as f:
            class_mapping = pickle.load(f)
        
        # Load model
        model = GrapevineDiseaseModel(num_classes=len(class_mapping['classes']))
        model.load(model_path)
        
        # Create test dataset
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = GrapevineDataset(data_dir, transform=test_transform)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['training']['batch_size'], 
            shuffle=False,
            num_workers=2
        )
        
        # Evaluate
        results = model.evaluate(test_loader)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(
            results['labels'], 
            results['predictions'], 
            results['class_names']
        )
        
        logger.info(f"Evaluation accuracy: {results['accuracy']:.4f}")
        logger.info("Classification Report:\n" + results['report'])
        
        return results
    
    def _plot_training_history(self, history: Dict[str, List[float]]):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_acc'], label='Train Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_path = Path(self.config['training']['output_dir']) / 'training_history.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
    
    def _plot_confusion_matrix(self, y_true, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plot_path = Path(self.config['training']['output_dir']) / 'confusion_matrix.png'
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {plot_path}")

def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Grapevine Disease Detection System - PNG Image Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model with default config
  python cli.py train
  
  # Make prediction on single image
  python cli.py predict --model ./models/grapevine_model.pth --image ./test_image.png
  
  # Evaluate model on test data
  python cli.py evaluate --model ./models/grapevine_model.pth --data-dir ./test_data
        """
    )
    
    # Global arguments
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make prediction on an image')
    predict_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model file')
    predict_parser.add_argument('--image', type=str, required=True,
                               help='Path to input image file')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model file')
    evaluate_parser.add_argument('--data-dir', type=str, required=True,
                                help='Path to test data directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate paths
    if not validate_paths(config):
        sys.exit(1)
    
    # Initialize CLI handler
    cli = CLICommands(config)
    
    try:
        # Execute command
        if args.command == 'train':
            results = cli.train()
            print("\n✅ Training completed successfully!")
            print(f"Model saved to: {results['model_path']}")
            
        elif args.command == 'predict':
            prediction = cli.predict(args.model, args.image)
            print(f"\n{prediction}")
            
        elif args.command == 'evaluate':
            results = cli.evaluate(args.model, args.data_dir)
            print(f"\nEvaluation Accuracy: {results['accuracy']:.4f}")
            print("\nClassification Report:")
            print(results['report'])
    
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