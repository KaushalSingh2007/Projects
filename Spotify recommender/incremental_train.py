"""
Script for incremental training of the emotion recognition model.

This script trains the model and saves checkpoints every 10 epochs,
deleting previous checkpoints to save storage space. This allows
resuming training from the most recent checkpoint.

Usage:
    python incremental_train.py --train_dir data/train --val_dir data/test --epochs 50

Author: Moodify Team
"""

import os
import argparse
import tensorflow as tf
import numpy as np
import glob
import re

def get_latest_checkpoint():
    """
    Find the most recent checkpoint file based on epoch number.
    
    Returns:
        str or None: Path to the latest checkpoint file, or None if no checkpoints exist
    """
    # Find all checkpoint files
    checkpoint_files = glob.glob('checkpoint_epoch_*.keras')
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the highest
    epoch_numbers = []
    for file in checkpoint_files:
        # Extract epoch number from filename (e.g., checkpoint_epoch_10_val_acc_0.45.keras)
        match = re.search(r'checkpoint_epoch_(\d+)_val_acc', file)
        if match:
            epoch_numbers.append((int(match.group(1)), file))
    
    if not epoch_numbers:
        return None
    
    # Return the file with the highest epoch number
    latest_file = max(epoch_numbers, key=lambda x: x[0])[1]
    return latest_file

def delete_old_checkpoints(keep_checkpoint):
    """
    Delete all checkpoint files except the one to keep.
    
    Args:
        keep_checkpoint (str): Path to the checkpoint file to keep
    """
    checkpoint_files = glob.glob('checkpoint_epoch_*.keras')
    
    for file in checkpoint_files:
        if file != keep_checkpoint:
            try:
                os.remove(file)
                print(f"Deleted old checkpoint: {file}")
            except Exception as e:
                print(f"Warning: Could not delete {file}: {str(e)}")

def create_model(num_classes=7):
    """
    Create a CNN model for emotion classification.
    
    Args:
        num_classes (int): Number of emotion classes to classify
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Dense layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data(train_dir, validation_dir, batch_size=32):
    """
    Prepare data generators for training with data augmentation.
    
    Args:
        train_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_generator, validation_generator)
    """
    # Data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(48, 48),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, validation_generator

class CheckpointCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to save model every 10 epochs and delete previous checkpoints.
    """
    def __init__(self, save_freq=10):
        super().__init__()
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch.
        """
        # Save checkpoint every 'save_freq' epochs (starting from epoch 1)
        if (epoch + 1) % self.save_freq == 0:
            # Create checkpoint filename
            val_acc = logs.get('val_accuracy', 0)
            checkpoint_name = f'checkpoint_epoch_{epoch+1:02d}_val_acc_{val_acc:.4f}.keras'
            
            # Save model
            self.model.save(checkpoint_name)
            print(f"\nSaved checkpoint: {checkpoint_name}")
            
            # Delete old checkpoints
            delete_old_checkpoints(checkpoint_name)

def train_model(train_dir, validation_dir, epochs=30, batch_size=32, 
                model_save_path="emotion_model.keras", start_epoch=0):
    """
    Train the emotion recognition model with incremental checkpointing.
    
    Args:
        train_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_save_path (str): Path to save the final trained model
        start_epoch (int): Starting epoch number (for resuming training)
        
    Returns:
        tuple: (model, history)
    """
    # Validate input directories
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    if not os.path.exists(validation_dir):
        raise FileNotFoundError(f"Validation directory not found: {validation_dir}")
    
    print(f"Training data directory: {train_dir}")
    print(f"Validation data directory: {validation_dir}")
    
    # Check if there's a latest checkpoint to resume from
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"Loading model from latest checkpoint: {latest_checkpoint}")
        model = tf.keras.models.load_model(latest_checkpoint)
        # Extract epoch number from checkpoint filename
        match = re.search(r'checkpoint_epoch_(\d+)_val_acc', latest_checkpoint)
        if match:
            start_epoch = int(match.group(1))
            print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Starting training from scratch")
        model = create_model()
    
    print("\nModel Architecture:")
    model.summary()
    
    # Prepare data
    print("\nPreparing data generators...")
    train_generator, validation_generator = prepare_data(train_dir, validation_dir, batch_size)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        # Custom checkpoint callback
        CheckpointCallback(save_freq=10)
    ]
    
    # Calculate total epochs to train
    total_epochs = start_epoch + epochs
    
    # Train model
    print(f"\nStarting training from epoch {start_epoch} for {epochs} more epochs (up to epoch {total_epochs})...")
    history = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=start_epoch,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(model_save_path)
    print(f"\nFinal model saved as {model_save_path}")
    
    # Also save in HDF5 format for compatibility
    model.save(model_save_path.replace('.keras', '.h5'))
    print(f"Model also saved as {model_save_path.replace('.keras', '.h5')} for compatibility")
    
    return model, history

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description='Train emotion recognition model with incremental checkpointing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python incremental_train.py --train_dir data/train --val_dir data/test --epochs 50
  python incremental_train.py --train_dir data/train --val_dir data/test --epochs 30 --batch_size 16
        """
    )
    
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--model_path', type=str, default="emotion_model.keras",
                        help='Path to save the trained model (default: emotion_model.keras)')
    
    args = parser.parse_args()
    
    try:
        print("Starting incremental emotion recognition model training...")
        model, history = train_model(
            train_dir=args.train_dir,
            validation_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_path
        )
        print("\nTraining completed successfully!")
        
        # Print final metrics
        print("Available metrics in history:", list(history.history.keys()))
        # Handle different possible metric names
        train_acc_key = 'accuracy' if 'accuracy' in history.history else 'acc' if 'acc' in history.history else None
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_acc' if 'val_acc' in history.history else None
        
        if train_acc_key:
            final_train_acc = max(history.history[train_acc_key])
            print(f"Best training accuracy: {final_train_acc:.4f}")
        else:
            print("Training accuracy metric not found")
            
        if val_acc_key:
            final_val_acc = max(history.history[val_acc_key])
            print(f"Best validation accuracy: {final_val_acc:.4f}")
        else:
            print("Validation accuracy metric not found")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()