"""
Script to train a custom CNN model for facial emotion recognition using the FER2013 dataset.

This script trains a convolutional neural network for emotion classification using the FER2013 dataset.
The model can recognize 7 emotions: angry, disgust, fear, happy, sad, surprise, and neutral.

Usage:
    python train.py --train_dir data/train --val_dir data/test --epochs 50

Author: Moodify Team
"""

import os
import argparse
import tensorflow as tf
import numpy as np

def create_model(num_classes=7):
    """
    Create a CNN model for emotion classification.
    
    Args:
        num_classes (int): Number of emotion classes to classify
        
    Returns:
        tf.keras.Model: Compiled Keras model
    """
    # Use the standard tf.keras approach that works with TensorFlow 2.x
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

def train_model(train_dir, validation_dir, epochs=30, batch_size=32, model_save_path="emotion_model.keras"):
    """
    Train the emotion recognition model.
    
    Args:
        train_dir (str): Path to training data directory
        validation_dir (str): Path to validation data directory
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        model_save_path (str): Path to save the trained model
        
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
    
    # Create model
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
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_save_path.replace('.keras', '_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # Add checkpoint callback to save model after each epoch
        tf.keras.callbacks.ModelCheckpoint(
            filepath='checkpoint_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras',
            save_freq='epoch',
            save_best_only=False,  # Save every epoch, not just best
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nStarting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save model in the new Keras format
    model.save(model_save_path)
    print(f"\nModel saved as {model_save_path}")
    
    # Also save in HDF5 format for compatibility
    model.save(model_save_path.replace('.keras', '.h5'))
    print(f"Model also saved as {model_save_path.replace('.keras', '.h5')} for compatibility")
    
    return model, history

def main():
    """Main function to parse arguments and start training."""
    parser = argparse.ArgumentParser(
        description='Train emotion recognition model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Example usage:
  python train.py --train_dir data/train --val_dir data/test --epochs 50
  python train.py --train_dir data/train --val_dir data/test --epochs 30 --batch_size 16
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
        print("Starting emotion recognition model training...")
        model, history = train_model(
            train_dir=args.train_dir,
            validation_dir=args.val_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_path
        )
        print("\nTraining completed successfully!")
        
        # Print final metrics
        final_train_acc = max(history.history['accuracy'])
        final_val_acc = max(history.history['val_accuracy'])
        print(f"Best training accuracy: {final_train_acc:.4f}")
        print(f"Best validation accuracy: {final_val_acc:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()