"""
Module for predicting emotions using a trained CNN model.

This module provides functions to load a trained emotion recognition model
and predict emotions from facial images.

Author: Moodify Team
"""

import cv2
import numpy as np
import os
import tensorflow as tf

# Emotion classes
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_model(model_path="emotion_model.keras"):
    """
    Load the trained emotion recognition model.
    
    Args:
        model_path (str): Path to the trained model file
        
    Returns:
        tf.keras.Model: Loaded Keras model
        
    Raises:
        FileNotFoundError: If the model file is not found
        Exception: If there's an error loading the model
    """
    if not os.path.exists(model_path):
        # Try alternative model format
        h5_model_path = model_path.replace('.keras', '.h5')
        if os.path.exists(h5_model_path):
            model_path = h5_model_path
        else:
            raise FileNotFoundError(f"Model file {model_path} not found. Please train the model first.")
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def predict_emotion(frame, model=None):
    """
    Predict emotion from a given frame using the trained model.
    
    Args:
        frame (numpy.ndarray): Input image frame (BGR format)
        model (tf.keras.Model, optional): Pre-loaded model
        
    Returns:
        tuple: (emotion, confidence) or (None, None) if prediction fails
    """
    try:
        # Load model if not provided
        if model is None:
            model = load_model()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Resize to model input size
        face = cv2.resize(gray, (48, 48))
        
        # Normalize pixel values
        face = face / 255.0
        
        # Reshape for model input (batch_size, height, width, channels)
        face = np.expand_dims(face, axis=0)  # Add batch dimension
        face = np.expand_dims(face, axis=-1)  # Add channel dimension
        
        # Predict
        preds = model.predict(face, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        confidence = preds[emotion_idx]
        
        return emotion, confidence
    except Exception as e:
        print(f"Error predicting emotion: {str(e)}")
        return None, None