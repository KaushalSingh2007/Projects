"""
Main Streamlit application for Moodify - AI Mood-Based Music Recommender.

This application uses a trained CNN model to detect emotions from facial expressions
and recommends Spotify playlists based on the detected mood.

Author: Moodify Team
"""

import streamlit as st
import cv2
import numpy as np
import os

# Import utility functions
from utils.emotion_predictor import predict_emotion, load_model
from utils.spotify_utils import get_playlist_url

# Set page config
st.set_page_config(
    page_title="Moodify - AI Mood-Based Music Recommender",
    page_icon="🎧",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1DB954;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .emotion-text {
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
    .playlist-container {
        text-align: center;
        margin-top: 2rem;
    }
    .confidence-bar {
        margin: 1rem 0;
    }
    .capture-button {
        display: flex;
        justify-content: center;
        margin: 1rem 0;
    }
    .history-item {
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        background-color: #f0f0f0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    # Main header
    st.markdown('<h1 class="main-header">🎧 AI Mood-Based Music Recommender</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'emotion_history' not in st.session_state:
        st.session_state.emotion_history = []
    
    # Try to load the trained model
    try:
        model = load_model("emotion_model.keras")
        st.success("Model loaded successfully!")
    except FileNotFoundError:
        st.warning("Trained model not found. Please train the model first using train.py")
        model = None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model = None
    
    # Webcam input
    img = st.camera_input("Capture your face", key="camera")
    
    # Manual emotion selection as fallback
    st.subheader("Or select your mood manually:")
    manual_emotion = st.selectbox("Select emotion", 
                                 ["happy", "sad", "angry", "neutral", "surprise", "fear", "disgust"],
                                 index=None,
                                 placeholder="Choose your mood...")
    
    # Process captured image or manual selection
    if img is not None or manual_emotion is not None:
        emotion = None
        confidence = 0.0
        
        if img is not None and model is not None:
            # Process webcam image
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            with st.spinner("Analyzing your mood..."):
                emotion, confidence = predict_emotion(frame, model)
                
            if emotion and confidence is not None:
                st.success(f"Detected emotion: **{emotion}** ({confidence*100:.1f}% confidence)")
            else:
                st.error("Could not detect emotion. Please try again or select manually.")
                emotion = None
        
        if manual_emotion is not None:
            emotion = manual_emotion
            confidence = 1.0
            st.info(f"Manually selected emotion: **{emotion}**")
        
        # If we have an emotion, recommend playlist
        if emotion:
            # Add to history
            confidence_percent = confidence * 100 if isinstance(confidence, float) else confidence
            st.session_state.emotion_history.append((emotion, confidence_percent))
            if len(st.session_state.emotion_history) > 5:  # Keep only last 5
                st.session_state.emotion_history.pop(0)
            
            # Get playlist
            playlist_name = {
                'happy': 'Feel-Good Pop',
                'sad': 'Chill Vibes',
                'angry': 'Rock Essentials',
                'neutral': 'Lo-Fi Beats',
                'surprise': 'Party Anthems',
                'fear': 'Calm Piano',
                'disgust': 'Focus Flow'
            }.get(emotion, 'Chill Vibes')
            
            playlist_url = get_playlist_url(emotion)
            
            # Display recommendation
            st.markdown(f'<div class="emotion-text">You seem {emotion}! Here\'s your playlist:</div>', 
                       unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center;'>🎵 {playlist_name}</h2>", unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown('<div class="confidence-bar">', unsafe_allow_html=True)
            confidence_value = confidence if isinstance(confidence, (int, float)) else 0.0
            st.progress(float(confidence_value) if confidence_value <= 1.0 else confidence_value/100.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Embed Spotify player
            embed_url = f"https://open.spotify.com/embed/playlist/{playlist_url.split('/')[-1]}"
            
            st.markdown('<div class="playlist-container">', unsafe_allow_html=True)
            st.markdown(f'<iframe src="{embed_url}" width="100%" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Open in Spotify button
            st.markdown(f'<div style="text-align: center; margin: 1rem 0;">'
                       f'<a href="{playlist_url}" target="_blank" style="background-color: #1DB954; color: white; padding: 10px 20px; border-radius: 30px; text-decoration: none; font-weight: bold;">Open in Spotify</a>'
                       f'</div>', unsafe_allow_html=True)
    
    # Show emotion history
    if st.session_state.emotion_history:
        st.sidebar.header("Recent Detections")
        for i, (emo, conf) in enumerate(reversed(st.session_state.emotion_history)):
            st.sidebar.markdown(f'<div class="history-item">{i+1}. {emo.capitalize()} ({conf:.1f}%)</div>', unsafe_allow_html=True)
    
    # Instructions
    st.sidebar.header("How it works")
    st.sidebar.write("""
    1. Allow camera access when prompted
    2. Position your face in the frame
    3. Click "Capture My Mood"
    4. Get a personalized playlist based on your mood!
    """)
    
    # Training instructions
    st.sidebar.header("Training the Model")
    st.sidebar.write("""
    To train your own emotion recognition model:
    1. Download the FER2013 dataset
    2. Extract it to the data/ directory
    3. Run: `python train.py --train_dir data/train --val_dir data/test`
    """)

if __name__ == "__main__":
    main()