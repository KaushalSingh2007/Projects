"""
Module for Spotify integration utilities.

This module provides functions to interact with the Spotify Web API
and map emotions to playlists.

Author: Moodify Team
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_spotify_client():
    """
    Initialize Spotify client with credentials.
    
    Returns:
        spotipy.Spotify or None: Spotify client or None if initialization fails
    """
    try:
        client_id = os.getenv('SPOTIPY_CLIENT_ID')
        client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')
        
        if not client_id or not client_secret:
            st.warning("Spotify credentials not found in .env file")
            return None
            
        client_credentials_manager = SpotifyClientCredentials(
            client_id=client_id, 
            client_secret=client_secret
        )
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception as e:
        st.error(f"Error initializing Spotify client: {str(e)}")
        return None

def get_playlist_url(emotion):
    """
    Map emotion to Spotify playlist URL.
    
    Args:
        emotion (str): Detected emotion
        
    Returns:
        str: Spotify playlist URL
    """
    mapping = {
        "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
        "sad": "https://open.spotify.com/playlist/37i9dQZF1DWVrtsSlLKzro",
        "angry": "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP",
        "fear": "https://open.spotify.com/playlist/37i9dQZF1DWX83CujKHHOn",
        "surprise": "https://open.spotify.com/playlist/37i9dQZF1DX0BcQWzuB7ZO",
        "disgust": "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
        "neutral": "https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6"
    }
    return mapping.get(emotion, mapping["neutral"])