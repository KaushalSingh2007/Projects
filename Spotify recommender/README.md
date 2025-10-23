# Moodify - AI Mood-Based Music Recommender

A Streamlit application that recommends Spotify playlists based on your mood detected from facial expressions in real-time.

## 🎯 Features

- **Real-time Facial Emotion Detection**: Uses a custom-trained CNN model on the FER2013 dataset
- **Spotify Integration**: Maps detected emotions to curated playlists
- **Webcam Integration**: Capture your mood directly from your webcam
- **Manual Override**: Select your mood manually if detection fails
- **Emotion History**: Keeps track of your recent mood detections

## 📁 Project Structure

```
moodify/
├── app.py                 # Main Streamlit application
├── train.py               # Script to train the emotion recognition model
├── requirements.txt       # Python dependencies
├── .env                   # Spotify API credentials (not included in repo)
├── README.md              # This file
├── emotion_model.keras    # Trained emotion recognition model (generated after training)
├── emotion_model.h5       # Trained emotion recognition model (HDF5 format)
└── utils/
    ├── emotion_predictor.py  # Emotion prediction using trained model
    ├── spotify_utils.py      # Spotify API integration
└── data/
    ├── train/             # Training data (FER2013 dataset)
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── sad/
    │   ├── surprise/
    │   └── neutral/
    └── test/              # Testing data (FER2013 dataset)
        ├── angry/
        ├── disgust/
        ├── fear/
        ├── happy/
        ├── sad/
        ├── surprise/
        └── neutral/
```

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Spotify API Credentials

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Create a new app
3. Copy your Client ID and Client Secret
4. Create a `.env` file in the project root with your credentials:

```
SPOTIPY_CLIENT_ID=your_spotify_client_id
SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

## 🏋️ Training the Emotion Recognition Model

### Option 1: Use Pre-trained Model

If you have a pre-trained model file, place it in the project root directory as `emotion_model.keras` or `emotion_model.h5`.

### Option 2: Train Your Own Model

1. **Download the FER2013 dataset**:
   - Go to [FER2013 on Kaggle](https://www.kaggle.com/msambare/fer2013)
   - Click on "Download" to download the dataset
   - Extract the zip file to the `data/` directory

2. **Train the model**:
   ```bash
   python train.py --train_dir data/train --val_dir data/test --epochs 50
   ```

### Expected Dataset Structure

After downloading and extracting the FER2013 dataset, your folder structure should look like:

```
data/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

## ▶️ Running the Application

After training the model (or if you want to use the manual selection feature):

```bash
streamlit run app.py
```

## 🧠 How It Works

1. **Model Training**: A CNN is trained on the FER2013 dataset to recognize 7 emotions:
   - Angry
   - Disgust
   - Fear
   - Happy
   - Sad
   - Surprise
   - Neutral

2. **Emotion Detection**: When you capture an image, the trained model analyzes your facial expression to detect your emotion.

3. **Playlist Mapping**: Each emotion is mapped to a specific Spotify playlist:
   - Happy → Feel-Good Pop
   - Sad → Chill Vibes
   - Angry → Rock Essentials
   - Neutral → Lo-Fi Beats
   - Surprise → Party Anthems
   - Fear → Calm Piano
   - Disgust → Focus Flow

4. **Spotify Integration**: The app displays an embedded Spotify player for the recommended playlist.

## 🎵 Playlist Mappings

| Emotion   | Playlist           | Spotify URI                                 |
|-----------|--------------------|---------------------------------------------|
| Happy     | Feel-Good Pop      | 37i9dQZF1DXdPec7aLTmlC                      |
| Sad       | Chill Vibes        | 37i9dQZF1DWVrtsSlLKzro                      |
| Angry     | Rock Essentials    | 37i9dQZF1DX76Wlfdnj7AP                      |
| Neutral   | Lo-Fi Beats        | 37i9dQZF1DX4WYpdgoIcn6                      |
| Surprise  | Party Anthems      | 37i9dQZF1DX0BcQWzuB7ZO                      |
| Fear      | Calm Piano         | 37i9dQZF1DWX83CujKHHOn                      |
| Disgust   | Focus Flow         | 37i9dQZF1DX3rxVfibe1L0                      |

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning model training and inference
- **OpenCV**: Image processing
- **Spotipy**: Spotify Web API integration
- **Python-dotenv**: Environment variable management

## 📈 Model Performance

The CNN model achieves approximately 73% accuracy on the FER2013 dataset with the following architecture:

```python
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.