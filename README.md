# 🎵 Music Genre Classification Dashboard

A Streamlit web application for classifying music genres from audio files using machine learning. This application allows you to upload audio files (WAV or MP3) and predict their music genre using a pre-trained Support Vector Machine model.

## 🚀 Features

- **Audio File Processing**: Upload and analyze WAV or MP3 audio files
- **Pre-trained SVM Model**: Uses a Support Vector Machine model trained on the GTZAN dataset
- **Feature Extraction**: Automatically extracts audio features (chroma, spectral centroid, bandwidth, etc.)
- **Genre Prediction**: Predicts music genre with confidence scores
- **Audio Visualization**: Displays waveform visualization of the uploaded audio
- **Real-time Analysis**: Instant genre prediction with visual feedback

## 🎧 Supported Genres

The application can classify audio files into 10 different music genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock

## 📁 Requirements

- Python 3.7 or higher
- Streamlit
- Librosa (for audio processing)
- Scikit-learn
- Numpy
- Pandas
- Matplotlib

## 🛠️ Installation

1. Clone or download this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit librosa scikit-learn pandas numpy matplotlib
```

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open your browser to the URL provided (typically http://localhost:8502)

## 🎯 Usage

1. **Upload Audio File**: Click "Browse files" to upload a WAV or MP3 audio file
2. **Audio Playback**: Listen to your uploaded audio file directly in the app
3. **Automatic Analysis**: The app will automatically extract features and predict the genre
4. **View Results**: See the predicted genre and confidence score
5. **Detailed Probabilities**: View probability distribution across all genres
6. **Audio Visualization**: See the waveform visualization of your audio

## 📈 Expected Performance

The pre-trained SVM model achieves approximately 68% accuracy on the test set. The app will display the confidence level for each prediction.

## 🧠 Technical Details

- **Model**: Support Vector Machine (SVM) trained on GTZAN dataset
- **Feature Extraction**: Chroma STFT, RMSE, Spectral Centroid, Spectral Bandwidth, Rolloff, Zero Crossing Rate, and 20 MFCCs
- **Audio Processing**: Librosa library for feature extraction
- **Preprocessing**: StandardScaler for feature normalization

## 📝 Notes

- The application works best with 30-second audio clips (same duration as training data)
- For best results, use clean audio recordings without background noise
- The model was trained on the GTZAN dataset, so it works best with similar music styles
- MP3 files will be converted to WAV format for processing

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements. Suggestions for enhancements include:
- Additional audio features
- More advanced ML models
- Improved visualization options
- Support for longer audio files

## 📄 License

This project is open source and available under the MIT License.