# Speech Emotion Recognition with Mood-Based Music Recommendation
A complete end-to-end system that detects 9 emotions from speech audio and recommends songs to uplift the user's mood. 
Built with Python, scikit-learn, librosa and Streamlit.

## Features
- 9 Emotion Detection: neutral, calm, happy, sad, angry, fear, disgust, surprise (ps), boredom
- 4 ML Models: SVM (95.5%), Random Forest (95.8%), MLP (96.1%), Gradient Boosting (93.6%)
- 218 Audio Features: MFCCs, mel-spectrogram, chroma, pitch, spectral, temporal, prosodic
- Song Recommendations: Content-based filtering with cosine similarity across 38 curated songs
- Streamlit Dashboard: Upload audio or record live, see predictions and get recommendations
- 3 Datasets: RAVDESS (2,452 samples), TESS (2,800 samples), EMO-DB (535 samples)

## Quick Start
1. Clone the repository
https://github.com/TheSalmanShariff/speech-emotion-recognition.git

2. Install dependencies
install -r requirements.txt

3. Run the Jupyter notebook (training + evaluation)
jupyter notebook speech_emotion_recognition.ipynb

Run all cells from top to bottom. This loads the datasets, extracts features, trains models, evaluates performance
and saves the best model.

4. Run the Streamlit dashboard
streamlit run dashboard.py

Opens in your browser at http://localhost:8501.
Upload a WAV/MP3 file or record live through your microphone.

## How It Works
1. Audio is loaded and padded/trimmed to 3 seconds at 22,050 Hz sample rate
2. 218 acoustic features are extracted (MFCCs, mel-spectrogram, chroma, pitch, spectral, temporal, prosodic)
3. Features are standardized and fed into the trained ML model
4. The predicted emotion is matched to a target mood profile
5. Songs are ranked by weighted cosine similarity and the top 5 are recommended

## Datasets
| Dataset | Samples | Emotions                                                  | Language |
|---------|---------|-----------------------------------------------------------|----------|
| RAVDESS | 2,452   | neutral, calm, happy, sad, angry, fear, disgust, surprise | English  |
| TESS    | 2,800   | neutral, happy, sad, angry, fear, disgust, surprise       | English  |
| EMO-DB  | 535     | neutral, happy, sad, angry, fear, disgust, boredom        | German   |
| Total   | 5,787.  | 9 unique emotions.                                        |          |

## Results
| Model             | Accuracy |
|-------------------|----------|
| MLP               | 96.1%    |
| Random Forest     | 95.8%.   |
| SVM               | 95.5%    |
| Gradient Boosting | 93.6%    |

Best per-class: Pleasant Surprise (F1: 0.997), Angry (F1: 0.975), Disgust (F1: 0.967)
Weighted average: Precision 96.2%, Recall 96.1%, F1-score 96.1%

## Dashboard Features
- Upload audio files (WAV, MP3, OGG, FLAC)
- Live microphone recording
- Model selection (MLP, SVM, Random Forest, Gradient Boosting)
- Emotion probability bar chart with color coding
- Dominant emotion card with emoji and confidence
- Audio waveform visualization
- Top 5 song recommendations with match scores
- Analysis history tracking

## Song Recommendation
The system recommends songs to uplift mood, not match it.
- Sad mood means recommendations suggests upbeat, energetic songs
- Angry mood means recommendations suggests calm, relaxing songs
- Fear mood means recommendations suggests comforting, peaceful songs
- Boredom mood means recommendations suggests exciting, high-energy songs

Uses weighted cosine similarity with 6 audio features: valence (0.30), energy (0.25), tempo (0.15), danceability (0.15), 
acousticness (0.10), instrumentalness (0.05).

## Technologies Used
- Python 3.8+
- librosa — audio feature extraction
- scikit-learn — machine learning models
- Streamlit — interactive web dashboard
- matplotlib / seaborn — visualization
- NumPy / Pandas — data processing
- SoundDevice / SoundFile — live audio recording

## Author
Salman Shariff — CS 5100, Foundations of Artificial Intelligence, Northeastern University

## References
- Burkhardt et al. (2005). A database of German emotional speech. Interspeech.
- Livingstone & Russo (2018). The RAVDESS. PLoS ONE, 13(5).
- Dupuis & Pichora-Fuller (2010). Toronto Emotional Speech Set (TESS).
- McFee et al. (2015). librosa: Audio and music signal analysis in Python.
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. JMLR.