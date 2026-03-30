#Song Recommendation System for Speech Emotion Recognition.
#Recommends songs to uplift mood based on detected emotions.
#Uses content-based filtering with cosine similarity between emotion mood profiles and song audio features.

import numpy as np
from typing import List, Dict
from collections import defaultdict

# What kind of music should we recommend for each emotion?
# The idea is to uplift, so sad people get happy songs, angry people get calm songs, etc.
EMOTION_TARGET_PROFILES = {
    'happy': {
        'valence': 0.8, 'energy': 0.7, 'tempo': 120,
        'danceability': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.1
    },
    'sad': {
        'valence': 0.75, 'energy': 0.7, 'tempo': 115,
        'danceability': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.1
    },
    'angry': {
        'valence': 0.6, 'energy': 0.4, 'tempo': 80,
        'danceability': 0.5, 'acousticness': 0.6, 'instrumentalness': 0.3
    },
    'fear': {
        'valence': 0.65, 'energy': 0.4, 'tempo': 85,
        'danceability': 0.5, 'acousticness': 0.6, 'instrumentalness': 0.3
    },
    'disgust': {
        'valence': 0.75, 'energy': 0.7, 'tempo': 115,
        'danceability': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.1
    },
    'neutral': {
        'valence': 0.6, 'energy': 0.6, 'tempo': 105,
        'danceability': 0.6, 'acousticness': 0.4, 'instrumentalness': 0.2
    },
    'calm': {
        'valence': 0.6, 'energy': 0.3, 'tempo': 75,
        'danceability': 0.4, 'acousticness': 0.7, 'instrumentalness': 0.4
    },
    'ps': {
        'valence': 0.8, 'energy': 0.8, 'tempo': 130,
        'danceability': 0.75, 'acousticness': 0.3, 'instrumentalness': 0.1
    },
    'boredom': {
        'valence': 0.7, 'energy': 0.75, 'tempo': 125,
        'danceability': 0.75, 'acousticness': 0.25, 'instrumentalness': 0.1
    }
}

# Our song database with audio features for each track
# In a production system you'd pull this from Spotify's API
SONG_DATABASE = [
    # Happy / Upbeat
    {'title': 'Happy', 'artist': 'Pharrell Williams', 'genre': 'Pop',
     'valence': 0.85, 'energy': 0.75, 'tempo': 160, 'danceability': 0.8, 'acousticness': 0.2, 'instrumentalness': 0.0},
    {'title': 'Walking on Sunshine', 'artist': 'Katrina & The Waves', 'genre': 'Rock',
     'valence': 0.9, 'energy': 0.8, 'tempo': 110, 'danceability': 0.75, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Good Vibrations', 'artist': 'The Beach Boys', 'genre': 'Pop',
     'valence': 0.8, 'energy': 0.7, 'tempo': 140, 'danceability': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.0},
    {'title': 'I Gotta Feeling', 'artist': 'Black Eyed Peas', 'genre': 'Pop',
     'valence': 0.85, 'energy': 0.9, 'tempo': 128, 'danceability': 0.85, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {"title": "Can't Stop the Feeling", 'artist': 'Justin Timberlake', 'genre': 'Pop',
     'valence': 0.9, 'energy': 0.8, 'tempo': 113, 'danceability': 0.8, 'acousticness': 0.15, 'instrumentalness': 0.0},
    {'title': 'Uptown Funk', 'artist': 'Bruno Mars', 'genre': 'Pop',
     'valence': 0.9, 'energy': 0.85, 'tempo': 115, 'danceability': 0.9, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Shake It Off', 'artist': 'Taylor Swift', 'genre': 'Pop',
     'valence': 0.85, 'energy': 0.8, 'tempo': 160, 'danceability': 0.85, 'acousticness': 0.15, 'instrumentalness': 0.0},
    {"title": "Don't Stop Me Now", 'artist': 'Queen', 'genre': 'Rock',
     'valence': 0.9, 'energy': 0.9, 'tempo': 156, 'danceability': 0.8, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Levitating', 'artist': 'Dua Lipa', 'genre': 'Pop',
     'valence': 0.8, 'energy': 0.85, 'tempo': 103, 'danceability': 0.85, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Blinding Lights', 'artist': 'The Weeknd', 'genre': 'Pop',
     'valence': 0.7, 'energy': 0.85, 'tempo': 171, 'danceability': 0.8, 'acousticness': 0.1, 'instrumentalness': 0.0},

    # Energetic / Motivational
    {'title': 'Eye of the Tiger', 'artist': 'Survivor', 'genre': 'Rock',
     'valence': 0.6, 'energy': 0.95, 'tempo': 109, 'danceability': 0.6, 'acousticness': 0.05, 'instrumentalness': 0.0},
    {'title': 'Stronger', 'artist': 'Kanye West', 'genre': 'Hip-Hop',
     'valence': 0.5, 'energy': 0.9, 'tempo': 100, 'danceability': 0.7, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'genre': 'Electronic',
     'valence': 0.4, 'energy': 0.95, 'tempo': 126, 'danceability': 0.75, 'acousticness': 0.05, 'instrumentalness': 0.0},
    {'title': 'We Will Rock You', 'artist': 'Queen', 'genre': 'Rock',
     'valence': 0.6, 'energy': 0.9, 'tempo': 81, 'danceability': 0.65, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Thunder', 'artist': 'Imagine Dragons', 'genre': 'Rock',
     'valence': 0.6, 'energy': 0.85, 'tempo': 85, 'danceability': 0.6, 'acousticness': 0.1, 'instrumentalness': 0.0},
    {'title': 'Lose Yourself', 'artist': 'Eminem', 'genre': 'Hip-Hop',
     'valence': 0.4, 'energy': 0.9, 'tempo': 86, 'danceability': 0.65, 'acousticness': 0.05, 'instrumentalness': 0.0},
    {'title': 'Fight Song', 'artist': 'Rachel Platten', 'genre': 'Pop',
     'valence': 0.7, 'energy': 0.75, 'tempo': 92, 'danceability': 0.6, 'acousticness': 0.2, 'instrumentalness': 0.0},
    {'title': 'Roar', 'artist': 'Katy Perry', 'genre': 'Pop',
     'valence': 0.7, 'energy': 0.8, 'tempo': 90, 'danceability': 0.65, 'acousticness': 0.15, 'instrumentalness': 0.0},

    # Calm / Relaxing
    {'title': 'Weightless', 'artist': 'Marconi Union', 'genre': 'Ambient',
     'valence': 0.5, 'energy': 0.1, 'tempo': 60, 'danceability': 0.2, 'acousticness': 0.9, 'instrumentalness': 0.95},
    {'title': 'Strawberry Swing', 'artist': 'Coldplay', 'genre': 'Indie',
     'valence': 0.7, 'energy': 0.3, 'tempo': 90, 'danceability': 0.5, 'acousticness': 0.6, 'instrumentalness': 0.1},
    {'title': 'Watermark', 'artist': 'Enya', 'genre': 'New Age',
     'valence': 0.5, 'energy': 0.2, 'tempo': 70, 'danceability': 0.3, 'acousticness': 0.8, 'instrumentalness': 0.7},
    {"title": "Gymnopédie No. 1", 'artist': 'Erik Satie', 'genre': 'Classical',
     'valence': 0.4, 'energy': 0.1, 'tempo': 60, 'danceability': 0.2, 'acousticness': 0.95, 'instrumentalness': 1.0},
    {'title': 'Clair de Lune', 'artist': 'Claude Debussy', 'genre': 'Classical',
     'valence': 0.5, 'energy': 0.15, 'tempo': 65, 'danceability': 0.25, 'acousticness': 0.95, 'instrumentalness': 1.0},
    {'title': 'River Flows in You', 'artist': 'Yiruma', 'genre': 'Piano',
     'valence': 0.6, 'energy': 0.2, 'tempo': 75, 'danceability': 0.3, 'acousticness': 0.95, 'instrumentalness': 1.0},
    {'title': 'Nuvole Bianche', 'artist': 'Ludovico Einaudi', 'genre': 'Classical',
     'valence': 0.5, 'energy': 0.2, 'tempo': 70, 'danceability': 0.3, 'acousticness': 0.9, 'instrumentalness': 1.0},

    # Comforting / Soothing
    {'title': 'Fix You', 'artist': 'Coldplay', 'genre': 'Rock',
     'valence': 0.3, 'energy': 0.4, 'tempo': 72, 'danceability': 0.4, 'acousticness': 0.5, 'instrumentalness': 0.0},
    {'title': 'Lean On Me', 'artist': 'Bill Withers', 'genre': 'Soul',
     'valence': 0.7, 'energy': 0.4, 'tempo': 78, 'danceability': 0.5, 'acousticness': 0.6, 'instrumentalness': 0.0},
    {'title': 'Stand by Me', 'artist': 'Ben E. King', 'genre': 'Soul',
     'valence': 0.65, 'energy': 0.35, 'tempo': 80, 'danceability': 0.5, 'acousticness': 0.5, 'instrumentalness': 0.0},
    {'title': 'Here Comes the Sun', 'artist': 'The Beatles', 'genre': 'Rock',
     'valence': 0.75, 'energy': 0.45, 'tempo': 84, 'danceability': 0.55, 'acousticness': 0.6, 'instrumentalness': 0.0},
    {'title': 'Three Little Birds', 'artist': 'Bob Marley', 'genre': 'Reggae',
     'valence': 0.8, 'energy': 0.4, 'tempo': 76, 'danceability': 0.6, 'acousticness': 0.5, 'instrumentalness': 0.0},
    {'title': 'What a Wonderful World', 'artist': 'Louis Armstrong', 'genre': 'Jazz',
     'valence': 0.7, 'energy': 0.2, 'tempo': 68, 'danceability': 0.35, 'acousticness': 0.7, 'instrumentalness': 0.0},
    {'title': 'Imagine', 'artist': 'John Lennon', 'genre': 'Rock',
     'valence': 0.55, 'energy': 0.25, 'tempo': 70, 'danceability': 0.35, 'acousticness': 0.75, 'instrumentalness': 0.0},

    # Moderate / Mixed
    {'title': 'Shape of You', 'artist': 'Ed Sheeran', 'genre': 'Pop',
     'valence': 0.75, 'energy': 0.7, 'tempo': 96, 'danceability': 0.8, 'acousticness': 0.2, 'instrumentalness': 0.0},
    {'title': 'As It Was', 'artist': 'Harry Styles', 'genre': 'Pop',
     'valence': 0.6, 'energy': 0.6, 'tempo': 173, 'danceability': 0.7, 'acousticness': 0.3, 'instrumentalness': 0.0},
    {'title': 'Heat Waves', 'artist': 'Glass Animals', 'genre': 'Indie',
     'valence': 0.5, 'energy': 0.6, 'tempo': 80, 'danceability': 0.65, 'acousticness': 0.4, 'instrumentalness': 0.0},
    {'title': 'Viva La Vida', 'artist': 'Coldplay', 'genre': 'Rock',
     'valence': 0.55, 'energy': 0.65, 'tempo': 138, 'danceability': 0.55, 'acousticness': 0.15, 'instrumentalness': 0.0},
    {'title': 'Bohemian Rhapsody', 'artist': 'Queen', 'genre': 'Rock',
     'valence': 0.4, 'energy': 0.7, 'tempo': 144, 'danceability': 0.4, 'acousticness': 0.3, 'instrumentalness': 0.0},
    {'title': 'A Sky Full of Stars', 'artist': 'Coldplay', 'genre': 'Electronic',
     'valence': 0.7, 'energy': 0.75, 'tempo': 125, 'danceability': 0.6, 'acousticness': 0.1, 'instrumentalness': 0.0},
]

# How much weight each feature gets when calculating similarity
FEATURE_WEIGHTS = {
    'valence': 0.30,
    'energy': 0.25,
    'tempo': 0.15,
    'danceability': 0.15,
    'acousticness': 0.10,
    'instrumentalness': 0.05
}


def cosine_similarity(profile, song):
    """
    Calculate how well a song matches the target mood profile.
    Uses weighted cosine similarity so valence and energy matter most.
    """
    features = ['valence', 'energy', 'tempo', 'danceability', 'acousticness', 'instrumentalness']

    vec1, vec2, weights = [], [], []
    for f in features:
        v1 = profile[f]
        v2 = song[f]
        # Normalize tempo from BPM to 0-1 range
        if f == 'tempo':
            v1 = (v1 - 60) / 120
            v2 = (v2 - 60) / 120
        vec1.append(v1)
        vec2.append(v2)
        weights.append(FEATURE_WEIGHTS[f])

    vec1 = np.array(vec1) * np.array(weights)
    vec2 = np.array(vec2) * np.array(weights)

    dot = np.dot(vec1, vec2)
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)

    if n1 == 0 or n2 == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (n1 * n2)))


def get_recommendations(emotion, num_recommendations=5):
    """
    Get song recommendations based on the detected emotion.
    Picks songs that best match the target mood profile for that emotion,
    while making sure we get a nice mix of genres.
    """
    if emotion not in EMOTION_TARGET_PROFILES:
        emotion = 'neutral'

    profile = EMOTION_TARGET_PROFILES[emotion]

    # Score every song against the target profile
    scored = []
    for song in SONG_DATABASE:
        sim = cosine_similarity(profile, song)
        scored.append((song, sim))

    # Sort by similarity (best matches first)
    scored.sort(key=lambda x: x[1], reverse=True)
    
    selected = []
    genres_used = defaultdict(int)

    for song, sim in scored:
        if len(selected) >= num_recommendations:
            break
        # Allow max 2 songs per genre to keep things varied
        if genres_used[song['genre']] < 2:
            selected.append({
                'title': song['title'],
                'artist': song['artist'],
                'genre': song['genre'],
                'similarity_score': round(sim, 3)
            })
            genres_used[song['genre']] += 1

    return selected


def format_recommendations(recommendations):
    #Print recommendations in a nice readable format.
    if not recommendations:
        return "No recommendations available."

    lines = ["Song Recommendations:\n"]
    for i, song in enumerate(recommendations, 1):
        lines.append(f"  {i}. {song['title']} — {song['artist']} ({song['genre']})  "
                     f"[Match: {song['similarity_score']:.1%}]")
    return "\n".join(lines)


# Quick test
if __name__ == "main":
    for emo in ['sad', 'angry', 'happy', 'fear', 'calm', 'neutral']:
        print(f"  Detected Emotion: {emo.upper()}")
        recs = get_recommendations(emo, num_recommendations=5)
        print(format_recommendations(recs))