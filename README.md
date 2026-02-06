# MoodSense Streamlit Dashboard

A comprehensive interactive dashboard for showcasing the MoodSense music mood classification project.

## Features

- **🏠 Home**: Project overview and quick navigation
- **📊 Project Overview**: Detailed objectives, datasets, and methodology
- **📈 EDA Insights**: Interactive visualizations of data patterns
- **🤖 Model Performance**: Algorithm comparison and SHAP analysis
- **🎯 Real-Time Classification**: Predict mood from audio features
- **📚 Literature & References**: Complete bibliography and benchmarks

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Optional: Place your datasets** (if available):
   - `Spotify_Tracks_Dataset.csv` (114K tracks with audio features)
   - `spotify_millsongdata.csv` (57K songs with lyrics)
   - `best_model.pkl` (trained Random Forest model)

   If these files are not present, the dashboard will use sample data for demonstration.

## Running the Dashboard

### Method 1: Streamlit Command
```bash
streamlit run streamlit_app.py
```

### Method 2: Python Command
```bash
python -m streamlit run streamlit_app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## Usage Guide

### 1. Navigation
Use the sidebar to navigate between different pages:
- Click on any page to explore different aspects of the project
- Use tabs within pages for detailed sub-sections

### 2. Real-Time Classification
Navigate to "🎯 Real-Time Classification" to:
- **Manual Input**: Enter audio features manually using sliders
- **Use Presets**: Select from pre-configured mood examples
- **Random Sample**: Generate random feature values for testing
- **Batch Classification**: Upload a CSV file to classify multiple songs

#### CSV Format for Batch Classification
Your CSV should have the following columns:
```
valence,energy,danceability,acousticness,speechiness,instrumentalness,liveness,tempo,loudness,mode
0.85,0.78,0.82,0.15,0.05,0.02,0.12,128.0,-5.2,1
0.25,0.32,0.45,0.68,0.04,0.15,0.11,85.0,-11.5,0
```

Optional columns: `track_name`, `artists`

### 3. Exploring EDA
The EDA section provides:
- **Feature Distributions**: Histograms colored by mood
- **Correlations**: Heatmap and scatter matrix
- **Genre Analysis**: Violin plots and mood distributions
- **Lyrics Analysis**: Thematic distribution and artist statistics

### 4. Model Performance
View comprehensive model evaluation:
- Performance metrics table
- Confusion matrices (counts and normalized)
- SHAP feature importance
- Learning curves and cross-validation results

## Project Structure

```
moodsense-dashboard/
│
├── streamlit_app.py          # Main dashboard application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── (Optional) Data files:
│   ├── Spotify_Tracks_Dataset.csv
│   ├── spotify_millsongdata.csv
│   └── best_model.pkl
│
└── (Optional) Outputs:
    └── mood_predictions.csv   # Generated from batch classification
```

## Features Explanation

### Audio Features (Spotify API)

| Feature | Range | Description |
|---------|-------|-------------|
| **Valence** | 0-1 | Musical positivity (high = happy, low = sad) |
| **Energy** | 0-1 | Intensity and activity level |
| **Danceability** | 0-1 | Suitability for dancing |
| **Acousticness** | 0-1 | Confidence of acoustic (non-electronic) sound |
| **Speechiness** | 0-0.5 | Presence of spoken words |
| **Instrumentalness** | 0-1 | Likelihood of no vocals |
| **Liveness** | 0-0.5 | Presence of audience |
| **Tempo** | 60-200 | Beats per minute (BPM) |
| **Loudness** | -20-0 | Overall loudness in decibels |
| **Mode** | 0 or 1 | Major (1) or Minor (0) |

### Mood Categories

1. **Happy** 😊: High valence, high energy, major mode
2. **Sad** 😢: Low valence, low energy, minor mode
3. **Energetic** ⚡: Very high energy, fast tempo
4. **Calm** 🧘: Low energy, high acousticness, slow tempo
5. **Angry** 😠: Very high energy, high loudness, low valence
6. **Neutral** 😐: Balanced features

## Customization

### Adding Your Own Data

Replace the sample data loading with your actual datasets in the `load_data()` function:

```python
@st.cache_data
def load_data():
    audio_df = pd.read_csv('path/to/your/Spotify_Tracks_Dataset.csv')
    lyrics_df = pd.read_csv('path/to/your/spotify_millsongdata.csv')
    return audio_df, lyrics_df
```

### Using Your Trained Model

Place your trained model pickle file (`best_model.pkl`) in the same directory and it will be automatically loaded for real-time classification.

## Troubleshooting

### Common Issues

**1. Module not found error**
```bash
pip install -r requirements.txt --upgrade
```

**2. Port already in use**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**3. CSV upload error**
Ensure your CSV has all required columns with correct spelling

**4. Display issues**
Try a different browser or clear browser cache

## Performance Tips

- The dashboard uses caching to improve performance
- Large datasets (>100K rows) may take time to load initially
- Batch classification of >1000 songs may take a few seconds

## Academic Context

**Course**: MS DSP 422 - Practical Machine Learning  
**Team**: Group 3  
- Ankit Mittal
- Albin Anto Jose
- Nandini Bag
- Kasheena Mulla

**Academic Year**: 2025-2026

## Key Results

- **Best Model**: Random Forest with PCA features
- **Accuracy**: 85.24%
- **F1-Score**: 84.71%
- **Dataset**: 114,000 Spotify tracks
- **Mood Categories**: 6 classes

## Future Enhancements

- [ ] Integration with Spotify API for live song analysis
- [ ] Lyrics sentiment analysis integration
- [ ] User feedback collection for label validation
- [ ] Export trained model for deployment
- [ ] Playlist generation based on mood
- [ ] Audio visualization with waveforms

## License

This project is developed for academic purposes as part of MS DSP 422.

## Support

For questions or issues:
1. Check this README
2. Review the dashboard's built-in tooltips (hover over '?' icons)
3. Consult the project documentation

## Acknowledgments

- Spotify API for audio features
- SHAP library for explainability
- Streamlit for the dashboard framework
- Literature references cited in the dashboard

---

**Made with ❤️ by Team Group 3**
