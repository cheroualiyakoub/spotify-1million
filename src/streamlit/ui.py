# File: streamlit_app.py
import streamlit as st
import pandas as pd
import requests
from io import StringIO
from typing import Dict, Any

# Configuration
API_URL = "http://localhost:8000/predict"

def main():
    st.title("🎵 Spotify Track Popularity Predictor")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎹 Manual Input", "📁 CSV Upload"])
    
    with tab1:
        handle_manual_input()
    
    with tab2:
        handle_csv_upload()

def handle_manual_input():
    st.markdown("Adjust the parameters below to predict track popularity!")
    
    # Create input form
    with st.form("track_features_form"):
        # Track metadata
        col1, col2, col3 = st.columns(3)
        with col1:
            track_id = st.text_input("Track ID", value="TRACK123")
            track_name = st.text_input("Track Name", value="My Awesome Track")
        with col2:
            artist_name = st.text_input("Artist Name", value="Famous Artist")
            genre = st.selectbox("Genre", ["pop", "rock", "electronic", "hiphop", "jazz"])
        with col3:
            year = st.slider("Year", 1950, 2024, 2023)
            duration_ms = st.number_input("Duration (ms)", min_value=30000, max_value=600000, value=180000)

        # Audio features - first row
        st.subheader("Audio Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            danceability = st.slider("Danceability", 0.0, 1.0, 0.7)
            energy = st.slider("Energy", 0.0, 1.0, 0.5)
            valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.7)
        with col2:
            speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
            acousticness = st.slider("Acousticness", 0.0, 1.0, 0.2)
            instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        with col3:
            liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
            loudness = st.slider("Loudness (dB)", -60.0, 0.0, -10.0)
            tempo = st.slider("Tempo (BPM)", 40.0, 220.0, 120.0)

        # Musical attributes
        st.subheader("Musical Attributes")
        col1, col2 = st.columns(2)
        with col1:
            key = st.selectbox("Key", options=list(range(0, 12)), format_func=lambda x: f"C {x}" if x == 0 else f"C# {x}" if x == 1 else f"D {x}" if x == 2 else f"D# {x}" if x == 3 else f"E {x}" if x == 4 else f"F {x}" if x == 5 else f"F# {x}" if x == 6 else f"G {x}" if x == 7 else f"G# {x}" if x == 8 else f"A {x}" if x == 9 else f"A# {x}" if x == 10 else f"B {x}")
            mode = st.radio("Mode", options=[0, 1], format_func=lambda x: "Minor" if x == 0 else "Major")
        with col2:
            time_signature = st.selectbox("Time Signature", options=[3, 4, 5, 6, 7], index=1)

        # Submit button
        if st.form_submit_button("Predict Popularity", type="primary"):
            track_data = {
                "track_id": str(track_id),
                "track_name": str(track_name),
                "artist_name": str(artist_name),
                "danceability": float(danceability),
                "energy": float(energy),
                "key": int(key),
                "genre": str(genre),
                "year": int(year),
                "loudness": float(loudness),
                "mode": int(mode),
                "speechiness": float(speechiness),
                "acousticness": float(acousticness),
                "instrumentalness": float(instrumentalness),
                "liveness": float(liveness),
                "valence": float(valence),  # Explicit float conversion
                "tempo": float(tempo),
                "duration_ms": int(duration_ms),
                "time_signature": int(time_signature)
            }

            st.write("Sending data types:", {k: type(v) for k,v in track_data.items()})
            
            with st.spinner("Analyzing track..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"tracks": [track_data]},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        results = response.json()["predictions"]
                        show_predictions(results)
                    else:
                        st.error(f"Prediction failed: {response.text}")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

def handle_csv_upload():
    st.markdown("Upload a CSV file with track features to predict popularity!")

    # Create template download
    template = """track_id,track_name,artist_name,danceability,energy,key,genre,year,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature
1234,Song Title,Artist Name,0.8,0.7,5,pop,2023,-6.0,1,0.05,0.25,0.0,0.1,0.9,120.0,200000,4"""
    
    st.download_button(
        label="Download CSV Template",
        data=template,
        file_name="spotify_template.csv",
        mime="text/csv"
    )

    # File upload section
    uploaded_file = st.file_uploader(
        "Upload your tracks CSV file",
        type=["csv"],
        help="Ensure the file matches the required format",
        key="csv_uploader"
    )

    if uploaded_file is not None:
        try:
            # Read and validate CSV
            df = pd.read_csv(uploaded_file)
            st.subheader("Uploaded Tracks Preview")
            st.dataframe(df.head(3))

            # Convert to API request format
            tracks = df.to_dict(orient='records')
            
            # Show prediction button
            if st.button("Predict Popularity from CSV", type="primary"):
                with st.spinner("Analyzing tracks..."):
                    response = requests.post(
                        API_URL,
                        json={"tracks": tracks},
                        timeout=10
                    )

                if response.status_code == 200:
                    results = response.json()["predictions"]
                    show_predictions(results)
                else:
                    st.error(f"Prediction failed: {response.text}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def show_predictions(results):
    st.subheader("Prediction Results")
    
    # Create dataframe for display
    results_df = pd.DataFrame([{
        "Track": r["track_name"],
        "Artist": r["artist"],
        "Popularity Score": r["popularity_score"],
        "Prediction": r["popularity_class"],
        "Hit Probability": r["popularity_score"]
    } for r in results])

    # Style the dataframe
    styled_df = results_df.style.format({
        "Hit Probability": "{:.2%}",
        "Popularity Score": "{:.2f}"
    }).applymap(lambda x: "background-color: #4CAF50" if x == "Hit" else "", 
              subset=["Prediction"])
    
    st.dataframe(styled_df)

    # Show summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Popularity Score", 
                 f"{results_df['Popularity Score'].mean():.2f}")
    
    with col2:
        hit_rate = results_df[results_df['Prediction'] == 'Hit'].shape[0] / len(results_df)
        st.metric("Hit Rate", f"{hit_rate:.1%}")

    # Show raw JSON toggle
    if st.checkbox("Show raw API response"):
        st.json(results)

if __name__ == "__main__":
    main()