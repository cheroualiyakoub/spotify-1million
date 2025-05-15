# File: predict_request.py
from pydantic import BaseModel
from typing import List

class Track(BaseModel):
    track_id: str
    track_name: str
    artist_name: str
    danceability: float
    energy: float
    key: int
    genre: str
    year: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    duration_ms: int
    time_signature: int

class PredictionRequest(BaseModel):
    tracks: List[Track]