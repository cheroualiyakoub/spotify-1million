import os
import zipfile
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

csv_filename = "spotify_data.csv"
zip_path = "spotify-1million-tracks.zip"
dataset_kaggle_name = "amitanshjoshi/spotify-1million-tracks"

def download_data(data_folder, dataset='amitanshjoshi/spotify-1million-tracks'):
    zip_path = os.path.join(data_folder, 'spotify-1million-tracks.zip')

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.exists(zip_path):
        print("Downloading dataset from Kaggle...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset, path=data_folder, unzip=True)
        print("Download complete.")
    else:
        print("Zip file already exists.")

    return zip_path


def load_data(data_folder: str) -> pd.DataFrame:
    print("Loading data...")
    
    if  os.path.exists(zip_path):
        print("Loading CSV data...")
        df = pd.read_csv(csv_path)
        print("Data loaded successfully.")
        return df
    
    csv_path = os.path.join(data_folder, csv_filename)

    download_data(data_folder)

    print("Loading CSV data...")
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")

    return df

if __name__ == "__main__":
    load_data("./data")