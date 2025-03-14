import os
import random
import librosa
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import audioread
import IPython.display as ipd

from tqdm import tqdm 

class GTZANAnalyzer:
    def __init__(self, dataset_path="gtzan\genres_original"):
        self.dataset_path = dataset_path
        self.df = self._load_metadata()

    def _load_metadata(self):
        """Load metadata of dataset by listing files and genres."""

        data = []
        genres = os.listdir(self.dataset_path)
        for genre in genres:
            genre_path = os.path.join(self.dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith('.wav'):
                        data.append({"filename": file, "genre": genre, "filepath": os.path.join(genre_path, file)})

        return pd.DataFrame(data)
    
    def plot_genre_distribution(self):
        """Plot distribution of genres."""
        plt.figure(figsize=(10,5))
        sns.countplot(y=self.df["genre"], order=self.df["genre"].value_counts().index, palette="pastel", hue=self.df["genre"], legend=False)
        plt.xlabel("Number of samples")
        plt.ylabel("Music styles")
        plt.title("Music styles distribution of GTZAN")
        plt.show()

    def play_random_sample(self):
        """Play a random sample from the dataset"""
        sample = self.df.sample(1).iloc[0]
        print(f"Playing: {sample['filename']} ({sample['genre']})")
        return ipd.Audio(sample['filepath'])
    
    def plot_spectrogram(self, filename):
        """Plot melspectrogram for a specific audio file"""
        file_path = self.df[self.df['filename'] == filename]['filepath'].values[0]
        y, sr = librosa.load(file_path, sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(10, 5))
        librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"MelSpectrogram of {filename}")
        plt.show()

    def extract_features(self, filename, plot=True):
        """Extract basic features for a given audio file"""
        file_path = self.df[self.df["filename"] == filename]["filepath"].values[0]
        y, sr = librosa.load(file_path, sr=None)

        # Features extraction
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) 
        chroma =librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr) 
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        if plot:
            # Affichage des features
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 2, 1)
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
            plt.title("MFCCs")
            
            plt.subplot(3, 2, 2)
            librosa.display.specshow(chroma, x_axis='time', y_axis='chroma')
            plt.colorbar()
            plt.title("Chroma")
            
            plt.subplot(3, 2, 3)
            plt.plot(spectral_contrast.mean(axis=1))
            plt.title("Spectral Contrast")
            
            plt.subplot(3, 2, 4)
            plt.plot(zero_crossing_rate[0])
            plt.title("Zero Crossing Rate")
            
            plt.subplot(3, 2, 5)
            plt.plot(rms[0])
            plt.title("RMS Energy")
            
            plt.tight_layout()
            plt.show()

        print(f"MFCC décrit le contenu fréquentiel du son sur une échelle perceptuelle.")
        print(f"Chroma indique la présence des différentes notes de musique.")
        print(f"Spectral Contrast mesure la différence d'énergie entre les pics et les creux dans le spectre.")
        print(f"Zero Crossing Rate indique la rapidité des variations de signal (utile pour détecter percussions vs. voix).")
        print(f"RMS mesure de l'énergie globale du signal.")

        return {
            "mfccs": mfccs,
            "chroma": chroma,
            "spectral_contrast": spectral_contrast,
            "zero_crossing_rate": zero_crossing_rate,
            "rms": rms
        }

    def clean_dataset(self):
        """Suppress all corrupted files"""
        genres = os.listdir(self.dataset_path)
        for genre in tqdm(genres):
            genre_path = os.path.join(self.dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(genre_path, file)
                        try:
                            wav, sr = librosa.load(file_path)
                        except (audioread.exceptions.NoBackendError) as e:
                            print(f"File corrupted: {file} -> Removing it.")
                            os.remove(file_path)

if __name__ == "__main__":
    analyzer = GTZANAnalyzer()
    #analyzer.plot_genre_distribution()
    #analyzer.clean_dataset()
    #analyzer.plot_genre_distribution()
    #ipd.display(analyzer.play_random_sample())
    #analyzer.plot_spectrogram("blues.00000.wav")
    analyzer.extract_features("blues.00000.wav")