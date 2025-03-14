import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

def pad_or_truncate(tensor, target_length=1293):
    """
    Ajuste la dernière dimension du tenseur à target_length.
    - Si la dimension est trop courte, ajoute du padding (zéros).
    - Si elle est trop longue, tronque.
    
    Args:
        tensor (torch.Tensor): Tensor de forme [C, H, W]
        target_length (int): Longueur cible de la dernière dimension (W)
    
    Returns:
        torch.Tensor: Tensor de forme [C, H, target_length]
    """
    _, h, w = tensor.shape  # Récupérer la taille actuelle

    if w < target_length:
        # Padding à droite
        pad_amount = target_length - w
        tensor = torch.nn.functional.pad(tensor, (0, pad_amount))  # Padding sur la dernière dim

    elif w > target_length:
        # Truncate
        tensor = tensor[:, :, :target_length]

    return tensor

class GTZANDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.genres = sorted(os.listdir(dataset_path))
        self.genre_to_idx = {genre: i for i, genre in enumerate(self.genres)}

        for genre in self.genres:
            genre_path = os.path.join(dataset_path, genre)
            if os.path.isdir(genre_path):
                for file in os.listdir(genre_path):
                    if file.endswith('.wav'):
                        self.data.append(os.path.join(genre_path, file))
                        self.labels.append(self.genre_to_idx[genre])

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file_path = self.data[idx]
        label = self.labels[idx]
        waveform, sr = librosa.load(file_path, sr=22050)

        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=128)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
        mel_spec = mel_spec.unsqueeze(0)
        mel_spec = pad_or_truncate(mel_spec)


        return mel_spec, label


    
