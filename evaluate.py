import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from model import GenreClassifierCNN, ClassifierResNET, ResidualBlock  # Assure-toi que le modèle est bien importé
from dataset import GTZANDataset  # Assure-toi que ton dataset est bien importé

# 🔹 Paramètres
MODEL_PATH = "resnet_best_model.pth"  # Chemin du modèle sauvegardé
DATA_PATH = "gtzan/genres_original"  # Chemin des données
BATCH_SIZE = 64  # Taille des batchs
CONF_MATRIX_PATH = "resnet_confusion_matrix.png"  # Fichier de sauvegarde

def load_model(model_path, num_classes):
    """Charge le modèle depuis un fichier .pth"""
    model = ClassifierResNET(block=ResidualBlock, layers=[2, 2, 2, 2, 2], num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()  # Mode évaluation
    return model

def evaluate(model, test_loader):
    """Évalue le modèle sur test_loader et retourne les prédictions et labels"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    return all_labels, all_preds

def plot_confusion_matrix(labels, preds, classes, save_path):
    """Affiche et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)  # Sauvegarde l'image
    plt.show()

if __name__ == "__main__":
    # Chargement des données
    dataset = GTZANDataset(dataset_path=DATA_PATH)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Chargement du modèle
    model = load_model(MODEL_PATH, num_classes=len(dataset.genres))

    # Évaluation
    labels, preds = evaluate(model, test_loader)

    # Affichage de la matrice de confusion
    plot_confusion_matrix(labels, preds, dataset.genres, CONF_MATRIX_PATH)
    print(f"📸 Matrice de confusion sauvegardée sous '{CONF_MATRIX_PATH}'")
