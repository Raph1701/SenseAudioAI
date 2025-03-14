from model import GenreClassifierCNN, ClassifierResNET, ResidualBlock
from dataset import GTZANDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import os


class EarlyStopper:
    """Stop l'entraînement si la perte de validation ne s'améliore pas après 'patience' époques"""
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train(model, train_loader, test_loader, criterion, optimizer, epochs=100, patience=5):
    writer = SummaryWriter()
    early_stopper = EarlyStopper(patience=patience)
    best_acc = 0.0
    best_model_path = "resnet_best_model.pth"
    last_model_path = "resnet_last_model.pth"
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}...")
        model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.clone().detach().long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

            # Enregistrement des métriques plusieurs fois par epoch
            if batch_idx % 10 == 0:
                acc = accuracy_score(all_labels, all_preds)
                writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + batch_idx)
                writer.add_scalar("Accuracy/train", acc, epoch * len(train_loader) + batch_idx)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        # Évaluation sur le set de test
        test_acc, test_loss = test(model, test_loader, writer, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
        writer.add_scalar("Accuracy/test", test_acc, epoch)

        # Sauvegarde du meilleur modèle
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved as '{best_model_path}'")

        # Early stopping
        if early_stopper.should_stop(test_loss):
            print("⏳ Early stopping triggered!")
            break

    # Sauvegarde du dernier modèle
    torch.save(model.state_dict(), last_model_path)
    print(f"💾 Last model saved as '{last_model_path}'")
    writer.close()


def test(model, test_loader, writer, epoch):
    print(f"Test - Epoch {epoch+1}")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calcul des métriques
    report = classification_report(all_labels, all_preds, output_dict=True)
    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]
    acc = report["accuracy"]

    print(f"Test Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    # Ajout dans TensorBoard
    writer.add_scalar("Accuracy/test", acc, epoch)
    writer.add_scalar("Precision/test", precision, epoch)
    writer.add_scalar("Recall/test", recall, epoch)
    writer.add_scalar("F1-score/test", f1, epoch)

    return all_labels, all_preds



def plot_confusion_matrix(labels, preds, classes, epoch, save_dir="conf_matrices_resnet"):
    """Trace et sauvegarde la matrice de confusion"""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    
    filename = os.path.join(save_dir, f"conf_matrix_epoch_{epoch}.png")
    plt.savefig(filename)
    plt.close()
    print(f"📊 Confusion matrix saved: {filename}")

    # Ne garder que les 3 dernières matrices
    existing_matrices = sorted(os.listdir(save_dir))[-3:]
    for f in os.listdir(save_dir):
        if f not in existing_matrices:
            os.remove(os.path.join(save_dir, f))


if __name__ == '__main__':
    data_path = 'gtzan/genres_original'
    dataset = GTZANDataset(dataset_path=data_path)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = ClassifierResNET(ResidualBlock, [2, 2, 2, 2, 2], num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, test_loader, criterion, optimizer)
    #plot_confusion_matrix(labels, preds, dataset.genres, epoch="final")

    print("🎵 Training complete!")
