from src.data_loader import *
from src.model import *
from src.utils import *

import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_from_directory
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import io
import os

config = load_config("config.yaml")

mean = [0.539342462170622, 0.4856492361751947, 0.43324973198145483]
std = [0.2270848481387292, 0.22017275884417153, 0.22362268174675773]

transform = transforms.Compose([
    transforms.Resize((config["height"], config["width"])),  # Resize to model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=mean, std=std),  # Normalize using dataset statistics
])

# Load the model
def load_model(model_path, device):
    """
    Load a pre-trained model from the specified path.
    """
    print(f"Loading model from: {model_path}")

    # Ensure the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = ResNet18().to(device)  # Initialize the model
    model.load_state_dict(checkpoint['model_state_dict'])  # Load model weights
    model.eval()  # Set the model to evaluation mode

    print("Model loaded successfully.")
    return model

model = load_model(config["model_path"], "cpu")
test_dataset = ImageFolder(config["test_dir"], transform=transform)
batch_size = 16
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize lists to store predictions and true labels
predicted_classes = []
true_classes = []

# Disable gradient computation for inference
with torch.no_grad():
    for images, labels in test_loader:
        # Forward pass
        outputs = model(images)
        # Get predicted class (index of the maximum logit)
        _, predicted = torch.max(outputs, 1)
        # Append predictions and true labels
        predicted_classes.extend(predicted.cpu().numpy())
        true_classes.extend(labels.cpu().numpy())

# Convert lists to numpy arrays
predicted_classes = np.array(predicted_classes)
true_classes = np.array(true_classes)

# Get class labels (folder names)
class_labels = test_dataset.classes

if __name__ == "__main__":
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('plots/confusion_matrix.png')