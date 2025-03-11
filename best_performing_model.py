from src.utils import *
from src.model import *
from src.data_loader import *
from torch import optim
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings("ignore")

config = load_config("config.yaml")

def preprocess():
    mean, std = compute_mean_std(config["input_dir"])  # Compute from full dataset

    transformer = create_transformers(
        mean,
        std,
        config["height"],
        config["width"],
        config["random_rotation_degrees"],
        config["random_affine_degrees"],
        config["random_translation"],
        config["brightness"],
        config["contrast"],
        config["saturation"],
        config["hue"]
    )[0]  # Only need train_transformer

    dataset = ImageFolder(config["input_dir"], transform=transformer)
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

    return data_loader

def train(model, device, data_loader, optimizer, criterion, epochs):
    model.train()

    for epoch in range(epochs):
        print(f"EPOCH {epoch+1}/{epochs}")
        pbar = tqdm(data_loader)

        total_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(desc=f'Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100 * correct / processed:.2f}%')

        epoch_loss = total_loss / len(data_loader)
        epoch_acc = 100 * correct / processed

        print(f'Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.2f}%')

    return model

if __name__ == "__main__":
    data_loader = preprocess()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = ResNet18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.5)

    # Train on full dataset
    trained_model = train(model, device, data_loader, optimizer, criterion, config["epochs"])

    # Save final model
    os.makedirs("models", exist_ok=True)
    final_model_path = "models/final_model.pth"
    torch.save(trained_model.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
