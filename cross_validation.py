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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
import numpy as np

import warnings
warnings.filterwarnings("ignore")

config = load_config("config.yaml")

base_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToTensor()
])

def compute_dataset_mean_std(dataset, indices):
    loader = DataLoader(Subset(dataset, indices), batch_size=config["batch_size"], shuffle=False, pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.shape[1], -1)
        mean += images.mean(dim=[0, 2]) * batch_samples
        std += images.std(dim=[0, 2]) * batch_samples
        total_samples += batch_samples
    
    mean /= total_samples
    std /= total_samples
    return mean, std

def preprocess():
    dataset = ImageFolder(config["train_dir"], transform=base_transform)
    return dataset

def train_fold(model, device, train_loader, val_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        model.train()
        correct, processed = 0, 0
        train_losses, train_acc = [], []
        
        pbar = tqdm(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            pbar.set_description(f'Loss={loss.item():.4f} Acc={100 * correct / processed:.2f}%')
            train_losses.append(loss.item())
            train_acc.append(100 * correct / processed)
        
        test_loss, test_accuracy = evaluate(model, device, val_loader, criterion)
        print(f'Epoch {epoch}: Val Loss={test_loss:.4f} Val Acc={test_accuracy:.2f}%')

def evaluate(model, device, loader, criterion):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(loader.dataset)
    accuracy = 100. * correct / len(loader.dataset)
    return test_loss, accuracy

if __name__ == "__main__":
    dataset = preprocess()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dataset.samples, dataset.targets)):
        print(f'Fold {fold+1}/5')
        mean, std = compute_dataset_mean_std(dataset, train_idx)
        print(f'Fold {fold+1} Mean: {mean}, Std: {std}')
        
        train_transformer, test_transformer = create_transformers(
            mean, std,
            config["height"], config["width"],
            config["random_rotation_degrees"], config["random_affine_degrees"],
            config["random_translation"], config["brightness"],
            config["contrast"], config["saturation"], config["hue"]
        )
        
        dataset.transform = train_transformer
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
        
        model = ResNet18().to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        train_fold(model, device, train_loader, val_loader, optimizer, criterion, config["epochs"])
