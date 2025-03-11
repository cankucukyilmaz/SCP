from src.utils import *
from src.data_loader import *
from src.model import *
import optuna
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2
import datetime
import os
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

def mean_std(dataset):
    transform = v2.Compose([v2.Resize((224, 224)), v2.ToTensor()])
    dataset.transform = transform
    
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    n_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_samples += batch_samples

    mean /= n_samples
    std /= n_samples

    return mean, std

def objective(trial):
    config = load_config("config.yaml")

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    step_size = trial.suggest_int("step_size", 3, 10)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_transform = v2.Compose([v2.Resize((224, 224)), v2.ToTensor()])
    dataset = ImageFolder(config["train_dir"], transform=base_transform)
    targets = np.array(dataset.targets)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    val_scores = []
    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(targets, targets)):
        tqdm.write(f"\nTraining on Fold {fold + 1}/{skf.get_n_splits()}...\n")
        train_dataset = ImageFolder(config["train_dir"], transform=base_transform)
        val_dataset = ImageFolder(config["train_dir"], transform=base_transform)

        mean, std = mean_std(train_dataset)
        train_transformer, val_transformer = create_transformers(
            mean, std, config["height"], config["width"],
            config["random_rotation_degrees"], config["random_affine_degrees"],
            config["random_translation"], config["brightness"],
            config["contrast"], config["saturation"], config["hue"]
        )

        train_dataset.transform = train_transformer
        val_dataset.transform = val_transformer

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)

        model = ResNet18().to(device)
        
        match optimizer_name:
            case "SGD":
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
                momentum = trial.suggest_float("momentum", 1e-5, 1, log=True)
                optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
            case "Adam":
                beta1 = trial.suggest_float("beta1", 0.8, 0.99)
                beta2 = trial.suggest_float("beta2", 0.9, 0.999)
                epsilon = trial.suggest_float("epsilon", 1e-9, 1e-6, log=True)
                optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
            case "RMSprop":
                alpha = trial.suggest_float("alpha", 0.8, 0.999)
                epsilon = trial.suggest_float("epsilon", 1e-9, 1e-4, log=True)
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
                optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, eps=epsilon, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            
            for data, target in pbar:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                y_pred = model(data)
                loss = criterion(y_pred, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                pbar.set_postfix(loss=loss.item(), acc=100 * correct / total)
            
            train_losses.append(running_loss / len(train_loader))
            train_acc.append(100 * correct / total)
            scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_losses.append(val_loss / len(val_loader))
        val_acc.append(100 * correct / total)
        val_scores.append(correct / total)
    
    save_plots(train_losses, train_acc, val_losses, val_acc)
    return np.mean(val_scores)

def save_plots(train_losses, train_acc, val_losses, val_acc):
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join("plots", f"training_results_{timestamp}.png")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(train_losses, label='Train Loss', color='blue')
    axs[0].plot(val_losses, label='Validation Loss', color='red')
    axs[0].set_title("Loss")
    axs[0].legend()
    
    axs[1].plot(train_acc, label='Train Accuracy', color='blue')
    axs[1].plot(val_acc, label='Validation Accuracy', color='red')
    axs[1].set_title("Accuracy")
    axs[1].legend()
    
    plt.savefig(plot_path)
    plt.close()
    print(f"Plots saved at {plot_path}")

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best hyperparameters: ", study.best_params)
