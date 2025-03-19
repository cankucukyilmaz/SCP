import json
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import os

from src.utils import *
from src.model import *
from src.data_loader import *

import warnings
warnings.filterwarnings("ignore")

def load_hyperparameters(json_path="best_hyperparameters.json"):
    with open(json_path, "r") as f:
        hyperparams = json.load(f)
    return hyperparams

def preprocess(config, hyperparams):
    mean, std = compute_mean_std(config["train_dir"])

    train_transformer, test_transformer = create_transformers(
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
    )

    train_data = ImageFolder(config["train_dir"], transform=train_transformer)
    test_data = ImageFolder(config["test_dir"], transform=test_transformer)

    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=hyperparams["batch_size"], shuffle=False, pin_memory=True)

    return train_loader, test_loader

def topk_accuracy(output, target, k):
    with torch.no_grad():
        _, pred = output.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(target.view(-1, 1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()
    
def train(model, device, train_loader, optimizer, criterion, epoch, config):
    model.train()
    pbar = tqdm(train_loader)
    
    correct = 0
    processed = 0
    total_loss = 0
    topk_correct = 0

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

        topk_correct += topk_accuracy(y_pred, target, k=config["topk"]) * len(data)

        pbar.set_description(desc=f'Loss={loss.item():.4f} Batch={batch_idx} Accuracy={100 * correct / processed:.2f}%')

    epoch_train_loss = total_loss / len(train_loader)
    epoch_train_acc = 100 * correct / processed
    epoch_topk_acc = 100 * topk_correct / processed

    train_losses.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    train_topk_acc.append(epoch_topk_acc)

    print(f'Epoch {epoch+1}: Loss={epoch_train_loss:.4f}, Accuracy={epoch_train_acc:.2f}%, Top-{config["topk"]} Accuracy={epoch_topk_acc:.2f}%')

def test(model, device, test_loader, criterion, config):
    model.eval()
    test_loss = 0
    correct = 0
    topk_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            topk_correct += topk_accuracy(output, target, k=config["topk"]) * len(data)

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_topk_acc.append(100. * topk_correct / len(test_loader.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Top-{} Accuracy: {:.2f}%\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset), config["topk"],
        100. * topk_correct / len(test_loader.dataset)
    ))

def save_plots(train_losses, train_acc, test_losses, test_acc, train_topk_acc, test_topk_acc, topk):
    os.makedirs("plots", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join("plots", f"training_results_{timestamp}.png")
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    axs[0].plot(train_losses, label='Train Loss', color='green')
    axs[0].plot(test_losses, label='Test Loss', color='red')
    axs[0].set_title("Loss")
    axs[0].legend()
    
    axs[1].plot(train_acc, label='Train Accuracy', color='green')
    axs[1].plot(test_acc, label='Test Accuracy', color='red')
    axs[1].set_title("Accuracy")
    axs[1].legend()
    
    axs[2].plot(train_topk_acc, label=f'Train Top-{topk} Accuracy', color='blue')
    axs[2].plot(test_topk_acc, label=f'Test Top-{topk} Accuracy', color='purple')
    axs[2].set_title(f"Top-{topk} Accuracy")
    axs[2].legend()
    
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    hyperparams = load_hyperparameters("best_hyperparameters.json")

    config = load_config("config.yaml")

    train_loader, test_loader = preprocess(config, hyperparams)
    
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    train_topk_acc = []
    test_topk_acc = []

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    model = ResNet18().to(device)

    match hyperparams["optimizer"]:
        case "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=hyperparams["lr"],
                momentum=hyperparams["momentum"],
                weight_decay=hyperparams["weight_decay"]
            )
        case "Adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=hyperparams["lr"],
                betas=(hyperparams["beta1"], hyperparams["beta2"]),
                eps=hyperparams["epsilon"]
            )
        case "RMSprop":
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=hyperparams["lr"],
                alpha=hyperparams["alpha"],
                eps=hyperparams["epsilon"],
                weight_decay=hyperparams["weight_decay"]
            )
        case _:
            raise ValueError(f"Unsupported optimizer: {hyperparams['optimizer']}")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hyperparams["step_size"],
        gamma=hyperparams["gamma"]
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        train(model, device, train_loader, optimizer, criterion, epoch, config)
        scheduler.step()
        print('Current Learning Rate: ', optimizer.state_dict()["param_groups"][0]["lr"])
        test(model, device, test_loader, criterion, config)

    save_plots(train_losses, train_acc, test_losses, test_acc, train_topk_acc, test_topk_acc, config["topk"])