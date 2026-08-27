import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os
from classifier_S4_group_02 import ClassifierS4Group02


class FixEMNIST:
    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = torch.flip(img, [2])
        img = torch.rot90(img, 1, [1, 2])
        return img

def get_data(batch_size):
    train_data = datasets.EMNIST(
        root="data", split="balanced", train=True, download=True, transform=FixEMNIST()
    )
    test_data = datasets.EMNIST(
        root="data", split="balanced", train=False, download=True, transform=FixEMNIST()
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader

# TRAIN FUNCTION
def run_experiment(config, save_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = get_data(config["batch_size"])

    model = ClassifierS4Group02(dropout=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    loss_fn = nn.CrossEntropyLoss()

    # ===== scheduler =====
    scheduler = None

    if config["scheduler"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    elif config["scheduler"] == "Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=config["factor"], patience=config["patience"], threshold=config["threshhold"],
        )

    metrics = []
    total_time = 0

    for epoch in range(1, config["epoch"] + 1):

        start = time.time()
        model.train()

        train_loss, train_correct = 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            logits = model.forward_logits(X)
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)

        # ===== TEST =====
        model.eval()
        test_loss, test_correct = 0, 0

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                logits = model.forward_logits(X)
                test_loss += loss_fn(logits, y).item() * X.size(0)
                test_correct += (logits.argmax(1) == y).sum().item()

        test_loss /= len(test_loader.dataset)
        test_acc = test_correct / len(test_loader.dataset)

        # ===== scheduler step =====
        if config["scheduler"] == "Plateau":
            scheduler.step(test_loss)
        elif config["scheduler"] == "StepLR":
            scheduler.step()

        epoch_time = time.time() - start
        total_time += epoch_time

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"[{config['name']}] Epoch {epoch} | LR={current_lr:.5f} | "
              f"TrainAcc={train_acc:.4f} | TestAcc={test_acc:.4f}")

        metrics.append({
            **config,
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "lr": current_lr
        })

    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # ===== 1. LR SCAN =====
    for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
        config = {
            "name": f"lr_{lr}",
            "learning_rate": lr,
            "batch_size": 64,
            "epoch": 15,
            "weight_decay": 1e-4,
            "dropout": 0.5,
            "scheduler": "None"
        }
        run_experiment(config, f"results/lr_{lr}.csv")
    
    # ===== 2. PLATEAU =====

    config = {
        "name": "plateau",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epoch": 30,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "scheduler": "Plateau",
        "threshhold": 1e-2,
        "patience": 1,
        "factor": 0.2
    }
    run_experiment(config, "results/plateau_f02_thh2.csv")
    config = {
        "name": "plateau",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epoch": 30,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "scheduler": "Plateau",
        "threshhold": 1e-3,
        "patience": 1,
        "factor": 0.2
    }
    run_experiment(config, "results/plateau_f02_thh3.csv")
    config = {
        "name": "plateau",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epoch": 30,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "scheduler": "Plateau",
        "threshhold": 1e-3,
        "patience": 1,
        "factor": 0.5
    }
    run_experiment(config, "results/plateau_f05_thh3.csv")
    config = {
        "name": "plateau",
        "learning_rate": 1e-3,
        "batch_size": 64,
        "epoch": 30,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "scheduler": "Plateau",
        "threshhold": 1e-2,
        "patience": 1,
        "factor": 0.5
    }
    run_experiment(config, "results/plateau_f05_thh2.csv")
