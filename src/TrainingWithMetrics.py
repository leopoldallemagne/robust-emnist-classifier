import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os
from classifier_S4_group_2 import ClassifierS4Group02

class Oracle(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x.requires_grad_(True)
        logits = self.model(x)
        return logits

def training(init_metrics, metrics_file, oracle_file, dir):
    # -------------------------
    # PARAMETRES
    # -------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    learning_rate = init_metrics["learning_rate"][0]
    batch_size = init_metrics["batch_size"][0] 
    epochs = init_metrics["epoch"][0]
    weight_decay = init_metrics["weight_decay"][0]
    dropout = init_metrics["dropout"][0]

    metrics_file = dir + metrics_file
    oracle_file = dir + oracle_file

    # -------------------------
    # DATASET
    # -------------------------
    class FixEMNIST:
        def __call__(self, img):
            img = transforms.ToTensor()(img)
            img = torch.flip(img, [2])
            img = torch.rot90(img, 1, [1, 2])
            return img

    training_data = datasets.EMNIST(
        root="data",
        split="balanced",
        train=True,
        download=True,
        transform=FixEMNIST()
    )

    test_data = datasets.EMNIST(
        root="data",
        split="balanced",
        train=False,
        download=True,
        transform=FixEMNIST()
    )

    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    model = ClassifierS4Group02(dropout=dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5)

    # -------------------------
    # Metrics pandas
    # -------------------------
    metrics = pd.DataFrame(columns=[
        "epoch", "train_loss", "train_acc", "test_loss", "test_acc",
        "learning_rate", "batch_size", "weight_decay", "dropout"
    ])

    # -------------------------
    # TRAIN & TEST LOOP
    # -------------------------
    def train_loop(dataloader, model, loss_fn, optimizer):
        model.train()
        size = len(dataloader.dataset)
        running_loss, running_correct = 0, 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_logits = model.forward_logits(X)
            loss = loss_fn(pred_logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item() * X.size(0)
            running_correct += (pred_logits.argmax(1) == y).sum().item()
        return running_loss / size, running_correct / size

    def test_loop(dataloader, model, loss_fn):
        model.eval()
        size = len(dataloader.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred_logits = model.forward_logits(X)
                test_loss += loss_fn(pred_logits, y).item() * X.size(0)
                correct += (pred_logits.argmax(1) == y).sum().item()
        return test_loss / size, correct / size

    # -------------------------
    # TRAINING
    # -------------------------
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}\n-------------------------------")

        train_loss, train_acc = train_loop(
            train_dataloader, model, loss_fn, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss_fn)
        scheduler.step()

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(
            f"Test  Loss: {test_loss:.4f} | Test  Acc: {100*test_acc:.2f}%\n")

        # Save metrics
        metrics = pd.concat([
            metrics,
            pd.DataFrame([{
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "dropout": dropout
            }])
        ], ignore_index=True)

    # -------------------------
    # Export metrics
    # -------------------------
    metrics.to_csv(metrics_file, index=False)
    print(f"✅ Metrics saved to {metrics_file}")

    # -------------------------
    # ORACLE EXPORT
    # -------------------------

    model.eval()
    oracle_model = Oracle(model).to("cpu")
    scripted_oracle = torch.jit.script(oracle_model)
    scripted_oracle.save(oracle_file)
    print("🚀 Oracle exporté après entraînement :", oracle_file)




if __name__ == "__main__":
    init_metrics = {
        "learning_rate": [1e-3],
        "batch_size": [64],
        "epoch": [3],
        "weight_decay": [1e-4],
        "dropout": [0.5]
    }
    dir = "results/test1/"
    metrics_file = "training_metrics.csv"
    oracle_file = "oracle.pt"
    training(init_metrics, metrics_file, oracle_file, dir)
