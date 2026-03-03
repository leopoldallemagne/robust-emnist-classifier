import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pandas as pd
import os
from classifier_S4_group_02 import ClassifierS4Group02
import matplotlib.pyplot as plt

class Oracle(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x.requires_grad_(True)
        logits = self.model(x)
        return logits




def training(init_metrics, subject, metrics_file, oracle_file, dir):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    all_metrics_files = []

    # config de base
    base_config = {
        "epoch": init_metrics["epoch"][0],
        "learning_rate": init_metrics["learning_rate"][0],
        "batch_size": init_metrics["batch_size"][0],
        "weight_decay": init_metrics["weight_decay"][0],
        "dropout": init_metrics["dropout"][0]
    }

    for idx, value in enumerate(init_metrics[subject]):

        print(f"\n--- Training config {idx} | {subject}={value} ---")

        config = base_config.copy()
        config[subject] = value

        current_metrics_file = os.path.join(dir, f"{metrics_file}_{idx}.csv")
        current_oracle_file = os.path.join(dir, f"{oracle_file}_{idx}.pt")

        # DATASET
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

        train_loader = DataLoader(training_data, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_data, batch_size=config["batch_size"])

        model = ClassifierS4Group02(dropout=config["dropout"]).to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        metrics_df = pd.DataFrame()

        # TRAIN LOOP
        for epoch in range(1, base_config["epoch"] + 1):

            model.train()
            train_loss, train_correct = 0, 0
            size = len(train_loader.dataset)

            for X, y in train_loader:
                X, y = X.to(device), y.to(device)

                logits = model.forward_logits(X)
                loss = loss_fn(logits, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item() * X.size(0)
                train_correct += (logits.argmax(1) == y).sum().item()

            train_loss /= size
            train_acc = train_correct / size

            # TEST
            model.eval()
            test_loss, test_correct = 0, 0
            size = len(test_loader.dataset)

            with torch.no_grad():
                for X, y in test_loader:
                    X, y = X.to(device), y.to(device)
                    logits = model.forward_logits(X)
                    test_loss += loss_fn(logits, y).item() * X.size(0)
                    test_correct += (logits.argmax(1) == y).sum().item()

            test_loss /= size
            test_acc = test_correct / size

            scheduler.step()

            row = {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                **config,
                "epoch": epoch,
            }

            metrics_df = pd.concat([metrics_df, pd.DataFrame([row])], ignore_index=True)
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

        metrics_df.to_csv(current_metrics_file, index=False)
        all_metrics_files.append(current_metrics_file)

        # EXPORT ORACLE
        model.eval()
        oracle_model = Oracle(model).to("cpu")
        scripted = torch.jit.script(oracle_model)
        scripted.save(current_oracle_file)

        print(f"Saved: {current_metrics_file}")
        print(f"Saved: {current_oracle_file}")

    return all_metrics_files


def rnaking_metrics(metrics_files):
    results = []

    for file in metrics_files:

        df = pd.read_csv(file)

        # dernière epoch
        last_row = df.iloc[-1]

        results.append({
            "file": os.path.basename(file),
            "test_acc": last_row["test_acc"],
            "test_loss": last_row["test_loss"],
            "train_acc": last_row["train_acc"],
            "train_loss": last_row["train_loss"],
            "learning_rate": last_row["learning_rate"],
            "batch_size": last_row["batch_size"],
            "weight_decay": last_row["weight_decay"],
            "dropout": last_row["dropout"]
        })

    results_df = pd.DataFrame(results)

    # tri : accuracy décroissante puis loss croissante
    results_df = results_df.sort_values(
        by=["test_acc", "test_loss"],
        ascending=[False, True]
    ).reset_index(drop=True)

    return results_df

 



def compare_metrics(metrics_files,
                    subject="batch_size",
                    performance_metric="test_acc",
                    title="Comparison",
                    save_path="comparison.png"):
    """
    Compare plusieurs fichiers CSV en regroupant par un hyperparamètre (subject).

    Parameters
    ----------
    metrics_files : list[str]
        Liste des CSV
    subject : str
        Hyperparamètre à comparer ("batch_size", "dropout", "learning_rate", etc.)
    performance_metric : str
        Metric à optimiser ("test_acc", "train_acc", "test_loss", etc.)
    title : str
        Titre du graphique
    save_path : str
        Nom du fichier image exporté

    Returns
    -------
    pd.DataFrame trié du meilleur au pire
    """

    rows = []

    for file in metrics_files:
        df = pd.read_csv(file)

        value = df[subject].iloc[0]

        if "acc" in performance_metric:
            best_perf = df[performance_metric].max()
        else:
            best_perf = df[performance_metric].min()

        rows.append({
            subject: value,
            "best_performance": best_perf
        })

    results_df = pd.DataFrame(rows)

    # Regrouper par hyperparamètre (au cas où plusieurs runs ont la même valeur)
    if "acc" in performance_metric:
        results_df = results_df.groupby(subject).max().reset_index()
        results_df = results_df.sort_values("best_performance", ascending=False)
    else:
        results_df = results_df.groupby(subject).min().reset_index()
        results_df = results_df.sort_values("best_performance", ascending=True)

    # -------------------------
    # GRAPH
    # -------------------------
    plt.figure(figsize=(8,6))
    plt.plot(results_df[subject], results_df["best_performance"], marker="o")
    plt.xlabel(subject)
    plt.ylabel(performance_metric)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"📊 Graph saved to {save_path}")

    return results_df


if __name__ == "__main__":
    init_metrics = {
        "learning_rate": [1e-3],
        "batch_size": [64, 32],
        "epoch": [2],
        "weight_decay": [1e-4],
        "dropout": [0.5]
    }
    subject = "batch_size"
    dir = "results/"+subject+"/"

    os.makedirs(dir, exist_ok=True)
    metrics_file = f"training_{subject}"

    oracle_file = "oracle"

    files = training(init_metrics, subject, metrics_file, oracle_file, dir)
    
    ranking = rnaking_metrics(files)

    compare_metrics(files, subject=subject, performance_metric="test_acc", title=f"Test Accuracy vs {subject}", save_path=os.path.join(dir, f"comparison_{subject}.png"))

    print(ranking)