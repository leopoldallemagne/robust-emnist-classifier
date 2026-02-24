import pandas as pd
import matplotlib.pyplot as plt


def plot_training_metrics(csv_file):
    """
    Lit un fichier CSV contenant les metrics de training et test par epoch,
    puis trace les courbes de loss et accuracy avec les hyperparamètres en légende.
    """
    # Charger le CSV
    df = pd.read_csv(csv_file)

    # Extraire les hyperparamètres pour la légende
    lr = df["learning_rate"].iloc[0]
    bs = df["batch_size"].iloc[0]
    wd = df["weight_decay"].iloc[0]
    do = df["dropout"].iloc[0]

    legend_text = f"lr={lr}, batch_size={bs}, weight_decay={wd}, dropout={do}"

    epochs = df["epoch"]

    plt.figure(figsize=(12, 5))

    # --- Loss plot ---
    plt.subplot(1, 2, 1)
    plt.plot(epochs, df["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs, df["test_loss"], label="Test Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per Epoch")
    plt.legend(title=legend_text)
    plt.grid(True)

    # --- Accuracy plot ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, df["train_acc"]*100, label="Train Accuracy", marker='o')
    plt.plot(epochs, df["test_acc"]*100, label="Test Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy per Epoch")
    plt.legend(title=legend_text)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    csv_file ="training_metrics.csv"
    plot_training_metrics(csv_file)
