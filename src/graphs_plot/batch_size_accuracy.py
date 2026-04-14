# Ce code a été généré, avant correction, par le prompt suivant dans une IA:
# "Crée un graphe avec une échelle log en x qui représente l'accuracy en fonction de la batch size"
# Avant de modifier nous-même le titre, les axes, etc..
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = ".\\results\\batch_size\\training_batch_size_30_epochs.csv"
    output_path = ".\\results\\batch_size\\batch_size_accuracy_s4_30epochs.pdf"
    df = pd.read_csv(csv_path)
    df = df.sort_values("batch_size")

    x = df["batch_size"]
    y = df["test_acc"] * 100.0
    total_time = df["total_time"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(x, y, marker="o", linewidth=2, color="#1f77b4", label="Précision")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Taille de batch")
    ax1.set_ylabel("Précision (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(x, total_time, marker="s", linewidth=2, color="#d62728", label="Temps total")
    ax2.set_ylabel("Temps total (s)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    ax1.set_title("Modèle S4 - Précision et temps total en fonction de la taille de batch (sur 30 epochs)")
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


if __name__ == "__main__":
	main()
