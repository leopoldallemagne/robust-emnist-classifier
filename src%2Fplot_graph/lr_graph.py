import pandas as pd
import matplotlib.pyplot as plt
import glob
import os



"""
Ce script charge tous les CSV générés par TrainingWithSchedulers.py, extrait les courbes de test_acc et les compare sur un même graphe.
"""



RESULTS_DIR = "results"

plt.figure()

all_results = []

# =========================
# LOAD & PLOT ALL FILES
# =========================
files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

for file in files:
    df = pd.read_csv(file)
    name = os.path.basename(file).replace(".csv", "")

    scheduler = df["scheduler"].iloc[0] if "scheduler" in df.columns else "unknown"

    # =========================
    # LABEL BUILDING
    # =========================
    label = scheduler

    if scheduler == "Plateau":
        factor = df.get("factor", [None])[0]
        patience = df.get("patience", [None])[0]

        if "threshold" in df.columns:
            threshold = df["threshold"].iloc[0]
        elif "threshhold" in df.columns:
            threshold = df["threshhold"].iloc[0]
        else:
            threshold = None

        label = f"Plateau f={factor}, th={threshold}, p={patience}"

    elif scheduler == "StepLR":
        if "step_size" in df.columns and "gamma" in df.columns:
            step_size = df["step_size"].iloc[0]
            gamma = df["gamma"].iloc[0]
            label = f"StepLR s={step_size}, g={gamma}"
        else:
            label = name

    # =========================
    # PLOT
    # =========================
    linestyle = "--" if scheduler == "StepLR" else "-"
    plt.plot(df["epoch"], df["test_acc"], linestyle=linestyle, label=label)

    # =========================
    # METRICS
    # =========================
    best_acc = df["test_acc"].max()
    final_acc = df["test_acc"].iloc[-1]

    result = {
        "file": name,
        "scheduler": scheduler,
        "best_acc": best_acc,
        "final_acc": final_acc,
        "label": label
    }

    all_results.append(result)

# =========================
# RESULTS ANALYSIS
# =========================
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values(by="best_acc", ascending=False)

# =========================
# PRINT BEST
# =========================
print("\n🏆 BEST CONFIG:")
best = results_df.iloc[0]
for k, v in best.items():
    print(f"{k}: {v}")

# =========================
# PRINT TOP 5
# =========================
print("\n🔥 TOP 5 CONFIGS:\n")
print(results_df.head(5))

# =========================
# PLOT SETTINGS
# =========================
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Schedulers Comparison")
plt.legend()
plt.grid()

plt.show()