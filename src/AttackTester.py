import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import glob
import os
import pandas as pd
import argparse


# ============================================================
# PGD ATTACK
# ============================================================

def pgd_attack(model, image, label, epsilon=0.15, alpha=0.03, iters=5, device="cpu"):
    x = image.unsqueeze(0).to(device)
    y = torch.tensor([label]).to(device)

    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad = True

        output = model(x_adv)
        loss = -torch.log(output[0, y])

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad

        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, -1, 1).detach()

    return x_adv.squeeze(0)


# ============================================================
# MAIN TEST FUNCTION
# ============================================================
def TestAttackerOnModels(
        oracle_files,
        sample_size=15,
        plot_samples=False,
        eps_plot=1.0,
        epsilons=[0.5, 1.0, 1.5],
        attacker_name="attacker",
        dir="./attack_results"):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # -----------------------
    # FIX EMNIST ORIENTATION
    # -----------------------
    class FixEMNIST:
        def __call__(self, img):
            img = transforms.ToTensor()(img)
            img = torch.flip(img, [2])
            img = torch.rot90(img, 1, [1, 2])
            return img

    dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=False,
        download=True,
        transform=FixEMNIST()
    )

    classes = dataset.classes
    metrics_df = pd.DataFrame()
    os.makedirs(dir, exist_ok=True)

    #csv of labels of not successful attacks by oracle 
    not_successful_attacks_label_count = pd.DataFrame(0, index=classes, columns=oracle_files)

    for oracle_file in oracle_files:

        print(f"\n🔎 Testing {oracle_file}")

        model = torch.jit.load(oracle_file, map_location=device).to(device)
        model.eval()

        # =====================
        # RANDOM SAMPLES
        # =====================
        random.seed(20)
        sample_indices = random.sample(range(len(dataset)), min(sample_size, 15))

        # =====================
        # CLEAN ACCURACY
        # =====================
        clean_correct = 0
        for idx in sample_indices:
            image, label = dataset[idx]
            with torch.no_grad():
                pred_clean = model(image.unsqueeze(0).to(device)).argmax(1).item()
            if pred_clean == label:
                clean_correct += 1
        clean_accuracy = clean_correct / len(sample_indices)
        print(f"✅ Clean accuracy ({len(sample_indices)} samples): {clean_accuracy:.3f}")

        # =====================
        # ATTACK SUCCESS RATE & PLOTS
        # =====================
        for eps in epsilons:

            success_count = 0
            if plot_samples:
                plt.figure(figsize=(12, 4))
                plt.suptitle(f"{os.path.basename(oracle_file)} | Epsilon={eps}")

            for i, idx in enumerate(sample_indices):
                image, label = dataset[idx]

                attacked_image = pgd_attack(
                    model,
                    image,
                    label,
                    epsilon=eps,
                    device=device
                )

                with torch.no_grad():
                    pred_clean = model(image.unsqueeze(0).to(device)).argmax(1).item()
                    pred_adv = model(attacked_image.unsqueeze(0).to(device)).argmax(1).item()

                if pred_adv != pred_clean:
                    success_count += 1
                else:
                    not_successful_attacks_label_count.loc[classes[label], oracle_file] += 1
                # --------------------
                # PLOT
                # --------------------
                if plot_samples:
                    orig_img = image.squeeze().cpu()
                    adv_img = attacked_image.squeeze().cpu()

                    plt.subplot(2, len(sample_indices), i+1)
                    plt.imshow(orig_img, cmap="gray")
                    plt.title(f"C: {classes[label]}\nP: {classes[pred_clean]}")
                    plt.axis("off")

                    plt.subplot(2, len(sample_indices), i+1+len(sample_indices))
                    plt.imshow(adv_img, cmap="gray")
                    plt.title(f"P_adv: {classes[pred_adv]}\n{'X' if pred_adv!=pred_clean else ''}")
                    plt.axis("off")

            attack_success_rate = success_count / len(sample_indices)

            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame([{
                    "files_name": os.path.basename(oracle_file),
                    "attacker_name": attacker_name,
                    "sample_size": len(sample_indices),
                    "clean_accuracy": clean_accuracy,
                    "attack_success_rate": attack_success_rate,
                    "epsilon": eps
                }])
            ], ignore_index=True)

            # Sauvegarde plot
            if plot_samples:
                plot_path = os.path.join(dir, f"{os.path.basename(oracle_file)}_eps{eps}.png")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(plot_path)
                plt.close()
                print(f"📌 Plot saved: {plot_path}")

    # =====================
    # SAVE CSV
    # =====================
    csv_path = os.path.join(dir, f"{attacker_name}.csv")
    metrics_df.to_csv(csv_path, index=False)
    not_successful_attacks_label_count.to_csv(os.path.join(dir, f"{attacker_name}_not_successful_attacks_label_count.csv")) 
    print(f"\n📊 Metrics saved: {csv_path}")

# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PGD Attack Tester")

    parser.add_argument("--samples", type=int, default=500,
                        help="Number of samples per oracle")

    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting of adversarial samples")

    parser.add_argument("--eps_plot", type=float, default=0.3,
                        help="Epsilon used for plotting")

    parser.add_argument("--attacker_name", type=str,
                        default="pgd_attack_group",
                        help="Name for output CSV")

    parser.add_argument("--oracle_dir", type=str,
                        default="./groupe_oracles",
                        help="Directory containing oracle .pt files")

    args = parser.parse_args()

    oracle_files = glob.glob(os.path.join(args.oracle_dir, "classifier_*.pt"))

    print("Found oracles:", oracle_files)

    TestAttackerOnModels(
        oracle_files,
        sample_size=args.samples,
        plot_samples=args.plot,
        eps_plot=args.eps_plot,
        attacker_name=args.attacker_name,
        dir="./attack_results"
    )