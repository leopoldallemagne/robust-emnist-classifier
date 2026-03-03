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

    # ============================================================
    # LOOP SUR CHAQUE ORACLE
    # ============================================================

    for oracle_file in oracle_files:

        print(f"\n🔎 Testing {oracle_file}")

        model = torch.jit.load(oracle_file, map_location=device).to(device)
        model.eval()

        # ========================================================
        # CLEAN ACCURACY (sur les samples choisis)
        # ========================================================

        random.seed(20)
        indices = random.sample(range(len(dataset)), sample_size)

        clean_correct = 0

        for idx in indices:
            image, label = dataset[idx]
            with torch.no_grad():
                pred_clean = model(image.unsqueeze(0).to(device)).argmax(1).item()
            if pred_clean == label:
                clean_correct += 1

        clean_accuracy = clean_correct / sample_size
        print(f"✅ Clean accuracy ({sample_size} samples): {clean_accuracy:.3f}")

        # ========================================================
        # OPTIONAL PLOT
        # ========================================================

        if plot_samples:

            plt.figure(figsize=(15, 6))

            for i, idx in enumerate(indices):

                image, label = dataset[idx]
                original_image = image.clone()


                # ----------- ATTACK CHOISI
                attacked_image = pgd_attack(
                    model,
                    image,
                    label,
                    epsilon=eps_plot,
                    device=device
                )

                with torch.no_grad():
                    pred_clean = model(image.unsqueeze(0).to(device)).argmax(1).item()
                    pred_adv = model(attacked_image.unsqueeze(0).to(device)).argmax(1).item()

                orig = original_image.squeeze().cpu() * 0.3081 + 0.1307
                adv = attacked_image.squeeze().cpu() * 0.3081 + 0.1307

                true_char = classes[label]
                pred_char_clean = classes[pred_clean]
                pred_char_adv = classes[pred_adv]

                # ORIGINAL
                plt.subplot(2, sample_size, i + 1)
                plt.imshow(orig, cmap="gray")
                plt.title(f"O\nL:{true_char}\nP:{pred_char_clean}")
                plt.axis("off")

                # ADVERSARIAL
                plt.subplot(2, sample_size, i + 1 + sample_size)
                plt.imshow(adv, cmap="gray")
                plt.title(
                    f"A\nP:{pred_char_adv}\n{'✔' if pred_adv != label else ''}"
                )
                plt.axis("off")

            plt.suptitle(
                f"{os.path.basename(oracle_file)}\n"
                f"Clean Acc: {clean_accuracy:.2f} | PGD eps={eps_plot}"
            )

            fig_path = os.path.join(
                dir,
                f"{os.path.basename(oracle_file)}_{eps_plot}_{attacker_name}_samples.png"
            )

            plt.tight_layout()
            plt.savefig(fig_path)
            plt.close()

            print(f"📸 Samples saved: {fig_path}")

        # ========================================================
        # ATTACK SUCCESS RATE
        # ========================================================

        for eps in [0.5, 1.0, 1.5]:

            success_count = 0

            for idx in indices:

                image, label = dataset[idx]

                attacked_image = pgd_attack(
                    model,
                    image,
                    label,
                    epsilon=eps,
                    device=device
                )

                with torch.no_grad():
                    pred_adv = model(attacked_image.unsqueeze(0).to(device)).argmax(1).item()

                if pred_adv != label:
                    success_count += 1

            attack_success_rate = success_count / sample_size

            metrics_df = pd.concat([
                metrics_df,
                pd.DataFrame([{
                    "files_name": os.path.basename(oracle_file),
                    "attacker_name": attacker_name,
                    "sample_size": sample_size,
                    "clean_accuracy": clean_accuracy,
                    "attack_success_rate": attack_success_rate,
                    "epsilon": eps
                }])
            ], ignore_index=True)

    csv_path = os.path.join(dir, f"{attacker_name}.csv")
    metrics_df.to_csv(csv_path, index=False)

    print(f"\n📊 Metrics saved: {csv_path}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PGD Attack Tester")

    parser.add_argument("--samples", type=int, default=15,
                        help="Number of samples per oracle")

    parser.add_argument("--plot", action="store_true",
                        help="Enable plotting of adversarial samples")

    parser.add_argument("--eps_plot", type=float, default=1.0,
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