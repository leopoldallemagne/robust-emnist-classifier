import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os, time

# -------------------------
# FIX EMNIST ORIENTATION
# -------------------------

class FixEMNIST:
    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = torch.flip(img, [2])
        img = torch.rot90(img, 1, [1, 2])
        return img


# -------------------------
# PGD ATTACK
# -------------------------

def pgd_attack(model, image, label, epsilon=0.2, alpha=0.01, iters=20, device="cpu"):

    x = image.unsqueeze(0).to(device)
    y = torch.tensor([label]).to(device)

    x_adv = x.clone().detach()

    for _ in range(iters):

        x_adv.requires_grad_(True)

        output = model(x_adv)

        loss = torch.nn.CrossEntropyLoss()(output, y)

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()

        x_adv = x_adv + alpha * grad

        delta = torch.clamp(x_adv - x, -epsilon, epsilon)

        x_adv = torch.clamp(x + delta, -1, 1).detach()

    return x_adv.squeeze(0)


# -------------------------
# GENERATE DATASET
# -------------------------

def generate_dataset(model_path, save_path, n_samples=20000):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = torch.jit.load(model_path, map_location=device)
    model.eval()

    dataset = datasets.EMNIST(
        root="./data",
        split="balanced",
        train=True,
        download=True,
        transform=FixEMNIST()
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    adv_images = []
    adv_labels = []

    for i, (img, label) in enumerate(loader):

        if i >= n_samples:
            break

        img = img.squeeze(0)
        label = label.item()

        adv_img = pgd_attack(model, img, label, device=device)

        adv_images.append(adv_img.cpu())
        adv_labels.append(label)

        if i % 1000 == 0:
            print("Generated", i)

    adv_images = torch.stack(adv_images)
    adv_labels = torch.tensor(adv_labels)

    torch.save({
        "images": adv_images,
        "labels": adv_labels
    }, save_path)

    print("Dataset saved to:", save_path)


if __name__ == "__main__":

    generate_dataset(
        model_path="classifier_S4_group_02.pt",
        save_path="adv_dataset.pt",
        n_samples=20000
    )