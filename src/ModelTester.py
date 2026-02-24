import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# MODEL LOAD
# -----------------------
oracle_file = "oracle.pt"
model = torch.jit.load(oracle_file, map_location=device)
model.eval()

# -----------------------
# TRANSFORM (SANS AUGMENTATION)
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
classes = dataset.classes  # mapping officiel EMNIST


# -----------------------
# PGD ATTACK
# -----------------------

def corrupte_adversarial(image):
    img = image.clone()
    # correction orientation EMNIST
    img = torch.flip(img, (1,))
    img = torch.rot90(img, 1, (0, 1))
    return img
def pgd_attack(model, image, label, epsilon=0.15, alpha=0.03, iters=5, device="cpu"):
    x = image.unsqueeze(0).to(device)
    y = torch.tensor([label]).to(device)

    x_adv = x.clone().detach()

    for _ in range(iters):
        x_adv.requires_grad = True

        output = model(x_adv)
        #loss = nn.CrossEntropyLoss()(output, y)
        loss = -torch.log(output[0, y])

        model.zero_grad()
        loss.backward()

        grad = x_adv.grad.sign()
        x_adv = x_adv + alpha * grad

        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, -1, 1).detach()

    return x_adv.squeeze(0)

# -----------------------
# RANDOM SAMPLES
# -----------------------
random.seed(20)
indices = random.sample(range(len(dataset)), 15)

plt.figure(figsize=(12, 5))

for i, idx in enumerate(indices):
    image, label = dataset[idx]

    original_image = image.clone()

    attacked_image = pgd_attack(
        model, image, label,
        epsilon=0.5,
        alpha=0.03,
        iters=5,
        device=device
    )

    with torch.no_grad():
        output_adv = model(attacked_image.unsqueeze(0).to(device))
        pred_adv = output_adv.argmax(1).item()
        output  = model(image.unsqueeze(0).to(device))
        pred = output.argmax(1).item()

    # -----------------------
    # DENORMALIZE
    # -----------------------
    orig = original_image.squeeze().cpu() * 0.3081 + 0.1307
    adv = attacked_image.squeeze().cpu() * 0.3081 + 0.1307

    
    true_char = classes[label]
    pred_char = classes[pred_adv]
    pred_char_clean = classes[pred]
    # -----------------------
    # ORIGINAL IMAGE
    # -----------------------
    plt.subplot(2, len(indices), i + 1)
    plt.imshow(orig, cmap="gray")
    plt.title(f"Original\nLabel: {true_char}\n Pred: {pred_char_clean}")
    plt.axis("off")

    # -----------------------
    # ADVERSARIAL IMAGE
    # -----------------------
    plt.subplot(2, len(indices), i + 1 + len(indices))
    plt.imshow(adv, cmap="gray")
    plt.title(f"Attack\nPred: {pred_char} \n {'X' if pred_adv != label else ''}")
    plt.axis("off")



plt.tight_layout()
plt.show()

