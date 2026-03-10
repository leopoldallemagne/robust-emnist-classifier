import torch
import matplotlib.pyplot as plt
from att import attack 
import torchvision
from torchvision import transforms

# Configuration
device = torch.device("cpu")
f_string = "D:\\BAC 3\\P4\\classifier_S4_group_A.pt"
eps = 0.25
mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)

idx = torch.randint(0, len(test_set), (1,)).item()
image_raw, label = test_set[idx]
image = image_raw
img = torch.flip(image_raw, [2])            # flip horizontal
img = torch.rot90(img, 1, [1, 2])    # rotate 90° pour mettre debout
 
image=img.squeeze() 
# --- ATTAQUE ---
delta = attack(image, f_string, eps)
image_attaquee = torch.clamp(image + delta, 0, 1)

# --- PRÉDICTIONS ---
model_oracle = torch.jit.load(f_string)
model_oracle.eval()

with torch.no_grad():
    input_orig = image.unsqueeze(0)
    orig_pred = torch.argmax(model_oracle(input_orig)).item()
    input_adv = image_attaquee.unsqueeze(0)
    adv_pred = torch.argmax(model_oracle(input_adv)).item()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().numpy().T, cmap='gray') 
plt.title(f"ORIGINAL\nVrai Label: {mapping[label]}\nPred modèle: {mapping[orig_pred]}")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image_attaquee.squeeze().detach().numpy().T, cmap='gray')
plt.title(f"ATTAQUÉ\nNouvelle Pred: {mapping[adv_pred]}")
plt.axis('off')

plt.tight_layout()
plt.show()
if orig_pred != adv_pred:
    print(f" Succès ! {mapping[orig_pred]} est devenu {mapping[adv_pred]}")
else:
    print(" Échec. Le modèle est resté robuste.")