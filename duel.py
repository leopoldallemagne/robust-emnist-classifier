import torch
import matplotlib.pyplot as plt
from classifier_S4_group_XX import ClassifierS4GroupXX 
from att import attack 
from torchvision import datasets, transforms
import torchvision
device = torch.device("cpu")
f_string = "classifier_S4_group_XX.pt"
eps = 0.25
transform = transforms.Compose([transforms.ToTensor()])
test_set = torchvision.datasets.EMNIST(
    root='./data', 
    split='balanced', 
    train=False, 
    download=True, 
    transform=transform
)
idx = torch.randint(0, len(test_set), (1,)).item()
image, label = test_set[idx]
# Ici on lance l'attaque et on obtient la perturbation
delta = attack(image, f_string, eps)
#on verifie que delta focntionce comme dans l'énncé
norm_l2 = torch.norm(delta, p=2).item()
print(f"Norme L2 de la perturbation : {norm_l2:.4f} (Limite eps: {eps})")
#c'est ici qu'on crée dont l'image attaquée
image_attaquee = torch.clamp(image + delta, 0, 1)
# Comparaison des prédictions du modèle sur l'image originale et l'image attaquée
model_oracle = torch.jit.load(f_string)
model_oracle.eval()

with torch.no_grad():# 
    input_orig = image.unsqueeze(0) if image.dim() == 3 else image.unsqueeze(0).unsqueeze(0)
    orig_probs = model_oracle(input_orig)
    orig_pred = torch.argmax(orig_probs).item()
    input_adv = image_attaquee.unsqueeze(0) if image_attaquee.dim() == 3 else image_attaquee.unsqueeze(0).unsqueeze(0)
    adv_probs = model_oracle(input_adv)
    adv_pred = torch.argmax(adv_probs).item()
mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(image.numpy().T, cmap='gray')
plt.title(f"Original\nPred: {mapping[orig_pred]}")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_attaquee.numpy().T, cmap='gray')
plt.title(f"Attaqué\nPred: {mapping[adv_pred]}")
plt.axis('off')
plt.show()
if orig_pred != adv_pred:
    print(" Succès ! Le modèle a été trompé.")
else:
    print(" Échec. Le modèle est resté robuste.")