import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from classifier_S4_group_XX import ClassifierS4GroupXX

transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_set = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

model = ClassifierS4GroupXX()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Début de l'entraînement...")
model.train()
for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        if outputs.dim() == 1: outputs = outputs.unsqueeze(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Époque {epoch+1}/10 terminée - Erreur moyenne : {running_loss/len(train_loader):.4f}")

model.eval()
m = torch.jit.script(model)
m.save("classifier_S4_group_XX.pt")
print("\nModèle sauvegardé sous classifier_S4_group_XX.pt")

# (Accuracy)
print("Calcul de la précision sur l'ensemble de test...")
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        if outputs.dim() == 1: 
            outputs = outputs.unsqueeze(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"PRÉCISION FINALE : {accuracy:.2f}%")

#AFFICHAGE
mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"
idx = torch.randint(0, len(test_set), (1,)).item()
image, label = test_set[idx]

with torch.no_grad():
    logits = model(image.unsqueeze(0))
    output_probs = torch.softmax(logits, dim=-1)
    if output_probs.dim() > 1:
        output_probs = output_probs.squeeze(0)

pred_idx = torch.argmax(output_probs).item()
confidence = output_probs[pred_idx].item() * 100

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image.squeeze().numpy().T, cmap='gray')
color = 'green' if pred_idx == label else 'red'
plt.title(f"Vrai: {mapping[label]}\nPred: {mapping[pred_idx]} ({confidence:.1f}%)", color=color)
plt.axis('off')
plt.subplot(1, 2, 2)
top5_probs, top5_idxs = torch.topk(output_probs, 5)
top5_chars = [mapping[i] for i in top5_idxs]
plt.barh(top5_chars, top5_probs.numpy(), color='skyblue')
plt.title('Top 5 Certitudes')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()