import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ClassifierS4Group02(nn.Module):
    def __init__(self):
        """
        Couches du réseau de neurones.
        """

        super().__init__()
        # Première couche: 784 -> 512
        self.fc1 = nn.Linear(784, 512)
        self.bn1 = nn.BatchNorm1d(512) # Normalisation autour de 0 avec var de 1 (pour éviter les valeurs immenses)
        
        # Deuxième couche, 512 -> 512
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        # Troisième couche: 512 -> 47
        self.fc3 = nn.Linear(512, 47)

        # Dropout 
        self.dropout = nn.Dropout(p=0.1) # Désactive aléatoirement 10% des neurones pendant l'entrainement pour rendre le modèle plus robuste

    def forward(self, x):
        """
        Inférence du réseau de neurones.

        Entrée: Une image x enregistrée comme un tensor.FloatTensor d'une des dimensions suivantes:
                    - (28, 28)
                    - (B, 28, 28)
                    - (B, 1, 28, 28)
                et dont les entrées sont comprises entre 0 et 1.

        Sortie: Un vecteur y enregistré comme un tensor.FloatTensor de dimension
                    - (47,) si B = 1
                    - (B, 47) si B > 1
                avec les probabilités de classification associées à chaque classe.

        Contrainte: y doit satisfaire à la définition de probabilités.
        """

        # Conversion automatique en (B, 1, 28, 28)
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        x = x.view(x.size(0), -1)

        # Première couche
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Deuxième
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Troisième
        x = self.fc3(x)

        
        y = F.softmax(x, dim=1)

        # Si dim 1 on change en un vecteur
        if y.size(0) == 1:
            y = y.squeeze(0)

        return y
    

def train():
    # Configuration
    BATCH_SIZE = 128 # taille du "paquet" sur lequel on calcule l'erreur
    LEARNING_RATE = 0.001 # taille de la modification du poids
    EPOCHS = 15 # nombre d'itérations du modèle
    
    # On essaie d'abord de faire tourner sur gpu, (windows ou apple), ou sur un cpu si pas dipso.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomRotation(5), # Rotation très légère +-5deg pour améliorer le modèle
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
    val_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transforms.ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ClassifierS4Group02().to(device)
    
    criterion = nn.NLLLoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model.train()
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            probs = model(inputs)
            
            log_probs = torch.log(probs + 1e-8)
            
            loss = criterion(log_probs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(probs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Mise à jour du learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        acc = 100 * correct / total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_probs = model(val_inputs)
                val_log_probs = torch.log(val_probs + 1e-8)
                val_loss += criterion(val_log_probs, val_labels).item()
                _, val_predicted = torch.max(val_probs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        # Sauvegarder le meilleur modèle
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "classifier_S4_group_02_best.pt")
        
        model.train()
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {acc:.2f}% | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

    # Sauvegarde le meilleur modèle
    model.load_state_dict(torch.load("classifier_S4_group_02_best.pt"))
    model.to("cpu") 
    model.eval()
    m = torch.jit.script(model)
    m.save("classifier_S4_group_02_MLP_512_512.pt")
    print(f"Modèle sauvegardé : classifier_S4_group_02_MLP_512_512.pt (Best Val Acc: {best_val_acc:.2f}%)")

if __name__ == '__main__':
    train()