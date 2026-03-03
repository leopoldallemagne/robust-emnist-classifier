import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class FixEMNIST:
    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = torch.flip(img, [2])            # flip horizontal
        img = torch.rot90(img, 1, [1, 2])    # rotate 90° pour mettre debout
        return img


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
batch_size = 32
epochs = 30
weight_decay = 1e-4
dropout=0.5

oracle_file="classifier_S4_group_02.pt"

class FixEMNIST:
    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = torch.flip(img, [2])            # flip horizontal
        img = torch.rot90(img, 1, [1, 2])    # rotate 90° pour mettre debout
        return img


training_data = datasets.EMNIST(
    root="data",
    split="balanced",
    train=True,
    download=True,
    transform=FixEMNIST()
)

test_data = datasets.EMNIST(
    root="data",
    split="balanced",
    train=False,
    download=True,
    transform=FixEMNIST()
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# -------------------------
# CNN MODEL
# -------------------------
class ClassifierS4Group02(nn.Module):
    """
    Modèle de classification d'images.

    Contrainte: Le classifieur génère un vecteur de probabilités en moins de 0.1 seconde pour une image, sur un ordinateur 
                personnel de base (équipé par exemple d’un processeur CPU de type Core i5 et de 8 Go de mémoire RAM).

    Export: Après avoir entrainé le modèle, utilisez le code suivant afin de le sauver comme un oracle:

            model = ClassifierS4GroupXX()
            m = torch.jit.script(model)
            m.save("classifier_S4_group_XX.pt")
    """
    def __init__(self, dropout=0.5):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 47),
        )

    def forward_logits(self, x):
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(1)
    
        x = self.conv_stack(x)
        return self.classifier(x)
    
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
        original_dim = x.dim()

        logits = self.forward_logits(x)
        if original_dim == 2:
            probs = probs.squeeze(0)
        return nn.functional.softmax(logits, dim=1)

@torch.no_grad()
def init_weights(m):
    if type(m) is nn.Linear:
        nn.init.kaiming_uniform_(tensor=m.weight, mode='fan_in', nonlinearity='relu')

model = ClassifierS4Group02().to(device)
model.apply(init_weights)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# -------------------------
# TRAIN LOOP
# -------------------------

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        pred_logits = model.forward_logits(X)
        
        loss = loss_fn(pred_logits, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 200 == 0:
            print(f"loss: {loss.item():>7f} [{batch * len(X):>5d}/{size:>5d}]")

# -------------------------
# TEST LOOP
# -------------------------

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_logits = model.forward_logits(X)
            test_loss += loss_fn(pred_logits, y).item()

            pred_probs = model(X)
            correct += (pred_probs.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Accuracy: {(100*correct):>0.1f}% | Avg loss: {test_loss:>8f}\n")
    return test_loss

if __name__ == "__main__":
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss =test_loop(test_dataloader, model, loss_fn)
        
        scheduler.step(val_loss)

    print("Done!")

    m = torch.jit.script(model)
    m.save("classifier_S4_group_02.pt")
    print(f"Fichier Oracle généré : {oracle_file}")