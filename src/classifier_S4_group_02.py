import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  88.1%
# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-4 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  86.9%  1
# dropout=0.2 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  88.1%  2

# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 512 -> 256 -> 47 :  88.5%  3
# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] (+BatchNorm) -> 512 -> 256 -> 47 :  88.4%
# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 and divided by 2 each 5 Epochs | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] (+BatchNorm) -> 512 -> 256 -> 47 :  88.6%

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
batch_size = 64
epochs = 15
weight_decay = 1e-4
dropout=0.5

weight_file="weight.pth"
oracle_file="oracle5.pt"

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
    def __init__(self):
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
        x = self.conv_stack(x)
        return self.classifier(x)
    
    def forward(self, x):
        logits = self.forward_logits(x)
        return nn.functional.softmax(logits, dim=1)

model = ClassifierS4Group02().to(device)

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