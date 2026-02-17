import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  88.1%
# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-4 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  86.9%  1
# dropout=0.2 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 128 -> 47 :  88.1%  2

# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] -> 512 -> 256 -> 47 :  88.5%  3
# dropout=0.5 | weight_decay = 1e-4 | batch_size=64 | ln=1e-3 | 28x28 -> [ 32x (14x14) -> 64x (7x7) ] (+BatchNorm) -> 512 -> 256 -> 47 :  88.4%

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

learning_rate = 1e-3
batch_size = 64
epochs = 15
weight_decay = 1e-4
dropout=0.5

weight_file="weight.pth"
oracle_file="oracle5.pt"

training_data = datasets.EMNIST(
    root="data",
    split="balanced",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

test_data = datasets.EMNIST(
    root="data",
    split="balanced",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# -------------------------
# CNN MODEL
# -------------------------

class CNN(nn.Module):
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

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.classifier(x)
        return x

model = CNN().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# -------------------------
# TRAIN LOOP
# -------------------------

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

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
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Accuracy: {(100*correct):>0.1f}% | Avg loss: {test_loss:>8f}\n")



for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

print("Done!")

model.to('cpu')
torch.save(model, oracle_file)
print(f"Fichier Oracle généré : {oracle_file}")