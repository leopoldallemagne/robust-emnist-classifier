import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib

def attack(x, f_string, eps):
    """
    Génère une perturbation adversariale pour une image donnée en moins de 10 secondes par image, sur un ordinateur 
    personnel de base (équipé par exemple d’un processeur CPU de type Core i5 et de 8 Go de mémoire RAM).

    Entrée: - Une image x enregistrée comme un tensor.FloatTensor de dimension (28, 28) et dont les entrées
              sont comprises entre 0 et 1.

            - Un string f_string qui contient le chemin vers le classifieur enregistré sous format f.pt que
              vous devez attaquer (potentiellement jamais vu auparavant).

            - Une taille de perturbation eps encodée comme un np.float entre 0 et 2.
        
    Sortie: La perturbation delta enregistrée comme un tensor.FloatTensor de dimension (28, 28) et dont 
            les entrées sont comprises entre 0 et 1.

    Contrainte: - La perturbation delta doit satisfaire ||delta||_2 <= eps.
                
                - L'image perturbée x+delta doit être un tensor.FloatTensor de dimension (28, 28) et dont
                  les entrées sont comprises entre 0 et 1.
    """

    # Chargement du modèle
    model = torch.jit.load(f_string)
    model.eval()

    if x.dim() == 2:
      x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
      x = x.unsqueeze(1)

    # ============================================================================================
    # TODO: Définir ici la méthode d'attaque adversariale pour générer une perturbation delta.
    # ============================================================================================
    w = torch.zeros_like(x, requires_grad=True)
    max_iteration = 1000
    label = model(x).argmax(1)
    lr = 1e-5
    c = 1e-1
    confidence = 0
    delta = None
    t_k = torch.tensor(10.0)

    def f(x):
      output = model(x)
      one_hot_labels = torch.eye(len(output[0]))[label].to(x.device)
      i,_ = torch.max((1 - one_hot_labels) * output, dim=1)
      j = torch.masked_select(output, one_hot_labels.bool())
      return torch.clamp(i-j, min=0)

    def g(x, t_k):
      if torch.norm(x) <= eps:
        return 0
      else:
        t_k *= 10
        return (t_k/eps) * torch.norm(x)

    
    optimizer = torch.optim.Adam([w], lr = lr)

    for step in range(max_iteration):
      a = 1/2 * (torch.nn.Tanh()(w) + 1)

      loss1 = torch.nn.MSELoss(reduction="sum")(a, x)
      loss2 = torch.sum(c * f(a))
      loss3 = g(w, t_k)

      cost = loss1 + loss2 + loss3
      optimizer.zero_grad()
      cost.backward()
      optimizer.step()

    attack_images = 1/2*(torch.nn.Tanh()(w) + 1)
    
    # w est équivalent au delta de l'énoncé, attack_images est juste là pour des tests rapides
    return w, attack_images


# Testing
class FixEMNIST:
    def __call__(self, img):
        img = transforms.ToTensor()(img)
        img = torch.flip(img, [2])            # flip horizontal
        img = torch.rot90(img, 1, [1, 2])    # rotate 90° pour mettre debout
        return img

test_data = datasets.EMNIST(
    root="data",
    split="balanced",
    train=False,
    download=True,
    transform=FixEMNIST()
)

batch_size = 128
test_dataloader = DataLoader(test_data, batch_size=batch_size)
"""
C'est du code que j'ai écrit quand je testais l'attaque donc pas très utile mais je l'efface pas au cas où 
je voudrais faire d'autres tests rapides
for X, Y in test_dataloader:
  if 4 in Y:
    for i in range(len(Y)):
      if Y[i] == 4:
        x,y = X[i], Y[i]
        break


eps = 2
delta, adv_im = attack(x, "classifier_S4_group_D.pt", eps)
print(torch.norm(delta))
cnt = 0
#print(delta)
#print(x)
model = torch.jit.load("classifier_S4_group_D.pt")
model.eval()
label = model(delta + x).argmax(1)
#print(2.is 0)
plt.imshow((x).numpy(force=True).reshape((28,28)), cmap='gray')
#plt.show()
X = np.linspace(0, 1, 28)
Y = np.linspace(0, 1, 28)
#myColorMap = matplotlib.cm.mpl.
X,Y = np.meshgrid(X, Y)
plt.figure()
delta = delta.numpy(force=True).reshape((28,28))
delta = np.rot90(delta, k=2)
delta = np.fliplr(delta)
#print(delta)
plt.title(r"Heatmap of the pertubation $\delta$ with $\epsilon = $" + str(eps) + f"\n The predicted label is {label.item()}")
cs = plt.contourf(X, Y, delta, cmap='CMRmap')
plt.colorbar(cs)
plt.xlabel(r"x [m]")
plt.ylabel(r"y [m]")
plt.axis("off")
#plt.show()
"""
def plot_accuracy_vs_epsilon(epsilons, model_path, num_images=100):
    """
    Plots the model accuracy on adversarial examples for different epsilon values.
    Args:
        epsilons (list or np.array): List of epsilon values to test.
        model_path (str): Path to the classifier model.
        num_images (int): Number of images to test per epsilon.
    """
    accuracies = []
    model = torch.jit.load(model_path)
    model.eval()
    # Prepare test data loader
    test_data = datasets.EMNIST(
        root="data",
        split="balanced",
        train=False,
        download=True,
        transform=FixEMNIST()
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    for eps in epsilons:
        correct = 0
        total = 0
        for X, Y in test_loader:
            x = X[0]
            y = Y[0]
            delta, adv_im = attack(x, model_path, eps)
            with torch.no_grad():
                pred = model(delta + x).argmax(1).item()
            if pred == y.item():
                correct += 1
            total += 1
            if total >= num_images:
                break
        accuracy = correct / total
        accuracies.append(accuracy)
        print(f"Epsilon: {eps}, Accuracy: {accuracy}")
    # Plot
    plt.figure()
    plt.plot(epsilons, accuracies, marker='o')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.title('Model accuracy vs epsilon (adversarial attack)')
    plt.grid(True)
    plt.show()

plot_accuracy_vs_epsilon([0.1, 0.5, 1, 1.5, 2], "classifier_S4_group_D.pt", num_images=100)