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
