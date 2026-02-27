import torch

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

    # ============================================================================================
    # TODO: Définir ici la méthode d'attaque adversariale pour générer une perturbation delta.
    # ============================================================================================

    delta = None

    return delta