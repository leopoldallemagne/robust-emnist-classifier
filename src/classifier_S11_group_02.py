import torch.nn as nn

class ClassifierS11GroupXX(nn.Module):
    """
    Modèle robuste de classification d'images.

    Contrainte: Le classifieur génère un vecteur de probabilités en moins de 0.1 seconde pour une image, sur un ordinateur 
                personnel de base (équipé par exemple d’un processeur CPU de type Core i5 et de 8 Go de mémoire RAM).

    Export: Après avoir entrainé le modèle, utilisez le code suivant afin de le sauver comme un oracle:

            model = ClassifierS11GroupXX()
            m = torch.jit.script(model)
            m.save("classifier_S11_group_XX.pt")
    """

    def __init__(self):
        """
        Couches du réseau de neurones.
        """

        super().__init__()

        # ======================================================
        # TODO: Définir ici les couches du réseau de neurones.
        # ======================================================

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

        # ==========================================================================================================
        # TODO: Définir ici l'inférence du réseau de neurones permettant de générer un vecteur de probabilités y.
        # ==========================================================================================================

        y = None

        return y