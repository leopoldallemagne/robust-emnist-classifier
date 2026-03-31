import torch
import torch.nn.functional as F
import numpy as np

def load_model(f_string):
    """Charge le modèle """
    try:
        model = torch.jit.load(f_string, map_location='cpu')
    except Exception:
        model = torch.load(f_string, map_location='cpu')
    return model.eval()


def prepare_input(x):
    """
    Assure que x est un FloatTensor 4D (1, 1, 28, 28) ou 3D (1, 28, 28)
    selon ce qu'attend le modèle  on détecte à la volée.
    """
    if x.dim() == 2:
        return x.unsqueeze(0).unsqueeze(0).clone().detach()
    elif x.dim() == 3:
        return x.unsqueeze(0).clone().detach()
    return x.clone().detach()


def model_forward(model, x_input):
    """
    Appel du modèle en gérant les deux formats d'entrée possibles
    (28x28 ou batch 1x1x28x28). Retourne des logits 2D (1, 47).
    """
    try:
        out = model(x_input)
    except Exception:
        out = model(x_input.squeeze(0))
    if out.dim() == 1:
        out = out.unsqueeze(0)
    return out


def project_l2(delta, eps):
    """Projette delta sur la boule L2 de rayon eps."""
    norm = torch.norm(delta, p=2)
    if norm > eps:
        delta = delta * (eps / norm)
    return delta


def clamp_and_reproject(x_orig, delta, eps):
    """
    Ordre  des deux contraintes :
    1. clamp x_adv dans [0,1]  (contrainte image valide)
    2. recalculer delta
    3. reprojeter sur la boule L2  (contrainte perturbation)
    """
    x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
    delta = x_adv - x_orig
    delta = project_l2(delta, eps)
    return delta


def dlr_loss(logits, label):
    """
    Difference of Logits Ratio non-ciblee.
    Invariante au scale des logits -> plus stable sur modeles inconnus.
    logits : (1, 47),  label : int
    """
    z = logits[0]
    sorted_vals, sorted_idx = torch.sort(z, descending=True)
    z_y = z[label]
    z_top = sorted_vals[0] if sorted_idx[0] != label else sorted_vals[1]
    z_third = sorted_vals[2]
    denom = (z_y - z_third).abs().clamp(min=1e-8)
    return -(z_y - z_top) / denom


def dlr_loss_targeted(logits, label, target):
    """
    DLR ciblee : pousse vers `target` ET eloigne de `label`.
    Plus efficace a faible eps car elle ne gaspille pas de gradient
    sur des classes lointaines.
    logits : (1, 47),  label : int,  target : int
    """
    z = logits[0]
    sorted_vals, _ = torch.sort(z, descending=True)
    z_y      = z[label]
    z_target = z[target]
    denom = (z_y - sorted_vals[2]).abs().clamp(min=1e-8)
    return -(z_target - z_y) / denom


def get_best_challenger(logits, label):
    """
    Retourne la classe la plus menaçante dans l'etat courant de x_adv :
    celle dont le logit est le plus eleve apres exclusion de la vraie classe.
    Recalculer dynamiquement a chaque restart evite de rester colle
    sur un challenger qui n'est plus le plus accessible.
    """
    z = logits[0].detach().clone()
    z[label] = -1e10
    return int(torch.argmax(z).item())

def pgd_run(x_orig, model, eps, label, n_iter, alpha,decay, mode, use_adaptive_step):
    """
    Un run PGD avec momentum reel.
    Retourne (best_delta, best_loss).

    mode : int
        0 -> DLR non-ciblee      (stable, explore large)
        1 -> loss margin         (max concurrent - vrai,)
        2 -> DLR ciblee          (challenger dynamique, precise a faible eps)
    """
    noise = torch.randn_like(x_orig)
    noise = noise / (torch.norm(noise, p=2) + 1e-10)
    r = eps * (torch.rand(1).item() ** (1.0 / x_orig.numel()))
    delta = noise * r
    delta = clamp_and_reproject(x_orig, delta, eps)

    momentum = torch.zeros_like(x_orig)
    best_delta = delta.clone() 
    best_loss  = -1e10
    stagnation = 0
    prev_loss  = None
    current_alpha = alpha

    for i in range(n_iter):
        delta = delta.detach().requires_grad_(True)
        x_adv = torch.clamp(x_orig + delta, 0.0, 1.0)
        logits = model_forward(model, x_adv)
        challenger = get_best_challenger(logits.detach(), label)

        #Choix du loss selon le mode du restart 
        if mode == 0:
            loss = dlr_loss(logits, label)
        elif mode == 1:
            z = logits[0]
            mask = torch.ones(z.shape[0], dtype=torch.bool)
            mask[label] = False
            loss = z[mask].max() - z[label]
        else:  # mode == 2
            loss = dlr_loss_targeted(logits, label, challenger)

        loss.backward()

        with torch.no_grad():
            grad = delta.grad.detach()

            #Momentum REEL 
            # Normalisation du gradient avant accumulation
            g_norm = torch.norm(grad, p=2)
            grad_n = grad / (g_norm + 1e-10)
            momentum = decay * momentum + grad_n
            step = momentum  

            # --- Pas adaptatif ---
            if use_adaptive_step and prev_loss is not None:
                if abs(loss.item() - prev_loss) < 1e-6:
                    stagnation += 1
                else:
                    stagnation = 0
                if stagnation >= 5:
                    current_alpha = max(current_alpha * 0.5, alpha * 0.05)
                    stagnation = 0
            prev_loss = loss.item()

            # --- Mise a jour + double projection ---
            new_delta = delta.detach() + current_alpha * step
            new_delta = clamp_and_reproject(x_orig, new_delta, eps)

            # Sauvegarde si meilleur
            if loss.item() > best_loss:
                best_loss  = loss.item()
                best_delta = new_delta.clone()

        delta = new_delta

    return best_delta.detach(), best_loss


# ---------------------------------------------------------------------------
# Fonction principale 
# ---------------------------------------------------------------------------

def attack(x, f_string, eps):
    """
    Attaque adversariale PGD+ avec multi-start.
    -----------------------------------------------
    x        : torch.FloatTensor  (28x28, valeurs dans [0.0, 1.0])
    f_string : str                chemin vers le classifieur .pt cible
    eps      : np.float           rayon boule L2, entre 0.0 et 2.0

    Sortie
    ------
    delta    : torch.FloatTensor  (28x28)  avec  ||delta||_2 <= eps
                                  et  x + delta dans [0, 1]^(28x28)
    """
    eps = float(eps)
    model = load_model(f_string)
    x_input = prepare_input(x)

    # Classe predite (cible de l'attaque)
    with torch.no_grad():
        out = model_forward(model, x_input)
        label = int(torch.argmax(out, dim=1).item())

    # Hyper-parametres selon eps
    if eps < 0.3:
        n_restarts, n_iter = 6, 100
    elif eps < 1.0:
        n_restarts, n_iter = 6, 70
    else:
        n_restarts, n_iter = 3, 50

    alpha = eps * 2.5 / n_iter  # pas initial

    best_delta = torch.zeros_like(x_input)
    best_loss  = -1e10
    for r in range(n_restarts):
        mode = r % 3 
        delta_cand, loss_cand = pgd_run(
            x_input, model, eps, label,
            n_iter=n_iter,
            alpha=alpha,
            decay=0.9,
            mode=mode,
            use_adaptive_step=True,
        )

        # Evaluation : cross-entropy sur la vraie classe
        with torch.no_grad():
            x_adv = torch.clamp(x_input + delta_cand, 0.0, 1.0)
            logits_adv = model_forward(model, x_adv)
            ce_loss = F.cross_entropy(
                logits_adv, torch.tensor([label])
            ).item()
            fooled = (int(torch.argmax(logits_adv, dim=1).item()) != label)

        # Priorite : (1) a t on trompe le modele ? (2) loss la plus haute
        score = (1e6 if fooled else 0.0) + ce_loss
        if score > best_loss:
            best_loss  = score
            best_delta = delta_cand

        if fooled and torch.norm(best_delta).item() < eps * 0.5:
            break
    best_delta = clamp_and_reproject(x_input, best_delta, eps)
    best_delta = best_delta.view_as(x)
    return best_delta.detach()