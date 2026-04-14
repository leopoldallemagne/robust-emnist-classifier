import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms 
import matplotlib.gridspec as gridspec

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
    selon ce qu'attend le modèle.
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
    Difference of Logits Ratio non-ciblees.
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
    Recalculer dynamiquement a chaque restart evite de rester collé
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
    le mode represente le type d'attaque que l'on effectue a chaque restart, 
    on fait un cycle complet du mode sur les restarts
    pour diversifier les attaques(ciblé et non-ciblé) et eviter de rester bloque sur une approche qui ne fonctionne pas 

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

            # Pas adaptatif
            if use_adaptive_step and prev_loss is not None:
                if abs(loss.item() - prev_loss) < 1e-6:
                    stagnation += 1
                else:
                    stagnation = 0
                if stagnation >= 5:
                    current_alpha = max(current_alpha * 0.5, alpha * 0.05)
                    stagnation = 0
            prev_loss = loss.item()

            #Mise a jour + double projection 
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

        score = (1e6 if fooled else 0.0) + ce_loss
        if score > best_loss:
            best_loss  = score
            best_delta = delta_cand

        if fooled and torch.norm(best_delta).item() < eps * 0.5:
            break
    best_delta = clamp_and_reproject(x_input, best_delta, eps)
    best_delta = best_delta.view_as(x)
    return best_delta.detach()

if __name__ == "__main__":
    f_string = "D:\\BAC 3\\P4\\classifier_S4_group_02.pt"
    eps      = 0.5
    mapping  = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabdefghnqrt"

    # Chargement des données 
    transform = transforms.Compose([transforms.ToTensor()])
    test_set  = torchvision.datasets.EMNIST(
        root='./data', split='balanced', train=False,
        download=True, transform=transform
    )

    # Tirage d'une image aléatoire
    idx          = torch.randint(0, len(test_set), (1,)).item()
    image_raw, label = test_set[idx]

    img = torch.flip(image_raw, [2])
    img = torch.rot90(img, 1, [1, 2])
    image = img.squeeze()   # (28, 28)
    try:
        model = torch.jit.load(f_string, map_location='cpu')
    except Exception:
        model = torch.load(f_string, map_location='cpu')
    model.eval()

    def predict(img_2d):
        """Prédit la classe d'une image (28,28), gère les formats d'entrée."""
        with torch.no_grad():
            x = img_2d.unsqueeze(0).unsqueeze(0)   
            try:
                logits = model(x)
            except Exception:
                logits = model(x.squeeze(0))        
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)[0]
            pred  = int(torch.argmax(probs).item())
            conf  = float(probs[pred].item())
        return pred, conf, probs

    # Prédiction originale 
    orig_pred, orig_conf, orig_probs = predict(image)

    #Attaque 
    delta         = attack(image, f_string, eps)
    image_adv     = torch.clamp(image + delta, 0.0, 1.0)
    bruit_reel    = image_adv - image  
    norme_l2      = torch.norm(bruit_reel, p=2).item()
    norme_linf    = bruit_reel.abs().max().item()

    #Prédiction après attaque
    adv_pred, adv_conf, adv_probs = predict(image_adv)
    success = (orig_pred != adv_pred)

    #Visualisation
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(
        2, 4,
        figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.06, right=0.97, top=0.88, bottom=0.08
    )

    title_color   = '#e0e0e0'
    subtitle_color= '#a0a0c0'
    success_color = '#2ecc71' if success else '#e74c3c'
    ax_bg         = '#16213e'

    # ── Ligne du haut : les 3 images 

    # 1. Image originale
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image.numpy(), cmap='gray', vmin=0, vmax=1)
    ax1.set_title(
        f"Image originale\n"
        f"Vrai label : {mapping[label]}   |   Prédit : {mapping[orig_pred]}\n"
        f"Confiance : {orig_conf*100:.1f}%",
        color=title_color, fontsize=9, pad=6
    )
    ax1.axis('off')
    ax1.set_facecolor(ax_bg)

    # 2. Bruit (perturbation réelle après clamp)
    bruit_np = bruit_reel.numpy()
    vmax_bruit = max(abs(bruit_np.min()), abs(bruit_np.max()), 1e-6)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(bruit_np, cmap='RdBu_r', vmin=-vmax_bruit, vmax=vmax_bruit)
    ax2.set_title(
        f"Perturbation delta\n"
        f"‖delta‖2 = {norme_l2:.4f} / epsilon = {eps}   ({norme_l2/eps*100:.1f}% du budget)\n",
        color=title_color, fontsize=9, pad=6
    )
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.axis('off')
    ax2.set_facecolor(ax_bg)

    # 3. Image attaquée
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(image_adv.numpy(), cmap='gray', vmin=0, vmax=1)
    status_str = "SUCCÈS" if success else "ÉCHEC"
    ax3.set_title(
        f"Image attaquée  [{status_str}]\n"
        f"Prédit : {mapping[adv_pred]}\n"
        f"Confiance : {adv_conf*100:.1f}%",
        color=success_color, fontsize=9, fontweight='bold', pad=6
    )
    ax3.axis('off')
    ax3.set_facecolor(ax_bg)

    # 4. Bruit amplifié ×10 pour la lisibilité
    ax4 = fig.add_subplot(gs[0, 3])
    bruit_amplifie = (bruit_reel * 100 + 0.5).clamp(0, 1)
    ax4.imshow(bruit_amplifie.numpy(), cmap='gray', vmin=0, vmax=1)
    ax4.set_title(
        "Perturbation x100\n(amplifiée pour visualisation)\n ",
        color=subtitle_color, fontsize=9, pad=6
    )
    ax4.axis('off')
    ax4.set_facecolor(ax_bg)

    # Ligne du bas : top-5 probabilités avant / après 
    def plot_top5(ax, probs, true_label, pred_label, title):
        top5_vals, top5_idx = torch.topk(probs, 5)
        top5_vals = top5_vals.numpy() * 100
        top5_labels = [mapping[i] for i in top5_idx.numpy()]

        colors = []
        for i in top5_idx.numpy():
            if i == true_label and i == pred_label:
                colors.append('#3498db')   # bleu : correct
            elif i == pred_label:
                colors.append('#e74c3c')   # rouge : prédit mais faux
            elif i == true_label:
                colors.append('#2ecc71')   # vert : vrai label non prédit
            else:
                colors.append('#7f8c8d')   # gris : autre

        bars = ax.barh(range(5), top5_vals[::-1], color=colors[::-1], height=0.6)
        ax.set_yticks(range(5))
        ax.set_yticklabels(top5_labels[::-1], color=title_color, fontsize=11, fontweight='bold')
        ax.set_xlabel("Probabilité (%)", color=subtitle_color, fontsize=8)
        ax.set_xlim(0, 105)
        ax.set_facecolor(ax_bg)
        ax.tick_params(colors=subtitle_color, labelsize=8)
        ax.spines[:].set_color('#444466')
        ax.set_title(title, color=title_color, fontsize=9, pad=6)

        for bar, val in zip(bars, top5_vals[::-1]):
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va='center', color=title_color, fontsize=8)

    ax5 = fig.add_subplot(gs[1, :2])
    plot_top5(ax5, orig_probs, label, orig_pred, f"Top-5 avant attaque  (vrai : {mapping[label]})")

    ax6 = fig.add_subplot(gs[1, 2:])
    plot_top5(ax6, adv_probs, label, adv_pred, f"Top-5 après attaque  (prédit : {mapping[adv_pred]})")

    #Titre global
    result_txt = f"Attaque réussie : {mapping[orig_pred]}->{mapping[adv_pred]}" if success \
                else f"Attaque échouée — {mapping[orig_pred]} résiste (epsilon = {eps})"
    fig.suptitle(
        f"Visualisation PGD+ |epsilon = {eps}|{result_txt}",
        color=success_color, fontsize=13, fontweight='bold', y=0.95
    )

    plt.savefig("visualisation_attaque.png", dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.show()

    #Résumé console 
    print("─" * 55)
    print(f"  Image n°{idx}  |  Vrai label : {mapping[label]}")
    print(f"  Avant attaque : {mapping[orig_pred]}  ({orig_conf*100:.1f}%)")
    print(f"  Après attaque : {mapping[adv_pred]}  ({adv_conf*100:.1f}%)")
    print(f"  ‖delta‖_2  = {norme_l2:.4f}  (budget : {eps}  —  {norme_l2/eps*100:.1f}% utilisé)")
    print(f"  {'SUCCÈS' if success else 'ÉCHEC'}")
    print("─" * 55)
        
    