import torch
import torch.nn.functional as F

def attack(x, f_string, eps):
    model = torch.jit.load(f_string)
    model.eval()
    x_img = x.clone().detach()
    if x_img.dim() == 2: x_input = x_img.unsqueeze(0).unsqueeze(0)
    elif x_img.dim() == 3: x_input = x_img.unsqueeze(0)
    else: x_input = x_img
    do=2
    with torch.no_grad():
        out = model(x_input)
        if out.dim() == 1: out = out.unsqueeze(0)
        label = torch.argmax(out, dim=1)
        # On récupère les cibles potentielles pour les restarts
        _, top_indices = torch.topk(out, k=do, dim=1)
        challengers = top_indices[1] 

    num_restarts = 40
    iters = 100
    alpha = 2 * eps / iters
    decay = 1.0
    
    best_delta = torch.zeros_like(x_input)

    for r in range(num_restarts):
        # On commence avec un petit bruit (Random Start)
        delta = torch.empty_like(x_input).uniform_(-eps, eps)
        # On projette pour respecter la norme L2 dès le début
        d_norm = torch.norm(delta, p=2)
        if d_norm > eps: delta = delta * (eps / d_norm)
        
        # x_adv est la variable qu'on va optimiser
        x_adv = (x_input + delta).detach().clamp(0, 1)
        momentum = torch.zeros_like(x_input)
        # Le symbole % (modulo) permet de rester entre 0 et 10, même si r vaut 60
        target_class = challengers[r % do] if r > 0 else label
        
        is_targeted = (r > 0)

        for i in range(iters):
            x_adv.requires_grad = True
            outputs = model(x_adv)
            if outputs.dim() == 1: outputs = outputs.unsqueeze(0)            
            loss = F.cross_entropy(outputs, target_class.view(-1))
            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                if grad is None: break
                grad = grad / (torch.norm(grad, p=2) + 1e-10)
                momentum = decay * momentum + grad
                step = alpha * (momentum / (torch.norm(momentum, p=2) + 1e-10))
                if is_targeted:
                    x_adv = x_adv - step
                else:
                    x_adv = x_adv + step
                
                actual_delta = x_adv - x_input
                norm = torch.norm(actual_delta, p=2)
                if norm > eps:
                    actual_delta = actual_delta * (eps / norm)
                x_adv = torch.clamp(x_input + actual_delta, 0, 1)
        with torch.no_grad():
            final_out = model(x_adv)
            if final_out.dim() == 1: final_out = final_out.unsqueeze(0)
            new_pred = torch.argmax(final_out, dim=1)
            
            if new_pred != label:
                return (x_adv - x_input).squeeze().detach()
            best_delta = (x_adv - x_input).squeeze().detach()

    return best_delta