import torch
import torch.nn.functional as F

def attack(x, f_string, eps):
    model = torch.jit.load(f_string)
    model.eval()
    if x.dim() == 2:
        x_input = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x_input = x.unsqueeze(0)
    else:
        x_input = x
    
    x_input = x_input.clone().detach()
    
    do=2
    with torch.no_grad():
        out = model(x_input)
        if out.dim() == 1: out = out.unsqueeze(0)
        label = torch.argmax(out, dim=1)
        # On récupère les cibles potentielles pour les restarts
        _, top_indices = torch.topk(out, k=do, dim=1)
        challengers = top_indices[0] 

    num_restarts = 10
    iters = 40
    alpha = eps / 10
    decay = 0.9
    
    best_delta = torch.zeros_like(x_input)

    for r in range(num_restarts):
        delta = torch.empty_like(x_input).uniform_(-eps, eps)
        d_norm = torch.norm(delta, p=2)
        if d_norm > eps: delta = delta * (eps / d_norm)
        
        x_adv = (x_input + delta).detach().clamp(0, 1) 
        momentum = torch.zeros_like(x_input)
        target_class = challengers[r % do] if (r % 2) != 0 else label
        is_targeted = (r % 2) != 0
        max_loss = -1e10 if not is_targeted else 1e10
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
                step=step.sign()*alpha
                
                if is_targeted:
                    x_adv = x_adv - step
                else:
                    x_adv = x_adv + step
                
                actual_delta = x_adv - x_input
                norm = torch.norm(actual_delta, p=2)
                if norm > eps:
                    actual_delta = actual_delta * (eps / norm)
                x_adv = torch.clamp(x_input + actual_delta, 0, 1) # on s'assure que x_adv reste dans les bonnes bornes(0,1)
        with torch.no_grad():
            final_out = model(x_adv)
            if final_out.dim() == 1: final_out = final_out.unsqueeze(0)
            current_loss = F.cross_entropy(final_out, target_class.view(-1)).item()
            new_pred = torch.argmax(final_out, dim=1)
            if new_pred != label:
                return (x_adv - x_input).squeeze().detach()
            if current_loss > max_loss:
                max_loss = current_loss
                best_delta = (x_adv - x_input).squeeze().detach()

    return best_delta