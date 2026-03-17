import torch
import torch.nn.functional as F

def attack(x, f_string, eps):
    model = torch.jit.load(f_string).eval()
    
    if x.dim() == 2: x_input = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3: x_input = x.unsqueeze(0)
    else: x_input = x
    x_input = x_input.clone().detach()
    
    with torch.no_grad():
        out = model(x_input)
        if out.dim() == 1: out = out.unsqueeze(0)
        label = torch.argmax(out, dim=1)
        _, top_indices = torch.topk(out, k=2, dim=1)
        challenger_class = top_indices[0][1]
    num_restarts = 2
    iters = 70
    decay = 0.9
    best_delta = torch.zeros_like(x_input)
    max_loss_global = -1e10

    for r in range(num_restarts):
        delta = torch.empty_like(x_input).uniform_(-eps, eps)
        d_norm = torch.norm(delta, p=2)
        if d_norm > eps: delta = delta * (eps / d_norm)        
        x_adv = (x_input + delta).detach().clamp(0, 1) 
        momentum = torch.zeros_like(x_input)
        is_targeted = (r % 2) != 0        
        for i in range(iters):
            current_alpha = (eps / 10) if i < (iters * 0.75) else (eps / 40)
            x_adv.requires_grad = True
            logits = model(x_adv)
            if logits.dim() == 1: logits = logits.unsqueeze(0)
            current_label_logit = logits[0, label]
            if is_targeted:
                loss = logits[0, challenger_class] - current_label_logit
            else:
                mask = torch.ones_like(logits, dtype=torch.bool); mask[0, label] = False
                loss = torch.max(logits[mask]) - current_label_logit
            
            model.zero_grad(); loss.backward()

            with torch.no_grad():
                grad = x_adv.grad
                if grad is None: break
                grad = grad / (torch.norm(grad, p=2) + 1e-10)
                momentum = decay * momentum + grad
                x_adv = x_adv + momentum.sign() * current_alpha
                actual_delta = x_adv - x_input
                norm = torch.norm(actual_delta, p=2)
                if norm > eps: actual_delta = actual_delta * (eps / norm)
                x_adv = torch.clamp(x_input + actual_delta, 0, 1)
        with torch.no_grad():
            final_out = model(x_adv)
            if final_out.dim() == 1: final_out = final_out.unsqueeze(0)
            new_pred = torch.argmax(final_out, dim=1)
            loss_vs_true = F.cross_entropy(final_out, label.view(-1)).item()
            
            if new_pred != label: return (x_adv - x_input).squeeze().detach()
            if loss_vs_true > max_loss_global:
                max_loss_global = loss_vs_true
                best_delta = (x_adv - x_input).squeeze().detach()

    return best_delta