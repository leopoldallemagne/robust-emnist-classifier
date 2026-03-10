import torch
import torch.nn.functional as F

def attack(x, f_string, eps):

    model = torch.jit.load(f_string)
    model.eval()

    x_img = x.clone().detach()
    if x_img.dim() == 2:
        x_adv = x_img.unsqueeze(0).unsqueeze(0)
    elif x_img.dim() == 3:
        x_adv = x_img.unsqueeze(0)
    else:
        x_adv = x_img
        
    x_adv.requires_grad = True

    outputs = model(x_adv)
    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    label = torch.argmax(outputs, dim=1)

    
    iters = 20
    alpha = eps / 10  

    for i in range(iters):
        if x_adv.grad is not None:
            x_adv.grad.zero_grad()
            
        outputs = model(x_adv)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(0)
            
        loss = F.cross_entropy(outputs, label)
        loss.backward()

        with torch.no_grad():
           
            grad = x_adv.grad
            norm_grad = torch.norm(grad, p=2)
            if norm_grad > 1e-10:
                grad = grad / norm_grad
            
            x_adv = x_adv + alpha * grad
            delta = x_adv - x_img.view_as(x_adv)
            delta_norm = torch.norm(delta, p=2)
            if delta_norm > eps:
                delta = delta * (eps / delta_norm)
            x_adv = torch.clamp(x_img.view_as(x_adv) + delta, 0, 1)
            x_adv.requires_grad = True
    final_delta = (x_adv - x_img.view_as(x_adv)).squeeze().detach()

    check_norm = torch.norm(final_delta, p=2)
    if check_norm > eps:
        final_delta = final_delta * (eps / check_norm)
    print (f"Norme L2 finale de la perturbation : {torch.norm(final_delta, p=2):.4f} (<= {eps})")

    return final_delta