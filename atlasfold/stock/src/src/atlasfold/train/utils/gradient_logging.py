import torch


@torch.no_grad()
def gradient_norm(module: torch.nn.Module) -> float:
    # Only compute over parameters that are being trained
    grads = [
        p.grad for p in module.parameters() if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return 0.0
    norms = torch._foreach_norm(grads, ord=2)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return total_norm.item()


@torch.no_grad()
def parameter_norm(module: torch.nn.Module) -> float:
    # Get all parameters that are being trained
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        return 0.0
    norms = torch._foreach_norm(parameters, ord=2)
    total_norm = torch.linalg.vector_norm(torch.stack(norms), ord=2)
    return total_norm.item()
