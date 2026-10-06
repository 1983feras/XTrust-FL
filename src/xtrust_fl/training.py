from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).cpu() for p in model.parameters()])


def parameter_delta(local_model: nn.Module, global_model: nn.Module) -> torch.Tensor:
    return flatten_parameters(local_model) - flatten_parameters(global_model)


def load_flat_parameters(model: nn.Module, flat: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        values = flat[offset: offset + n].view_as(p).to(p.device, dtype=p.dtype)
        p.data.copy_(values)
        offset += n
    if offset != flat.numel():
        raise ValueError("Flat parameter vector has incorrect length")


def local_train(
    global_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    fedprox_mu: float = 0.0,
    device: str = "cpu",
) -> tuple[nn.Module, float]:
    # Never move/mutate the shared global model. Each client receives a deep copy;
    # FedProx anchors are independent tensors placed on the requested device.
    model = deepcopy(global_model).to(device)
    model.train()
    reference = [p.detach().clone().to(device) for p in global_model.parameters()]
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    last_loss = 0.0
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if fedprox_mu > 0:
                prox = sum(torch.sum((p - p0) ** 2) for p, p0 in zip(model.parameters(), reference))
                loss = loss + 0.5 * fedprox_mu * prox
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
    return model.cpu(), last_loss


def apply_delta(global_model: nn.Module, delta: torch.Tensor) -> nn.Module:
    model = deepcopy(global_model).cpu()
    load_flat_parameters(model, flatten_parameters(global_model) + delta.cpu())
    return model
