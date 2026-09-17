"""Exponential moving-average weights for validation and final inference."""
from __future__ import annotations
from copy import deepcopy
import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = .999):
        self.model = deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = self.decay * (1 - torch.exp(torch.tensor(-self.updates / 2000.0)).item())
        source = model.state_dict()
        for name, value in self.model.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(incoming, alpha=1 - decay)
            else:
                value.copy_(incoming)

    def state_dict(self):
        return {"model": self.model.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.updates = int(state.get("updates", 0))
