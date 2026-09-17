from collections.abc import Sequence
from typing import Any

import torch


class ExponentialMovingAverage:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        submodules_to_ignore: Sequence[str] | None = None,
        submodules_to_update: Sequence[str] | None = None,
    ):
        """
        Parameters
        ----------
        model: torch.nn.Module
            The model to maintain the moving average of parameters for.
        decay: float
            The decay rate for the moving average. Should be between 0 and 1.
        submodules_to_ignore: list[str] | None
            Optional list of submodule names to ignore.
        submodules_to_update: list[str] | None
            Optional list of submodule names to update.
        """
        self.decay = decay
        self.prefixes_to_ignore: tuple[str, ...] | None = (
            tuple(submodules_to_ignore) if submodules_to_ignore is not None else None
        )
        self.prefixes_to_update: tuple[str, ...] | None = (
            tuple(submodules_to_update) if submodules_to_update is not None else None
        )
        if self.prefixes_to_ignore is not None:
            self.params: dict[str, torch.Tensor] = {
                name: p.clone().detach()
                for name, p in model.named_parameters()
                if not name.startswith(self.prefixes_to_ignore)
            }
        else:
            self.params = {
                name: p.clone().detach() for name, p in model.named_parameters()
            }
        self.device: torch.device = next(iter(self.params.values())).device

    def to(self, device: torch.device):
        self.device = device
        self.params = {k: v.to(device) for k, v in self.params.items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Parameters
        ----------
        model: torch.nn.Module
            The model to update the moving average of parameters for.
        """
        prefixes = self.prefixes_to_update
        to_update = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.params:
                continue
            if prefixes is not None and not name.startswith(prefixes):
                # If prefixes_to_update is specified, only update parameters
                # that start with one of the specified prefixes
                continue
            to_update[name] = p
        if not to_update:
            raise ValueError(
                f"No parameters to update. Check if the specified prefixes are correct.\n"
                f"Available parameters: {list(self.params.keys())}\n"
                f"Specified prefixes: {prefixes}\n"
            )

        for name, p_new in to_update.items():
            p = self.params[name]
            diff = p - p_new
            diff *= 1 - self.decay
            p -= diff

    def state_dict(self):
        return {
            "decay": self.decay,
            "params": self.params,
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
    ):
        self.decay = state_dict["decay"]
        for k, p in state_dict["params"].items():
            if k not in self.params:
                if strict:
                    raise KeyError(
                        f"Parameter {k} from state_dict not found in EMA parameters."
                    )
                else:
                    print(
                        f"Warning: Parameter {k} from state_dict not found "
                        f"in EMA parameters. Ignoring."
                    )
                    continue
            self.params[k] = p.clone()
