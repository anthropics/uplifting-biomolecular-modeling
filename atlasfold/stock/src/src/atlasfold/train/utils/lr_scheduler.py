import torch


class AlphaFoldLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_lr: float = 1.8e-3,
        warmup_steps: int = 1000,
        decay_steps: int = 50000,
        decay_factor: float = 0.95,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer.
        max_lr : float
            The max learning rate, by default 1.8e-3
        warmup_steps : int
            The number of warmup steps, by default 1000
        decay_steps : int
            The number of steps for decay, by default 50000
        decay_factor : float
            The decay factor, by default 0.95
        """
        assert warmup_steps >= 0, "num_warmup_steps must be non-negative"
        assert decay_steps > 0, "decay_steps must be positive"
        self.max_lr: float = max_lr
        self.warmup_steps: int = warmup_steps
        self.decay_steps: int = decay_steps
        self.decay_factor: float = decay_factor
        super().__init__(optimizer, last_epoch)

    def state_dict(self) -> dict:
        state_dict = {k: v for k, v in self.__dict__.items() if k not in ["optimizer"]}
        return state_dict

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        step = self.last_epoch
        if step == -1:
            lr_ratio = 0.0
        elif step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps
        else:
            num_decays = step // self.decay_steps
            lr_ratio = self.decay_factor**num_decays
        lr = lr_ratio * self.max_lr
        return [lr for _ in self.optimizer.param_groups]
