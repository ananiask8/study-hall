import torch
from os.path import join as join_path
from typing import NamedTuple, Optional, Callable


class Checkpoint(NamedTuple):
    epoch: int = 0
    acc: float = 0.
    loss: float = float('inf')
    f1: float = 0.
    opts: Optional[dict] = None
    model: Optional[str] = None
    optimizer: Optional[dict] = None
    model_state_dict: Optional[dict] = None
    optimizer_state_dict: Optional[dict] = None


class CheckpointStateOpts(NamedTuple):
    compare: Callable
    patience: float
    max_epochs: int
    ckpt_path: str
    ckpt: Optional[Checkpoint] = None


class CheckpointState:
    def __init__(self, opts: CheckpointStateOpts):
        self.compare = opts.compare
        self.ckpt_path = opts.ckpt_path
        self.ckpt = opts.ckpt
        if opts.ckpt is None:
            self.ckpt = Checkpoint()
        self.last_updated = 0
        self.patience = opts.patience
        self.max_epochs = opts.max_epochs

    def update_best(self, ckpt: Checkpoint):
        self.last_updated += 1
        if self.compare(new=ckpt, old=self.ckpt):
            self.last_updated = 0
            self.ckpt = ckpt
            torch.save(ckpt._asdict(), join_path(self.ckpt_path, 'best.pth'))

    def update_nth(self, ckpt: Checkpoint):
        if ckpt.epoch % ckpt.opts['nth_epoch'] == 0:
            torch.save(ckpt._asdict(), join_path(self.ckpt_path, f'ckpt_{ckpt.epoch}.pth'))

    def get_best(self):
        return self.ckpt.epoch, self.ckpt.acc, self.ckpt.loss, self.ckpt.f1

    def is_patience_exhausted(self):
        return self.last_updated / self.max_epochs > self.patience

    def remaining(self):
        return int(self.max_epochs * self.patience - self.last_updated)
