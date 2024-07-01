from dataclasses import dataclass


@dataclass
class State:
    device: str
    distill_steps: int
    distill_epochs: int
    batch_size: int
    seq_len: int
    distill_lr: int
    lr: int
    vocab_size: int
    decay_epochs: int
    decay_factor: int
    epochs: int
    checkpoint_interval: int