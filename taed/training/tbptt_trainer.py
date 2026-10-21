"""Truncate recurrent policy gradients while retaining complete-episode returns."""

from training.rl_trainer import RLTrainer
from utils.reproducibility import require_int


class TBPTTRLTrainer(RLTrainer):
    def __init__(self, *args, tbptt_chunk_size=100, **kwargs):
        require_int(tbptt_chunk_size, "tbptt_chunk_size")
        if "truncation_steps" in kwargs:
            raise ValueError(
                "Use tbptt_chunk_size rather than truncation_steps with TBPTTRLTrainer."
            )
        super().__init__(*args, truncation_steps=tbptt_chunk_size, **kwargs)
        self.tbptt_chunk_size = tbptt_chunk_size
