"""The actual original Stage 0–4 networks and runtime neural head."""
import torch

from isa_lab.full_models import TrainableSelectors
from pan_isa.native import native_heads


class PANISASelectors(TrainableSelectors):
    name = "pan_isa"
    implementation = "original-pan-stage0-4-and-runtime-heads-v1"

    def __init__(self, seed=17):
        super().__init__()
        torch.manual_seed(seed)
        self.heads = native_heads()
