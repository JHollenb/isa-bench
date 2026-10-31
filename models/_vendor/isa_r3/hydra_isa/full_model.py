"""Instruction-local Hydra-style CPU reference learner."""
import torch
from torch import nn

from isa_lab.full_models import HEAD_SIZES, TrainableSelectors
from isa_lab.ssm import HydraMixer, SSMRuntimeHead, SSMStageHead
from pan_isa.native import runtime_module, stages


class HydraISASelectors(TrainableSelectors):
    name = "hydra"
    implementation = "hydra-style-shifted-bidirectional-ssd-cpu-v1"

    def __init__(self, seed=17):
        super().__init__()
        torch.manual_seed(seed)
        self.heads = nn.ModuleDict({str(i): SSMStageHead(len(s.OPS), HEAD_SIZES[i], i in (2, 3), HydraMixer)
                                    for i, s in enumerate(stages())})
        self.heads["runtime"] = SSMRuntimeHead(len(runtime_module().NUMERIC_FEATURE_NAMES), HydraMixer)
