"""Instruction-local Mamba-style CPU reference learner."""
import torch
from torch import nn

from isa_lab.full_models import HEAD_SIZES, TrainableSelectors
from isa_lab.ssm import MambaMixer, SSMRuntimeHead, SSMStageHead
from pan_isa.native import runtime_module, stages


class MambaISASelectors(TrainableSelectors):
    name = "mamba"
    implementation = "mamba-style-selective-scan-cpu-v1"

    def __init__(self, seed=17):
        super().__init__()
        torch.manual_seed(seed)
        self.heads = nn.ModuleDict({str(i): SSMStageHead(len(s.OPS), HEAD_SIZES[i], i in (2, 3), MambaMixer)
                                    for i, s in enumerate(stages())})
        self.heads["runtime"] = SSMRuntimeHead(len(runtime_module().NUMERIC_FEATURE_NAMES), MambaMixer)
