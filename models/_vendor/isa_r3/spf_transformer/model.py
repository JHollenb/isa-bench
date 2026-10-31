from __future__ import annotations

import numpy as np
from torch import nn

from isa_lab.contracts import ACTIONS, Abstained
from isa_lab.dependencies import spf_modules
from isa_lab.profile import instruction_key, proposal


class SPFSelector(nn.Module):
    name = "spf_transformer"
    implementation = "spf-native-phase-read-selector-v1"

    def __init__(self, seed=17, device="cpu"):
        super().__init__()
        if device != "cpu":
            raise ValueError("the initial SPF consumer is CPU/NumPy integer")
        self.core, self.circuits, self.select = spf_modules()
        self.routes = {}
        self._build()

    def _address(self, key):
        codes = np.array([key[0], key[1] if len(key) == 2 else 0, len(key)], dtype=np.uint16)
        words = self.core.pack_words(np.zeros((1, 3), dtype=np.int16), (codes * 256).reshape(1, 3))
        return self.circuits.execute_phase_character_features(words,
                    characters=(((0, 1),), ((1, 1),), ((2, 1),)))

    def _build(self):
        entries = sorted(self.routes.items())
        self.keys = (np.concatenate([self._address(bytes.fromhex(key)) for key, _ in entries], axis=0)
                     if entries else np.empty((0, 3), dtype=np.uint32))
        phases = np.array([(ACTIONS.index(action) + 1) * 4096 for _, action in entries], dtype=np.uint16)
        self.values = self.core.pack_words(np.zeros((len(entries), 1), dtype=np.int16), phases.reshape(-1, 1))

    def get_extra_state(self):
        return {"routes": dict(self.routes), "keys": self.keys.tolist(), "values": self.values.tolist()}

    def set_extra_state(self, state):
        self.routes = dict(state["routes"])
        self._build()
        if self.keys.tolist() != state["keys"] or self.values.tolist() != state["values"]:
            raise ValueError("SPF route program and packed planes disagree")

    def update(self, observations, routes, optimizer=None):
        self.routes = dict(routes)
        self._build()
        return {"learned_records": len(routes), "physical_examples": len(observations), "optimizer_steps": 0}

    def propose(self, instruction, before):
        if not len(self.keys):
            raise Abstained("empty SPF route memory")
        query = self._address(instruction_key(instruction))
        result = self.select.execute_select_head(query, self.keys, self.values,
                    channel_weights=np.ones(3, dtype=np.int64), score="exact_match", min_score=0, causal=False)
        if not bool(result.found[0, 0]):
            raise Abstained("SPF route not found")
        # Explicit packed phase-code view becomes an integer candidate-control ID.
        code = int(self.core.unpack_theta(result.values)[0, 0, 0])
        action = ACTIONS[(code // 4096) - 1]
        return proposal(action, instruction, before,
                        {"packed_route_word": int(result.values[0, 0, 0]),
                         "pairs_scored": result.counters.pairs_scored,
                         "coherent_additions": result.counters.coherent_additions})

    def export(self):
        return {"implementation": self.implementation, "state": self.get_extra_state(), "parameter_count": 0,
                "packed_plane_bytes": self.keys.nbytes + self.values.nbytes,
                "boundary": "native phase/read component; full contextual SPFTransformer remains planned"}

    def restore_export(self, artifact):
        self.set_extra_state(artifact["state"])
