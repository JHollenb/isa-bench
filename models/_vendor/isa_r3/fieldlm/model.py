from __future__ import annotations

from torch import nn

from isa_lab.contracts import Abstained
from isa_lab.dependencies import field_modules
from isa_lab.profile import instruction_key, proposal


class FieldLMSelector(nn.Module):
    name = "fieldlm"
    implementation = "fieldfabric-learned-opcode-records-v1"

    def __init__(self, seed=17, device="cpu"):
        super().__init__()
        if device != "cpu":
            raise ValueError("the initial FieldLM consumer is CPU")
        self.fabric_api, self.chart_api = field_modules()
        self.routes = {}
        self._build()

    def _build(self):
        self.fabric = self.fabric_api.FieldFabric()
        field = self.chart_api.RoleChartField.from_state({"charts": [], "classes": {}, "merge_log": [],
                    "role_class_merges": [], "records": [], "active_axes": None})
        self.fabric.register_view("routes", field)
        self.fabric.manage_view("routes")
        records = [self.chart_api.Record("isa", "x86-al8-v1", 0, (key, action),
                                        f"discovery-route-{key}", index)
                   for index, (key, action) in enumerate(sorted(self.routes.items()))]
        self.fabric.mutate_records("routes", add=records)

    def get_extra_state(self):
        return {"routes": dict(self.routes), "fabric": self.fabric.state()}

    def set_extra_state(self, state):
        self.routes = dict(state["routes"])
        self.fabric = self.fabric_api.FieldFabric.from_state(state["fabric"])

    def update(self, observations, routes, optimizer=None):
        self.routes = dict(routes)
        self._build()
        return {"structural_records": len(routes), "physical_examples": len(observations),
                "optimizer_steps": 0}

    def propose(self, instruction, before):
        key = instruction_key(instruction).hex()
        program = self.fabric_api.FabricProgram("read-isa-route", (
            self.fabric_api.ReadStep("routes", 0, {0: "$opcode"}, {1: "action"}),), ("action",))
        # Use a detached consumer so serving/evaluation cannot mutate trainable cache state.
        # Reconstruction overhead is part of this starter adapter's measured inference time.
        reader = self.fabric_api.FieldFabric.from_state(self.fabric.state())
        result = reader.run(program, {"opcode": key}, "isa", "x86-al8-v1")
        if result["status"] != "resolved":
            raise Abstained(f"FieldLM route {result['status']} for {key}")
        return proposal(result["value"], instruction, before,
                        {"sources": result["sources"], "read_trace": result["trace"]})

    def export(self):
        return {"implementation": self.implementation, "state": self.get_extra_state(),
                "parameter_count": 0, "learned_record_count": len(self.routes)}

    def restore_export(self, artifact):
        self.set_extra_state(artifact["state"])
