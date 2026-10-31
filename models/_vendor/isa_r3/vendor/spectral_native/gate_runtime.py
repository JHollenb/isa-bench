"""Typed rule cartridges, numerical evidence selection and reversible program memory.

Program addresses and feature vectors are declared inputs. Associations propose
whole cartridges; current scoped graph evidence determines their applicability.
All state is JSON-capturable, including query dependencies and observation history.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .field_fabric import FabricProgram, FieldFabric, ReadStep, _digest
from .numerical_laws import DIMENSIONLESS, Quantity, RegimeLaw, evaluate_expression
from .role_chart_field import Record, RoleChartField


def _candidates(result):
    if result["status"] == "resolved":
        return [{"bindings": result["bindings"], "sources": result["sources"]}]
    return result.get("candidates", []) if result["status"] == "ambiguous" else []


def _semantic(value):
    """Cache diagnostics are not consumer behavior; witnesses and sources are."""
    if isinstance(value, dict):
        return {
            key: _semantic(item)
            for key, item in value.items()
            if key not in {"cached", "recomputed", "fingerprints", "address"}
        }
    if isinstance(value, list):
        return [_semantic(item) for item in value]
    return value


def rule_outcome(result):
    """Declared monitor target: decision status and complete returned bindings."""
    if result is None:
        return None
    return {
        "status": result["status"],
        **{
            key: sorted([row["bindings"] for row in result[key]], key=_digest)
            for key in ("allowed", "blocked", "unknown")
        },
    }


@dataclass(frozen=True)
class RuleCartridge:
    body: FabricProgram
    exceptions: tuple[FabricProgram, ...] = ()
    truth_variable: str = "truth"

    def __post_init__(self):
        object.__setattr__(self, "exceptions", tuple(self.exceptions))
        if any(self.truth_variable not in program.returns for program in self.exceptions):
            raise ValueError("exception must return the declared truth variable")

    def state(self):
        return {
            "body": self.body.state(),
            "exceptions": [p.state() for p in self.exceptions],
            "truth_variable": self.truth_variable,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            FabricProgram.from_state(state["body"]),
            tuple(FabricProgram.from_state(p) for p in state["exceptions"]),
            state["truth_variable"],
        )


class GateRuntime:
    STATE_SCHEMA = "fieldlm-gate-runtime-v1"

    def __init__(self, fabric):
        self.fabric = fabric
        self.rules = {}
        self.banks = {}
        self.memory = None

    def install_rule(self, name, cartridge):
        if not name or not isinstance(cartridge, RuleCartridge):
            raise ValueError("invalid rule cartridge")
        self.fabric._validate_program(cartridge.body)
        for program in cartridge.exceptions:
            self.fabric._validate_program(program)
        self.rules[name] = cartridge.state()
        if self.memory is not None and name not in self.memory["names"]:
            self.memory["names"].append(name)
            self.memory["matrix"].append([0.0] * len(self.memory["keys"][0]))

    def execute_rule(self, name, inputs, *, namespace, scope, budget=None):
        cartridge = RuleCartridge.from_state(self.rules[name])
        total_reads = len(cartridge.body.steps) + sum(len(p.steps) for p in cartridge.exceptions)
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("budget must be a non-negative integer")
        # Budget is per complete candidate path, as in FabricProgram.
        body = self.fabric.run(cartridge.body, inputs, namespace, scope, budget)
        if body["status"] == "budget-exhausted" or (budget is not None and budget < total_reads):
            return {"status": "budget-exhausted", "allowed": [], "blocked": [], "unknown": [], "body": body}
        result = {"status": "missing", "allowed": [], "blocked": [], "unknown": [], "body": body}
        for candidate in _candidates(body):
            bindings = {**inputs, **candidate["bindings"]}
            checks, states = [], []
            for program in cartridge.exceptions:
                read = self.fabric.run(program, bindings, namespace, scope)
                truths = {item["bindings"][cartridge.truth_variable] for item in _candidates(read)}
                state = "blocked" if truths == {"true"} else "clear" if truths == {"false"} else "unknown"
                states.append(state)
                checks.append(read)
            category = "blocked" if "blocked" in states else "unknown" if "unknown" in states else "allowed"
            result[category].append({**copy.deepcopy(candidate), "exception_reads": checks})
        if result["allowed"] or result["blocked"] or result["unknown"]:
            result["status"] = "unresolved" if result["unknown"] else "evaluated"
        return result

    def install_bank(self, name, laws, *, namespace, scope):
        """Install immutable law payloads; evidence only changes the active set."""
        laws = list(laws)
        if (
            not name
            or not laws
            or name in self.banks
            or any(not isinstance(law, RegimeLaw) for law in laws)
            or len({law.name for law in laws}) != len(laws)
        ):
            raise ValueError("invalid or duplicate numerical bank")
        if len({law.output_dimension for law in laws}) != 1:
            raise ValueError("hypotheses must predict compatible quantities")
        view_name = "numeric:" + name
        if view_name in self.fabric.views:
            raise ValueError("numerical view name is already in use")
        field = RoleChartField.from_state(
            {"charts": [], "classes": {}, "merge_log": [], "role_class_merges": [], "records": [], "active_axes": None}
        )
        records = []
        for law in laws:
            for regime, coefficients in enumerate(law.coefficients):
                for term, coefficient in sorted(coefficients.items()):
                    roles = (
                        law.name,
                        str(regime),
                        term,
                        coefficient,
                        json.dumps(law.expressions[term], sort_keys=True),
                    )
                    records.append(Record(namespace, scope, 0, roles, _digest(law.state()), len(records)))
        self.fabric.register_view(view_name, field)
        self.fabric.manage_view(view_name)
        self.fabric.mutate_records(view_name, add=records)
        program = FabricProgram(
            "law:" + name,
            (ReadStep(view_name, 0, {0: "$law", 1: "$regime"}, {2: "term", 3: "coefficient", 4: "expression"}),),
            ("term", "coefficient", "expression"),
        )
        self.banks[name] = {
            "namespace": namespace,
            "scope": scope,
            "laws": {law.name: law.state() for law in laws},
            "active": [law.name for law in laws],
            "program": program.state(),
            "history": [],
        }

    def _bank(self, name, namespace, scope):
        bank = self.banks[name]
        if (namespace, scope) != (bank["namespace"], bank["scope"]):
            raise ValueError("numerical bank context mismatch")
        return bank

    def predict_bank(self, name, quantities, *, namespace, scope):
        bank = self._bank(name, namespace, scope)
        predictions = {}
        program = FabricProgram.from_state(bank["program"])
        for identity in bank["active"]:
            law = RegimeLaw.from_state(bank["laws"][identity])
            regime = law.regime(quantities)
            read = self.fabric.run(program, {"law": identity, "regime": str(regime)}, namespace, scope)
            total, sources = Quantity(0, law.output_dimension), set()
            expected = law.coefficients[regime]
            rows = _candidates(read)
            found = {row["bindings"]["term"] for row in rows}
            if found != set(expected) or len(rows) != len(expected):
                raise ValueError("numerical payload is missing, duplicated or ambiguous")
            for row in rows:
                binding = row["bindings"]
                term = binding["term"]
                if (
                    binding["coefficient"] != expected[term]
                    or json.loads(binding["expression"]) != law.expressions[term]
                ):
                    raise ValueError("numerical payload disagrees with installed law")
                value = evaluate_expression(json.loads(binding["expression"]), quantities)
                total = total + value * Quantity(Fraction(binding["coefficient"]), DIMENSIONLESS)
                sources.update(row["sources"])
            predictions[identity] = {
                "quantity": total.state(),
                "regime": regime,
                "sources": sorted(sources),
                "read": read,
            }
        unique = {_digest(value["quantity"]): value["quantity"] for value in predictions.values()}
        status = "inconsistent" if not predictions else "resolved" if len(unique) == 1 else "ambiguous"
        return {
            "status": status,
            "quantity": next(iter(unique.values())) if len(unique) == 1 else None,
            "predictions": predictions,
        }

    def suggest_observation(self, name, menu, *, namespace, scope):
        bank = self._bank(name, namespace, scope)
        actions = list(menu)
        if len({item["id"] for item in actions}) != len(actions):
            raise ValueError("observation action IDs must be unique")
        for item in actions:
            if Fraction(item["cost"]) <= 0:
                raise ValueError("observation cost must be positive")
        parent = _digest(bank)
        ranked = []
        for action in actions:
            predictions = self.predict_bank(name, action["quantities"], namespace=namespace, scope=scope)
            counts = Counter(_digest(row["quantity"]) for row in predictions["predictions"].values())
            n = sum(counts.values())
            reduction = Fraction(n) - sum(Fraction(c * c, n) for c in counts.values()) if n else Fraction(0)
            ranked.append(
                {
                    "id": str(action["id"]),
                    "cost": str(Fraction(action["cost"])),
                    "quantities": {key: value.state() for key, value in action["quantities"].items()},
                    "predictions": {key: value["quantity"] for key, value in predictions["predictions"].items()},
                    "reduction_per_cost": str(reduction / Fraction(action["cost"])),
                }
            )
        if not ranked or max(Fraction(item["reduction_per_cost"]) for item in ranked) == 0:
            return {"status": "no-discriminating-action", "bank": name, "menu": ranked}
        chosen = sorted(ranked, key=lambda item: (-Fraction(item["reduction_per_cost"]), item["id"]))[0]
        ticket = {"bank": name, "bank_sha256": parent, "namespace": namespace, "scope": scope, **chosen}
        ticket["sha256"] = _digest(ticket)
        return {"status": "request-observation", "ticket": ticket, "menu": ranked}

    def record_observation(self, ticket, quantity, *, source):
        ticket = copy.deepcopy(ticket)
        seal = ticket.pop("sha256")
        if seal != _digest(ticket) or not source or not isinstance(quantity, Quantity):
            raise ValueError("invalid observation ticket or quantity")
        bank = self._bank(ticket["bank"], ticket["namespace"], ticket["scope"])
        if ticket["bank_sha256"] != _digest(bank):
            raise ValueError("stale observation ticket")
        quantities = {key: Quantity.from_state(value) for key, value in ticket["quantities"].items()}
        actual = self.predict_bank(ticket["bank"], quantities, namespace=ticket["namespace"], scope=ticket["scope"])
        predictions = {key: value["quantity"] for key, value in actual["predictions"].items()}
        if predictions != ticket["predictions"]:
            raise ValueError("observation predictions drifted")
        if any(quantity.dimension != Quantity.from_state(value).dimension for value in predictions.values()):
            raise ValueError("observation dimension mismatch")
        retained = [key for key, value in predictions.items() if value == quantity.state()]
        event = {
            "ticket_sha256": seal,
            "action": ticket["id"],
            "cost": ticket["cost"],
            "source": source,
            "observed": quantity.state(),
            "before": list(bank["active"]),
            "retained": retained,
            "predictions": predictions,
        }
        bank["history"].append(event)
        bank["active"] = retained
        return copy.deepcopy(event)

    def configure_memory(self, keys, payloads, *, ridge=0.0):
        """Supplied feature keys fit program scores; this is not a signal carrier."""
        keys = np.asarray(keys, dtype=float)
        payloads = list(payloads)
        if (
            keys.ndim != 2
            or not len(keys)
            or keys.shape[1] == 0
            or len(keys) != len(payloads)
            or not np.isfinite(keys).all()
            or np.any(np.linalg.norm(keys, axis=1) == 0)
            or any(name not in self.rules for name in payloads)
            or not np.isfinite(ridge)
            or ridge < 0
        ):
            raise ValueError("invalid program memory")
        names = sorted(self.rules)
        targets = np.eye(len(names))[[names.index(name) for name in payloads]].T
        matrix = (
            targets @ np.linalg.pinv(keys.T)
            if ridge == 0
            else targets @ keys @ np.linalg.inv(keys.T @ keys + ridge * np.eye(keys.shape[1]))
        )
        self.memory = {
            "keys": keys.tolist(),
            "payloads": payloads,
            "names": names,
            "matrix": matrix.tolist(),
            "ridge": float(ridge),
        }

    def recall(self, query, *, margin=0.0):
        if self.memory is None:
            raise ValueError("working memory is not configured")
        query = np.asarray(query, dtype=float)
        matrix = np.asarray(self.memory["matrix"])
        if query.shape != (matrix.shape[1],) or not np.isfinite(query).all() or not np.isfinite(margin) or margin < 0:
            raise ValueError("invalid memory query")
        scores = matrix @ query
        if not np.isfinite(scores).all():
            raise ValueError("nonfinite memory read")
        order = np.argsort(-scores, kind="stable")
        gap = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None
        resolved = float(scores[order[0]]) > 0 and (gap is None or gap > margin)
        return {
            "status": "resolved" if resolved else "ambiguous",
            "cartridge": self.memory["names"][order[0]] if resolved else None,
            "scores": dict(zip(self.memory["names"], scores.tolist(), strict=True)),
            "margin": gap,
        }

    def execute_memory(self, query, inputs, *, namespace, scope, budget=None, margin=0.0):
        proposal = self.recall(query, margin=margin)
        result = (
            self.execute_rule(proposal["cartridge"], inputs, namespace=namespace, scope=scope, budget=budget)
            if proposal["status"] == "resolved"
            else None
        )
        return {"proposal": proposal, "result": result}

    def trial_write(self, key_index, payload, target, collateral, *, dose=1.0, max_regressions=0):
        """Preview through native rule consumers; always restore the entire closure.

        Target and collateral rows are caller-declared monitor queries. A checked
        write can commit only against this exact parent and the explicit policy.
        No held labels are implicitly consulted. Losing futures remain in receipts.
        """
        if (
            self.memory is None
            or type(key_index) is not int
            or not 0 <= key_index < len(self.memory["keys"])
            or payload not in self.memory["names"]
            or not np.isfinite(dose)
            or type(max_regressions) is not int
            or max_regressions < 0
        ):
            raise ValueError("invalid corrective write")
        collateral = list(collateral)
        if not collateral or "expected" not in target:
            raise ValueError("corrective writes require collateral monitor queries")
        parent = self.state()
        parent_sha = _digest(parent)
        rows = [{key: value for key, value in target.items() if key != "expected"}, *collateral]
        measurements = []
        try:
            before = [self.execute_memory(**row) for row in rows]
            matrix = np.asarray(self.memory["matrix"])
            key = np.asarray(self.memory["keys"][key_index])
            desired = np.eye(len(self.memory["names"]))[self.memory["names"].index(payload)]
            matrix += float(dose) * np.outer(desired - matrix @ key, key) / float(key @ key)
            if not np.isfinite(matrix).all():
                raise ValueError("nonfinite corrective write")
            self.memory["matrix"] = matrix.tolist()
            self.memory["payloads"][key_index] = payload
            candidate_memory = copy.deepcopy(self.memory)
            after = [self.execute_memory(**row) for row in rows]
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                measurements.append(
                    {
                        "role": "target" if index == 0 else "collateral",
                        "before": old,
                        "after": new,
                        "changed": _digest(_semantic(old.get("result"))) != _digest(_semantic(new.get("result"))),
                        "route_changed": old["proposal"]["cartridge"] != new["proposal"]["cartridge"],
                    }
                )
            repaired = (
                after[0]["proposal"]["cartridge"] == payload and rule_outcome(after[0]["result"]) == target["expected"]
            )
            regressions = sum(row["changed"] or row["route_changed"] for row in measurements[1:])
            accepted = repaired and regressions <= max_regressions
            receipt = {
                "schema": "fieldlm-working-write-v1",
                "parent_sha256": parent_sha,
                "before_memory": parent["memory"],
                "candidate_memory": candidate_memory,
                "key_index": key_index,
                "payload": payload,
                "dose": float(dose),
                "policy": {"max_regressions": max_regressions},
                "target_repaired": repaired,
                "collateral_regressions": regressions,
                "eligible": accepted,
                "measurements": measurements,
            }
            receipt["sha256"] = _digest(receipt)
        finally:
            self._restore(parent)
        if _digest(self.state()) != parent_sha:
            raise RuntimeError("working-memory preview failed exact restoration")
        return receipt

    @staticmethod
    def _check_receipt(receipt):
        value = copy.deepcopy(receipt)
        if value.pop("sha256") != _digest(value) or value.get("schema") != "fieldlm-working-write-v1":
            raise ValueError("working-memory receipt integrity mismatch")
        return value

    def commit_write(self, receipt):
        checked = self._check_receipt(receipt)
        if not checked["eligible"] or _digest(self.state()) != checked["parent_sha256"]:
            raise ValueError("ineligible or stale working-memory write")
        self.memory = copy.deepcopy(checked["candidate_memory"])
        return {"committed": True, "receipt_sha256": receipt["sha256"]}

    def rollback_write(self, receipt):
        checked = self._check_receipt(receipt)
        if self.memory != checked["candidate_memory"]:
            raise ValueError("working memory changed since this write")
        self.memory = copy.deepcopy(checked["before_memory"])
        return {"rolled_back": True, "receipt_sha256": receipt["sha256"]}

    def state(self):
        return copy.deepcopy(
            {
                "schema": self.STATE_SCHEMA,
                "fabric": self.fabric.state(),
                "rules": self.rules,
                "banks": self.banks,
                "memory": self.memory,
            }
        )

    @classmethod
    def from_state(cls, state, *, fabric=None):
        if state["schema"] != cls.STATE_SCHEMA:
            raise ValueError("unsupported gate runtime state")
        obj = cls(FieldFabric.from_state(state["fabric"]) if fabric is None else fabric)
        obj.rules, obj.banks, obj.memory = copy.deepcopy((state["rules"], state["banks"], state["memory"]))
        for value in obj.rules.values():
            RuleCartridge.from_state(value)
        for bank in obj.banks.values():
            for value in bank["laws"].values():
                RegimeLaw.from_state(value)
        if obj.memory is not None and not np.isfinite(np.asarray(obj.memory["matrix"])).all():
            raise ValueError("nonfinite working memory state")
        return obj

    def _restore(self, state):
        obj = self.from_state(state)
        self.fabric.__dict__.clear()
        self.fabric.__dict__.update(obj.fabric.__dict__)
        self.rules, self.banks, self.memory = obj.rules, obj.banks, obj.memory


__all__ = ["GateRuntime", "RuleCartridge", "rule_outcome"]
