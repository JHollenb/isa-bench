"""Promoted exact quantity/expression mechanics and declared switching laws.

The Quantity and expression evaluator originate in the 2026-10-08 math/physics
organism. This is scalar rational computation, not an SPF signal kernel.
Expressions and units are supplied; no general equation discovery is claimed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction

MASS = (1, 0, 0)
LENGTH = (0, 1, 0)
TIME = (0, 0, 1)
VELOCITY = (0, 1, -1)
ACCELERATION = (0, 1, -2)
FORCE = (1, 1, -2)
STIFFNESS = (1, 0, -2)
DRAG = (1, 0, -1)
DIMENSIONLESS = (0, 0, 0)


@dataclass(frozen=True)
class Quantity:
    value: Fraction
    dimension: tuple[int, int, int]

    def __post_init__(self):
        object.__setattr__(self, "value", Fraction(self.value))
        object.__setattr__(self, "dimension", tuple(self.dimension))
        if len(self.dimension) != 3 or any(type(v) is not int for v in self.dimension):
            raise ValueError("dimension must contain three integer exponents")

    def __add__(self, other):
        if self.dimension != other.dimension:
            raise ValueError("cannot add quantities with different dimensions")
        return Quantity(self.value + other.value, self.dimension)

    def __mul__(self, other):
        return Quantity(
            self.value * other.value, tuple(a + b for a, b in zip(self.dimension, other.dimension, strict=True))
        )

    def __truediv__(self, other):
        if not other.value:
            raise ValueError("zero divisor")
        return Quantity(
            self.value / other.value, tuple(a - b for a, b in zip(self.dimension, other.dimension, strict=True))
        )

    def state(self):
        return {"value": str(self.value), "dimension": list(self.dimension)}

    @classmethod
    def from_state(cls, state):
        return cls(Fraction(state["value"]), tuple(state["dimension"]))


def evaluate_expression(expression, quantities):
    if isinstance(expression, str):
        return quantities[expression]
    operation, left, right = expression
    a = evaluate_expression(left, quantities)
    b = evaluate_expression(right, quantities)
    if operation == "mul":
        return a * b
    if operation == "div":
        return a / b
    raise ValueError("unsupported supplied numerical operation")


def validate_expression(expression):
    if isinstance(expression, str) and expression:
        return
    if not isinstance(expression, (list, tuple)) or len(expression) != 3 or expression[0] not in {"mul", "div"}:
        raise ValueError("invalid numerical expression")
    validate_expression(expression[1])
    validate_expression(expression[2])


@dataclass(frozen=True)
class RegimeLaw:
    """Two coefficient maps selected by a dimensioned absolute-value gate."""

    name: str
    variable: str
    threshold: Quantity
    expressions: dict
    coefficients: tuple[dict, dict]
    output_dimension: tuple[int, int, int]

    def __post_init__(self):
        if not self.name or not self.variable or self.threshold.value < 0 or len(self.coefficients) != 2:
            raise ValueError("invalid numerical gate")
        dimension = tuple(self.output_dimension)
        Quantity(0, dimension)
        object.__setattr__(self, "output_dimension", dimension)
        expressions = copy.deepcopy(self.expressions)
        for expression in expressions.values():
            validate_expression(expression)
        coefficients = tuple(
            {str(key): str(Fraction(value)) for key, value in row.items()} for row in self.coefficients
        )
        if any(set(row) - set(expressions) for row in coefficients):
            raise ValueError("coefficient has no declared expression")
        object.__setattr__(self, "expressions", expressions)
        object.__setattr__(self, "coefficients", coefficients)

    def regime(self, quantities):
        gate = quantities[self.variable]
        if gate.dimension != self.threshold.dimension:
            raise ValueError("gate dimension mismatch")
        return int(abs(gate.value) > self.threshold.value)

    def state(self):
        return {
            "name": self.name,
            "variable": self.variable,
            "threshold": self.threshold.state(),
            "expressions": copy.deepcopy(self.expressions),
            "coefficients": copy.deepcopy(list(self.coefficients)),
            "output_dimension": list(self.output_dimension),
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            state["name"],
            state["variable"],
            Quantity.from_state(state["threshold"]),
            state["expressions"],
            tuple(state["coefficients"]),
            tuple(state["output_dimension"]),
        )
