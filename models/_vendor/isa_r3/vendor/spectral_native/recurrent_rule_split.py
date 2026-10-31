"""Whole-permutation split for the per-world rule task, with shortcut checks.

Training bindings are the 12 even permutations of the four behaviors; held-out
bindings are the 12 odd ones.  Each coset is sharply 2-transitive (the only two
12-sets with that property), so every (symbol, behavior) pair appears three
times in training and every ordered pair of (symbol, behavior) assignments
appears exactly once in each coset.  Single-symbol and pairwise statistics are
therefore identical in training and held-out.  Conditioning on two symbols'
behaviors fixes the parity-consistent completion, so a two-given predictor
learned from training is wrong on every held-out case; ``shortcut_checks``
reports this order-3 anti-shortcut rather than hiding it.  Three entries
determine a binding, so no three-given context of a held-out binding ever
occurs in training.
"""

from __future__ import annotations

import itertools
from collections import Counter

from .recurrent_rule import PERMUTATIONS


def parity(perm: tuple[int, ...]) -> int:
    inversions = sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
    return 1 if inversions % 2 == 0 else -1


EVEN_PERMS = tuple(p for p in PERMUTATIONS if parity(p) == 1)
ODD_PERMS = tuple(p for p in PERMUTATIONS if parity(p) == -1)
PERM_TRAIN = EVEN_PERMS
PERM_HELDOUT = ODD_PERMS


def _majority_accuracy(train, heldout, given: tuple[int, ...], target: int) -> float:
    """Expected accuracy of predicting target's behavior from the given symbols' behaviors
    by training-majority lookup, ties split uniformly; unseen contexts fall back to the
    training marginal of the target."""
    marginal = Counter(p[target] for p in train)
    total = 0.0
    for perm in heldout:
        context = tuple(perm[s] for s in given)
        counts = Counter(p[target] for p in train if tuple(p[s] for s in given) == context) or marginal
        best = [b for b, v in counts.items() if v == max(counts.values())]
        total += (perm[target] in best) / len(best)
    return total / len(heldout)


def shortcut_checks(train=PERM_TRAIN, heldout=PERM_HELDOUT) -> dict:
    """Held-out accuracy of training-statistics predictors of one symbol's behavior."""
    symbols = range(4)
    order1 = [_majority_accuracy(train, heldout, (), s) for s in symbols]
    order2 = [_majority_accuracy(train, heldout, (g,), s) for s in symbols for g in symbols if g != s]
    order3 = [_majority_accuracy(train, heldout, g, s) for s in symbols
              for g in itertools.combinations([x for x in symbols if x != s], 2)]
    order4 = [_majority_accuracy(train, heldout, tuple(x for x in symbols if x != s), s) for s in symbols]
    pairs_in_train = all(any(p[s] == b for p in train) for s in symbols for b in range(4))
    return {
        "train_bindings": [list(p) for p in train], "heldout_bindings": [list(p) for p in heldout],
        "overlap": len(set(train) & set(heldout)), "every_symbol_behavior_pair_in_train": pairs_in_train,
        "order1_per_symbol_majority": {"mean": sum(order1) / len(order1), "chance": 0.25, "values": order1},
        "order2_one_given": {"mean": sum(order2) / len(order2), "chance": 1 / 3, "min": min(order2), "max": max(order2)},
        "order3_two_given": {"mean": sum(order3) / len(order3), "chance": 0.5, "min": min(order3), "max": max(order3)},
        "order4_three_given": {"mean": sum(order4) / len(order4), "chance": 0.25,
                               "note": "three entries determine a binding, so no held-out context occurs in training; lookup falls back to the marginal"},
    }
