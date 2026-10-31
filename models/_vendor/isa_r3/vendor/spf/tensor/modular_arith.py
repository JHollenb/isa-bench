"""Modular number-theory primitives shared by the character surfaces.

``primitive_root`` / ``discrete_log_tables`` underpin both the additive-readout
multiplicative resolution (:mod:`spf.tensor.modular_readout`) and the
multiplicative character surface (:mod:`spf.tensor.character`'s
``DiscreteLogCharacterSurface``). They live here so the discrete-log change of
basis has a **single source of truth** — and a single primality guard.

The guard matters: ``Z_p^*`` is cyclic (so a primitive root, and a clean
``dlog`` table, exist) only for prime ``p`` (and a few special moduli SPF does
not use). For a composite ``p`` the additive loop would silently leave holes in
the table and the discrete-log basis would be wrong. These functions raise
rather than return that garbage.
"""

from __future__ import annotations

import numpy as np

__all__ = ["discrete_log_tables", "dlog_table", "is_prime", "primitive_root"]


def is_prime(n: int) -> bool:
    """Trial-division primality test (the moduli here are small task primes)."""
    n = int(n)
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def primitive_root(p: int) -> int:
    """Smallest primitive root ``g`` of the prime ``p`` (a generator of ``Z_p^*``).

    Raises ``ValueError`` if ``p`` is not prime — ``Z_p^*`` then has no single
    generator, so the discrete-log table the callers build would be silently
    wrong (holes for the non-coprime residues).
    """
    p = int(p)
    if not is_prime(p):
        raise ValueError(f"p={p} is not prime; Z_p^* has no primitive root for the dlog basis")
    if p == 2:
        return 1
    phi = p - 1
    # Prime factors of φ(p) = p−1; g is primitive iff g^(φ/q) ≠ 1 for every such q.
    factors: list[int] = []
    n = phi
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    for g in range(2, p):
        if all(pow(g, phi // q, p) != 1 for q in factors):
            return g
    raise ValueError(f"no primitive root found for p={p}")


def discrete_log_tables(p: int, generator: int | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """``(dlog, exp_table, g)`` for ``Z_p^* = ⟨g⟩``.

    ``exp_table[k] = g^k mod p`` over ``k ∈ [0, p−1)``; ``dlog[x]`` is its inverse
    for ``x ∈ [1, p)`` (``dlog[0] = −1``, the off-group sentinel). The pair is the
    change of basis that carries multiplication mod ``p`` to addition mod ``p−1``.

    Raises ``ValueError`` for a non-prime ``p`` (even when ``generator`` is given
    explicitly — the table is only well-defined over a cyclic ``Z_p^*``).
    """
    p = int(p)
    if not is_prime(p):
        raise ValueError(f"p={p} is not prime; the discrete-log table is undefined over a non-cyclic Z_p^*")
    g = primitive_root(p) if generator is None else int(generator)
    exp_table = np.empty(p - 1, dtype=np.int64)
    dlog = np.full(p, -1, dtype=np.int64)
    acc = 1
    for k in range(p - 1):
        exp_table[k] = acc
        dlog[acc] = k
        acc = (acc * g) % p
    return dlog, exp_table, g


def dlog_table(p: int, generator: int | None = None) -> list[int]:
    """List form of ``dlog``: ``dlog[x] = log_g(x)`` for ``x∈[1,p)``; ``dlog[0] = -1``."""
    dlog, _, _ = discrete_log_tables(int(p), generator)
    return [int(v) for v in dlog]
