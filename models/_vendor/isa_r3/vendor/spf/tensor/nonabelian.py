"""SPFTensor v3 NONABELIAN_OPERATOR block (Axis 2) — a non-abelian generator.

The abelian primitives (``SEPARABLE_PHASE``, ``ITERATED_OPERATOR``) store a clock:
a generator that *commutes with itself*, replayed as phase addition. Rich relational
meaning — analogy chains, syntax, function composition — almost certainly does **not**
commute (``A∘B ≠ B∘A``). The non-abelian generalisation: a relation operator is an
element of a small **matrix subalgebra** ``𝒜 = span{G₁…G_d}`` that is **closed under
composition** (products of operators stay in ``𝒜``). Store the algebra basis once; each
relation — and every composite — is then ``d`` coordinates, ``A = Σ cᵢ Gᵢ``. The
saturation dimension ``d`` is the non-abelian analogue of "number of clock frequencies":
small ``d`` ⇔ a low-complexity generator ⇔ the relation system **decompiles**.

The decisive instrument is :func:`composition_closure` — the non-abelian productivity
gate. It measures (a) **non-commutativity** (is this genuinely beyond the abelian/clock
regime) and (b) **closure**: does the subspace spanned by the operators stop growing
once their products are included (a small algebra), or keep expanding toward the full
``n²`` (no generator → memorisation). This is "closure under composition of a
low-complexity generator" made mechanical for the non-abelian case.

Reserved id range ``0x8000–0xBFFF`` (design §5.1); this fills ``0x8000``.
Registration happens at import; :mod:`spf.tensor.v3` triggers it lazily.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm, logm

from .core import SPFError, encode_complex
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    register_base_model,
)

# --------------------------------------------------------------------------- #
# Closure / algebra helpers (numpy SVD only — no matrix log, no branch cuts).
# --------------------------------------------------------------------------- #


def _orthonormalise(vectors: np.ndarray, tol: float) -> np.ndarray:
    """Orthonormal basis (rows) of the row-span of ``vectors`` via SVD."""
    v = np.asarray(vectors)
    if v.shape[0] == 0:
        return v
    _, s, Vh = np.linalg.svd(v, full_matrices=False)
    smax = float(s[0]) if s.size else 0.0
    rank = int(np.sum(s > tol * max(smax, 1e-12)))
    return Vh[:rank]


def _log_generators(ops: list[np.ndarray]) -> list[np.ndarray]:
    """Lie-algebra elements ``Xᵢ = log(Aᵢ)``. Falls back to the tangent ``Aᵢ − I`` when
    the matrix logarithm is ill-conditioned (eigenvalues on the negative real axis),
    which still spans the correct algebra to first order for near-identity relations."""
    gens = []
    for A in ops:
        n = A.shape[0]
        try:
            X = logm(A)
            if not np.all(np.isfinite(X)):
                raise ValueError
        except (ValueError, np.linalg.LinAlgError):
            X = A - np.eye(n)
        gens.append(np.asarray(X, dtype=np.complex128))
    return gens


def composition_closure(
    operators: Sequence[np.ndarray],
    *,
    tol: float = 1e-2,
    max_iter: int = 12,
) -> dict[str, Any]:
    """Non-abelian productivity gate: do relation operators close into a small **Lie
    algebra** under the commutator?

    The associative algebra a group generates is generically the whole of ``M_n``
    (Burnside) — so the right "few non-abelian generators" test is the **Lie** closure:
    take ``Xᵢ = log(Aᵢ)``, span them, then keep adjoining commutators ``[Xᵢ, Xⱼ]`` until
    the span stops growing. The closure dimension ``d`` is the Lie-algebra dimension —
    the non-abelian analogue of "number of clock frequencies":

    * ``d ≪ n²`` and **saturated** ⇒ a low-complexity non-abelian generator. Store the
      ``d`` algebra basis matrices once; each relation/composite is ``d`` coordinates.
    * ``d`` grows toward ``n²`` ⇒ no small generator ⇒ the relations are independent and
      must be **memorised**.

    Also reports **non-commutativity** (median ``‖[Aᵢ,Aⱼ]‖ / ‖AᵢAⱼ‖``): ≈0 ⇒ abelian
    (a clock — use ``ITERATED_OPERATOR``); large ⇒ genuinely non-abelian.

    Returns ``{n, ambient, closure_dim, saturated, fill_fraction, n_generators,
    noncommutativity, passed}``. ``passed`` ⇔ non-abelian **and** closes at ``d`` small
    relative to ambient ``n²`` **and** smaller than the number of operators catalogued.
    """
    ops = [np.asarray(A, dtype=np.complex128) for A in operators]
    if not ops:
        raise SPFError("composition_closure needs ≥1 operator")
    n = ops[0].shape[0]
    for A in ops:
        if A.shape != (n, n):
            raise SPFError("all operators must be square and the same size")
    ambient = n * n

    nc = []
    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            comm = ops[i] @ ops[j] - ops[j] @ ops[i]
            nc.append(float(np.linalg.norm(comm) / (np.linalg.norm(ops[i] @ ops[j]) + 1e-12)))
    noncommutativity = float(np.median(nc)) if nc else 0.0

    gens = _log_generators(ops)
    basis = _orthonormalise(np.stack([X.reshape(-1) for X in gens]), tol)  # (d, n²)
    saturated = False
    for _ in range(max_iter):
        d = basis.shape[0]
        mats = basis.reshape(d, n, n)
        comms = [
            (mats[i] @ mats[j] - mats[j] @ mats[i]).reshape(-1)
            for i in range(d) for j in range(i + 1, d)
        ]
        if not comms:
            saturated = True
            break
        augmented = np.concatenate([basis, np.stack(comms)], axis=0)
        new_basis = _orthonormalise(augmented, tol)
        if new_basis.shape[0] == d:
            saturated = True
            basis = new_basis
            break
        basis = new_basis
    closure_dim = int(basis.shape[0])
    fill_fraction = closure_dim / ambient
    passed = bool(
        noncommutativity > 0.05
        and saturated
        and fill_fraction < 0.5
        and closure_dim < len(ops)
    )
    return {
        "n": n,
        "ambient": ambient,
        "closure_dim": closure_dim,
        "saturated": saturated,
        "fill_fraction": fill_fraction,
        "n_generators": closure_dim,
        "n_operators": len(ops),
        "noncommutativity": noncommutativity,
        "passed": passed,
    }


def lie_algebra_basis(
    operators: Sequence[np.ndarray], *, tol: float = 1e-2, max_iter: int = 12
) -> np.ndarray:
    """Orthonormal (Frobenius) basis ``(d, n, n)`` of the Lie algebra generated by the
    operators under commutator — the stored non-abelian generator set."""
    ops = [np.asarray(A, dtype=np.complex128) for A in operators]
    n = ops[0].shape[0]
    gens = _log_generators(ops)
    basis = _orthonormalise(np.stack([X.reshape(-1) for X in gens]), tol)
    for _ in range(max_iter):
        d = basis.shape[0]
        mats = basis.reshape(d, n, n)
        comms = [
            (mats[i] @ mats[j] - mats[j] @ mats[i]).reshape(-1)
            for i in range(d) for j in range(i + 1, d)
        ]
        if not comms:
            break
        new_basis = _orthonormalise(np.concatenate([basis, np.stack(comms)], axis=0), tol)
        if new_basis.shape[0] == d:
            break
        basis = new_basis
    return basis.reshape(basis.shape[0], n, n)


# Back-compat alias: the stored generator set is the Lie-algebra basis.
def algebra_basis(operators: Sequence[np.ndarray], **kw: Any) -> np.ndarray:
    """Alias for :func:`lie_algebra_basis` — the non-abelian generator set to store."""
    return lie_algebra_basis(operators, **kw)


def coords_in_basis(A: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Coordinates of ``log(A)`` in an orthonormal Lie-algebra ``basis`` (d, n, n):
    ``cᵢ = ⟨Gᵢ, log A⟩_F``, so ``A ≈ expm(Σ cᵢ Gᵢ)`` when ``A`` is in the group."""
    X = _log_generators([np.asarray(A, dtype=np.complex128)])[0].reshape(-1)
    B = np.asarray(basis, dtype=np.complex128).reshape(basis.shape[0], -1)
    return B.conj() @ X


# --------------------------------------------------------------------------- #
# The base model.
# --------------------------------------------------------------------------- #


class NonAbelianOperatorModel:
    """Axis-2 base model: a relation operator as an element of a stored matrix algebra.

    ``params``:
      * ``basis_re`` / ``basis_im`` — the algebra basis ``(d, n, n)`` (real / imag parts;
        stored once, shared across every relation that references it).
      * ``coords_re`` / ``coords_im`` — this operator's coordinates ``c`` (length ``d``).

    Reconstructs ``A = Σ cᵢ Gᵢ`` and emits it as the ``(L, Θ)`` planes (residual NONE),
    the same "law generates the planes" pattern as the abelian operators. ``d`` real
    coordinates per relation; the basis is amortised across the whole relation family.
    """

    kind = int(BaseModelKind.NONABELIAN_OPERATOR)

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(dims) != 2 or dims[0] != dims[1]:
            raise SPFError(f"NONABELIAN_OPERATOR block requires a square matrix, got dims={tuple(dims)!r}")
        basis = np.asarray(params["basis_re"], dtype=np.float64) + 1j * np.asarray(
            params["basis_im"], dtype=np.float64
        )
        coords = np.asarray(params["coords_re"], dtype=np.float64) + 1j * np.asarray(
            params["coords_im"], dtype=np.float64
        )
        if basis.ndim != 3 or basis.shape[0] != coords.shape[0]:
            raise SPFError("NONABELIAN_OPERATOR basis/coords shape mismatch")
        X = np.tensordot(coords, basis, axes=(0, 0))  # Σ cᵢ Gᵢ  (a Lie-algebra element)
        # A = expm(X): the non-abelian generalisation of the phase accumulator —
        # exp of a sum of generators, where the abelian case A = exp(i·Σ k ωᵢ) is the
        # diagonal (commuting) special case.
        A = expm(X) if params.get("exp", True) else X
        if A.shape != (dims[0], dims[1]):
            raise SPFError("NONABELIAN_OPERATOR reconstructed operator does not match dims")
        n = dims[0] * dims[1]
        baseL, baseT = encode_complex(A.reshape(-1))
        return np.arange(n, dtype=np.int64), baseL.astype(np.int64), baseT.astype(np.int64)

    def storage_bits(self, params: Mapping[str, Any]) -> int:
        # Per-relation cost is the coordinates only — the basis is amortised across the
        # whole family (charge it once via shares_basis=False on a single block).
        coords = np.asarray(params.get("coords_re", ()))
        bits = int(coords.size) * 2 * 32  # complex coords
        if not params.get("shares_basis", False):
            basis = np.asarray(params.get("basis_re", ()))
            bits += int(basis.size) * 2 * 32
        return bits


# --------------------------------------------------------------------------- #
# Constructor + reader.
# --------------------------------------------------------------------------- #


def build_nonabelian_operator_block(
    A: np.ndarray,
    basis: np.ndarray,
    *,
    shares_basis: bool = False,
    representation: int = int(Representation.POLAR_SPF32),
) -> SPFBlockV3:
    """Store operator ``A`` as coordinates in a non-abelian algebra ``basis`` (d, n, n).

    ``shares_basis=True`` marks this block as referencing a basis stored once elsewhere
    in the family, so :meth:`storage_bits` charges only the ``d`` coordinates — the
    whole point: ``d`` numbers per relation, the generator amortised across the family.
    """
    A = np.asarray(A, dtype=np.complex128)
    basis = np.asarray(basis, dtype=np.complex128)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise SPFError(f"operator must be a square matrix, got shape {A.shape}")
    n = A.shape[0]
    if basis.ndim != 3 or basis.shape[1:] != (n, n):
        raise SPFError(f"basis must be (d, {n}, {n})")
    coords = coords_in_basis(A, basis)
    meta = {
        "basis_re": np.ascontiguousarray(basis.real),
        "basis_im": np.ascontiguousarray(basis.imag),
        "coords_re": np.ascontiguousarray(coords.real),
        "coords_im": np.ascontiguousarray(coords.imag),
        "shares_basis": bool(shares_basis),
        "exp": True,
    }
    base = BaseModelRef(int(BaseModelKind.NONABELIAN_OPERATOR), meta)
    residual = ResidualRef(int(ResidualCodecKind.NONE), None)
    return SPFBlockV3(
        dims=(n, n),
        origin=(0, 0),
        representation=RepresentationRef(int(representation)),
        base=base,
        residual=residual,
    )


def nonabelian_meta(block: SPFBlockV3) -> dict[str, Any]:
    """Return the parameter/metadata dict carried by a NONABELIAN_OPERATOR block."""
    if int(block.base.profile_id) != int(BaseModelKind.NONABELIAN_OPERATOR):
        raise SPFError("block is not a NONABELIAN_OPERATOR block")
    return dict(block.base.params)


register_base_model(NonAbelianOperatorModel())


__all__ = [
    "NonAbelianOperatorModel",
    "algebra_basis",
    "build_nonabelian_operator_block",
    "composition_closure",
    "coords_in_basis",
    "lie_algebra_basis",
    "nonabelian_meta",
]
