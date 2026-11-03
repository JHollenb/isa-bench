"""Factor-grid held-out splits for the v2 generator (round 5 and later).

Composition used to be measured on seven hand-picked (form, register) pairs. The grid instead holds out cells
along each factor of an instruction, so every factor gets its own held-out panel:

  dest_form   (form, destination register)    one or more per register-writing form
  src_form    (form, source register)         register-register forms
  literal     (form, low immediate byte)      immediate forms (the global HELD_BYTES stay the held-literal panel)
  width       whole (operation, width) forms  e.g. SUB r32, imm8 while ADD r32, imm8 and SUB r64, imm8 train
  pairs7      the original seven pairs         kept as a named panel for continuity with rounds 1-4

Training and feedback streams never contain an instruction that hits any held cell. Each panel contains only
instructions that hit exactly its own kind of cell, so a failure is attributable to one factor.
"""
from __future__ import annotations

import math
import random

import numpy as np

from direct_effect import profile as P
from direct_effect import v2

KINDS = ("dest_form", "src_form", "literal", "width", "pairs7")
SRC_FORMS = ("xor_r32_r32", "mov_r64_r64", "mov_r32_r32", "mov_r64_rm", "xor_r64_r64")
NO_DEST = ("nop", "syscall", "store_m64_r64")
# Width cells: the REX.W twin of a trained form. SUB r32, imm8 (83 E8+r) vs trained SUB r64, imm8 (48 83 E8+r) and
# ADD r32, imm8; XOR r64, r64 (48 31) vs trained XOR r32, r32 (31) and MOV r64, r64 (48 89).
WIDTH_FORMS = ("sub_r32_8", "xor_r64_r64")
SPECIAL_BYTES = (0, 1, 8, 60)


def dest_forms(width=WIDTH_FORMS):
    return tuple(f for f in v2.FORM_NAMES if f not in P.AL8 and f not in NO_DEST and f not in width)


def make_grid(seed=20261015, dest_per_form=1, src_per_form=1, literal_per_form=2, width=WIDTH_FORMS):
    """Deterministic, JSON-serializable grid. Held destinations are spread so each register is held about equally
    often and no form loses more than dest_per_form registers beyond the original seven pairs."""
    rng = random.Random(seed)
    grid = {"seed": seed, "width": list(width), "pairs7": sorted([list(p) for p in v2.HELDOUT_PAIRS])}
    dest, counts = [], [0] * 8
    for form in dest_forms(width):
        taken = {r for f, r in v2.HELDOUT_PAIRS if f == form}
        for _ in range(dest_per_form):
            options = [r for r in range(8) if r not in taken]
            low = min(counts[r] for r in options)
            r = rng.choice([r for r in options if counts[r] == low])
            taken.add(r)
            counts[r] += 1
            dest.append([form, r])
    grid["dest_form"] = sorted(dest)
    src = []
    for form in SRC_FORMS:
        if form in width:
            continue
        src += [[form, r] for r in rng.sample(range(8), src_per_form)]
    grid["src_form"] = sorted(src)
    pool = [b for b in range(256) if b not in v2.HELD_BYTES and b not in SPECIAL_BYTES]
    grid["literal"] = sorted([f, b] for f in v2.FORM_NAMES if v2.FORMS[f][2] > 0 and f not in width
                             for b in rng.sample(pool, literal_per_form))
    return grid


class _Index:
    def __init__(self, grid):
        self.width = set(grid["width"])
        self.pairs7 = {tuple(p) for p in grid["pairs7"]}
        self.dest = {tuple(p) for p in grid["dest_form"]}
        self.src = {tuple(p) for p in grid["src_form"]}
        self.literal = {tuple(p) for p in grid["literal"]}


_CACHE = {}


def _index(grid):
    key = id(grid)
    if key not in _CACHE:
        _CACHE[key] = (grid, _Index(grid))
    return _CACHE[key][1]


def cells(ins, grid):
    """The set of held kinds an instruction hits (empty = trainable)."""
    ix, hit = _index(grid), set()
    if ins.form in ix.width:
        hit.add("width")
    if (ins.form, ins.dest) in ix.pairs7:
        hit.add("pairs7")
    if ins.form not in P.AL8 and ins.form not in NO_DEST and (ins.form, ins.dest) in ix.dest:
        hit.add("dest_form")
    if ins.form in SRC_FORMS and (ins.form, ins.src) in ix.src:
        hit.add("src_form")
    if v2.FORMS[ins.form][2] > 0 and (ins.form, ins.imm & 0xFF) in ix.literal:
        hit.add("literal")
    return hit


def random_instruction(rng, grid, split="train", mode="train", mix=None, forms=v2.FORM_NAMES):
    """split="train": never hits a held cell. split=<kind>: hits exactly that kind of cell."""
    if split == "train":
        while True:
            ins = v2.random_instruction(rng, mode, forms, pairs="all", mix=mix)
            if not cells(ins, grid):
                return ins
    if split not in KINDS:
        raise ValueError(split)
    pool = {"width": [[f] for f in grid["width"]], "pairs7": grid["pairs7"], "dest_form": grid["dest_form"],
            "src_form": grid["src_form"], "literal": grid["literal"]}[split]
    pool = [c for c in pool if c[0] in forms]
    while True:
        cell = rng.choice(pool)
        form = cell[0]
        reg = 0 if form in P.AL8 else rng.randrange(8)
        src = rng.randrange(8)
        if split in ("dest_form", "pairs7"):
            reg = cell[1]
        if split == "src_form":
            src = cell[1]
        imm = v2.raw_imm(rng, v2.FORMS[form][2], mode)
        if split == "literal":
            imm = (imm & ~0xFF) | cell[1]
        ins = v2.encode(form, reg, imm, src)
        if cells(ins, grid) == {split}:
            return ins


def _chunk(args):
    n, seed, grid, split, mode, mix = args
    rng, emu = random.Random(seed), v2.Emulator()
    return v2.pack([v2.make_example(rng, emu, random_instruction(rng, grid, split, mode, mix)) for _ in range(n)])


def generate(n, seed, grid, split="train", mode="train", workers=4, mix=None):
    import multiprocessing as mp
    per = math.ceil(n / workers)
    jobs = [(min(per, n - i * per), seed * 1000 + i, grid, split, mode, mix) for i in range(workers) if n - i * per > 0]
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_chunk, jobs)
    else:
        parts = [_chunk(j) for j in jobs]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
