"""Train one direct-effect executor variant (runs inside a GPU lease via ``isa_bench.worker``).

Port of ``isa/scripts/direct_effect_v2.py child()`` for a single architecture: generate the
Unicorn-labelled instruction streams, train with the Saturn checkpoint controller
(``direct_effect.v2.train``), score the held panels, and write the artifact the adapter loads:

  artifact/model.pt       accepted state dict
  artifact/result.json    {"arch", "config", "training", "panels", "data_gen_s"}

Every knob is the model config's ``train`` block, so a variant is one Hydra override away:
  python -m isa_bench stage=train model=de_mamba model.train.seed=674 +model.variant=s674
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

DATA_KEYS = ("train_n", "feedback_n", "test_n", "probe_n", "gen_workers", "cuda", "vram_mb", "ram_mb", "est_wall_s",
             "cpu_threads")


def main(model: dict, root: Path, out: Path, emit=print) -> dict:
    here = Path(root) / model["code"]
    for p in (here / "src", Path(root) / "models/_vendor/isa_r3"):
        sys.path.insert(0, str(p))
    import torch
    from direct_effect import v2 as V

    spec = dict(model["train"])
    torch.set_num_threads(int(spec.get("cpu_threads", 4)))
    device = "cuda" if torch.cuda.is_available() and spec.get("cuda", True) else "cpu"
    seed, workers = int(spec["seed"]), int(spec.get("gen_workers", 4))
    geometry = model.get("architecture") or {}
    cfg = {k: v for k, v in spec.items() if k not in DATA_KEYS}
    cfg.update(seed=seed, aligned=bool(geometry.get("aligned", spec.get("aligned", False))))
    cfg.setdefault("max_attempt_steps", int(cfg["max_updates"] * 1.5))
    t0 = time.perf_counter()
    gen = lambda n, s, mode="train": V.generate(n, s, mode=mode, workers=workers)  # noqa: E731
    train = V.to_device(gen(int(spec["train_n"]), seed * 10 + 1), device)
    feedback = V.to_device(gen(int(spec.get("feedback_n", 4096)), seed * 10 + 2), device)
    test_n = int(spec.get("test_n", 8192))
    panels = {"iid": V.to_device(gen(test_n, seed * 10 + 5), device),
              "heldout_literal": V.to_device(gen(test_n, seed * 10 + 6, mode="held"), device),
              "heldout_pairs": V.to_device(V.generate(test_n // 2, seed * 10 + 7, pairs="only", workers=workers), device)}
    probe = V.to_device(gen(int(spec.get("probe_n", 1500)), seed * 10 + 9), device)
    gen_s = time.perf_counter() - t0
    emit("ISA_BENCH_TRAIN_DATA:" + json.dumps({"train_n": spec["train_n"], "gen_s": round(gen_s, 1), "device": device}))
    work = Path(out).parent / "train-work"
    model_obj, info = V.train(model["arch"], cfg, work, device, emit=emit, data=(train, feedback, probe))
    result = {"arch": model["arch"], "config": cfg, "training": info, "data_gen_s": gen_s,
              "panels": {k: V.exactness(model_obj, v) for k, v in panels.items()},
              "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model_obj.state_dict().items()}, out / "model.pt")
    (out / "result.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))
    shutil.rmtree(work / "futures", ignore_errors=True)  # controller futures are large; the receipts stay in training
    emit("ISA_BENCH_TRAINED:" + json.dumps({"arch": model["arch"], "seed": seed,
                                             "panels": {k: v.get("state") for k, v in result["panels"].items()}}))
    return result
