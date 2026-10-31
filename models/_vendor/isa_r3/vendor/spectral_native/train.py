"""Train the spectral-native model, its Cartesian ablations and a transformer baseline.

Tasks
  mqar  — multi-query associative recall: k1 v1 … kn vn, then queried keys; predict each value
  copy  — random tokens, a separator, then the same tokens again; predict the second half
  lm    — byte-level WikiText-2 (bits per byte)

Arms
  spectral            all native components, Adamax + phase wrap
  spectral-softattn   attention superposes values (Cartesian read)
  spectral-relu       linear + ReLU instead of the tropical MLP
  spectral-addres     Cartesian value-add residual instead of binding
  spectral-adamw      AdamW instead of Adamax + phase wrap
  v2, v2-*            slot register file (model_v2); same ablations
  v3                  v2 plus the audit fixes (model_v3): law offsets, recency, key gate, strict causal
  v3-softattn / v3-relu / v3-adamw   ablations as above
  v3-tropical         MLP = collapsed one-layer tropical max − max over all slots instead of the lookup
  v3-freepos          position query as 32 free phases instead of an integer offset on the law
  v3b, v3b-*          v3 plus the second audit's fixes (soft-backward magnitude selection, live binding/carry,
                      table-sized lookup temperature, quadrature ablation inputs, small magnitude init)
  v3b-singlegrad       v3b with the accidental second selector-logit gradient path removed
  v3b-qrecency         v3b with the nonnegative recency coefficient snapped to the L grid
  v3c                  both third-audit changes; hard routing and token argmax admit an integer lowering
  transformer         a standard pre-norm transformer of similar width
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import Config, SpectralNative, TinyTransformer, wrap_phase_params
from .model_v2 import ConfigV2, SpectralNativeV2, wrap_phase_params_v2
from .model_v3 import ConfigV3, SpectralNativeV3, wrap_phase_params_v3

WIKITEXT = ("/home/beast/.cache/huggingface/hub/datasets--wikitext/snapshots/"
            "b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/{split}-00000-of-00001.parquet")


def batch_mqar(rng, b, n_pairs, n_q, vocab=256):
    half = vocab // 2
    seq, tgt = [], []
    for _ in range(b):
        keys = rng.choice(half, n_pairs, replace=False)
        vals = rng.integers(half, vocab, n_pairs)
        qs = rng.integers(0, n_pairs, n_q)
        s = np.empty(2 * n_pairs + 2 * n_q, dtype=np.int64)
        s[0:2 * n_pairs:2], s[1:2 * n_pairs:2] = keys, vals
        s[2 * n_pairs::2] = keys[qs]
        s[2 * n_pairs + 1::2] = vals[qs]
        t = np.full_like(s, -100)
        t[2 * n_pairs::2] = vals[qs]            # predict the value right after each queried key
        seq.append(s), tgt.append(t)
    return np.stack(seq), np.stack(tgt)


def batch_copy(rng, b, n, vocab=256):
    sep = vocab - 1
    x = rng.integers(0, vocab - 1, (b, n))
    s = np.concatenate([x, np.full((b, 1), sep), x], 1)
    t = np.full_like(s, -100)
    t[:, n:2 * n] = x                            # after the separator, predict the repeat
    return s, t


class LMData:
    def __init__(self, split):
        import pyarrow.parquet as pq
        text = "\n".join(pq.read_table(WIKITEXT.format(split=split)).column("text").to_pylist())
        self.data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int64)

    def batch(self, rng, b, t):
        i = rng.integers(0, len(self.data) - t - 1, b)
        s = np.stack([self.data[j:j + t] for j in i])
        return s[:, :-1], s[:, 1:]


def make_model(arm, args):
    if arm.startswith("v3"):
        cfg = ConfigV3(
            slot_ch=args.slot_ch,
            heads=args.heads,
            layers=args.layers,
            hidden=args.hidden,
            max_len=args.seq + 8,
            mlp_keys=args.mlp_keys,
        )
        if arm == "v3b-singlegrad":
            cfg.rev = 2
            cfg.selector_grad = "single"
            arm = "v3"
        elif arm == "v3b-qrecency":
            cfg.rev = 2
            cfg.recency = "quantized"
            arm = "v3"
        elif arm.startswith("v3c"):
            cfg.rev = 2
            cfg.selector_grad = "single"
            cfg.recency = "quantized"
            arm = "v3" + arm[3:]
        elif arm.startswith("v3b"):
            cfg.rev = 2
            arm = "v3" + arm[3:]
        if arm == "v3-softattn":
            cfg.attn = "soft"
        if arm == "v3-relu":
            cfg.mlp = "relu"
        if arm == "v3-tropical":
            cfg.mlp = "tropical"
        if arm == "v3-freepos":
            cfg.pos = "free"
        return SpectralNativeV3(cfg)
    if arm.startswith("v2"):
        cfg = ConfigV2(
            slot_ch=args.slot_ch,
            heads=args.heads,
            layers=args.layers,
            hidden=args.hidden,
            max_len=args.seq + 8,
        )
        if arm == "v2-softattn":
            cfg.attn = "soft"
        if arm == "v2-relu":
            cfg.mlp = "relu"
        return SpectralNativeV2(cfg)
    if arm == "transformer":
        return TinyTransformer(256, args.channels, args.heads, args.layers, args.seq + 8)
    cfg = Config(channels=args.channels, heads=args.heads, layers=args.layers, hidden=args.hidden, max_len=args.seq + 8)
    if arm == "spectral-softattn":
        cfg.attn = "soft"
    if arm == "spectral-relu":
        cfg.mlp = "relu"
    if arm == "spectral-addres":
        cfg.residual = "add"
    return SpectralNative(cfg)


def get_batch(task, rng, args, lm):
    if task == "mqar":
        s, t = batch_mqar(rng, args.batch, args.pairs, args.queries)
    elif task == "copy":
        s, t = batch_copy(rng, args.batch, args.seq // 2)
    else:
        s, t = lm.batch(rng, args.batch, args.seq + 1)
    return torch.as_tensor(s), torch.as_tensor(t)


def evaluate(model, task, args, lm, dev, seed):
    rng = np.random.default_rng(seed + 12345)
    model.eval()
    tot, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(args.eval_batches):
            s, t = get_batch(task, rng, args, lm)
            s, t = s.to(dev), t.to(dev)
            logits = model(s).float()
            m = t != -100
            tot += F.cross_entropy(logits[m], t[m], reduction="sum").item()
            correct += (logits.argmax(-1)[m] == t[m]).sum().item()
            n += int(m.sum())
    model.train()
    return {"loss": tot / n, "acc": correct / n, "bpb": tot / n / math.log(2)}


def grid_exactness(model, task, args, lm, dev, seed):
    """Spectral arms: float32 forward vs float64 forward. This tests that predictions are robust to float
    precision, NOT an integer forward (none exists yet): score reductions are float sums (max logit diff
    ~5e-7), so a near-tie could still flip under true integer arithmetic."""
    if isinstance(model, TinyTransformer):
        return None
    rng = np.random.default_rng(seed + 999)
    s, _target = get_batch(task, rng, args, lm)
    s = s.to(dev)
    model.eval()
    with torch.no_grad():
        a = model(s).float()
        model.double()
        b = model(s).float()
        model.float()
    model.train()
    return {"argmax_agree": float((a.argmax(-1) == b.argmax(-1)).float().mean()),
            "max_abs_logit_diff": float((a - b).abs().max())}


def run(arm, task, seed, args, dev, lm_train, lm_val):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_model(arm, args).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    # integer law offsets (v3 δ) step in positions, not in fractions of a turn: own learning rate
    delta = [p for n, p in model.named_parameters() if n.endswith(".delta")]
    rest = [p for n, p in model.named_parameters() if not n.endswith(".delta")]
    groups = [{"params": rest}] + ([{"params": delta, "lr": args.lr * args.delta_lr_mult}] if delta else [])
    if arm in ("transformer", "spectral-adamw", "v2-adamw", "v3-adamw"):
        opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.Adamax(groups, lr=args.lr)
    curve, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        s, t = get_batch(task, rng, args, lm_train)
        s, t = s.to(dev), t.to(dev)
        logits = model(s)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), t.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if isinstance(model, SpectralNative) and arm != "spectral-adamw":
            wrap_phase_params(model)
        if isinstance(model, SpectralNativeV2) and arm != "v2-adamw":
            wrap_phase_params_v2(model)
        if isinstance(model, SpectralNativeV3) and arm != "v3-adamw":
            wrap_phase_params_v3(model)
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(model, task, args, lm_val, dev, seed)
            curve.append({"step": step, "train_loss": float(loss), **ev, "sec": round(time.time() - t0, 1)})
            print(json.dumps({"arm": arm, "task": task, "seed": seed, **curve[-1]}), flush=True)
    return {"arm": arm, "task": task, "seed": seed, "params": n_params, "curve": curve, "final": curve[-1],
            "grid_exactness": grid_exactness(model, task, args, lm_val, dev, seed),
            "peak_mem_mb": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="mqar,copy,lm")
    ap.add_argument(
        "--arms",
        default="spectral,spectral-softattn,spectral-relu,spectral-addres,spectral-adamw,transformer",
    )
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--queries", type=int, default=16)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--slot-ch", type=int, default=32)
    ap.add_argument("--mlp-keys", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--delta-lr-mult", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm_train = lm_val = None
    if "lm" in args.tasks:
        lm_train, lm_val = LMData("train"), LMData("validation")
    result = {"host": platform.node(), "device": str(dev), "torch": torch.__version__, "args": vars(args), "runs": []}
    for task in args.tasks.split(","):
        for arm in args.arms.split(","):
            for seed in [int(x) for x in args.seeds.split(",")]:
                result["runs"].append(run(arm, task, seed, args, dev, lm_train, lm_val))
                if dev.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(result, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
