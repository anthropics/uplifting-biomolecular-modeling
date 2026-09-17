"""Score the sequences of a FASTA file with Evo 2 — the kit tree's command line over the upstream API (evo2 ships none):
``Evo2(model_name, local_path)`` then ONE ``model.score_sequences(seqs, batch_size)`` call (evo2/models.py:121; the mean per-token
log-likelihood of each sequence, upstream's zero-shot variant-effect score), the scores written in input order.

    python route/evo2_route.py --model_name evo2_7b --input windows.fa --out_dir out/ [--mode exact|fast|off] [--batch_size N] [--local_path CKPT]

Modes: ``exact`` (default) = the kit (``EVO2_OPT=exact`` must be in the environment: ``bash run.sh score`` puts it there) on the stock's
construction; ``off`` = the stock: ``Evo2(model_name, use_kernels=True)`` — the README's documented speed switch — for the one-device models,
the constructor's defaults for ``evo2_40b`` (``use_kernels=True`` scores NaN over its two-device layer split, STOCK.md), in a process where
``EVO2_OPT`` is unset and nothing of the kit is imported (``ENV-CLEAN``). ``--batch_size`` is upstream's ``score_sequences`` argument
(default 1); ``--local_path`` the checkpoint (default
``$EVO2_OPT_WEIGHTS/<model_name>.pt`` when that file exists, else upstream's own Hugging Face download).
Output: ``<out_dir>/scores.jsonl`` — one row per record in input order {id, index, length, score, score_hex} (score = repr of upstream's
float32, score_hex its little-endian bit pattern).
Lines: ``[evo2-route] START …``, ``[evo2-route stock] ENV-CLEAN ok: …`` (off), ``[evo2-route] READY …`` (the model constructed),
``[evo2-route] EXIT … status=ok|not-finite``. Exit: 0 ok · 1 a failure or non-finite scores · 2 usage · 3 the kit refused (its NOT ACTIVE /
KIT REFUSED line names why).
"""
import argparse
import json
import math
import os
import struct
import sys

PREFIX = "[evo2-route]"
STOCK_PREFIX = "[evo2-route stock]"
MODES = ("exact", "fast", "off")
KIT_MODES = ("exact", "fast")                                # fast scores exactly as exact (the same levers); it differs only in generate()
KIT_ENV = "EVO2_OPT"
WEIGHTS_ENV = "EVO2_OPT_WEIGHTS"
KERNELS_OFF = {"evo2_40b": "use_kernels=True scores NaN over the two-device layer split (STOCK.md): the stock is the constructor's defaults",
               "evo2_40b_base": "use_kernels=True scores NaN over the two-device layer split (STOCK.md): the stock is the constructor's defaults"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def read_fasta(path):
    """[(id, sequence)] in file order; the id is the header's first token, the sequence uppercased with whitespace removed."""
    recs, cur, buf = [], None, []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if cur is not None:
                    recs.append((cur, "".join(buf).upper()))
                cur, buf = (line[1:].split() or [""])[0], []
            else:
                buf.append(line)
    if cur is not None:
        recs.append((cur, "".join(buf).upper()))
    return recs


def construction(mode, model_name):
    """The constructor keywords of the arm: every mode builds the stock — use_kernels=True unless the model's stock is the constructor's defaults."""
    if model_name in KERNELS_OFF:
        return {}
    return {"use_kernels": True}


def env_clean_line():
    """The off mode's ENV-CLEAN line: nothing of the kit is engaged in this process; else (None, what is engaged)."""
    engaged = [m for m in ("evo2_opt.activation", "evo2_opt.kit", "evo2_opt.gen") if m in sys.modules]
    import evo2.models as em
    wrapped = hasattr(em.Evo2.__init__, "__wrapped__")
    if os.environ.get(KIT_ENV) not in (None, "", "off") or engaged or wrapped:
        return None, f"{KIT_ENV}={os.environ.get(KIT_ENV)!r} kit modules imported={engaged} Evo2.__init__ wrapped={wrapped}"
    return f"{STOCK_PREFIX} ENV-CLEAN ok: {KIT_ENV} unset, no kit module imported, Evo2.__init__ is upstream's", None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="evo2_route.py", description="Score a FASTA file's sequences with Evo 2 (score_sequences).")
    ap.add_argument("--model_name", default="evo2_7b")
    ap.add_argument("--mode", default="exact", choices=MODES)
    ap.add_argument("--input", required=True, help="FASTA file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=1, help="score_sequences batch_size (upstream's default 1)")
    ap.add_argument("--local_path", default=None, help="checkpoint path (default: $EVO2_OPT_WEIGHTS/<model_name>.pt if present, else upstream's download)")
    a = ap.parse_args(argv)
    env_mode = os.environ.get(KIT_ENV)
    if a.mode in KIT_MODES and env_mode != a.mode:
        log(f"{PREFIX} usage: --mode {a.mode} runs with {KIT_ENV}={a.mode} in the environment (bash run.sh score sets it); got {KIT_ENV}={env_mode!r}"); return 2
    if a.mode not in KIT_MODES and env_mode not in (None, "", "off"):
        log(f"{PREFIX} usage: --mode {a.mode} is a stock arm: unset {KIT_ENV} (got {KIT_ENV}={env_mode!r})"); return 2
    local_path = a.local_path
    if local_path is None and os.environ.get(WEIGHTS_ENV):
        cand = os.path.join(os.environ[WEIGHTS_ENV], f"{a.model_name}.pt")
        local_path = cand if os.path.isfile(cand) else None
    recs = read_fasta(a.input)
    if not recs:
        log(f"{PREFIX} usage: no FASTA records in {a.input}"); return 2
    kw = construction(a.mode, a.model_name)
    why = f" ({KERNELS_OFF[a.model_name]})" if a.model_name in KERNELS_OFF else ""
    log(f"{PREFIX} START mode={a.mode} model={a.model_name} construction=Evo2({a.model_name!r}{''.join(f', {k}={v}' for k, v in kw.items())}){why} "
        f"n={len(recs)} batch_size={a.batch_size} local_path={local_path or 'upstream download'}")
    os.makedirs(a.out_dir, exist_ok=True)
    from evo2 import Evo2
    try:
        model = Evo2(a.model_name, local_path=local_path, **kw)
    except Exception as e:                                   # the kit's refusal out of the constructor (its line printed): exit 3; anything else re-raised
        if type(e).__name__ == "Evo2OptRefused":
            log(f"{PREFIX} EXIT mode={a.mode} model={a.model_name} n_items=0 status=kit-refused"); return 3
        raise
    if a.mode not in KIT_MODES:
        line, bad = env_clean_line()
        if line is None:
            log(f"{PREFIX} usage: the stock arm's process is not clean: {bad}"); return 2
        log(line)
    log(f"{PREFIX} READY model={a.model_name} constructed; scoring {len(recs)} sequences")
    scores = model.score_sequences([s for _, s in recs], batch_size=a.batch_size)
    n_bad = 0
    with open(os.path.join(a.out_dir, "scores.jsonl"), "w", encoding="utf-8") as out:
        for i, ((rid, seq), sc) in enumerate(zip(recs, scores)):
            x = float(sc)
            if not math.isfinite(x):
                n_bad += 1
            out.write(json.dumps({"id": rid, "index": i, "length": len(seq), "score": repr(x), "score_hex": struct.pack("<f", x).hex()}) + "\n")
    status = "ok" if n_bad == 0 else f"not-finite({n_bad}/{len(recs)})"
    with open(os.path.join(a.out_dir, "run.json"), "w", encoding="utf-8") as out:          # the run's record beside scores.jsonl: what was scored, how many, the status word of the EXIT line
        json.dump({"model": a.model_name, "mode": a.mode, "batch_size": a.batch_size, "n_sequences": len(recs), "n_scored": len(recs) - n_bad, "status": status}, out)
    log(f"{PREFIX} EXIT mode={a.mode} model={a.model_name} n_items={len(recs)} status={status}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
