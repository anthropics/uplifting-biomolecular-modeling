"""H200 exact-tier vouch rows (add-only) -- kernels.trimul / kernels.triattn / kernels.ln / kernels.apb cell tables.

Three guards:
  1. test_tables_additions_only: each touched table at the parent of the commit that last changed it vs the working tree: every pre-existing
     path -> value is identical; lists only grew by APPENDED entries, and (for the H200 vouch landing) every appended entry is an H200 stack word.
     Needs git history (skips in an exported tree).
  2. test_h200_rows_are_parity_subset: every row of tests/fixtures/h200_exact_vouch_rows.json is present in its table, is spelled as an H200 stack
     word no H100 / A100 process constructs, and its H100 spelling of the SAME torch / triton / cuequivariance stack is vouched in the same cell for
     the same arm (the H200 claim is never wider than the H100 claim it mirrors); no vouched_on list holds a duplicate.
  3. test_h100_a100_selection_golden: select() of the four families enumerated over every cell x {fast, big, exact} x every recorded NON-H200
     stack word at the bucket size -> a pinned digest.  A data change that moves any H100 / A100 decision changes the digest (re-pin deliberately,
     with the reason in the commit message); the H200 rows do not.
"""
import hashlib
import json
import os
import re
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.dirname(HERE)
K = os.path.join(CORE, "opt_core", "kernels")
TABLES = {"trimul": os.path.join(K, "trimul", "TRIMUL_CELLS.json"), "triattn": os.path.join(K, "triattn", "TRIATTN_CELLS.json"),
          "ln": os.path.join(K, "ln", "LN_CELLS.json"), "apb": os.path.join(K, "apb", "APB_CELLS.json")}
FIXTURE = os.path.join(HERE, "fixtures", "h200_exact_vouch_rows.json")
LANDING_SUBJECT = "H200 exact vouch rows"

# select() over every cell x tier x recorded non-H200 stack word (bucket size), sha256 of the decision stream; re-pin only with a stated reason.
GOLDEN_H100_A100 = "4d1a9197fa7398af577776a1a78b6f490b3b7d9e6171a7fcb75be02f66f71252"   # 59577 decisions: kernels.triattn row triattn_exact vouched on 21 stack keys (7 stacks x H100 / H200 / A100 80GB) and named the exact winner of the 34 cc-9.0 and 32 cc-8.0 cells bf16|D32|H2..H12|N<=...4096|fwd -> +1000 decisions for new stack strings; of the 58577 earlier decisions 58417 are byte-identical and exactly 160 move, all in those cells under the words exact / exact_headsplit on already-recorded stacks now vouched (8.0|torch2.10.0+cu128|cueq0.10.0, 8.0|torch2.12.0+cu130|cueq0.10.0, 8.0|torch2.13.0+cu130|cueq0.11.1, 8.0|torch2.7.1+cu126|cueq0.10.0, 8.0|torch2.7.1+cu128|cueq0.10.0): the earlier winner (cueq or exact_headsplit) -> triattn_exact, or on an unvouched cc-8.0 stack exact_headsplit -> cueq by name = the flip taking effect. Earlier note: 57785 decisions: select() of the four families over every cell x {fast, big, exact} x every recorded non-H200 stack; the exact tier of triangle attention answers the stock op by name or exact_headsplit


def _git(*args):
    try:
        r = subprocess.run(["git"] + list(args), cwd=CORE, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _walk(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from _walk(v, path + (k,))
    elif isinstance(o, list):
        yield path, ("list", o)
    else:
        yield path, ("leaf", o)


def _get(o, path):
    for k in path:
        o = o[k]
    return o


@pytest.mark.parametrize("fam", sorted(TABLES))
def test_tables_additions_only(fam):
    path = TABLES[fam]
    rel = os.path.relpath(path, CORE)
    if _git("rev-parse", "--is-inside-work-tree") is None:
        pytest.skip("no git work tree: additions-only is checked against history")
    last = (_git("log", "-1", "--format=%H%x00%s", "--", rel) or "").strip()
    if not last:
        pytest.skip("no history for %s" % rel)
    sha, subject = last.split("\x00", 1)
    old_raw = _git("show", "%s~1:./%s" % (sha, rel))
    if old_raw is None:
        pytest.skip("parent of %s does not carry %s" % (sha[:12], rel))
    old = json.loads(old_raw)
    new = json.load(open(path, encoding="utf-8"))
    if LANDING_SUBJECT not in subject:
        pytest.skip("%s: last change %s is not an H200 vouch landing (%r)" % (rel, sha[:12], subject[:60]))
    h200_landing = True
    bad = []
    for p, (kind, val) in _walk(old):
        try:
            nv = _get(new, p)
        except (KeyError, TypeError, IndexError):
            bad.append(("missing", p)); continue
        if kind == "leaf":
            if nv != val:
                bad.append(("changed", p))
        else:
            if not isinstance(nv, list):
                bad.append(("type", p)); continue
            if nv == val:
                continue
            if nv[:len(val)] != val:
                bad.append(("list-reordered-or-edited", p)); continue
            if h200_landing and not all(isinstance(x, str) and "H200" in x for x in nv[len(val):]):
                bad.append(("appended-non-H200", p))
        if len(bad) > 20:
            break
    assert not bad, "%s: last change %s (%r) is not additions-only: %s" % (rel, sha[:12], subject, bad[:20])


def _h100_spelling(fam, word):
    if fam == "trimul":
        assert word.startswith("NVIDIA_H200:"), word
        return "H100:" + word[len("NVIDIA_H200:"):]
    if fam == "triattn":
        assert word.endswith("|H200") and word.count("|") == 3, word
        return word[:-len("|H200")]
    assert word.startswith("H200:torch"), word
    return "H100:" + word[len("H200:"):]


def test_h200_rows_are_parity_subset():
    rows = json.load(open(FIXTURE, encoding="utf-8"))["rows"]
    assert rows, "empty fixture"
    tabs = {f: json.load(open(p, encoding="utf-8")) for f, p in TABLES.items()}
    for r in rows:
        fam, key, arm, word = r["family"], r["cell"], r["arm"], r["stack"]
        cell = tabs[fam]["cells"][key]
        lst = (cell.get("vouched_on") or {}).get(arm)
        assert isinstance(lst, list) and word in lst, (fam, key, arm, word)
        assert lst.count(word) == 1, ("duplicate", fam, key, arm, word)
        h100 = _h100_spelling(fam, word)
        if fam == "trimul":
            cls = str(((cell.get("class") or {}).get(arm) or {}).get(h100, ""))
            assert h100 in lst or cls.startswith("bitwise"), ("H200 row wider than the H100 claim", fam, key, arm, word)
        else:
            assert h100 in lst, ("H200 row wider than the H100 claim", fam, key, arm, word)
    for fam, t in tabs.items():
        for key, cell in t["cells"].items():
            for arm, lst in (cell.get("vouched_on") or {}).items():
                if isinstance(lst, list):
                    assert len(lst) == len(set(lst)), ("duplicate vouch entries", fam, key, arm)


def _bucket(key):
    m = re.search(r"\|N<=(\d+)", key)
    return int(m.group(1)) if m else None


def golden_digest():
    os.environ.setdefault("MODEL_OPT_CENSUS", "0")
    from opt_core.kernels import trimul as KT, triattn as KA, ln as KL, apb as KP
    h = hashlib.sha256()
    n = 0

    def rec(*parts):
        nonlocal n
        n += 1
        h.update(json.dumps(parts, sort_keys=True, default=str).encode())

    def res(fn):
        try:
            s = fn()
        except Exception as e:  # noqa: BLE001  a Refusal by name (or an argument error) is the decision
            return ("R", type(e).__name__, str(getattr(e, "row", "")), str(getattr(e, "fallback", "")), str(getattr(e, "kind", ""))[:80])
        return ("S", str(getattr(s, "row", "")), str(getattr(s, "variant", "") or getattr(s, "config_word", "") or ""))

    def stacks(fam, t):
        st = set()
        for c in t["cells"].values():
            for fld in ("fast_per_stack", "exact_per_stack", "big_per_stack", "stock_row", "stacks"):
                v = c.get(fld)
                if isinstance(v, dict):
                    st |= set(v.keys())
            for lst in (c.get("vouched_on") or {}).values():
                if isinstance(lst, list):
                    st |= set(lst)
        return sorted(s for s in st if isinstance(s, str) and "H200" not in s and "A100_40" not in s and "A100-40GB" not in s)

    t = KT.table()
    sts = stacks("trimul", t)
    for key in sorted(t["cells"]):
        p = key.split("|")
        if len(p) < 7 or "+" in key:
            continue
        cc, prec, C, H, N, dirw, pas = p[0], p[1], int(p[2][1:]), int(p[3][1:]), _bucket(key), p[5], p[6]
        sdt = "bf16" if prec in ("bf16", "f32z_bf16") else "fp32"
        kw = dict(residency=("fp32" if prec == "f32z_bf16" else None), backward=(pas == "fwdbwd"), tf32=(prec == "tf32"))
        for st in sts:
            if (cc == "9.0") != st.startswith("H100:"):
                continue
            for w in ("fast", "big", "exact"):
                rec("trimul", key, w, st, res(lambda: KT.select(cc, sdt, C, H, N, "outgoing" if dirw != "in" else "incoming", word=w, stack=st, **kw)))
    t = KA.table()
    eks = set()
    for c in t["cells"].values():
        for lst in (c.get("vouched_on") or {}).values():
            eks |= set(x for x in lst if isinstance(x, str) and x.count("|") == 2)
    for R in t["rows"].values():
        eks |= set(x for x in (R.get("vouched_on") or []) if isinstance(x, str) and x.count("|") == 2)
    for key in sorted(t["cells"]):
        p = key.split("|")
        cc, dtype, D, H, N, dirw = p[0], p[1], int(p[2][1:]), int(p[3][1:]), _bucket(key), p[5]
        cct = tuple(int(x) for x in cc.split("."))
        for ek in sorted(e for e in eks if e.startswith(cc + "|")):
            lib = ek.split("|")[2]
            for w in ("exact", "exact_headsplit"):
                rec("triattn", key, w, ek, res(lambda: KA.select(cct, dtype, D, H, N, dirw, word=w, exact_stack=ek, lib=lib)))
    t = KL.table()
    sts = stacks("ln", t)
    for key in sorted(t["cells"]):
        p = key.split("|")
        cc, dtw, cellw, N, timing, pas = p[0], p[1], p[2], _bucket(key), p[4], p[5]
        dt, widen, outw = {"fp32": ("fp32", False, None), "bf16": ("bf16", False, None), "bf16w": ("bf16", True, None), "bf16o": ("bf16", True, "bf16")}.get(dtw, (dtw, False, None))
        m = re.search(r"_c(\d+)", cellw)
        C = int(m.group(1)) if m else None
        cct = tuple(int(x) for x in cc.split("."))
        for st in sts:
            if (cc == "9.0") != st.startswith("H100:"):
                continue
            for w in ("fast", "big", "exact"):
                rec("ln", key, w, st, res(lambda: KL.select(cct, dt, cellw, N, word=w, widen=widen, out=outw, timing=timing, pass_=pas, stack=st, C=C)))
    t = KP.table()
    sts = stacks("apb", t)
    for key in sorted(t["cells"]):
        p = key.split("|")
        cc, dtype, cellw, S, N, timing = p[0], p[1], p[2], int(p[3][1:]), _bucket(key), p[5]
        Hh = Dd = cz = None
        m = re.match(r"(dit|pf|msarow|atom)_h(\d+)d(\d+)", cellw)
        if m:
            Hh, Dd = int(m.group(2)), int(m.group(3))
        m = re.match(r"bias_c(\d+)h(\d+)", cellw)
        if m:
            Hh, cz = int(m.group(2)), int(m.group(1))
        cct = tuple(int(x) for x in cc.split("."))
        for st in sts:
            if (cc == "9.0") != st.startswith("H100:"):
                continue
            for w in ("fast", "big", "exact"):
                rec("apb", key, w, st, res(lambda: KP.select(cct, dtype, cellw, N, word=w, samples=S, timing=timing, stack=st, head_dim=Dd, heads=Hh, c_z=cz)))
    return h.hexdigest(), n


def test_h100_a100_selection_golden():
    digest, n = golden_digest()
    assert n > 50000, n
    assert digest == GOLDEN_H100_A100, "H100/A100 selection stream moved (%d decisions): %s -- re-pin GOLDEN_H100_A100 only with the reason stated" % (n, digest)


if __name__ == "__main__":
    d, n = golden_digest()
    print(d, n)
