#!/usr/bin/env python
"""Build the pipeline_tf kit's entry script from the STOCK bytes: calico/borzoi @5c93582 ``borzoi_sad.py`` (sha256 asserted) + the
kit's insertions, each anchored on an exact stock line that occurs once — the stock lines themselves are never edited, only added to.
``<frozen>/borzoi_sad.py`` is exactly ``render()``; ``tests/test_build.py`` holds the shipped entry to it byte for byte, so the entry and
this file change together or not at all.

    python build.py                      # say whether <frozen>/borzoi_sad.py equals render() (exit 1 if not)
    python build.py --write [--force]    # write <frozen>/borzoi_sad.py (refuses to overwrite a file that differs unless --force)
    python build.py --stock-dir <borzoi checkout>/src/scripts ...   # read the stock script from a checkout instead of stock/'s archive
"""
import argparse
import hashlib
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))                 # borzoi/
FROZEN = "v17"
ENTRY = "borzoi_sad.py"
STOCK_COMMIT = "5c9358222b5026abb733ed5fb84f3f6c77239b37"
STOCK_ARCHIVE = os.path.join(KIT_ROOT, "stock", "borzoi-5c93582.tar.gz")
STOCK_MEMBER = "borzoi-5c93582/src/scripts/" + ENTRY
STOCK_SHA256 = "10c7016704d9972a225311ddda5b872ee585c76e1112f067a232a40ad5957911"

# ---- the insertions, in file order: (where, anchor = an exact stock line, block). "after": the block follows the anchor line;
# ---- "before": the block precedes it. Every block is bracketed `# ---- kit … ----` … `# ---- end kit … ----` in the entry.
PROBE_START = '''# ---- kit: the deliverable-cores probe STARTS in a subprocess before the TensorFlow import and is joined at the first post (kitlib/sad_post.py) ----
import os as _kit_os0, sys as _kit_sys0
_kit_sys0.path.insert(0, _kit_os0.path.dirname(_kit_os0.path.abspath(__file__)))
from kitlib import sad_post as _kit_post0
_KIT_PROBE_H = _kit_post0.start_probe_async()
# ---- end kit probe ----
'''

KIT_IMPORT = '''# ---- pipeline_tf kit, frozen dir v17 (opt/datapath/pipeline_tf): every line above and below that is not marked
# ---- "kit" is calico/borzoi @5c93582 verbatim. Inserted by build.py at the anchors named there.
import atexit as _kit_atexit, json as _kit_json, os as _kit_os, sys as _kit_sys, time as _kit_time
_kit_sys.path.insert(0, _kit_os.path.dirname(_kit_os.path.abspath(__file__)))
from kitlib import onehot as _kit_onehot, sad_post as _kit_post
_KIT_STAMP = {"kit": "pipeline_tf.v17", "script": _kit_os.path.basename(__file__), "onehot": _kit_onehot.install(), "t0": _kit_time.time()}
def _kit_dump():
    _KIT_STAMP["wall_main_s"] = _kit_time.time() - _KIT_STAMP["t0"]
    if _KIT_STAMP.get("post_obj") is not None:
        _KIT_STAMP["post"] = _KIT_STAMP.pop("post_obj").stamp()
    print("KIT_STAMP " + _kit_json.dumps(_KIT_STAMP, sort_keys=True, default=str), flush=True)
    d = _kit_os.environ.get("KIT_STAMP_DIR")
    if d:
        _kit_json.dump(_KIT_STAMP, open(_kit_os.path.join(d, "kit_stamp.json"), "w"), indent=1, sort_keys=True, default=str)
_kit_atexit.register(_kit_dump)
# ---- end kit import block ----
'''

FORWARD_LEVER = '''    # ---- kit: the forward call's host side (kitlib/forward.py): KIT_FWD=1 = the stock's eager call returned without its second
    # ---- float32 copy (astype only on a dtype change); default off = the stock call ----
    from kitlib import forward as _kit_forward
    seqnn_model, _KIT_STAMP["forward"] = _kit_forward.install(seqnn_model)
    # ---- end kit ----
'''

POST_SETUP = '''    # ---- kit: the pipelined chunked host post (supported option set only; anything else = the stock path below, untouched) ----
    _KIT_POST = None
    if _kit_post.applies(options, sum_strand, sum_length):
        _kit_untr = None if options.no_untransform else (dataset.untransform_preds1 if options.untransform_old else dataset.untransform_preds)
        _kit_pool = _kit_post.pool_size(handle=_KIT_PROBE_H)
        _KIT_POST = _kit_post.PipelinedChunkedPost(targets_df, strand_transform, options.sad_stats, _kit_untr, write_snp_len, _kit_pool["threads"], seq_len=targets_length)
        _KIT_STAMP["post_obj"] = _KIT_POST; _KIT_STAMP["pool"] = _kit_pool
    _KIT_STAMP["post_applies"] = _KIT_POST is not None
    # ---- end kit ----
'''

POST_SUBMIT = '''        # ---- kit: queue this variant's post on the pool; the loop goes on to the next variant's one-hot + forward ----
        if _KIT_POST is not None and _KIT_POST.submit(ref_preds, alt_preds, sad_out, si):
            continue
        # ---- end kit (the stock post follows) ----
'''

POST_FLUSH = '''    # ---- kit: write the pending rows (main thread) before the percentile pass reads every row ----
    if _KIT_POST is not None:
        _KIT_POST.flush()
    # ---- end kit ----
'''

INSERTIONS = (
    ('after', 'from __future__ import print_function\n', PROBE_START),
    ('after', 'from baskerville import vcf as bvcf\n', KIT_IMPORT),
    ('after', '    seqnn_model.build_ensemble(options.rc, options.shifts)\n', FORWARD_LEVER),
    ('after', '    genome_open = pysam.Fastafile(options.genome_fasta)\n', POST_SETUP),
    ('before', '        # untransform predictions\n', POST_SUBMIT),
    ('before', '    write_pct(sad_out, options.sad_stats)\n', POST_FLUSH),
)


def stock_source(stock_dir=None) -> str:
    """The stock script's text: from a checkout's scripts dir when given, else from the archive in stock/; sha256 asserted either way."""
    if stock_dir:
        with open(os.path.join(stock_dir, ENTRY), "rb") as f:
            raw = f.read()
    else:
        with tarfile.open(STOCK_ARCHIVE) as tf:
            raw = tf.extractfile(STOCK_MEMBER).read()
    got = hashlib.sha256(raw).hexdigest()
    if got != STOCK_SHA256:
        raise SystemExit(f"stock {ENTRY} sha256 {got} != {STOCK_SHA256} (calico/borzoi @{STOCK_COMMIT})")
    return raw.decode()


def transform(src: str) -> str:
    """The stock text with every insertion applied; an anchor that does not occur exactly once is refused by name."""
    for where, anchor, block in INSERTIONS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(f"anchor {anchor.strip()!r} occurs {n} times in the stock script (expected exactly 1)")
        src = src.replace(anchor, anchor + block if where == "after" else block + anchor, 1)
    return src


def render(stock_dir=None) -> str:
    return transform(stock_source(stock_dir))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="build or check the kit entry <frozen>/borzoi_sad.py = the stock script + the kit's insertions")
    ap.add_argument("--stock-dir", default=None, help="a calico/borzoi checkout's src/scripts (default: the archive in stock/)")
    ap.add_argument("--frozen", default=FROZEN)
    ap.add_argument("--write", action="store_true", help="write <frozen>/borzoi_sad.py")
    ap.add_argument("--force", action="store_true", help="with --write: overwrite an entry that differs")
    a = ap.parse_args(argv)
    out = render(a.stock_dir)
    path = os.path.join(HERE, a.frozen, ENTRY)
    have = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            have = f.read()
    if not a.write:
        same = have == out
        print(f"{path}: {'equals' if same else 'DIFFERS from'} the stock script + the kit's insertions ({len(INSERTIONS)} anchors)")
        return 0 if same else 1
    if have is not None and have != out and not a.force:
        raise SystemExit(f"{path} exists and differs: frozen means frozen (pass --force to overwrite)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(out)
    os.chmod(path, 0o755)
    print(f"BUILT {path} (from stock sha256 {STOCK_SHA256[:12]}…, {len(INSERTIONS)} insertions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
