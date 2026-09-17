"""`run.sh install --weights DIR` — fetch the five AlphaFold-Multimer v3 parameter files into DIR/params/ and check each against stock/PINS.json
`weights.files` (sha256 + byte count). DIR is then the params root every route reads (COLABDESIGN_PARAMS_DIR, or --params-dir; STOCK.md).

    python -m colabdesign_opt.weights DIR                     (run.sh install --weights DIR)

Upstream ships no download routine for these files that a program can call: BindCraft's installer fetches DeepMind's parameter archive and
unpacks it (`wget` + `tar` in install_bindcraft.sh), ColabDesign's notebooks do the same, and the AlphaFold-Multimer v3 parameters are five
members of that one archive. This step therefore streams that archive — stock/PINS.json `weights.source`, the URL BindCraft's installer names —
once, front to back, and writes the five pinned members to DIR/params/, each through a temporary file beside its target renamed into place
when complete, so an interrupted fetch leaves no file that could pass for a finished one; nothing else of the archive is kept. Files already
present are kept and only checked; with all five present nothing is fetched. The comparison is stock/check_pins.py `check_weights` — the one
the routes' `weights=<name> sha256=<12> (pinned)` lines come from (cli.check_pins_module, digesting now, no memo): a file that is not the
pinned one is REFUSED here by name and digest (exit 1) and left in place for inspection — the install step vouches for the pinned weights only,
while the design routes accept other weights and word them NOT PINNED; all five pinned prints
`WEIGHTS OK: 5/5 files pinned in DIR — export COLABDESIGN_PARAMS_DIR=DIR`. No GPU is touched and nothing upstream is imported.
"""
import os
import shutil
import sys
import tarfile
from urllib.request import urlopen

from . import cli

TAG = "[colabdesign-opt install]"
PARAMS_SUBDIR = "params"                                                   # the layout every route reads: <root>/params/params_model_{1..5}_multimer_v3.npz (stock/check_pins.py check_weights)


def _say(out, msg):
    out.write(msg + "\n"); out.flush()


def extract_members(source, params_dir, wanted, out=sys.stdout, opener=urlopen):
    """Stream the tar archive at `source` and write every member whose base name is in `wanted` to params_dir/<name> — a `.part` file beside
    the target, renamed into place once the member is complete. The stream is read once and left as soon as every wanted member is written.
    Returns the set of names written."""
    remaining, written = set(wanted), set()
    if not remaining:
        return written
    with opener(source) as resp, tarfile.open(fileobj=resp, mode="r|*") as tar:
        for member in tar:
            name = os.path.basename(member.name)
            if name not in remaining or not member.isfile():
                continue
            dest = os.path.join(params_dir, name); tmp = dest + ".part"
            _say(out, f"{TAG} extracting {name} ({member.size / 2**20:.0f} MB) from the archive")
            src = tar.extractfile(member)
            try:
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(src, fh, length=1 << 20)
                os.replace(tmp, dest)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
            written.add(name); remaining.discard(name)
            if not remaining:
                break
    return written


def fetch(directory, files=None, source=None, out=sys.stdout, opener=urlopen, matcher=None):
    """Fetch what is missing of `files` (name -> {"sha256", "bytes"}; default stock/PINS.json `weights.files`) from the archive `source`
    (default PINS `weights.source`) into <directory>/params/, then check every file with `matcher(directory) -> (bad, detail)` (default:
    stock/check_pins.py check_weights against PINS, digested now). Returns the exit code: 0 = every file present and pinned; 1 = a member the
    archive does not hold, or a file that is not the pinned one (named, left in place)."""
    root = os.path.abspath(os.path.expanduser(directory))
    params = os.path.join(root, PARAMS_SUBDIR)
    os.makedirs(params, exist_ok=True)
    module = pins = None
    if files is None or source is None or matcher is None:
        module, pins = cli.check_pins_module()                          # stock/check_pins.py as a module + stock/PINS.json: the ONE home of the weights pins and their check
    files = dict(pins["weights"]["files"]) if files is None else files
    source = pins["weights"]["source"] if source is None else source
    if matcher is None:
        matcher = lambda d: module.check_weights(pins, d)            # noqa: E731 — digest now (no memo): the install step's word is authoritative
    missing = []
    for name in files:
        if os.path.isfile(os.path.join(params, name)):
            _say(out, f"{TAG} present  {PARAMS_SUBDIR}/{name} (kept; checked below)")
        else:
            missing.append(name)
            _say(out, f"{TAG} missing  {PARAMS_SUBDIR}/{name}")
    if missing:
        _say(out, f"{TAG} fetching {len(missing)} file(s) from {source} — the archive BindCraft's installer unpacks; streamed once, only the pinned members are written")
        written = extract_members(source, params, missing, out=out, opener=opener)
        short = [n for n in missing if n not in written]
        if short:
            for n in short:
                _say(out, f"{TAG} FAILED  {PARAMS_SUBDIR}/{n}: the archive at {source} holds no member of that name")
            _say(out, f"{TAG} FAILED: {len(short)}/{len(files)} pinned files could not be fetched into {root} (stock/PINS.json weights.source names the archive; STOCK.md has the layout to place them by hand)")
            return 1
    bad, detail = matcher(root)
    off = []
    for name in files:
        d = detail.get(name)
        if isinstance(d, dict) and d.get("pinned"):
            continue
        off.append(name)
        if isinstance(d, dict):
            _say(out, f"{TAG} REFUSED {PARAMS_SUBDIR}/{name}: sha256 {d['sha256'][:16]} ({d['bytes']} bytes) is not the pinned {d['pinned_sha256'][:16]} ({d['pinned_bytes']} bytes) — left in place; not the file this tree was tested with")
        else:
            _say(out, f"{TAG} REFUSED {PARAMS_SUBDIR}/{name}: missing after the fetch")
    if off:
        _say(out, f"{TAG} REFUSED: {len(off)}/{len(files)} files in {root} are not the pinned weights (stock/PINS.json weights.files): {', '.join(off)}")
        return 1
    _say(out, f"{TAG} WEIGHTS OK: {len(files)}/{len(files)} files pinned in {root} (sha256 = stock/PINS.json weights) — export {cli.ENV_PARAMS}={root}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        sys.stderr.write(f"usage: python -m colabdesign_opt.weights DIR   (run.sh install --weights DIR): the five AlphaFold-Multimer v3 parameter files into DIR/{PARAMS_SUBDIR}/, checked against stock/PINS.json\n")
        return 2
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
