"""python stock/check_pins.py [--weights DIR] — compare the stock/src tree digest with PINS.json (exit 1 on mismatch) and, optionally, print the weights' sha256 comparison as information (never affects the exit code: weights are not pinned or gated).
The digest covers source files only: build metadata that `pip install -e stock/src` writes into the tree (`*.egg-info/`, `build/`, `dist/`, `.eggs/`) and bytecode (`__pycache__/`, `*.pyc`) are excluded, so an editable install on a networked machine does not change it; any edit to a tracked file does."""
import hashlib, json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
EXCLUDED_DIRS = {"__pycache__", "build", "dist", ".eggs"}          # + any directory named *.egg-info
def _skip_dir(name):
    return name in EXCLUDED_DIRS or name.endswith(".egg-info")
def tree_digest(root):
    h = hashlib.sha256()
    for dp, dn, fn in sorted(os.walk(root)):                       # same walk order as the pinned digest
        rel_dir = os.path.relpath(dp, root)
        if rel_dir != "." and any(_skip_dir(part) for part in rel_dir.split(os.sep)):
            continue
        for f in sorted(fn):
            if f.endswith(".pyc"): continue
            p = os.path.join(dp, f); rel = os.path.relpath(p, root)
            h.update(rel.encode()); h.update(hashlib.sha256(open(p, "rb").read()).digest())
    return h.hexdigest()
def main(argv):
    pins = json.load(open(os.path.join(here, "PINS.json")))
    d = tree_digest(os.path.join(here, "src")); want = pins["upstream"]["stock_src_tree_sha256"]
    ok = d == want
    print(f"stock/src tree sha256 {'OK' if ok else 'MISMATCH'} {d[:16]} (pinned {want[:16]}; build metadata and bytecode excluded)")
    if "--weights" in argv:
        root = argv[argv.index("--weights") + 1]
        for w in pins["weights"]:
            if not w["file"].endswith(".pth"): continue
            p = os.path.join(root, w["repo"].split("/")[1], w["file"])
            if not os.path.isfile(p): print("WEIGHTS missing (information):", p); continue
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
            good = h.hexdigest() == w["sha256"]
            print(f"WEIGHTS {'same as PINS' if good else 'differs from PINS (information)'}: {p} {h.hexdigest()[:16]}")
    return 0 if ok else 1
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
