"""The pin check of the gpnstar tree (stdlib only; CPU; runnable alone):

    python stock/check_pins.py [--weights DIR [--variant v100-200m|ce11-25m|m447-200m|p243-200m|tair10-25m|all]] [--gpu] [--quiet]

(1) gpn is installed at the version AND from the commit stock/PINS.json pins (the dist-info direct_url.json of a
    `pip install "gpn[inference] @ git+...@<commit>"` records the commit), and its installed modules are byte-identical
    to the sources carried in stock/gpn-6f28c81b.tar.gz (src/gpn/*) — anything else is another stock: REFUSED, exit 3;
    transformers off upstream's own hard pin (pyproject: transformers==5.15.0) is also another stock: REFUSED;
(2) the library stack (torch / triton / huggingface_hub / safetensors / accelerate / numpy / zarr / cuDNN) against
    `pins`: a distribution absent or off its pinned version is REFUSED here, at install time (the pinned stack underwrites the
    kit's bitwise statement); at run time the kit names such drift on its ACTIVE line and proceeds;
(3) --weights DIR: every pinned checkpoint file under DIR against weights.<repo>.files (sha256 and size). DIR may be
    an HF_HOME (DIR/hub/models--songlab--<name>/snapshots/<commit>/), a hub cache directory (DIR/models--...), or one
    snapshot directory itself. The files the model loader reads (weights_layout.required_files) must be present and
    match: REFUSED otherwise; the other files of the snapshot are checked when present and named when absent;
(4) --gpu: the visible device against gpus (named listed / unlisted).
Exit status: 0 all pinned; 3 REFUSED (another stock, or checkpoint bytes off the pin); 2 usage error.
"""
import argparse
import fnmatch
import hashlib
import json
import os
import sys
import tarfile
from importlib import metadata as md

HERE = os.path.dirname(os.path.abspath(__file__))
PINNED_STACK = ("torch", "triton", "huggingface_hub", "safetensors", "accelerate", "numpy", "zarr", "nvidia_cudnn_cu13")   # refused when absent or off the pin
REFUSED_IF_OFF = ("transformers",)                                                                                    # upstream's own hard pin: another stock when off


def sha256_file(path, chunk=16 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dist_version(name):
    for n in (name, name.replace("_", "-"), name.replace("-", "_")):
        try:
            return md.version(n)
        except md.PackageNotFoundError:
            continue
    return None


def archive_sources(archive_path, package_dir="src/gpn"):
    """{path relative to site-packages (gpn/...): sha256} of every file under <prefix>/src/gpn in the carried archive."""
    out = {}
    with tarfile.open(archive_path, "r:gz") as t:
        for m in t.getmembers():
            if not m.isfile():
                continue
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(package_dir + "/"):
                continue
            rel = "gpn/" + parts[1][len(package_dir) + 1:]
            out[rel] = hashlib.sha256(t.extractfile(m).read()).hexdigest()
    return out


def check_gpn(pins):
    """(1): version, commit and byte equality of the installed package against the pin and the carried archive."""
    up = pins["upstream"]["gpn"]
    try:
        dist = md.distribution("gpn")
    except md.PackageNotFoundError:
        return [f"gpn: not installed (the stock is gpn {up['version']} from {up['repo']} @ {up['commit']})"], None
    bad = []
    if dist.version != up["version"]:
        bad.append(f"gpn {dist.version} is installed: the stock is {up['version']} @ {up['commit'][:8]}")
    commit = None
    try:
        raw = dist.read_text("direct_url.json")
        if raw:
            commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
    except (OSError, ValueError):
        commit = None
    if commit is None:
        commit_note = f"installed from the stock archive (files byte-identical), commit {up['commit'][:8]} per PINS"   # printed only when the byte comparison below passes
    elif commit != up["commit"]:
        bad.append(f"gpn was installed from commit {commit}: the stock is {up['commit']}")
        commit_note = f"commit {commit[:8]} (off the pin)"
    else:
        commit_note = f"commit {commit[:8]} == the pin"
    archive = os.path.join(HERE, os.path.basename(up["archive"]))
    if not os.path.exists(archive):
        return bad + [f"gpn: the carried archive {archive} is absent from stock/"], None
    got = sha256_file(archive)
    if got != pins["archive"]["sha256"]:
        return bad + [f"gpn: {os.path.basename(archive)} sha256 {got[:16]}... is not archive.sha256 {pins['archive']['sha256'][:16]}..."], None
    n_ok, modified, missing = 0, [], []
    for rel, want in sorted(archive_sources(archive, up.get("package_dir", "src/gpn")).items()):
        p = dist.locate_file(rel)
        if not os.path.exists(p):
            missing.append(rel)
        elif sha256_file(p) == want:
            n_ok += 1
        else:
            modified.append(rel)
    if modified or missing:
        bad.append(f"gpn: installed files differ from the carried archive — modified {modified[:5]}, missing {missing[:5]}")
    return bad, f"gpn {dist.version}: pinned ({commit_note}; {n_ok} installed files byte-identical to {os.path.basename(archive)})"


def check_stack(pins):
    """(1)+(2): transformers against upstream's hard pin, then every pinned distribution of the stack against pins — absent or off its
    version: refused (exit 3); at its version: named as pinned."""
    bad, notes = [], []
    for name in REFUSED_IF_OFF:
        want, got = pins["pins"].get(name), dist_version(name)
        if got is None:
            bad.append(f"{name}: not installed (the stock is {name}=={want})")
        elif str(got).split("+")[0] != str(want):
            bad.append(f"{name} {got} is installed: upstream pins {name}=={want} — another stock")
        else:
            notes.append(f"{name} {got}: pinned")
    for name in PINNED_STACK:
        want, got = pins["pins"].get(name), dist_version(name)
        if want is None:
            continue
        if got is None:
            bad.append(f"{name}: not installed (the pin is {want})")
        elif str(got).split("+")[0] != str(want).split("+")[0]:
            bad.append(f"{name} {got} is installed: the pin is {want} (pins) — install the pinned stack (environment/requirements.lock)")
        else:
            notes.append(f"{name} {got}: pinned")
    return bad, notes


def snapshot_dir(root, repo, commit):
    """The snapshot directory of repo@commit under root (an HF_HOME, a hub cache dir, or the snapshot dir itself), or None."""
    name = "models--" + repo.replace("/", "--")
    for cand in (os.path.join(root, "hub", name, "snapshots", commit), os.path.join(root, name, "snapshots", commit)):
        if os.path.isdir(cand):
            return cand
    if os.path.isfile(os.path.join(root, "config.json")) and os.path.isfile(os.path.join(root, "model.safetensors")):
        return root
    return None


def check_weights(root, keys, pins):
    bad, notes = [], []
    required = pins.get("weights_layout", {}).get("required_files") or ["config.json", "*.safetensors", "phylo_dist/pairwise.npy", "phylo_dist/in_clade.npy"]
    for key in keys:
        var = pins["variants"].get(key)
        if var is None:
            bad.append(f"--variant {key}: no variants.{key} pin (pinned: {sorted(pins['variants'])})")
            continue
        repo, commit = var["hf_repo"], var["snapshot_commit"]
        files = (pins["weights"].get(repo) or {}).get("files")
        if not files:
            notes.append(f"{key}: weights.{repo} carries no digests (optional model) — nothing to check")
            continue
        snap = snapshot_dir(root, repo, commit)
        if snap is None:
            bad.append(f"{key}: no snapshot of {repo} @ {commit[:8]} under {root} (expected hub/models--{repo.replace('/', '--')}/snapshots/{commit}/)")
            continue
        n_ok, absent_optional = 0, []
        for rel, want in sorted(files.items()):
            p = os.path.join(snap, rel)
            is_required = any(fnmatch.fnmatch(rel, pat) for pat in required)
            if not os.path.exists(p):
                if is_required:
                    bad.append(f"{key}: {rel} is absent from {snap} (read at load)")
                else:
                    absent_optional.append(rel)
                continue
            real = os.path.realpath(p)
            size, got = os.path.getsize(real), sha256_file(real)
            if got != want["sha256"] or size != want["size_bytes"]:
                bad.append(f"{key}: {rel} sha256 {got[:16]}... / {size} bytes is not the pin's {want['sha256'][:16]}... / {want['size_bytes']} bytes (weights.{repo})")
            else:
                n_ok += 1
        note = f"{key}: {snap} — {n_ok}/{len(files)} pinned files present and identical (weights.{repo} @ {commit[:8]})"
        if absent_optional:
            note += f"; absent, not read at load: {absent_optional}"
        notes.append(note)
    return bad, notes


def check_gpu(pins):
    try:
        import torch
    except ImportError:
        return ["--gpu: torch is not importable"]
    if not torch.cuda.is_available():
        return ["--gpu: no CUDA device visible"]
    notes = []
    for i in range(torch.cuda.device_count()):
        cc = torch.cuda.get_device_capability(i); sm = f"sm_{cc[0]}{cc[1]}"
        mib = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)
        cls = next((k for k, v in pins.get("gpus", {}).items() if v["sm"] == sm and abs(mib - v["memory_mib"]) <= 0.1 * v["memory_mib"]), None)
        notes.append(f"gpu {i}: {torch.cuda.get_device_name(i)} {sm} {mib} MiB — " + (f"listed ({cls})" if cls else "unlisted (not a class the kit was measured on: engaged and named)"))
    return notes


def main():
    ap = argparse.ArgumentParser(description="Check the installed gpn, the library stack and (optionally) the checkpoint files against stock/PINS.json.")
    ap.add_argument("--weights", default=None, help="HF_HOME, hub cache dir, or a snapshot dir holding the pinned checkpoints")
    ap.add_argument("--variant", default="v100-200m", help="a variants key of PINS.json, a comma list, or 'all' (with --weights)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-package", action="store_true", help="skip (1)/(2): check only --weights / --gpu (e.g. before the stack is installed)")
    a = ap.parse_args()
    pins = json.load(open(os.path.join(HERE, "PINS.json")))
    bad, notes = [], []
    if not a.skip_package:
        b, note = check_gpn(pins)
        bad += b
        if note and not b:
            notes.append(note)
        b, n = check_stack(pins)
        bad += b; notes += n
    if a.weights:
        keys = sorted(pins["variants"]) if a.variant == "all" else [k.strip() for k in a.variant.split(",") if k.strip()]
        b, n = check_weights(a.weights, keys, pins)
        bad += b; notes += n
    if a.gpu:
        notes += check_gpu(pins)
    if not a.quiet:
        for n in notes:
            print(f"check_pins: {n}")
    if bad:
        for b in bad:
            print(f"check_pins: REFUSED {b}", file=sys.stderr)
        sys.exit(3)
    print("check_pins: OK")


if __name__ == "__main__":
    main()
