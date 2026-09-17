"""run.sh install: its usage refusals (rc 2); the weights step stages with the hub downloader and checks digests against the pins —
a file off its digest fails by name (rc 1), a matching snapshot passes (rc 0), a model without pinned digests is staged but reported UNCHECKED (rc 1)."""
import json
import os
import subprocess

from gpnstar_opt import registry, weights
from gpnstar_opt.tests._paths import RUN_SH, env_clean, tree_or_skip


def _sh(*args, **env):
    return subprocess.run(["bash", RUN_SH, *args], capture_output=True, text=True, env=env_clean(**env), timeout=120)


def test_install_usage_refusals():
    tree_or_skip()
    for args in (["install", "--bogus"], ["install", "--model", "ce11-25m"], ["install", "--weights"], ["check", "--mode", "exact"], ["vep", "x"], []):
        p = _sh(*args)
        assert p.returncode == 2, (args, p.stdout, p.stderr)


def _fake_download(files):
    def download(repo, revision=None, allow_patterns=None, cache_dir=None):
        snap = os.path.join(cache_dir, f"models--{repo.replace('/', '--')}", "snapshots", revision)
        for rel, data in files.items():
            p = os.path.join(snap, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
        return snap
    return download


def test_weights_step_checks_every_pinned_file(tmp_path, monkeypatch, capsys):
    files = {"config.json": b"{}", "model.safetensors": b"tensor-bytes", "phylo_dist/pairwise.npy": b"dist"}
    import hashlib
    pins = {"weights": {"songlab/gpn-star-hg38-v100-200m": {"snapshot_commit": registry.models()["v100-200m"]["revision"],
            "files": {rel: {"sha256": hashlib.sha256(d).hexdigest(), "size_bytes": len(d)} for rel, d in files.items()}}}}
    monkeypatch.setattr(registry, "load_pins", lambda path=None: pins)
    rc = weights.main([str(tmp_path / "w")], download=_fake_download(files))
    err = capsys.readouterr().err
    assert rc == 0, err
    assert "[gpnstar-opt install] OK model=v100-200m files=3" in err and os.environ["HF_HOME"] == str(tmp_path / "w")
    ref = tmp_path / "w" / "hub" / "models--songlab--gpn-star-hg38-v100-200m" / "refs" / "main"
    assert ref.read_text() == registry.models()["v100-200m"]["revision"]                 # the repository id resolves offline under HF_HOME
    rc = weights.main([str(tmp_path / "w2")], download=_fake_download(dict(files, **{"model.safetensors": b"other-bytes"})))
    err = capsys.readouterr().err
    assert rc == 1 and "FAIL model=v100-200m" in err and "model.safetensors: sha256" in err
    assert os.path.isfile(tmp_path / "w2" / "hub" / "models--songlab--gpn-star-hg38-v100-200m" / "snapshots" / registry.models()["v100-200m"]["revision"] / "model.safetensors")   # left in place


def test_a_model_without_pinned_digests_is_staged_but_unchecked(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(registry, "load_pins", lambda path=None: {"weights": {}})
    rc = weights.main([str(tmp_path / "w"), "--model", "songlab/gpn-star-ce11-n135-25m"], download=_fake_download({"config.json": b"{}"}))
    err = capsys.readouterr().err
    assert rc == 1 and "DIGEST model=ce11-25m file=config.json sha256=" in err and "UNCHECKED" in err


def test_weights_refuses_an_unpinned_model(tmp_path, capsys):
    assert weights.main([str(tmp_path / "w"), "--model", "someone/else"], download=_fake_download({})) == 2
