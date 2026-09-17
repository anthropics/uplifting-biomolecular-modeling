"""``--upstream-fix <ID>``: the one switchable issue file of the tree (ESMIF1-001), the flag's parsing and refusal, the child hand-off tokens,
and the arm process's ``apply()`` + ``UPSTREAM-FIX <ID> applied`` line — exercised here with a stand-in issue file (the real one rebinds an
``esm`` method and needs the model stack)."""
import os
import subprocess
import sys

import pytest

from esm_if1_opt import settings, stack, upstream_fix

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
CORE = os.path.join(os.path.dirname(os.path.dirname(OPT)), "common", "opt_core")


def test_the_tree_carries_one_issue_file_with_the_documented_shape():
    avail = upstream_fix.available(stack.tree_home())
    assert list(avail) == ["ESMIF1-001"] and os.path.basename(avail["ESMIF1-001"]) == "ESMIF1-001_nogpu_device.py"
    src = open(avail["ESMIF1-001"], encoding="utf-8").read()
    for words in ("WHAT UPSTREAM DOES.", "EVIDENCE.", "WHAT THE FIX CHANGES.", "EXPECTED EFFECT ON OUTPUTS.", 'ID = "ESMIF1-001"', "def apply():"):
        assert words in src


def test_flag_parsing_resolution_and_refusal(tmp_path):
    assert upstream_fix.parse(None) == [] and upstream_fix.parse("") == [] and upstream_fix.parse(" ESMIF1-001 , ESMIF1-001") == ["ESMIF1-001"]
    home = stack.tree_home()
    [(fid, path)] = upstream_fix.resolve(["ESMIF1-001"], home)
    assert fid == "ESMIF1-001" and os.path.isfile(path) and upstream_fix.resolve([], home) == []
    with pytest.raises(upstream_fix.UnknownUpstreamFix) as e:
        upstream_fix.resolve(["ESMIF1-999"], home)
    assert "known: ESMIF1-001" in str(e.value)
    assert upstream_fix.child_tokens([]) == [] and upstream_fix.child_tokens([(fid, path)]) == ["--upstream-fix", path]
    assert upstream_fix.split_argv(["--upstream-fix", "/a.py,/b.py", "x.pdb", "--chain", "A"]) == (["/a.py", "/b.py"], ["x.pdb", "--chain", "A"])
    assert upstream_fix.split_argv(["x.pdb", "--upstream-fix", "X"]) == ([], ["x.pdb", "--upstream-fix", "X"])      # only a LEADING pair is the hand-off
    ns = settings.parser().parse_args(["x.pdb"])
    assert ns.upstream_fix is None
    assert settings.parser().parse_args(["x.pdb", "--upstream-fix", "ESMIF1-001"]).upstream_fix == "ESMIF1-001"


def test_apply_files_loads_by_path_and_prints_the_applied_line(tmp_path):
    issue = tmp_path / "TEST-001_standin.py"
    issue.write_text('ID = "TEST-001"\nWORDS = "stand-in"\nCALLS = []\ndef apply():\n    CALLS.append(1)\n    return {"id": ID, "applied": True}\n')
    code = ("import sys, json; sys.path[:0] = [%r]; from esm_if1_opt import upstream_fix as u; "
            "recs = u.apply_files([%r]); print(json.dumps(recs))" % (OPT, str(issue)))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONPATH=os.pathsep.join([OPT, CORE])))
    assert r.returncode == 0, r.stderr
    assert r.stderr.strip() == "[esm_if1-opt] UPSTREAM-FIX TEST-001 applied (stand-in)"
    assert '"id": "TEST-001"' in r.stdout and str(issue) in r.stdout
    # the kit child applies a leading hand-off pair before the driver parses the rest (here the rest is a usage error: exit 2 after the line)
    r = subprocess.run([sys.executable, "-m", "esm_if1_opt.kit_design", "--upstream-fix", str(issue), "--batch_size", "0", "x.pdb"], capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, PYTHONPATH=os.pathsep.join([OPT, CORE])))
    assert r.returncode == 2 and r.stderr.splitlines()[0] == "[esm_if1-opt] UPSTREAM-FIX TEST-001 applied (stand-in)", r.stderr


def test_an_unknown_id_is_a_usage_error_before_anything_launches(tmp_path):
    (tmp_path / "in.pdb").write_text("ATOM\n")
    r = subprocess.run([sys.executable, "-m", "esm_if1_opt", "design", "--mode", "off", str(tmp_path / "in.pdb"), "--outpath", str(tmp_path / "o" / "x.fasta"), "--upstream-fix", "NOPE-1"],
                       capture_output=True, text=True, timeout=60, env=dict(os.environ, PYTHONPATH=os.pathsep.join([OPT, CORE])))
    assert r.returncode == 2 and "unknown upstream fix 'NOPE-1'" in r.stderr and not (tmp_path / "o").exists(), r.stderr
