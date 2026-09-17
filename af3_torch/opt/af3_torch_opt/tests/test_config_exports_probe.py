"""configs/<card>.env capture the exports probe's stderr (the interpreter's own words, shown verbatim on a refusal) in a fresh private file from
mktemp — never a predictable name under the shared temporary directory, where a symbolic link planted by another account would be followed on
write — and remove it on both paths."""
import os
import stat
import subprocess

import pytest

from .conftest import HOME

STUB = """#!/bin/sh
# python -m af3_torch_opt exports: the words and rc the test names; anything else passes
if [ "$1 $2 $3" = "-m af3_torch_opt exports" ]; then [ -z "$STUB_WORDS" ] || echo "$STUB_WORDS" >&2; exit "${STUB_RC:-0}"; fi
exit 0
"""


def _source(tmp_path, card, script_prefix="", env=None):
    b = tmp_path / "bin"; b.mkdir(exist_ok=True)
    py = b / "python"; py.write_text(STUB); py.chmod(py.stat().st_mode | stat.S_IXUSR)
    t = tmp_path / "tmp"; t.mkdir(exist_ok=True)
    e = {"PATH": f"{b}:/usr/bin:/bin", "HOME": str(tmp_path), "TMPDIR": str(t), "MODEL_OPT_JIT_ROOT": "/a/jit"}; e.update(env or {})
    p = subprocess.run(["bash", "-c", f'{script_prefix}source "{HOME}/configs/{card}.env"; echo "rc=$?"'], env=e, capture_output=True, text=True, timeout=60)
    return p.stdout, p.stderr, sorted(os.listdir(t))


@pytest.mark.parametrize("card", ["h100", "a100", "h200"])
def test_probe_words_verbatim_and_nothing_left_behind(tmp_path, card):
    out, err, left = _source(tmp_path, card)
    assert out.strip() == "rc=0" and err == "" and left == []                                       # the pass: no file left under the temporary directory
    out, err, left = _source(tmp_path, card, env={"STUB_RC": "7", "STUB_WORDS": "af3_torch_opt: core_mismatch (the stub's words)"})
    assert out.strip() == "rc=7" and "af3_torch_opt: core_mismatch (the stub's words)" in err and f"configs/{card}.env: refused (rc 7)" in err and left == []
    cfg = open(os.path.join(HOME, "configs", f"{card}.env"), encoding="utf-8").read()
    assert "af3_torch_exports.$$" not in cfg and 'mktemp "${TMPDIR:-/tmp}/af3_torch_exports.' in cfg   # no pid-named path; a mktemp file


@pytest.mark.parametrize("card", ["h100", "a100", "h200"])
def test_a_link_planted_at_the_pid_name_is_never_written_through(tmp_path, card):
    """Another account that predicts the shell's pid plants <tmp>/af3_torch_exports.<pid> as a symbolic link to a file of its choosing; the
    config must not write through it (the old pid-named redirection did)."""
    target = tmp_path / "victim.txt"
    plant = 'ln -s "$VICTIM" "$TMPDIR/af3_torch_exports.$$"; '
    out, err, left = _source(tmp_path, card, plant, env={"VICTIM": str(target), "STUB_RC": "7", "STUB_WORDS": "words another account wanted written"})
    assert out.strip() == "rc=7" and "words another account wanted written" in err
    assert not target.exists()                                                                        # nothing went through the planted link …
    assert len(left) == 1 and os.path.islink(tmp_path / "tmp" / left[0])                              # … which is left alone (not the config's to remove); the private file is gone
