"""The levers' end-of-run report ($PTX_LEVER_REPORT: JSON lines appended at exit and read back) defaults to a file under the user's own
cache directory (env.sh) whose directory the package makes private before use (stack.private_report_dir) — never a fixed name in the
shared /tmp; a directory another account could have planted is refused by name and nothing is appended or read there."""
import os
import re
import stat

import pytest

from protenix_opt import stack

FPF = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "forward", "flashpairformer")


def test_env_sh_names_no_fixed_tmp_file_and_needs_no_command_for_the_default():
    text = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
    line = [l for l in text.splitlines() if l.startswith("export PTX_LEVER_REPORT=")]
    assert len(line) == 1 and ":-/tmp/" not in line[0] and "$(" not in line[0].split("#")[0] and "`" not in line[0].split("#")[0]
    assert line[0].split("#")[0].strip() == "export PTX_LEVER_REPORT=${PTX_LEVER_REPORT:-${XDG_CACHE_HOME:-$HOME/.cache}/protenix_v2_opt/fpf_lever_report.jsonl}"
    ref = open(os.path.join(FPF, "src", "fpf", "reference.py"), encoding="utf-8").read()
    assert not re.search(r"""["']/tmp/""", ref)


def test_the_report_directory_is_made_private(tmp_path):
    rep = tmp_path / "cache" / "protenix_v2_opt" / "fpf_lever_report.jsonl"
    env = {"PTX_LEVER_REPORT": str(rep)}
    old = os.umask(0o002)
    try:
        got = stack.private_report_dir(env)
    finally:
        os.umask(old)
    assert got == str(rep.parent) and env["PTX_LEVER_REPORT"] == str(rep)
    st = os.lstat(rep.parent)
    assert stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700
    assert stack.private_report_dir(env) == str(rep.parent)              # an existing private one: used again, silently


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o702])
def test_a_directory_open_to_group_or_others_is_refused_by_name(tmp_path, capsys, mode):
    d = tmp_path / "planted"; d.mkdir(); os.chmod(d, mode)
    env = {"PTX_LEVER_REPORT": str(d / "fpf_lever_report.jsonl")}
    assert stack.private_report_dir(env) is None and "PTX_LEVER_REPORT" not in env
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and f"lever report: REFUSED {d}: it is writable by group or other" in err and "PTX_LEVER_REPORT" in err
    assert stat.S_IMODE(os.stat(d).st_mode) == mode and list(d.iterdir()) == []


def test_a_symbolic_link_is_refused_by_name(tmp_path, capsys):
    e = tmp_path / "e"; e.mkdir(mode=0o700); d = tmp_path / "link"; os.symlink(e, d)
    env = {"PTX_LEVER_REPORT": str(d / "r.jsonl")}
    assert stack.private_report_dir(env) is None and "symbolic link" in capsys.readouterr().err and list(e.iterdir()) == []


def test_no_report_named_is_left_alone(tmp_path):
    env = {}
    assert stack.private_report_dir(env) is None and env == {}
