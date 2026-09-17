"""The kit's command surface is stock's plus the mode: `pred` takes a plain Boltz YAML in stock's own schema (no kit field, no role) with
`boltz predict`'s own options by name, `--mode`, `--n_gpu`, `--det`, `--allow-partial`, `--seeds`; there is no `--tag`, `--timeout`,
`warm --seed` or `--version`."""
import pytest

from .. import cli, report as rep, worker
from . import _stubs


def test_pred_takes_a_plain_stock_yaml_and_refuses_the_retired_kit_toggles(tmp_path, monkeypatch, capsys):
    y = _stubs.write_yamls(str(tmp_path), ("plain",))[0]
    text = open(y).read()
    assert "sequences:" in text and "role" not in text and "items" not in text, "a plain Boltz YAML in stock's schema"
    calls = []
    monkeypatch.setattr(worker, "run", lambda mode, yamls, out_dir, seeds, **kw: calls.append((mode, yamls, seeds, sorted(kw))) or 0)
    monkeypatch.setenv("BOLTZ2_OPT", "exact")
    assert cli.main(["pred", "--mode", "exact", "--input", y, "--out_dir", str(tmp_path / "o"), "--seed", "7"]) == 0
    assert calls == [("exact", [y], [7], ["allow_partial", "n_gpu", "settings"])], calls
    for gone in (["--tag", "x"], ["--timeout", "60"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["pred", "--mode", "exact", "--input", y, "--out_dir", str(tmp_path / "o")] + gone)
        assert e.value.code == 2 and "unrecognized arguments" in capsys.readouterr().err
    for gone in (["--timeout", "60"], ["--seed", "3"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["warm", "--mode", "exact", "--out", str(tmp_path / "w")] + gone)
        assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 2
