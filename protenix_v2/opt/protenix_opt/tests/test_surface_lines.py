"""The modes' exported switches, mode table, line formats and the multi-GPU line's constants / rank environment / launcher command /
ACTIVE line on an H100 stack are the recorded ones (tests/data/surface_lines.json, produced by _surface_render.render()): a change to
what exact / fast / big / big --n_gpu 2 export or print shows up here as a named key, never silently."""
import json

from protenix_opt.tests import _surface_render as S


def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flat(v, f"{prefix}/{k}"))
    else:
        out[prefix] = d
    return out


def test_rendered_surface_is_the_recorded_one():
    with open(S.DATA, encoding="utf-8") as fh:
        recorded = json.load(fh)
    now = json.loads(json.dumps(S.render(), sort_keys=True))
    rf, nf = _flat(recorded), _flat(now)
    changed = sorted(k for k in set(rf) | set(nf) if rf.get(k, "<absent>") != nf.get(k, "<absent>"))
    assert not changed, "surface lines differ from tests/data/surface_lines.json at: " + ", ".join(changed[:20]) + "".join(
        f"\n  {k}\n    recorded: {rf.get(k, '<absent>')!r}\n    now:      {nf.get(k, '<absent>')!r}" for k in changed[:6])
