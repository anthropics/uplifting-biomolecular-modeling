"""Every JSON table shipped in this package parses, carries no duplicate object keys (json keeps the last silently), and no file under the
package carries merge-conflict markers."""
import json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_reporting_duplicates(path):
    dups = []

    def hook(pairs):
        seen = {}
        for k, v in pairs:
            if k in seen:
                dups.append(k)
            seen[k] = v
        return seen

    with open(path) as fh:
        json.load(fh, object_pairs_hook=hook)
    return dups


def test_every_json_table_parses_and_has_no_duplicate_keys():
    bad = {}
    for p in list((ROOT / "opt_core").rglob("*.json")) + list((ROOT / "tests" / "fixtures").glob("*.json")):
        try:
            d = _load_reporting_duplicates(p)
        except Exception as exc:  # a table that does not parse is a release blocker
            bad[str(p.relative_to(ROOT))] = f"does not parse: {exc}"
            continue
        if d:
            bad[str(p.relative_to(ROOT))] = f"duplicate keys: {sorted(set(d))[:5]}"
    assert not bad, bad


def test_no_merge_conflict_markers_in_the_package():
    pat = re.compile(r"^(<{7} |>{7} |={7}$)", re.M)
    hits = []
    for p in list((ROOT / "opt_core").rglob("*")) + list((ROOT / "tests").rglob("*")):
        if p.is_file() and p.suffix in {".py", ".json", ".md", ".toml", ".txt", ".cfg"}:
            try:
                s = p.read_text(errors="ignore")
            except Exception:
                continue
            if pat.search(s):
                hits.append(str(p.relative_to(ROOT)))
    assert not hits, hits
