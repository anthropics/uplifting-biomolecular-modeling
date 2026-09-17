"""The kit's install recipes never pipe a downloaded installer straight into a shell: the one installer they fetch (uv's) comes from
a version-pinned URL, is written to a file, is checked against its SHA-256 with `sha256sum -c`, and runs only when the check passes
(README.md / STOCK.md route C, and environment/Dockerfile where the image recipe fetches it)."""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
RECIPES = [p for p in (os.path.join(KIT, "README.md"), os.path.join(KIT, "STOCK.md"),
                       os.path.join(KIT, "environment", "Dockerfile")) if os.path.isfile(p)]

_FETCH = re.compile(r"\b(curl|wget)\b[^\n]*?(astral\.sh/|/install\.sh\b|micro\.mamba\.pm)")
_PINNED = re.compile(r"https://astral\.sh/uv/\d+\.\d+\.\d+/install\.sh")
_PIPED = re.compile(r"\|\s*(sh|bash)\b")
_CHECK = re.compile(r'echo "[0-9a-f]{64}  [^"]+" \| sha256sum -c -')


def _logical_lines(path):
    """(first line number, text) of every logical line: backslash-newline continuations joined."""
    out, buf, start = [], [], None
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if start is None:
                start = n
            if line.endswith("\\"):
                buf.append(line[:-1])
                continue
            buf.append(line)
            out.append((start, " ".join(buf)))
            buf, start = [], None
    if buf:
        out.append((start, " ".join(buf)))
    return out


def _installer_lines():
    found = []
    for path in RECIPES:
        for n, text in _logical_lines(path):
            if _FETCH.search(text):
                found.append((os.path.relpath(path, KIT), n, text))
    return found


def test_recipes_fetch_an_installer_at_all():
    # the recipes this test guards do fetch uv's installer; if that ever changes, this test should change with it
    assert _installer_lines(), RECIPES


def test_no_installer_is_piped_into_a_shell():
    for rel, n, text in _installer_lines():
        assert not _PIPED.search(text), "%s:%d pipes a downloaded installer into a shell" % (rel, n)


def test_installer_url_is_version_pinned():
    for rel, n, text in _installer_lines():
        assert _PINNED.search(text), "%s:%d fetches an installer from an unpinned URL" % (rel, n)
        assert "astral.sh/uv/install.sh" not in text, "%s:%d" % (rel, n)


def test_installer_is_digest_checked_before_it_runs():
    for rel, n, text in _installer_lines():
        m = _CHECK.search(text)
        assert m, "%s:%d runs an installer without a sha256sum -c check" % (rel, n)
        run = re.search(r'\bsh ("\$f"|/tmp/uv-install\.sh)', text)
        assert run and run.start() > m.end(), "%s:%d: the check must come before the installer runs" % (rel, n)
