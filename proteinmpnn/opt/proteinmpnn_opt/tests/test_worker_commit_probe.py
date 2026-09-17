"""The exact worker's one child process — the checkout's commit for the ``.fa`` header, read with git (opt/forward/mpnn_exact_worker/addon/
mpnn_worker2.py, read as source: the worker imports torch, the tests do not) — is started from an argument list, never from a command string
handed to a shell. Upstream reads the same commit with a shell command string (protein_mpnn_run.py: ``git --git-dir {dir}/.git rev-parse
HEAD``), and the header is part of what ``exact`` repeats byte for byte, so the worker gives what upstream's line gives: the commit for a
clone path a shell leaves alone, ``unknown`` for one a shell would split or expand (spaces, quotes, ``$``, backticks, ``;`` ...) — with
nothing run at all. The worker's own statement is executed here on such paths, beside upstream's line where that is harmless, and no source
of this kit starts a process through a shell."""
import ast
import os
import shutil
import subprocess
import re
import tempfile
import types
import unittest

from proteinmpnn_opt import modes, stack

WORKER = os.path.join(stack.kit_home(), modes.WORKER_DIR, "addon", "mpnn_worker2.py")
ODD = "my \"odd\" checkout; touch pwned; $(touch pwned2) `touch pwned3` & x"                 # one directory name: spaces, quotes, ;, $(), backticks, &


def _process_calls(tree):
    """(call node, dotted name) of every ``subprocess.*`` / ``os.system`` / ``os.popen`` call of a module."""
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name):
            name = f"{n.func.value.id}.{n.func.attr}"
            if n.func.value.id == "subprocess" or name in ("os.system", "os.popen"):
                out.append((n, name))
    return out


def _through_a_shell(call, name):
    """The call hands a command string to a shell: os.system / os.popen / subprocess.getoutput / getstatusoutput, or any ``shell=`` other than a literal False."""
    if name in ("os.system", "os.popen", "subprocess.getoutput", "subprocess.getstatusoutput"):
        return True
    return any(k.arg == "shell" and not (isinstance(k.value, ast.Constant) and k.value.value is False) for k in call.keywords)


def _probe():
    """The worker's own statement — ``try: self.commit = <git probe> except Exception: self.commit = "unknown"`` — as a function of (MPNN_DIR) -> commit."""
    src = open(WORKER, encoding="utf-8").read()
    tree = ast.parse(src)
    stmt, = [n for n in ast.walk(tree) if isinstance(n, ast.Try) and any(
        isinstance(b, ast.Assign) and ast.get_source_segment(src, b.targets[0]) == "self.commit" for b in n.body)]
    code = compile(ast.Module(body=[stmt], type_ignores=[]), WORKER, "exec")

    def run(mpnn_dir):
        ns = {"subprocess": subprocess, "re": re, "MPNN_DIR": mpnn_dir, "self": types.SimpleNamespace()}
        exec(code, ns)
        return ns["self"].commit
    return run, stmt, src


def _checkout(repo):
    """A git checkout with one commit at ``repo`` (made with argument lists)."""
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "init.defaultBranch=main", "-c", "commit.gpgsign=false"]
    os.makedirs(repo)
    subprocess.run(git + ["init", "-q", repo], check=True, capture_output=True)
    subprocess.run(git + ["-C", repo, "commit", "-q", "--allow-empty", "-m", "x"], check=True, capture_output=True)
    return repo


class TestCommitProbe(unittest.TestCase):
    def test_the_worker_probes_git_with_an_argument_list(self):
        _, stmt, src = _probe()
        calls = _process_calls(stmt)
        self.assertEqual([name for _, name in calls], ["subprocess.check_output"])
        call = calls[0][0]
        self.assertIsInstance(call.args[0], ast.List, ast.get_source_segment(src, call))          # argv, not a string
        self.assertEqual([a.value for a in call.args[0].elts if isinstance(a, ast.Constant)], ["git", "--git-dir", "rev-parse", "HEAD"])
        self.assertEqual(len(call.args[0].elts), 5)                                               # ... and the one path argument between them
        self.assertNotIn("shell", [k.arg for k in call.keywords])
        self.assertEqual([c for c in _process_calls(ast.parse(src)) if _through_a_shell(*c)], [])  # the whole worker: no process through a shell

    @unittest.skipUnless(shutil.which("git"), "git not on PATH")
    def test_a_clone_path_holding_shell_syntax_is_unknown_and_nothing_in_it_runs(self):
        run, _, _ = _probe()
        with tempfile.TemporaryDirectory() as td:
            repo = _checkout(os.path.join(td, ODD))
            cwd = os.getcwd()
            os.chdir(td)                                                                          # anything the path's shell syntax started would write here
            try:
                got = run(repo)
            finally:
                os.chdir(cwd)
            self.assertEqual(got, "unknown")
            self.assertEqual(sorted(os.listdir(td)), [ODD])                                       # no pwned* beside the checkout: nothing in the path ran
            self.assertEqual(sorted(os.listdir(repo)), [".git"])

    @unittest.skipUnless(shutil.which("git") and os.path.exists("/bin/sh"), "git or /bin/sh absent")
    def test_the_probe_gives_what_upstreams_shell_line_gives(self):
        """Harmless names only (nothing in them is a command): a plain path, glob and brace characters a shell passes through, and the
        spaces / quote a shell splits on. Upstream's statement is run as it stands in protein_mpnn_run.py, through /bin/sh."""
        run, _, _ = _probe()
        with tempfile.TemporaryDirectory() as td:
            for name, want_commit in (("plain", True), ("run[1]", True), ("a#b", True), ("{x}", True), ("~u", True), ("my clone", False), ("it's", False)):
                with self.subTest(name=name):
                    repo = _checkout(os.path.join(td, name))
                    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
                    p = subprocess.run(["/bin/sh", "-c", f"git --git-dir {repo}/.git rev-parse HEAD"], capture_output=True, text=True)   # upstream's line
                    upstream = p.stdout.strip() if p.returncode == 0 else "unknown"
                    self.assertEqual(upstream, head if want_commit else "unknown")
                    self.assertEqual(run(repo), upstream)
                    self.assertEqual(len(head), 40)

    @unittest.skipUnless(shutil.which("git"), "git not on PATH")
    def test_a_directory_that_is_no_checkout_is_unknown_as_before(self):
        run, _, _ = _probe()
        with tempfile.TemporaryDirectory() as td:
            plain = os.path.join(td, "plain")
            os.makedirs(plain)
            self.assertEqual(run(plain), "unknown")
            self.assertEqual(run(os.path.join(td, "absent")), "unknown")
            self.assertEqual(sorted(os.listdir(td)), ["plain"])

    def test_no_source_of_this_kit_starts_a_process_through_a_shell(self):
        hits = []
        for root, dirs, files in os.walk(os.path.join(stack.tree_home(), "opt")):                 # the package and the carried executables (opt/forward)
            dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
            for f in sorted(files):
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    for c, name in _process_calls(ast.parse(open(path, encoding="utf-8").read())):
                        if _through_a_shell(c, name):
                            hits.append(f"{path}:{c.lineno} {name}")
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
