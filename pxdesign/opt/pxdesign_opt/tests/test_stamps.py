"""The kit's line grammar: the KERNELS example matches its pattern and tokenizes as pure key=value after the TAG; the probes name Protenix's two
LayerNorm classes verbatim; the batch facts are the record both routes file; report.py re-exports the pattern; every line writer starts on a fresh line."""
import re
import types

import pytest

from pxdesign_opt import stamps


def _pure_kv(line, tag):
    toks = stamps.tokens(line, tag)
    assert toks is not None, (tag, line)
    assert toks, (tag, line)                                                   # at least one token after the tag
    for t in toks:
        assert len(t) == 2 and t[0] and t[1] not in (None, ""), (tag, t, line)  # key=value, never a bare word or an empty value
    assert "(" not in line and ")" not in line                                # no parenthesised free text anywhere
    return dict(toks)


@pytest.mark.parametrize("tag", stamps.TAGS)
def test_examples_match_their_pattern_and_tokenize(tag):
    line = stamps.EXAMPLES[tag]
    assert line.startswith("[pxdesign-opt] ") or line.startswith("[pxdesign-opt stock] ")
    m = stamps.RES[tag].match(line)
    assert m, (tag, line)
    kvs = _pure_kv(line, tag)
    assert m.group("tag") in ("pxdesign-opt", "pxdesign-opt stock")
    for k, v in m.groupdict().items():                                        # every named group is one of the printed tokens (or the composite item)
        if k in ("tag", "cls", "name", "s", "item", "task", "seed"):
            continue
        assert kvs.get(k) == v, (tag, k, v, kvs)


def test_batch_facts_and_kv():
    assert stamps.batch_facts(20, "20", 1, 2) == {"n_sample": 20, "chunk": 20, "batch": 20, "tasks": 1, "seeds": 2, "expected_designs": 40}   # the record both routes file (an argv word or a configs value alike)
    assert stamps.batch_facts(5, None, 3, 2) == {"n_sample": 5, "chunk": None, "batch": 5, "tasks": 3, "seeds": 2, "expected_designs": 30}
    assert stamps.batch_facts(5, 10, 1, 1)["batch"] == 5 and stamps.batch_facts(5, "none", 1, 1)["chunk"] is None                          # a null chunk: one batch of N_sample
    assert stamps.kv(a=None, b="", c=True, d=[1, 2], e="x y") == "a=none b=none c=True d=1,2 e=x_y"        # never an empty value, never a blank inside one
    assert stamps.TAGS == ("KERNELS",) and set(stamps.RES) == set(stamps.EXAMPLES) == {"KERNELS"}


def test_layernorm_classes_verbatim(tree):
    """The two class names are Protenix's own (the vendored source), and the environment rule is primitives.py:49-51's."""
    import os
    prim = open(os.path.join(tree, "stock", "src", "Protenix", "protenix", "openfold_local", "model", "primitives.py")).read()
    ln = open(os.path.join(tree, "stock", "src", "Protenix", "protenix", "model", "layer_norm", "layer_norm.py")).read()
    assert "class OpenFoldLayerNorm(nn.Module):" in prim and stamps.PLAIN_CLASS == "protenix.openfold_local.model.primitives.OpenFoldLayerNorm"
    assert "class FusedLayerNorm(torch.nn.Module):" in ln and stamps.FUSED_CLASS == "protenix.model.layer_norm.layer_norm.FusedLayerNorm"
    assert 'fastln_is_installed = os.getenv("LAYERNORM_TYPE", None) == "fast_layernorm"' in prim
    assert stamps.layernorm_of_env({"LAYERNORM_TYPE": "fast_layernorm"}) == ("fused", stamps.FUSED_CLASS, "fast_layernorm")
    assert stamps.layernorm_of_env({}) == ("plain", stamps.PLAIN_CLASS, None)
    assert stamps.layernorm_of_env({"LAYERNORM_TYPE": "torch"}) == ("plain", stamps.PLAIN_CLASS, "torch")


def test_layernorm_of_model_reads_the_built_class():
    """The class in use is the first LEAF module named *LayerNorm*: Protenix's AdaptiveLayerNorm (a container holding the real one) is skipped."""
    class _M:                                                                  # a stand-in for nn.Module: children() and a pre-order modules() walk
        def __init__(self, *kids): self._kids = kids
        def children(self): return iter(self._kids)
        def modules(self):
            yield self
            for k in self._kids:
                yield from k.modules()
    Fused = type("FusedLayerNorm", (_M,), {"__module__": "protenix.model.layer_norm.layer_norm"})
    Plain = type("OpenFoldLayerNorm", (_M,), {"__module__": "protenix.openfold_local.model.primitives"})
    Adaptive = type("AdaptiveLayerNorm", (_M,), {"__module__": "protenix.model.modules.primitives"})
    Linear, Root = type("Linear", (_M,), {"__module__": "torch.nn"}), type("ProtenixDesign", (_M,), {"__module__": "pxdesign.model.pxdesign"})
    assert stamps.layernorm_of_model(Root(Linear(), Adaptive(Fused(), Linear()), Plain())) == ("fused", stamps.FUSED_CLASS)
    assert stamps.layernorm_of_model(Root(Adaptive(Plain(), Linear()), Fused())) == ("plain", stamps.PLAIN_CLASS)
    assert stamps.layernorm_of_model(Root(Linear(), Plain())) == ("plain", stamps.PLAIN_CLASS)
    assert stamps.layernorm_of_model(Root(Linear())) == ("plain", "none")
    assert stamps.layernorm_of_model(Root(Adaptive())) == ("plain", "protenix.model.modules.primitives.AdaptiveLayerNorm")   # no leaf found: the container is named rather than nothing


def test_kernels_line_grammar_without_torch(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)                            # torch not importable here: versions from metadata, flags n/a, source=metadata
    facts = stamps.kernels_facts(layernorm_kind="plain", ds4sci=False, source="caller_process", environ={})
    assert facts["source"] == "metadata" and facts["tf32_matmul"] == "n/a" and facts["alloc_conf"] == "unset" and facts["compile"] == "off"
    assert stamps.KERNELS_RE.match("[pxdesign-opt stock] " + stamps.kernels(facts)), stamps.kernels(facts)
    fake = types.SimpleNamespace(__version__="2.3.1+cu121", version=types.SimpleNamespace(cuda="12.1"),
                                 backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)), cudnn=types.SimpleNamespace(allow_tf32=True)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    facts = stamps.kernels_facts(layernorm_kind="fused", ds4sci=True, source="process", environ={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    line = "[pxdesign-opt] " + stamps.kernels(facts)
    m = stamps.KERNELS_RE.match(line)
    assert m and (m.group("torch"), m.group("cuda"), m.group("tf32_matmul"), m.group("tf32_cudnn"), m.group("alloc_conf"), m.group("ds4sci_evo_attention"), m.group("source")) == \
        ("2.3.1+cu121", "12.1", "False", "True", "expandable_segments:True", "on", "process")


def test_option_value_and_truthy():
    assert stamps.option_value(["--sample_diffusion_chunk_size", "5"], "sample_diffusion_chunk_size") == "5"
    assert stamps.option_value(["--infer_setting.sample_diffusion_chunk_size", "7", "--x", "1"], "sample_diffusion_chunk_size") == "7"
    assert stamps.option_value(["--N_sample", "5"], "sample_diffusion_chunk_size") is None
    assert stamps.option_value(["--use_deepspeed_evo_attention", "true"], "use_deepspeed_evo_attention") == "true"
    assert stamps.truthy("true") and stamps.truthy("True") and not stamps.truthy("false") and not stamps.truthy(False) and stamps.truthy(1)


def test_report_reexports_the_patterns():
    from pxdesign_opt import report
    assert report.KERNELS_RE is stamps.KERNELS_RE
    assert report.STAMP_EXAMPLES is stamps.EXAMPLES
    done = report.done_line({"mode": "exact", "n_designs": 10, "n_tasks": 1, "n_seeds": 2, "s_per_design": 4.1, "s_per_task": 20.5, "load_s": 60.0, "wall_s": 101.0, "out_dir": "/o", "scope": "job", "expected": 10})
    assert done.endswith(" out_dir=/o scope=job expected=10") and report.DONE_RE.match(done).group("expected") == "10"          # scope, then expected as the LAST field; the head is unchanged
    assert report.DONE_RE.match(done).group("scope") == "job"
    assert report.done_line({"mode": "off"}).endswith(" expected=n/a")


def test_stock_process_modules_are_core_free():
    """stamps.py imports nothing of the core, torch or upstream at module level (the stock caller imports it before its proof)."""
    import ast, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in ("stamps.py",):
        treeast = ast.parse(open(os.path.join(here, f)).read())
        top = [n for n in treeast.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = set()
        for n in top:
            names |= {a.name.split(".")[0] for a in n.names} if isinstance(n, ast.Import) else {(n.module or "").split(".")[0] or "."}
        assert not names & {"opt_core", "torch", "protenix", "pxdesign"}, (f, names)


BAR = "Sampling: 100%|" + "\u2588" * 10 + "| 400/400 [00:10<00:00, 38.21it/s]\r"      # a tqdm-style progress fragment: "\r"-terminated, no newline


def test_fresh_line_rule_after_a_progress_fragment():
    """A kit line emitted right after a "\\r"-terminated fragment on the same stream starts at column 0 (stamps.emit writes a newline first),
    so a reader that splits on "\\n" finds it; a reader that splits on "\\r" and "\\n" finds it with or without the rule."""
    import io
    line, rx = stamps.EXAMPLES["KERNELS"], stamps.KERNELS_RE
    glued = BAR + line + "\n"                                                   # what a bare write would leave on the stream
    assert not any(rx.match(p) for p in glued.split("\n"))                     # the failure mode: the line-anchored reader loses the glued line
    assert any(rx.match(p) for p in re.split(r"[\r\n]", glued))                # a reader splitting on both separators still finds it
    buf = io.StringIO(); buf.write(BAR)
    assert stamps.emit(line, stream=buf) == line
    text = buf.getvalue()
    assert text == BAR + "\n" + line + "\n"
    assert line in text.split("\n") and any(rx.match(p) for p in text.split("\n"))          # column 0 for the "\n" reader
    assert any(rx.match(p) for p in re.split(r"[\r\n]", text))


def test_every_kit_line_writer_uses_the_fresh_line_rule(capsys):
    """report.log / log_done / log_activation and the stock caller's log write through stamps.emit: their line starts at column 0 after a fragment."""
    import sys
    from pxdesign_opt import report, stock_infer
    facts = stamps.kernels_facts(layernorm_kind="plain", ds4sci=False, source="caller_process", environ={})
    sys.stderr.write(BAR); report.log(stamps.kernels(facts))
    sys.stderr.write(BAR); report.log_done({"mode": "exact", "n_designs": 10, "n_tasks": 1, "n_seeds": 2, "expected": 10})
    sys.stderr.write(BAR); report.log_activation({"active": False, "mode": "off", "reason": "stock"})
    sys.stderr.write(BAR); stock_infer.log(stamps.kernels(facts))
    err = capsys.readouterr().err.split("\n")
    assert sum(1 for l in err if stamps.KERNELS_RE.match(l)) == 2 and any(l.startswith(stock_infer.PREFIX + " KERNELS ") for l in err)
    assert any(report.DONE_RE.match(l) for l in err)
    assert any(l.startswith(report.PREFIX + " ") and l != report.PREFIX and "KERNELS" not in l and "DONE" not in l for l in err)   # the activation-class line


def test_kernels_tf32_readback_follows_the_cublas_override():
    """KERNELS tf32_matmul is the live readback torch.backends.cuda.matmul.allow_tf32 in the printing process: torch folds
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE into it at the first query (CPU torch included) — True with the override, False without."""
    import os, subprocess, sys
    pytest.importorskip("torch")
    code = ("import importlib.util as u; s = u.spec_from_file_location('stamps', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
            "k = m.kernels_facts(layernorm_kind='fused', ds4sci=False, source='caller_process'); print(k['tf32_matmul'], k['source'])") % stamps.__file__   # the module by path: core-free, no package import
    OVERRIDE = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"
    env = {k: v for k, v in os.environ.items() if k != OVERRIDE}
    run = lambda e: subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=120)   # noqa: E731
    plain, guard = run(env), run(dict(env, **{OVERRIDE: "1"}))
    assert plain.returncode == 0 and guard.returncode == 0, (plain.stderr[-2000:], guard.stderr[-2000:])
    assert plain.stdout.split() == ["False", "caller_process"], plain.stdout
    assert guard.stdout.split() == ["True", "caller_process"], guard.stdout
