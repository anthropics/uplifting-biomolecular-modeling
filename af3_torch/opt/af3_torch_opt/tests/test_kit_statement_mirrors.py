"""Two package levers whose correctness is that they RESTATE the kit's own statements (compared against the kit's source live,
at test time, never a stored hash): canonical_noise's two sampler methods mirror AlphaFold3's statement for statement except the two
shape-dependent draws (made at the input's own token count, laid into the padded length); template_dedupe's distinct-slot forward is
bitwise the stock per-slot loop's (same addends, same order), and restates TemplateEmbedding.forward's own lines."""
import ast
import dataclasses
import inspect
import os
import textwrap

import pytest

from af3_torch_opt import canonical_noise, stack, template_dedupe

CANONICAL_NOISE_KIT_FILE = "af3t/af3_torch/xfold/alphafold3.py"
TEMPLATE_DEDUPE_KIT_FILE = "af3t/af3_torch/xfold/nn/template.py"


# ---- canonical_noise ----------------------------------------------------------------------------------------------------------

def _func_src(path, qual):
    src = open(path, encoding="utf-8").read(); tree = ast.parse(src); lines = src.splitlines(keepends=True)
    node = tree
    for part in qual.split("."):
        node = next(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    return "".join(lines[node.lineno - 1: node.end_lineno])


def _stmts(fn_src):
    """the function body's statements as normalised source lines (docstring dropped)."""
    node = ast.parse(textwrap.dedent(fn_src)).body[0]
    body = node.body[1:] if isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) else node.body
    return [ast.unparse(s) for s in body]


def test_statements_are_the_stock_statements_but_the_draws():
    path = os.path.join(stack.forward_dir(), *CANONICAL_NOISE_KIT_FILE.split("/"))
    for qual, ours, n_draws in (("AlphaFold3._apply_denoising_step", canonical_noise._apply_denoising_step, 1), ("AlphaFold3._sample_diffusion", canonical_noise._sample_diffusion, 1)):
        stock = _stmts(_func_src(path, qual)); mine = _stmts(inspect.getsource(ours))
        mine = [s for s in mine if not s.startswith(("import torch", "from xfold.nn import diffusion_head"))]
        draws_stock = [s for s in stock if "torch.randn(" in s]; draws_mine = [s for s in mine if "torch.randn(" in s]
        assert len(draws_stock) == len(draws_mine) == n_draws, (qual, draws_stock, draws_mine)
        rest_stock = [s for s in stock if "torch.randn(" not in s]
        rest_mine = [s for s in mine if "torch.randn(" not in s and "_canonical_len" not in s and "COUNTS[" not in s and not s.startswith(("C = ", "n = ", "if C > n"))]
        assert rest_mine == rest_stock, (qual, [s for s in rest_mine if s not in rest_stock], [s for s in rest_stock if s not in rest_mine])
        for d in draws_mine:                                                      # the canonical draw: the input's token count leads the token axis, laid into the model's n rows (_rows)
            assert "C" in d and "_rows(" in d and ", n," in d, d


def test_canonical_draws_are_stocks_rows():
    """Stock runs at the input's own token count C; the kit pads to n >= C. The draw at C laid into n rows = stock's values on rows 0..C-1,
    zeros (no draw) on the padding rows, and the generator advanced exactly as stock's; at n == C the stock statement's values bit for bit."""
    torch = pytest.importorskip("torch")
    S, C, n = 2, 5, 8
    torch.manual_seed(3); stock = torch.randn((S, C, 24, 3)); step_stock = torch.randn((C, 24, 3)); after_stock = torch.randn(7)
    torch.manual_seed(3); ours = canonical_noise._rows(torch.randn((S, C, 24, 3)), n, 1); step_ours = canonical_noise._rows(torch.randn((C, 24, 3)), n, 0); after_ours = torch.randn(7)
    assert ours.shape == (S, n, 24, 3) and step_ours.shape == (n, 24, 3)
    assert torch.equal(ours[:, :C], stock) and torch.equal(step_ours[:C], step_stock) and torch.equal(after_ours, after_stock)   # stock's rows for the real tokens; the generator advanced identically
    assert not ours[:, C:].any() and not step_ours[C:].any()                                                                   # padding rows: zeros, no draw
    torch.manual_seed(3); at_c = canonical_noise._rows(torch.randn((S, C, 24, 3)), C, 1)
    torch.manual_seed(3); assert torch.equal(at_c, torch.randn((S, C, 24, 3)))                                                  # n == C: the stock statement's values bit for bit


def test_canonical_noise_install_refuses_a_changed_model():
    class M:
        pass
    with pytest.raises(RuntimeError, match="_sample_diffusion"):
        canonical_noise.install(M())


# ---- template_dedupe ----------------------------------------------------------------------------------------------------------

def test_restated_statements_are_the_stock_statements():
    src = open(os.path.join(stack.forward_dir(), TEMPLATE_DEDUPE_KIT_FILE), encoding="utf-8").read()
    assert "for template_idx in range(num_templates):" in src and "summed_template_embeddings += template_embedding" in src
    ours = inspect.getsource(template_dedupe.forward)
    for stmt in ("num_templates = templates.aatype.shape[0]", "summed_template_embeddings += template_embedding",
                 "embedding = summed_template_embeddings / (1e-7 + num_templates)", "embedding = torch.relu(embedding)", "embedding = self.output_linear(embedding)"):
        assert stmt in src and stmt in ours, stmt                                # the restated statements are the stock statements


def test_distinct_slots_bitwise_and_counted():
    torch = pytest.importorskip("torch")

    @dataclasses.dataclass
    class Templates:                                                            # the fields of xfold.features.Templates
        aatype: "torch.Tensor"
        atom_positions: "torch.Tensor"
        atom_mask: "torch.Tensor"

        def __getitem__(self, i):
            return Templates(self.aatype[i], self.atom_positions[i], self.atom_mask[i])

    class Single(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.lin = torch.nn.Linear(3, 8, bias=False)

        def forward(s, q, t, pm, mm):
            return torch.tanh(q[..., :8] * 0.1 + s.lin(t.atom_positions.mean(1))[None, :, :] * pm[..., None] + t.aatype.float()[:, None, None] * 1e-3)

    class Embedder(torch.nn.Module):                                            # the shape template_dedupe.install expects: single_template_embedding, output_linear, num_channels
        def __init__(s):
            super().__init__(); s.num_channels = 8; s.single_template_embedding = Single(); s.output_linear = torch.nn.Linear(8, 16, bias=False)

        def forward(s, query_embedding, templates, padding_mask_2d, multichain_mask_2d):    # the stock loop (xfold TemplateEmbedding.forward)
            num_templates = templates.aatype.shape[0]
            num_res, _, _ = query_embedding.shape
            summed = query_embedding.new_zeros(num_res, num_res, s.num_channels)
            for template_idx in range(num_templates):
                summed += s.single_template_embedding(query_embedding, templates[template_idx], padding_mask_2d, multichain_mask_2d)
            embedding = summed / (1e-7 + num_templates)
            return s.output_linear(torch.relu(embedding))

    class Model:                                                                # model.evoformer.template_embedding
        pass

    torch.manual_seed(0)
    N = 12
    emb = Embedder(); model = Model(); model.evoformer = Model(); model.evoformer.template_embedding = emb
    q = torch.randn(N, N, 16); pm = torch.ones(N, N); mm = torch.ones(N, N)
    real = lambda: (torch.randint(0, 20, (N,)), torch.randn(N, 24, 3), torch.ones(N, 24, dtype=torch.bool))
    dummy = (torch.zeros(N, dtype=torch.long), torch.zeros(N, 24, 3), torch.zeros(N, 24, dtype=torch.bool))
    r1, r2 = real(), real()
    cases = {"4 dummies": [dummy] * 4, "real+3 dummies": [r1, dummy, dummy, dummy], "all distinct": [r1, r2, real(), real()], "a,b,a,b": [r1, r2, r1, r2]}
    expect_evaluated = {"4 dummies": 1, "real+3 dummies": 2, "all distinct": 4, "a,b,a,b": 2}
    for name, slots in cases.items():
        T = Templates(torch.stack([s_[0] for s_ in slots]), torch.stack([s_[1] for s_ in slots]), torch.stack([s_[2] for s_ in slots]))
        stock = Embedder.forward(emb, q, T, pm, mm)
        template_dedupe.install(model)
        ours = emb(q, T, pm, mm)
        assert torch.equal(stock, ours), name                                   # bitwise: the same addends in the same order
        again = emb(q, T, pm, mm)                                               # a later call on the same template tensors (the trunk's next recycle): the item's map reused, no search
        assert torch.equal(stock, again), name
        c = template_dedupe.take()
        assert c == {"calls": 2, "slots": 8, "evaluated": 2 * expect_evaluated[name], "scans": 1}, (name, c)   # scans: the distinct-slot search ran once for the two calls
        assert template_dedupe.take() == {"calls": 0, "slots": 0, "evaluated": 0, "scans": 0}
        assert torch.equal(stock, emb(q, T, pm, mm)) and template_dedupe.take()["scans"] == 1, name   # take() forgot the map (a new item): searched again
        T2 = Templates(T.aatype.clone(), T.atom_positions.clone(), T.atom_mask.clone())              # other tensors with the same values (another item's batch): searched, never served the old map
        emb(q, T, pm, mm); emb(q, T2, pm, mm); emb(q, T2, pm, mm)
        assert template_dedupe.take()["scans"] == 2, name
    assert template_dedupe.install(model) == {"installed": True, "already": True, "tables": [], "device": None}
    assert template_dedupe.TABLES == ("RESTYPE_RIGIDGROUP_DENSE_ATOM_IDX", "RESTYPE_PSEUDOBETA_INDEX") and template_dedupe.place_tables(model) == {"tables": [], "device": None}   # a stand-in without CUDA parameters places nothing


def test_template_dedupe_install_refuses_a_changed_embedder():
    pytest.importorskip("torch")

    class M:
        pass
    m = M(); m.evoformer = M(); m.evoformer.template_embedding = M()
    with pytest.raises(RuntimeError, match="single_template_embedding"):
        template_dedupe.install(m)
