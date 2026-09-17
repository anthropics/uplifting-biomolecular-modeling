"""Test fixtures: the carried kit tree (required by most tests: the real pins module, the kit entry), a STUB upstream
(a site directory with dist-info metadata at the pinned versions — read from the pins module, never typed here — a stub `torch`, a stub
`flash_attn` package (its two functions, `__version__` at the pin, its `flash_attn_2_cuda` extension module) and a stub `E1` package whose
`tools/score.py` mirrors the real CLI's options and __main__ block and writes a scores.csv, and whose `model/` mirrors upstream's
accelerator dispatch sites — `model/flash_attention.py` (the guarded import + `is_flash_attention_available()` reading USE_FLASH_ATTN per
call), `model/attention.py` (`Attention._flash_attn` / `_flex_attn`, the `flash_attention_func` name), `model/flex_attention.py` (a
torch.compile-shaped callable), `modeling.layer_norm` (the hub kernel module loaded from the stub kernel snapshot: `rms_norm_fn`, the
`Autotuner`-typed `_layer_norm_fwd_1pass_kernel`) — so the KERNELS proof (accel.py) reads real bound objects in the children), stub caches
(the HF snapshot and the kernel snapshot at their pinned paths, with placeholder bytes), and a fake `nvidia-smi` on PATH.
No torch, no GPU, no network."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import textwrap
import unittest

from e1_opt import stack

HERE = os.path.dirname(os.path.abspath(__file__))
OPT_DIR = os.path.dirname(os.path.dirname(HERE))          # e1/opt (the package's install dir, for the children's PYTHONPATH)


def tree_or_skip():
    """The e1/ tree with the carried kit (opt/forward/...); skips the test when absent."""
    if not os.path.isfile(stack.kit_entry()):
        raise unittest.SkipTest(f"carried kit tree not found at {stack.forward_root()} (set MODEL_OPT to the e1/ directory)")
    return stack.tree_home()


def pins_or_skip():
    tree_or_skip()
    return stack.load_pins()


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------------------------------ stub site
STUB_TORCH = '''
__version__ = {torch_version!r}
class _Version:
    cuda = {cuda!r}
version = _Version()
class _Matmul:
    allow_tf32 = True
class _CudaBackend:
    matmul = _Matmul()
class _Cudnn:
    allow_tf32 = True
    benchmark = True
class _Backends:
    cuda = _CudaBackend()
    cudnn = _Cudnn()
backends = _Backends()
_DET = {{"enabled": False}}
_SEEDS = []
class _Cuda:
    @staticmethod
    def is_available(): return False
    @staticmethod
    def get_device_name(i=0): return {gpu_name!r}
    @staticmethod
    def manual_seed_all(s): _SEEDS.append(("cuda", s))
    @staticmethod
    def set_device(i): pass
    @staticmethod
    def current_device(): return 0
cuda = _Cuda()
def manual_seed(s): _SEEDS.append(("cpu", s))
def use_deterministic_algorithms(mode, warn_only=False): _DET["enabled"] = bool(mode)
def are_deterministic_algorithms_enabled(): return _DET["enabled"]
class device:
    def __init__(self, *a): self.args = a
float = "float32"
class Tensor: pass
'''

STUB_E1_TOOLS_SCORE = '''
"""Stub of E1/tools/score.py: the same options and __main__ block shape as the upstream CLI; writes id,context_id,score."""
import argparse
import logging
import sys

from .. import dist
from ..io import read_fasta_sequences
from ..modeling import E1ForMaskedLM
from ..scorer import E1Scorer

CALLS = []


def score(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--mutants-path", required=True)
    ap.add_argument("--parent-path", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--context-path", default=None)
    ap.add_argument("--max-batch-tokens", type=int, default=65536)
    ap.add_argument("--scoring-method", default="masked_marginal")
    ap.add_argument("--context-reduction", default="mean")
    a = ap.parse_args(argv)
    model = E1ForMaskedLM.from_pretrained(a.model_name, dtype="float32").to(dist.get_device())
    model.eval()
    parent = list(read_fasta_sequences(a.parent_path).items())
    assert len(parent) == 1, "Only one parent sequence is supported for E1 models"
    muts = list(read_fasta_sequences(a.mutants_path).items())
    assert len(muts) > 0, "No mutated sequences found in the input file"
    scorer = E1Scorer(model, method=a.scoring_method, max_batch_tokens=a.max_batch_tokens)
    rows = scorer.score(parent_sequence=parent[0][1], sequences=[s for _, s in muts], sequence_ids=[i for i, _ in muts])
    with open(a.output_path, "w") as fh:
        fh.write("id,context_id,score\\n")
        for r in rows:
            fh.write(f"{r['id']},{r['context_id']},{r['score']}\\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dist.setup_dist()
    try:
        score()
    finally:
        dist.destroy_process_group()
'''

STUB_E1_SCORER = '''
import torch
from .modeling import E1ForMaskedLM
from .predictor import E1Predictor


class E1Scorer:
    def __init__(self, model, method, data_prep_config=None, max_batch_tokens=65536):
        self.predictor = E1Predictor(model, max_batch_tokens=max_batch_tokens)
        self.method = method

    def score(self, parent_sequence, sequences, sequence_ids, context_seqs=None, context_reduction="mean"):
        out = []
        for sid, seq in zip(sequence_ids, sequences):
            diff = sum(1 for a, b in zip(parent_sequence, seq) if a != b) + abs(len(parent_sequence) - len(seq))
            out.append({"id": sid, "context_id": "", "score": -float(diff)})
        return out
'''

STUB_E1_PREDICTOR = '''
import torch
from .modeling import E1ForMaskedLM      # upstream's predictor imports the model module (which imports the attention modules)

INSTANCES = []


class E1Predictor:
    def __init__(self, model, data_prep_config=None, max_batch_tokens=65536, **kw):
        self.model = model
        self.max_batch_tokens = max_batch_tokens
        INSTANCES.append(self)

    def predict(self, seqs):
        return [{"logits": None} for _ in seqs]
'''

STUB_E1_MODELING = '''
import importlib.util
import logging
import os
import torch

from .model import attention             # upstream: modeling.py imports its attention layer (-> flash_attention, flex_attention)

logger = logging.getLogger(__name__)       # upstream: transformers' get_logger(__name__) == logging.getLogger("E1.modeling")
try:                                       # upstream L22-26: get_kernel("kernels-community/triton-layer-norm") or the logged torch fallback
    if os.environ.get("STUB_HUB_KERNEL_FAILS"):
        raise RuntimeError("stub: hub kernel fetch failed")
    _spec = importlib.util.spec_from_file_location("triton_layer_norm_stub", {kernel_file!r})
    layer_norm = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(layer_norm)
except Exception as e:
    logger.warning(f"Failed to load triton layer norm kernel: {{e}}; Will be using PyTorch RMSNorm implementation instead.")
    layer_norm = None


class _W:
    def __init__(self, n, h): self.shape = (n, h)


class _Emb:
    def __init__(self, h): self.weight = _W(64, h)


class E1ForMaskedLM:
    def __init__(self, name, hidden): self.name, self.hidden = name, hidden

    @classmethod
    def from_pretrained(cls, name, dtype=None, **kw):
        return cls(name, {hidden_size})

    def to(self, device): return self
    def eval(self): return self
    def get_input_embeddings(self): return _Emb(self.hidden)
'''

STUB_E1_FLASH_ATTENTION = '''
import os
import torch
try:                                       # upstream E1/model/flash_attention.py L5-9
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None


def is_flash_attention_available() -> bool:   # L14-17: both symbols AND the env, read per call
    return flash_attn_func is not None and flash_attn_varlen_func is not None and os.getenv("USE_FLASH_ATTN", "1") == "1"


def flash_attention_func(*a, **k):
    return flash_attn_varlen_func(*a, **k)
'''

STUB_E1_ATTENTION = '''
import torch
from .flash_attention import flash_attention_func, is_flash_attention_available   # upstream E1/model/attention.py L10
from .flex_attention import flex_attention


class Attention:
    def _flash_attn(self, *a, **k):        # L267-312: flash if is_flash_attention_available() else varlen flex (silent)
        return flash_attention_func(*a, **k) if is_flash_attention_available() else None

    def _flex_attn(self, *a, **k):
        return flex_attention(*a, **k)
'''

STUB_E1_FLEX_ATTENTION = '''
import torch


def _flex_attention_eager(*a, **k):
    return None


class _Compiled:                           # the shape of torch.compile's wrapper around a function (torch._dynamo marks the original)
    _torchdynamo_orig_callable = staticmethod(_flex_attention_eager)

    def __call__(self, *a, **k):
        return _flex_attention_eager(*a, **k)


flex_attention = _Compiled()               # upstream E1/model/flex_attention.py L4-11: torch.compile(flex_attention, dynamic=True) on CUDA
'''

STUB_FLASH_ATTN = '''
__version__ = {version!r}
import flash_attn_2_cuda                  # flash_attn_interface imports its CUDA extension at import


def flash_attn_varlen_func(*a, **k):
    return None


def flash_attn_func(*a, **k):
    return None
'''

STUB_HUB_KERNEL_FILE = '''
# stub of the hub kernel snapshot's layer_norm.py: the two objects the model's call reaches
class Autotuner:                           # triton.runtime.autotuner.Autotuner is the type of the real kernel object
    def __init__(self):
        self.configs = []


_layer_norm_fwd_1pass_kernel = Autotuner()


def rms_norm_fn(x, weight, bias, **kw):
    return x
'''

STUB_E1_DIST = '''
import torch
def setup_dist(): pass
def destroy_process_group(): pass
def get_device(): return torch.device("cpu")
def get_rank(): return 0
def barrier(): pass
'''

STUB_E1_IO = '''
def read_fasta_sequences(path):
    out, head, seq = {}, None, []
    for line in open(path):
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if head is not None: out[head] = "".join(seq)
            head, seq = line[1:], []
        else:
            seq.append(line)
    if head is not None: out[head] = "".join(seq)
    return out
'''


#: the KERNELS proof line a kit-route child prints once per process (accel.py via the runner), for the stub kit scripts of the CLI tests:
#: the words are the EXPECTED ones (the real reader is exercised on the stub upstream by the stock-route children and test_kernels_proof.py);
#: `kernels_print(mode)` words it under a kit mode (exact: the runner's prefix and route are the mode's), KERNELS_PRINT = exact's
def kernels_print(mode: str = "exact") -> str:
    return ('from e1_opt import accel, report, stack as _st; '
            f'print(report.kernels_line("{mode}", "{mode}", accel.expected("{mode}", _st.load_pins()), "kit_attn", True), flush=True)')


KERNELS_PRINT = kernels_print("exact")


def _dist_info(site: str, name: str, version: str, direct_url: dict = None, files=()) -> None:
    d = os.path.join(site, f"{name.replace('-', '_')}-{version}.dist-info")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "METADATA"), "w") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    with open(os.path.join(d, "RECORD"), "w") as fh:
        for f in files:
            fh.write(f"{f},,\n")
        fh.write(f"{os.path.basename(d)}/METADATA,,\n")
    if direct_url is not None:
        with open(os.path.join(d, "direct_url.json"), "w") as fh:
            json.dump(direct_url, fh)


def write_stub_site(root: str, pins, variant: str, kernel_file: str, gpu_name: str = None, commit: str = None) -> str:
    """A site directory: stub torch + E1 (modules and dist-info at the pinned versions; E1's direct_url.json at the pinned commit)."""
    site = os.path.join(root, "site")
    os.makedirs(site, exist_ok=True)
    st = pins.STACK
    gpu_name = gpu_name or next(iter(stack.CARDS))
    os.makedirs(os.path.join(site, "torch"), exist_ok=True)
    with open(os.path.join(site, "torch", "__init__.py"), "w") as fh:
        fh.write(STUB_TORCH.format(torch_version=st["torch_version_str"], cuda=st["cuda"], gpu_name=gpu_name))
    for pkg, key in (("torch", "torch"), ("triton", "triton"), ("transformers", "transformers"), ("tokenizers", "tokenizers"),
                     ("kernels", "kernels"), ("flash-attn", "flash_attn")):
        _dist_info(site, pkg, st[key])
    e1 = os.path.join(site, "E1")
    os.makedirs(os.path.join(e1, "tools"), exist_ok=True)
    open(os.path.join(e1, "__init__.py"), "w").close()
    open(os.path.join(e1, "tools", "__init__.py"), "w").close()
    with open(os.path.join(e1, "tools", "score.py"), "w") as fh:
        fh.write(STUB_E1_TOOLS_SCORE)
    with open(os.path.join(e1, "scorer.py"), "w") as fh:
        fh.write(STUB_E1_SCORER)
    with open(os.path.join(e1, "predictor.py"), "w") as fh:
        fh.write(STUB_E1_PREDICTOR)
    with open(os.path.join(e1, "modeling.py"), "w") as fh:
        fh.write(STUB_E1_MODELING.format(kernel_file=kernel_file, hidden_size=int(pins.WEIGHTS[variant]["hidden_size"])))
    os.makedirs(os.path.join(e1, "model"), exist_ok=True)
    open(os.path.join(e1, "model", "__init__.py"), "w").close()
    for name, body in (("flash_attention.py", STUB_E1_FLASH_ATTENTION), ("attention.py", STUB_E1_ATTENTION), ("flex_attention.py", STUB_E1_FLEX_ATTENTION)):
        with open(os.path.join(e1, "model", name), "w") as fh:
            fh.write(body)
    os.makedirs(os.path.join(site, "flash_attn"), exist_ok=True)
    with open(os.path.join(site, "flash_attn", "__init__.py"), "w") as fh:
        fh.write(STUB_FLASH_ATTN.format(version=st["flash_attn"]))
    with open(os.path.join(site, "flash_attn_2_cuda.py"), "w") as fh:
        fh.write("# stub of flash-attn's compiled extension module\n")
    os.makedirs(os.path.join(site, "kernels"), exist_ok=True)
    with open(os.path.join(site, "kernels", "__init__.py"), "w") as fh:                     # the hub-kernels package E1/modeling.py imports (get_kernel)
        fh.write(f"__version__ = {st['kernels']!r}\ndef get_kernel(repo_id, revision=None):\n    raise RuntimeError('stub: no hub access')\n")
    with open(os.path.join(e1, "dist.py"), "w") as fh:
        fh.write(STUB_E1_DIST)
    with open(os.path.join(e1, "io.py"), "w") as fh:
        fh.write(STUB_E1_IO)
    _dist_info(site, "E1", pins.STOCK["version"], direct_url={"url": pins.STOCK["repo"], "vcs_info": {"vcs": "git", "commit_id": commit or pins.STOCK["commit"]}},
               files=["E1/__init__.py", "E1/tools/score.py", "E1/scorer.py"])
    return site


def write_stub_caches(root: str, pins, variant: str) -> dict:
    """HF_HOME with the pinned snapshot (a placeholder model.safetensors) and a KERNELS_CACHE with the pinned kernel snapshot (a
    placeholder layer_norm.py). Returns {hf_home, kernels_cache, snapshot, weights_file, kernel_file, weights_sha256, weights_bytes,
    kernel_sha256} — the shas of the placeholders, for tests that patch the pins module in memory to accept them."""
    hf_home = os.path.join(root, "hf")
    snap = pins.weights_snapshot(variant, hf_home)
    os.makedirs(snap, exist_ok=True)
    wf = os.path.join(snap, "model.safetensors")
    with open(wf, "wb") as fh:
        fh.write(b"placeholder weights " + variant.encode())
    kc = os.path.join(root, "kernels_cache")
    k = pins.KERNEL
    kd = os.path.join(kc, "models--" + k["repo"].replace("/", "--"), "snapshots", k["rev"], "build", "torch-universal", "triton_layer_norm")
    os.makedirs(kd, exist_ok=True)
    kf = os.path.join(kd, "layer_norm.py")
    with open(kf, "w") as fh:
        fh.write(STUB_HUB_KERNEL_FILE)
    return {"hf_home": hf_home, "kernels_cache": kc, "snapshot": snap, "weights_file": wf, "kernel_file": kf,
            "weights_sha256": sha256(wf), "weights_bytes": os.path.getsize(wf), "kernel_sha256": sha256(kf)}


def write_fake_nvidia_smi(root: str, name: str, mib: int, cc: str = "9.0") -> str:
    """A directory holding an executable `nvidia-smi` that answers the package's query with one line; prepend it to PATH."""
    d = os.path.join(root, "bin")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "nvidia-smi")
    with open(p, "w") as fh:
        fh.write(f"#!/bin/sh\necho '{name}, {mib}, {cc}'\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
    return d


def patch_pins_for_stub(monkeypatch, pins, variant: str, caches: dict) -> None:
    """Make the real pins module accept the placeholder files (in memory, this process only)."""
    monkeypatch.setitem(pins.WEIGHTS[variant], "sha256", caches["weights_sha256"])
    monkeypatch.setitem(pins.WEIGHTS[variant], "bytes", caches["weights_bytes"])
    monkeypatch.setitem(pins.KERNEL, "layer_norm_py_sha256", caches["kernel_sha256"])


def setup_box(monkeypatch, tmp_path, variant: str = None, gpu: tuple = None, commit: str = None) -> dict:
    """The whole mocked box for an in-process run: stub site on sys.path (and PYTHONPATH for children), caches, fake nvidia-smi on
    PATH, a clean E1_* environment. Returns {pins, variant, site, caches, gpu}."""
    pins = pins_or_skip()
    variant = variant or sorted(pins.WEIGHTS)[0]
    name, mib = gpu or (next(iter(stack.CARDS)), stack.CARDS[next(iter(stack.CARDS))][1])
    caches = write_stub_caches(str(tmp_path), pins, variant)
    site = write_stub_site(str(tmp_path), pins, variant, caches["kernel_file"], gpu_name=name, commit=commit)
    bin_dir = write_fake_nvidia_smi(str(tmp_path), name, mib)
    for m in [m for m in sys.modules if m == "torch" or m.startswith("torch.") or m == "E1" or m.startswith("E1.") or m in ("flash_attn", "flash_attn_2_cuda", "triton_layer_norm_stub")]:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.syspath_prepend(site)
    import importlib
    importlib.invalidate_caches()
    monkeypatch.setenv("PATH", bin_dir + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("PYTHONPATH", site + os.pathsep + OPT_DIR)
    monkeypatch.setenv("HF_HOME", caches["hf_home"])
    monkeypatch.setenv("KERNELS_CACHE", caches["kernels_cache"])
    for k in list(os.environ):
        if k.startswith(("E1_OPT", "E1_KIT", "E1_VARIANT", "MODEL_OPT_TARGET")) or k in ("USE_FLASH_ATTN", "FAST_BLOCK_MASK", "STUB_HUB_KERNEL_FAILS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("MODEL_OPT", stack.tree_home())
    # the tree's own stock/PINS.json describes the real weights; the stub box's pins are patched to the stub files, so the
    # PINS.json cross-check is pointed at an absent file here (registry.cross_check: absent = nothing to compare)
    monkeypatch.setattr(stack, "pins_json_path", lambda: os.path.join(str(tmp_path), "absent_PINS.json"))
    stack._reset_for_tests()
    return {"pins": pins, "variant": variant, "site": site, "caches": caches, "gpu": (name, mib), "bin": bin_dir}


def child_env(box: dict, **extra) -> dict:
    """An environment for a subprocess on the mocked box (no E1_* variable unless passed in `extra`)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("E1_OPT", "E1_KIT", "E1_VARIANT")) and k not in ("USE_FLASH_ATTN", "FAST_BLOCK_MASK", "STUB_HUB_KERNEL_FAILS")}
    env.update({"PATH": box["bin"] + os.pathsep + os.environ.get("PATH", ""), "PYTHONPATH": box["site"] + os.pathsep + OPT_DIR,
                "HF_HOME": box["caches"]["hf_home"], "KERNELS_CACHE": box["caches"]["kernels_cache"], "MODEL_OPT": stack.tree_home(),
                "PYTHONDONTWRITEBYTECODE": "1"})
    env.update(extra)
    return env


def write_items(root: str, names=("item_b", "item_a"), n_mut: int = 3) -> str:
    """A directory of item directories (parent.fasta + mutants.fasta), returned; names are created in the given order."""
    d = os.path.join(root, "items")
    parent = "MKTAYIAKQRQISFVKSHFSRQ"
    for n in names:
        idir = os.path.join(d, n)
        os.makedirs(idir, exist_ok=True)
        with open(os.path.join(idir, "parent.fasta"), "w") as fh:
            fh.write(f">{n}_parent\n{parent}\n")
        with open(os.path.join(idir, "mutants.fasta"), "w") as fh:
            for i in range(n_mut):
                s = parent[:i] + "A" + parent[i + 1:]
                fh.write(f">{n}_m{i}\n{s}\n")
    return d
