"""Stubs for the CPU tests: a minimal ``boltzgen`` package (the classes and functions the kit modules patch at import, a
``configure`` command that writes ``steps.yaml`` + per-step configs, a ``resources/main.py`` whose ``main(config, [])`` writes a
marker) with a ``boltzgen-0.3.2.dist-info`` so ``importlib.metadata`` reports the pinned version, and a minimal
``pytorch_lightning`` (``Trainer.predict``, ``LightningModule``, ``seed_everything``). Real ``torch`` (CPU) is required: the kit
modules import it. ``materialize(root)`` writes the stubs under ``root`` and returns the directory to put first on PYTHONPATH.

The stubs let the real kit modules (``bg_hook``, ``bg_graph_patch``, ``xa_fastinit``, ``xa_hoist``, `big`'s ``sz_levers``, `fast`'s ``fl_levers``) import and
patch as they do in a real process, so the activation, refusal, late-activation and exit-tally rules are exercised on the kits' own bytes."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

FILES = {
    "boltzgen/__init__.py": '__version__ = "0.3.2"\n',
    "boltzgen/model/__init__.py": "",
    "boltzgen/model/models/__init__.py": "",
    "boltzgen/model/models/boltz.py": textwrap.dedent('''
        import pytorch_lightning as pl
        import torch

        class Boltz(pl.LightningModule):
            def __init__(self, width=4):
                super().__init__()
                self.lin = torch.nn.Linear(width, width)
            def load_state_dict(self, sd, strict=True):
                return super().load_state_dict(sd, strict=strict)
            @classmethod
            def load_from_checkpoint(cls, path, strict=True, map_location=None, **kw):
                m = cls()
                m.on_load_checkpoint({"state_dict": m.state_dict()})
                return m
            def on_load_checkpoint(self, checkpoint):
                return None
            def load_checkpoint_weights(self, checkpoint_path):
                self.loaded = getattr(self, "loaded", []) + [checkpoint_path]
            def predict_step(self, batch, *a, **kw):
                if isinstance(batch, dict) and batch.get("raise"):            # test seam: a call that raises
                    raise RuntimeError(batch["raise"])
                if isinstance(batch, dict) and batch.get("switch"):           # test seam: upstream's design-checkpoint switch inside predict_step (boltz.py l.1220-1235)
                    self.current_checkpoint_index += 1
                    if 0 <= self.current_checkpoint_index < len(self.checkpoint_paths):
                        self.load_checkpoint_weights(self.checkpoint_paths[self.current_checkpoint_index])
                return batch
        '''),
    "boltzgen/model/layers/__init__.py": "",
    "boltzgen/model/layers/initialize.py": textwrap.dedent('''
        def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
            weights.data.zero_()
        '''),
    "boltzgen/model/layers/triangular_attention/__init__.py": "",
    "boltzgen/model/layers/triangular_attention/primitives.py": textwrap.dedent('''
        def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
            weights.data.zero_()
        '''),
    "boltzgen/model/layers/attention.py": textwrap.dedent('''
        import torch

        class AttentionPairBias(torch.nn.Module):
            def __init__(self, c=4):
                super().__init__()
                self.proj_z = torch.nn.Linear(c, c)
            def forward(self, s, z, mask=None, k_in=None, multiplicity=1, to_keys=None, model_cache=None, attn_mask=None):
                return s
        '''),
    "boltzgen/model/modules/__init__.py": "",
    "boltzgen/model/modules/trunk.py": textwrap.dedent('''
        import torch

        class TokenDistanceModule(torch.nn.Module):                      # big's td_chunk patches .forward (sz_levers.install)
            def forward(self, z, feats, pair_mask, relative_position_encoding, use_kernels=False):
                return z
        '''),
    "boltzgen/model/layers/transition.py": textwrap.dedent('''
        import torch

        class Transition(torch.nn.Module):                               # upstream's pair Transition (boltzgen/model/layers/transition.py); no lever patches it
            def forward(self, x, chunk_size=None):
                return x
        '''),
    "boltzgen/model/modules/utils.py": textwrap.dedent('''
        def center(x, mask): return x
        def compute_random_augmentation(*a, **kw): return None
        def default(v, d): return d if v is None else v
        '''),
    "boltzgen/model/modules/diffusion.py": textwrap.dedent('''
        import torch

        class DiffusionModule(torch.nn.Module):
            def forward(self, *a, **kw):
                return None

        class AtomDiffusion(torch.nn.Module):                            # fast's fl_levers wraps .preconditioned_network_forward (the sampler context)
            def sample(self, atom_mask, num_sampling_steps=None, multiplicity=1, step_scale=None, noise_scale=None, **kw):
                return {}
            def preconditioned_network_forward(self, noised_atom_coords, sigma, network_condition_kwargs, training=True):
                fn = network_condition_kwargs.get("fn")
                return fn() if fn is not None else noised_atom_coords
        '''),
    "boltzgen/model/modules/encoders.py": textwrap.dedent('''
        import torch
        from torch import nn

        class SingleConditioning(nn.Module):                               # fast's cond_dedup wraps .forward; the statements of upstream's (encoders.py l.131-198) at toy width
            def __init__(self, token_s=8, dim_fourier=4):
                super().__init__()
                self.norm_single = nn.LayerNorm(2 * token_s)
                self.single_embed = nn.Linear(2 * token_s, 2 * token_s)
                self.fourier_proj = nn.Linear(1, dim_fourier)
                self.norm_fourier = nn.LayerNorm(dim_fourier)
                self.fourier_to_single = nn.Linear(dim_fourier, 2 * token_s, bias=False)
                self.transitions = nn.ModuleList([nn.Sequential(nn.LayerNorm(2 * token_s), nn.Linear(2 * token_s, 2 * token_s)) for _ in range(2)])
            def forward(self, times, s_trunk, s_inputs):
                s = torch.cat((s_trunk, s_inputs), dim=-1)
                s = self.single_embed(self.norm_single(s))
                normed_fourier = self.norm_fourier(torch.cos(self.fourier_proj(times.unsqueeze(-1))))
                s = self.fourier_to_single(normed_fourier).unsqueeze(1) + s
                for transition in self.transitions:
                    s = transition(s) + s
                return s, normed_fourier
        '''),
    "boltzgen/model/modules/transformers.py": textwrap.dedent('''
        import torch
        from torch import nn, sigmoid
        from boltzgen.model.layers.attention import AttentionPairBias

        class AdaLN(nn.Module):                                            # fast's cond_dedup replaces .forward; upstream's statements (transformers.py l.19-33) at toy width
            def __init__(self, dim, dim_single_cond):
                super().__init__()
                self.a_norm = nn.LayerNorm(dim, elementwise_affine=False, bias=False)
                self.s_norm = nn.LayerNorm(dim_single_cond, bias=False)
                self.s_scale = nn.Linear(dim_single_cond, dim)
                self.s_bias = nn.Linear(dim_single_cond, dim, bias=False)
            def forward(self, a, s):
                a = self.a_norm(a)
                s = self.s_norm(s)
                a = sigmoid(self.s_scale(s)) * a + self.s_bias(s)
                return a

        class ConditionedTransitionBlock(nn.Module):                      # .forward replaced (l.36-66)
            def __init__(self, dim_single, dim_single_cond, expansion_factor=2):
                super().__init__()
                dim_inner = int(dim_single * expansion_factor)
                self.adaln = AdaLN(dim_single, dim_single_cond)
                self.swish_gate = nn.Sequential(nn.Linear(dim_single, dim_inner * 2, bias=False), nn.SiLU())
                self.a_to_b = nn.Linear(dim_single, dim_inner * 2, bias=False)
                self.b_to_a = nn.Linear(dim_inner * 2, dim_single, bias=False)
                self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim_single), nn.Sigmoid())
            def forward(self, a, s):
                a = self.adaln(a, s)
                b = self.swish_gate(a) * self.a_to_b(a)
                a = self.output_projection(s) * self.b_to_a(b)
                return a

        class DiffusionTransformerLayer(nn.Module):                        # .forward replaced (l.141-205)
            def __init__(self, heads=2, dim=8, dim_single_cond=None, post_layer_norm=False):
                super().__init__()
                dim_single_cond = dim if dim_single_cond is None else dim_single_cond
                self.adaln = AdaLN(dim, dim_single_cond)
                self.pair_bias_attn = AttentionPairBias(dim)
                self.output_projection = nn.Sequential(nn.Linear(dim_single_cond, dim), nn.Sigmoid())
                self.transition = ConditionedTransitionBlock(dim_single=dim, dim_single_cond=dim_single_cond)
                self.post_lnorm = nn.LayerNorm(dim) if post_layer_norm else nn.Identity()
            def forward(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1):
                b = self.adaln(a, s)
                k_in = b
                if to_keys is not None:
                    k_in = to_keys(b)
                    mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
                b = self.pair_bias_attn(s=b, z=bias, mask=mask, multiplicity=multiplicity, k_in=k_in)
                b = self.output_projection(s) * b
                a = a + b
                a = a + self.transition(a, s)
                a = self.post_lnorm(a)
                return a
        class DiffusionTransformer(nn.Module):                             # upstream l.70-137: the layer loop (bias split per layer)
            def __init__(self, depth=2, heads=2, dim=8, dim_single_cond=None, post_layer_norm=False, activation_checkpointing=False, **kw):
                super().__init__()
                self.layers = nn.ModuleList([DiffusionTransformerLayer(heads, dim, dim_single_cond, post_layer_norm) for _ in range(depth)])
                self.activation_checkpointing = activation_checkpointing
            def forward(self, a, s, bias=None, mask=None, to_keys=None, multiplicity=1, use_uniform_bias=False):
                B, N, M, D = bias.shape
                L = len(self.layers)
                bias = bias.view(B, N, M, L, D // L) if not use_uniform_bias else bias
                for i, layer in enumerate(self.layers):
                    bias_l = bias[:, :, :, i]
                    a = layer(a, s, bias_l if not use_uniform_bias else bias, mask, to_keys, multiplicity)
                return a
        class AtomTransformer(nn.Module):                                  # exact's hoist replaces .forward (upstream l.212-263: windows folded into the batch, bias repeated)
            def __init__(self, attn_window_queries, attn_window_keys, **diffusion_transformer_kwargs):
                super().__init__()
                self.attn_window_queries = attn_window_queries
                self.attn_window_keys = attn_window_keys
                self.diffusion_transformer = DiffusionTransformer(**diffusion_transformer_kwargs)
            def forward(self, q, c, bias, to_keys, mask, multiplicity=1):
                W = self.attn_window_queries
                H = self.attn_window_keys
                B, N, D = q.shape
                NW = N // W
                q = q.view((B * NW, W, -1))
                c = c.view((B * NW, W, -1))
                mask = mask.view(B * NW, W)
                bias = bias.repeat_interleave(multiplicity, 0)
                bias = bias.view((bias.shape[0] * NW, W, H, -1))
                to_keys_new = lambda x: to_keys(x.view(B, NW * W, -1)).view(B * NW, H, -1)
                q = self.diffusion_transformer(a=q, s=c, bias=bias, mask=mask.float(), multiplicity=1, to_keys=to_keys_new)
                q = q.view((B, NW * W, D))
                return q
        '''),
    "boltzgen/model/loss/__init__.py": "",
    "boltzgen/model/loss/diffusion.py": "def weighted_rigid_align(*a, **kw): return None\n",
    "boltzgen/task/__init__.py": "",
    "boltzgen/task/predict/__init__.py": "",
    "boltzgen/task/predict/writer.py": textwrap.dedent('''
        import hashlib, json, os
        class BasePredictionWriter:
            def __init__(self, write_interval="batch"):
                self.write_interval = write_interval
            def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
                self.write_on_batch_end(trainer, pl_module, outputs, None, batch, batch_idx, dataloader_idx)
            def on_predict_end(self, trainer, pl_module):
                return None
        class DesignWriter(BasePredictionWriter):
            """The stub design writer: one .cif-named text file and one .npz-named byte file per design of the batch, whose bytes are a
            pure function of the tensors it is handed (so a background write can be compared byte for byte with a synchronous one)."""
            def __init__(self, output_dir):
                super().__init__(write_interval="batch")
                self.failed = 0
                self.init_outdir(output_dir)
            def init_outdir(self, outdir):
                self.outdir = outdir
                os.makedirs(outdir, exist_ok=True)
            def write_on_batch_end(self, trainer=None, pl_module=None, prediction=None, batch_indices=None, batch=None, batch_idx=None, dataloader_idx=0, sample_id=None):
                import torch
                if prediction.get("exception"):
                    self.failed += 1
                    return
                if os.environ.get("BGSTUB_WRITER_FAIL") == "1":          # test switch: the write itself fails (as a full disk or a bad path would)
                    raise OSError(28, "stub writer: injected write failure")
                mult = getattr(getattr(getattr(trainer, "datamodule", None), "cfg", None), "multiplicity", 1)
                coords = prediction["coords"]
                for n in range(coords.shape[0]):
                    c = coords[n].cpu()
                    name = f"{batch['id'][0]}_{batch_idx * mult + n}"
                    with open(os.path.join(self.outdir, name + ".cif"), "w") as fh:
                        fh.write("\\n".join(f"ATOM {i} {x:.3f} {y:.3f} {z:.3f}" for i, (x, y, z) in enumerate(c.tolist())) + "\\n")
                    with open(os.path.join(self.outdir, name + ".npz"), "wb") as fh:
                        fh.write(hashlib.sha256(c.numpy().tobytes()).hexdigest().encode() + b"|" + str(batch["mask"][0].sum().item()).encode())
        '''),
    "boltzgen/task/predict/predict.py": textwrap.dedent('''
        class Predict:
            def __init__(self, data=None):
                self.data = data
            def run(self, config=None):
                return "ran"
        '''),
    "boltzgen/resources/__init__.py": "",
    "boltzgen/resources/main.py": textwrap.dedent('''
        import json, os, sys

        def main(config, args):
            """The stub step: writes <run_dir>/<step>.done with what the process saw (env, sys.path, modules)."""
            run_dir = os.path.dirname(os.path.dirname(os.path.abspath(config)))
            step = os.path.splitext(os.path.basename(config))[0]
            rec = {"config": config, "step_env": os.environ.get("BOLTZGEN_PIPELINE_STEP"), "pythonpath": os.environ.get("PYTHONPATH"),
                   "kit_modules": sorted(n for n in sys.modules if n.startswith(("xa_", "bg_", "sz_levers", "fl_levers", "hl_levers"))),
                   "opt_modules": sorted(n for n in sys.modules if n.startswith("boltzgen_opt")),
                   "env_kit": sorted(k for k in os.environ if k.startswith(("XA_", "BG_", "SZ_", "FL_", "HL_")) or k in ("BOLTZGEN_OPT", "PYTHONPATH")),
                   "seeded": "pytorch_lightning" in sys.modules and getattr(sys.modules["pytorch_lightning"], "LAST_SEED", None)}
            with open(os.path.join(run_dir, step + ".done"), "w") as fh:
                json.dump(rec, fh)
            calls = os.environ.get("BOLTZGEN_STUB_CALLS")               # test seam: "128,100" = one triangle attention + one triangle multiplication per token count, through the front-end
            if calls:
                import torch
                for n in (int(x) for x in calls.split(",") if x):
                    from cuequivariance_torch.primitives.triangle import triangle_attention, triangle_multiplicative_update
                    q = torch.zeros(1, 1, 2, n, 32); bias = torch.zeros(1, 1, 2, n, n)
                    triangle_attention(q, q, q, bias)
                    triangle_multiplicative_update(torch.zeros(1, n, n, 8))

        if __name__ == "__main__":                                       # upstream's step entry `python main.py <config>`: how `boltzgen run` and the kits' runner (its CPU steps) start a step
            main(sys.argv[1], sys.argv[2:])
        '''),
    "boltzgen/cli/__init__.py": "",
    "boltzgen/cli/boltzgen.py": textwrap.dedent('''
        """Stub of upstream's CLI: `configure <spec> --output <dir> [--steps ...] [--num_designs N] ...` writes steps.yaml + config/<step>.yaml."""
        import argparse, os, sys

        STEPS = ["design", "inverse_folding", "design_folding", "folding", "analysis", "filtering"]

        def build_parser():
            p = argparse.ArgumentParser(prog="boltzgen")
            sub = p.add_subparsers(dest="command")
            c = sub.add_parser("configure")
            c.add_argument("spec"); c.add_argument("--output", required=True); c.add_argument("--steps", nargs="+", default=None)
            c.add_argument("--num_designs", type=int, default=10000); c.add_argument("--protocol", default="protein-anything")
            c.add_argument("--cache", default=None); c.add_argument("--moldir", default=None); c.add_argument("--num_workers", type=int, default=1)
            c.add_argument("--devices", type=int, default=None); c.add_argument("--design_checkpoints", nargs="+", default=None)
            c.add_argument("--inverse_fold_checkpoint", default=None); c.add_argument("--folding_checkpoint", default=None); c.add_argument("--budget", type=int, default=30)
            c.add_argument("--use_kernels", choices=["auto", "true", "false"], default="auto")
            return p

        def resolution_line(args):
            """Upstream's `Using kernels: <bool> [device capability: (M, m)]` (PipelineBuilder, cli/boltzgen.py l.920-931); the card is the
            test seam BOLTZGEN_STUB_KERNELS_CC ("9,0"; unset = a CPU box, printed as (0, 0) with kernels off)."""
            cc = tuple(int(x) for x in os.environ.get("BOLTZGEN_STUB_KERNELS_CC", "0,0").split(","))
            use = {"auto": cc[0] >= 8, "true": True, "false": False}[args.use_kernels]
            print(f"Using kernels: {use} [device capability: ({cc[0]}, {cc[1]})]")

        def configure_command(args):
            out = args.output
            resolution_line(args)
            os.makedirs(os.path.join(out, "config"), exist_ok=True)
            steps = args.steps or STEPS
            with open(os.path.join(out, "steps.yaml"), "w") as fh:
                fh.write("steps:\\n")
                for s in steps:
                    fh.write(f"  - name: {s}\\n    config_file: config/{s}.yaml\\n")
                    with open(os.path.join(out, "config", s + ".yaml"), "w") as c:
                        c.write(f"_target_: stub.{s}\\nnum_designs: {args.num_designs}\\ncache: {args.cache}\\nnum_workers: {args.num_workers}\\n")
            with open(os.path.join(out, "configure_argv.txt"), "w") as fh:
                fh.write(" ".join(sys.argv[1:]) + "\\n")
                fh.write("ENV " + ",".join(sorted(k for k in os.environ if k.startswith(("XA_", "BG_", "SZ_", "FL_", "HL_")) or k in ("BOLTZGEN_OPT", "PYTHONPATH"))) + "\\n")

        def main():
            args = build_parser().parse_args()
            if args.command == "configure":
                configure_command(args)
            else:
                sys.exit(2)

        if __name__ == "__main__":
            main()
        '''),
    "boltzgen-0.3.2.dist-info/METADATA": "Metadata-Version: 2.1\nName: boltzgen\nVersion: 0.3.2\n",
    "boltzgen-0.3.2.dist-info/RECORD": "",
    # --- the accelerator library, stubbed with the names the census binds (kernels.py): the package attributes upstream's front-end
    # resolves at call time, the two modules' reference implementations and thresholds, the _ext capability flags, a -cu13 dist-info.
    # Seams: STUB_CUEQ_BROKEN_BUILD=1 routes every call to the reference path (a build whose kernels never serve);
    # STUB_CUEQ_TRITON_BROKEN=1 binds triangle_multiplicative_update to the package's own raise-stub (its triton components failed).
    "cuequivariance_ops_torch/__init__.py": textwrap.dedent('''
        import os
        from ._version import __version__
        from . import _ext
        from .triangle_attention import triangle_attention
        IMPORT_EXCEPTION = None
        if os.environ.get("STUB_CUEQ_TRITON_BROKEN") == "1":
            IMPORT_EXCEPTION = "Traceback (stub)\\nImportError: triton components unavailable"
            def _raise_triton_import_error(*args, **kwargs):
                raise ImportError(IMPORT_EXCEPTION)
            triangle_multiplicative_update = _raise_triton_import_error
        else:
            from .triangle_multiplicative_update import triangle_multiplicative_update
        '''),
    "cuequivariance_ops_torch/_version.py": '__version__ = "0.11.1"\n',
    "cuequivariance_ops_torch/_ext.py": "def has_sm100f_support(): return False\ndef has_sm107f_support(): return False\ndef has_sm120f_support(): return False\n",
    "cuequivariance_ops_torch/triangle_attention.py": textwrap.dedent('''
        import enum, os
        import torch
        CUEQ_TRIATTN_FALLBACK_THRESHOLD = int(os.getenv("CUEQ_TRIATTN_FALLBACK_THRESHOLD", "100"))
        BROKEN = os.environ.get("STUB_CUEQ_BROKEN_BUILD") == "1"
        class FwdBackend(enum.Enum):
            GENERIC = 0
            SM100F = 1
        def _fallback_threshold():
            return CUEQ_TRIATTN_FALLBACK_THRESHOLD
        def _select_forward_backend(q, k, mask, actual_s_kv):
            return FwdBackend.GENERIC, [9, 0]
        def _triangle_attention_torch(q, k, v, bias, mask=None, scale=None, return_aux=False, actual_s_kv=None):
            return torch.zeros_like(q)
        def triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False):
            if q.shape[3] <= _fallback_threshold() or BROKEN:
                return _triangle_attention_torch(q, k, v, bias, mask, scale)
            return q.clone()
        '''),
    "cuequivariance_ops_torch/triangle_multiplicative_update.py": textwrap.dedent('''
        import os
        CUEQ_TRIMUL_FALLBACK_THRESHOLD = int(os.getenv("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "100"))
        BROKEN = os.environ.get("STUB_CUEQ_BROKEN_BUILD") == "1"
        def _tri_mul_torch(x, *args, **kwargs):
            return x
        def triangle_multiplicative_update(x, direction="outgoing", mask=None, **kwargs):
            if x.shape[-2] <= CUEQ_TRIMUL_FALLBACK_THRESHOLD or BROKEN:
                return _tri_mul_torch(x)
            return x.clone()
        '''),
    "cuequivariance_ops_torch_cu13-0.11.1.dist-info/METADATA": "Metadata-Version: 2.1\nName: cuequivariance-ops-torch-cu13\nVersion: 0.11.1\n",
    "cuequivariance_ops_torch_cu13-0.11.1.dist-info/RECORD": "",
    "cuequivariance_torch/__init__.py": "",
    "cuequivariance_torch/primitives/__init__.py": "",
    "cuequivariance_torch/primitives/triangle.py": textwrap.dedent('''
        def triangle_attention(q, k, v, bias, mask=None, scale=None, return_aux=False):
            from cuequivariance_ops_torch import triangle_attention as f          # the real front-end imports at call time too
            return f(q, k, v, bias, mask, scale)
        def triangle_multiplicative_update(x, **kwargs):
            from cuequivariance_ops_torch import triangle_multiplicative_update as f
            return f(x, **kwargs)
        '''),
    "pytorch_lightning/__init__.py": textwrap.dedent('''
        import torch
        LAST_SEED = None

        class LightningModule(torch.nn.Module):
            pass

        class Trainer:
            def __init__(self, *a, **kw):
                pass
            def predict(self, model=None, *a, **kw):
                return []

        def seed_everything(seed, workers=False):
            global LAST_SEED
            LAST_SEED = int(seed)
            torch.manual_seed(int(seed))
            return int(seed)
        '''),
}


CUEQ_PREFIXES = ("cuequivariance_ops_torch", "cuequivariance_torch")


def materialize(root: str, cueq: bool = True) -> str:
    """Write the stubs under ``root``; ``cueq=False`` replaces the accelerator library by a package that fails to import (the census then
    reads it as absent — also where a real cuequivariance is installed behind the stubs' path)."""
    site = os.path.join(root, "stub_site")
    for rel, content in FILES.items():
        if not cueq and rel.startswith(CUEQ_PREFIXES):
            if rel == "cuequivariance_ops_torch/__init__.py":
                content = 'raise ImportError("stub: cuequivariance_ops_torch is absent on this path")\n'
            else:
                continue
        p = os.path.join(site, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(content)
    return site


def opt_home() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def core_home() -> str:
    """common/opt_core: the directory the pinned core imports from (installed with the package by `run.sh install`)."""
    import opt_core
    return os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))


def clean_env(site: str, gpu: str = "NVIDIA H100 80GB HBM3,81559", extra_path: str = "", **more) -> dict:
    """An environment for a fresh interpreter: the stubs, the package and its core on PYTHONPATH, no kit switch, the test GPU seam."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("XA_", "BG_", "SZ_", "FL_", "HL_", "BOLTZGEN_OPT", "BOLTZGEN_PIPELINE", "CUEQ_")) and k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join(x for x in (site, opt_home(), core_home(), extra_path) if x)
    env["BOLTZGEN_OPT_TEST_GPU"] = gpu
    env["BOLTZGEN_OPT_HOME"] = opt_home()
    env["KMP_AFFINITY"] = "disabled"
    env["OMP_NUM_THREADS"] = "2"
    env["PYTHONDONTWRITEBYTECODE"] = "1"                                   # never leave __pycache__ inside the carried kit directories
    env.update(more)
    return env


PTH_NAME = "boltzgen_opt_test_stubs.pth"


def site_pth(site: str):
    """Make the stubs importable in a child that strips PYTHONPATH (the stock arm): a .pth in this interpreter's site-packages
    naming the stub directory. Needs a writable site-packages (run the tests in a venv with the package installed editable);
    returns the .pth path (remove it with ``remove_site_pth``) or raises with the reason."""
    import site as _site
    for d in _site.getsitepackages():
        if os.access(d, os.W_OK):
            p = os.path.join(d, PTH_NAME)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(site + "\n")
            return p
    raise RuntimeError("no writable site-packages for the stub .pth: run the tests in a venv (`pip install -e boltzgen/opt`)")


def user_site_pth(site: str, root: str) -> dict:
    """The stubs importable in a child whose PYTHONPATH the package replaces (the kits' process form, stack.mode_env) without a
    writable site-packages: the same .pth in a private user site under ``root``. Returns the names to put in the child's environment
    (``PYTHONUSERBASE``; ``PYTHONNOUSERSITE`` emptied). A child run with ``-s`` (the stock arm) ignores it: that one uses ``site_pth``."""
    base = os.path.join(root, "userbase")
    d = os.path.join(base, "lib", f"python{sys.version_info[0]}.{sys.version_info[1]}", "site-packages")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, PTH_NAME), "w", encoding="utf-8") as fh:
        fh.write(site + "\n")
    return {"PYTHONUSERBASE": base, "PYTHONNOUSERSITE": ""}


def remove_site_pth(path):
    try:
        os.remove(path)
    except OSError:
        pass


def package_installed() -> bool:
    """Whether ``python -s -m boltzgen_opt`` resolves without PYTHONPATH (the stock arm's child needs the package installed)."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-s", "-c", "import boltzgen_opt"], env=env, capture_output=True, text=True)
    return r.returncode == 0


def run_py(code: str, env: dict, timeout: float = 300, cwd=None, args=None):
    """Run ``code`` in a fresh interpreter (``python -c``); returns (rc, stdout, stderr)."""
    r = subprocess.run([sys.executable, "-c", code] + list(args or []), env=env, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return r.returncode, r.stdout, r.stderr
