"""The lever registry: one entry per lever of the kit, keyed by the kit's own name (P1, P2, P3, P5, and the per-step levers a tier composes).

The kit is a set of add-only files for a `mosaic.fast` sub-package (`mosaic_fast/`) and one driver that composes them (`tools/public_design_run.py`) through one recipe module (`tools/recipe.py`).
This registry only *describes* each lever — which file implements it, its class (forward / datapath / serving /
orchestration), its numerics tier, its class word (``klass``), the route by which a row switches it on (``route``), and the
field of the driver's own `results.json` that proves it ran in a process (`probe`). No lever value or mode composition lives here: modes are
the activation rows (modes.py) and the kit's own code applies the levers (the per-step levers through the ONE installer, levers.py).

``klass`` — the class words the kit's tables use (CHANGES.md):
  exact   bitwise identical to stock (the step's loss and gradient bytes at the same state and key)
  fast    same error class as stock at fixed states (never claimed bitwise)
  big   the memory tier: makes a problem size fit the card; its numerics word is fast-class unless shown bitwise
  qol     one-time-cost levers (weight load, featurized inputs, compilation cache): arithmetic-free by construction — they change a
          process's one-time costs (load · featurize · compile + first step), never the step's arithmetic or its steady rate
``route`` — how a row turns the lever on:
  env      the process environment before the interpreter starts (P1); in-process through `opt_core.jax_design.pcc` (stack.p1_lever)
  flag     a driver flag that replaces a call at the call site (P2, P3); no in-process form (an import hook cannot rewrite the caller's calls)
  install  a per-step lever: `mosaic_opt.levers` imports ``module`` and calls its ``install()`` — in-process (`enable()`, the `.pth` hook) and in
           the driver (its `--levers WORD[+ID[=SPEC],...]` flag), the same call either way

`probe` grammar (read by outputs.classify from the driver's results.json after a run):
  ("manifest", <dotted key>, <predicate>)   the value under results["manifest"][<key>] satisfies the predicate
                                            predicates: nonempty | startswith:<text> | equals:<text> | never
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

KIT_FAST_DIR = "mosaic_fast"          # the sub-package the kit adds (run.sh install copies it into site-packages/mosaic/fast/: mosaic_opt/leverfiles.py)
KLASSES = ("exact", "fast", "big", "qol")                      # the class words of the kit's tables (docstring)
ROUTES = ("env", "flag", "install")                              # how a row switches a lever on (docstring)
LEVERS_FLAG = "--levers"                                         # the driver's ONE per-step lever flag: `--levers WORD[+ID[=SPEC],...]` (tools/public_design_run.py; levers.parse_ids)


@dataclass(frozen=True)
class Lever:
    name: str                      # the kit's name (P1, P5, E1, …)
    title: str                     # what it does, in the kit's words
    kit_file: str                  # the file that implements it, relative to the kit directory (install route: relative to the mosaic/ tree)
    cls: str                       # forward | datapath | serving | orchestration
    tier: str                      # "bitwise" (Tier 1) | "tier2"
    switch: str                    # how the row turns it on (environment or driver flag), with the kit file:line
    probe: Tuple[str, str, str]    # ("manifest", dotted key, predicate name) — the driver's own record of the lever
    wired: bool                    # True = a mode composes it; False = shipped, no row switches it on
    doc: str                       # where the kit's docs state the lever's class and numerics
    klass: str = "qol"             # exact | fast | big | qol (KLASSES)
    route: str = "flag"            # env | flag | install (ROUTES)
    module: Optional[str] = None   # install route: the lever's module — `mosaic.fast.<stem>` (a kit file, through tools/recipe.py kit_module) or `opt_core.…`; its API is levers.py's module contract (ENV_REQUIRED · configure · install · uninstall · describe)
    flag: Optional[str] = None     # the exact driver-flag text the lever's row carries (None: environment-only, or not wired to the driver)
    origin: str = "kit"            # kit | core — where the implementation lives (the LEVER line's origin field; core = opt_core)
    requires: str = ""             # stack facts the lever needs beyond stock's pins (empty = none)
    needs: Tuple[str, ...] = ()    # levers that must be ON in the same process for this one to install (F8 needs F6): the ablation switch (levers.levers_off, `MODEL_OPT_LEVERS_OFF`) refuses by name a request that switches a needed lever off and keeps this one
    env: Tuple[str, ...] = ()      # env route: the environment variable NAMES the lever's row assignments carry (modes.ROWS holds the values; the resolver drops exactly these when the lever is switched off or steps aside)


LEVERS: Dict[str, Lever] = {
    "P1": Lever("P1", "persistent JAX compilation cache + XLA autotune results dumped by the first process and loaded by every later one",
                f"{KIT_FAST_DIR}/repro_cache.py", "serving", "bitwise",
                "environment before the process starts: the row's assignments (modes.ROWS, B_p1populate_p2 populate / C_p1warm_p2 warm — the "
                "compilation-cache variables and the autotune dump/load flag); in-process: mosaic.fast.repro_cache.enable(DIR) before the first jax "
                "computation (repro_cache.py:3-8)",
                ("manifest", "identity_key_pre.jax_compilation_cache_dir", "nonempty"), True, "CHANGES.md lever table (repro_cache.py)", klass="qol", route="env",
                env=("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "XLA_FLAGS")),
    "P2": Lever("P2", "Boltz-2 weight load without the random-init / torch-checkpoint conversion cost (identical parameters)",
                f"{KIT_FAST_DIR}/fastload.py", "serving", "bitwise",
                "driver flag --weights fastinit (tools/public_design_run.py `--weights` → tools/recipe.py load_model): Boltz2() replaced by "
                "mosaic.fast.fastload.load_stock_fast_init()",
                ("manifest", "load_path", "startswith:P2"), True, "CHANGES.md lever table (fastload.py)", klass="qol", route="flag", flag="--weights fastinit"),
    "P3": Lever("P3", "frozen featurized inputs (npz + sha256) loaded instead of re-featurizing (ref_pos is re-drawn by the featurizer otherwise)",
                f"{KIT_FAST_DIR}/frozen.py", "datapath", "bitwise",
                "driver flags --features-in NPZ --features-sha SHA: model.binder_features(...) replaced by the driver's inline load of the npz after its "
                "sha256 check (tools/public_design_run.py `--features-in` / `--features-sha` → tools/recipe.py load_frozen_features: the inline load), the in-line equivalent of "
                "frozen.load_features (frozen.py:9-15; the lever file, imported by no row)",
                ("manifest", "features.source", "startswith:frozen npz"), True, "CHANGES.md lever table (frozen.py)", klass="qol", route="flag",
                flag='--features-in "$F1" --features-sha "$FSHA"'),
    "P5": Lever("P5", "memory reschedule for large token counts — triangle-attention query-row chunking under lax.map, grouped two-level pairformer "
                "rematerialisation, optional sub-block remat and carry offload; the default setting `pf8+sub` makes a token count that does not fit the card "
                "under stock fit, at slower steps; accumulation order changes under chunking — fast numerics class unless shown bitwise",
                f"{KIT_FAST_DIR}/memlevers.py", "forward", "tier2",
                "driver flag --levers big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): the fast tier's levers, then "
                "mosaic.fast.memlevers.install() + configure(None = its default setting) after the stack import, before any trace; in-process: mosaic_opt.levers.install(\"big\")",
                ("manifest", "levers.P5.state", "equals:on"), True, "CHANGES.md §Lever map", klass="big", route="install", module="mosaic.fast.memlevers",
                flag="--levers big", origin="kit"),
    "E1": Lever("E1", "exact algebra: the template module's identically-zero addend removed when the featurized template mask is all zero (licensed before "
                "tracing from the features the step traces; a real template refused by name; an unlicensed trace refused fail-closed); loss / terms / refold bitwise "
                "everywhere tested, grad bitwise at n400 (2 items), n200 grad rel_l2 ≈1e-6 = autotune-pick noise from the changed HLO; re-measured 0.3.7 under the exact "
                "tier's own step recipe (one step from the stock trajectory's states, --det 1, the trajectory's XLA cache + autotune results loaded, H100): cd45 (431 "
                "tokens) bitwise on every unit at 4/4 checkpoints, pdl1 (195 tokens) loss / every term / argmax bitwise but the gradient differs (max_abs 3.0e-6..6.9e-6, "
                "rel_l2 3.1e-6..8.9e-6) and the post-update PSSM by up to 5.7e-7 at 4/4 checkpoints — the changed backward HLO compiles to a different executable "
                "at that size; not bitwise vs `off` at every representative size, so it stays fast-class INSIDE the fast tier; the `exact` row does not compose it",
                f"{KIT_FAST_DIR}/dead_template.py", "orchestration", "tier2",
                "driver flag --levers fast / --levers big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): "
                "mosaic.fast.dead_template.install() + configure(None = its default setting `skip`) first of the row, after the stack import, before any trace",
                ("manifest", "levers.E1.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.dead_template",
                flag="--levers fast", origin="kit"),
    "P6": Lever("P6", "per-region matmul precision for joltz (trunk · diffusion · confidence; words = opt_core's matmul vocabulary highest|high|medium): "
                "stock runs the 25-step sampler at HIGHEST (IEEE fp32, mosaic's own `float32` island) and everything else at DEFAULT; the setting of "
                "record `diffusion=high` runs the sampler's dots at HIGH (TF32) — fast numerics class by construction, never bitwise with stock",
                f"{KIT_FAST_DIR}/precision.py", "forward", "tier2",
                "driver flag --levers fast (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.precision.install() + "
                "configure(None = its default setting) after the stack import, before any trace; in-process: mosaic_opt.levers.install(\"fast\")",
                ("manifest", "levers.P6.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.precision",
                flag="--levers fast", origin="kit"),
    "K1": Lever("K1", "triangle attention (trunk pairformer, MSA / template / confidence pairformers; starting and ending node): the attention core — logits, masks, softmax and "
                "the value contraction, which stock materialises as an [rows, heads, N, N] float32 array — served forward AND backward by the shared core's JAX-family provider "
                "(opt_core.kernels.pallas: one measured cell table over every carried row — the XLA-FFI triangle-attention bridge triattn_xla with its differentiable row, the Pallas flash-attention rows, "
                "the stock statement) at the row's TIER WORD: the fast row pins `fast` (per call the fastest measured row of this jax line, card, operand dtype — bfloat16 under P7, float32 elsewhere —, "
                "head geometry and token bucket), the big row `big` (the lowest-peak row not slower than the stock statement); the module keeps its own LayerNorm and projections and names no kernel, "
                "tile or launch setting. A provider row or arm word (`K1=<row>`, e.g. triattn_xla@vjp, pallas_attn, cd_triatt) pins ONE row for the ablation of the tier's choice; a row that refuses the "
                "module's call class is refused by name at configure. Fast numerics class by construction (online softmax), never bitwise with stock",
                f"{KIT_FAST_DIR}/flashattn.py", "forward", "tier2",
                "driver flag --levers fast (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.flashattn.install() + "
                "configure(the row's tier word, modes.KIT_MODES specs) after the stack import, before any trace; in-process: mosaic_opt.levers.install(\"fast\")",
                ("manifest", "levers.K1.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.flashattn",
                flag="--levers fast", origin="core", requires="opt_core 0.5.125.1+ (kernels.pallas: the JAX-family provider with the tier words fast | big | exact on the attention face)"),
    "E10": Lever("E10", "the diffusion module's scanned layer stacks (joltz DiffusionTransformer2: token transformer 24 layers x the sampling steps, atom encoder/decoder "
                       "transformers) applied as a static Python loop so every layer's weights are read at static offsets instead of per-iteration dynamic-slice copies; "
                       "upstream's layers, order and per-layer checkpoint; not bitwise (XLA fuses across the exposed layer boundaries) — fast-class; "
                       "COST beside the number: the sampler's executables compile several-fold longer once per process (results.json compile_plus_first_s at N=400, one H100: "
                       "refold 79.5 s under fast+E10 vs 17.0 s under fast vs 19.4 s stock)",
                 f"{KIT_FAST_DIR}/layer_unroll.py", "forward", "tier2",
                 "driver flag --levers fast | big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.layer_unroll.install() + "
                 "configure(None = 'sampler') after recipe.import_stack() and before any trace, after the tier's other per-step levers; in-process: mosaic_opt.levers.install(\"fast\")",
                 ("manifest", "levers.E10.state", "equals:on"), True, f"{KIT_FAST_DIR}/layer_unroll.py:1-24 (module docstring: class and numerics)", klass="fast", route="install", module="mosaic.fast.layer_unroll",
                 flag="--levers fast", origin="kit"),
    "F6": Lever("F6", "triangle multiplication (outgoing and incoming; trunk pairformer, MSA and template modules) kept in ONE channel-major layout: the input "
                "projections contracted so the GEMM writes [2C, N, N], gate / mask / a|b split in that layout, the triangle product as the channel-batched matmul it is, "
                "norm_out over the leading channel axis (joltz's LayerNorm arithmetic along that axis), the output projection contracting the channel axis back to "
                "channel-minor — XLA materialises none of stock's [N,N,C]<->[C,N,N] transposes; the same contractions with other dimension numbers, so kernel and algorithm "
                "picks differ from stock's — fast numerics class unless shown bitwise; default setting `cmajor`",
                f"{KIT_FAST_DIR}/trimul_layout.py", "forward", "tier2",
                "driver flag --levers fast | big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.trimul_layout.install() + "
                "configure(None = its default setting) after the stack import and the tier's other per-step levers, before any trace (before P5 on the big row: P5's sub-block "
                "rematerialisation wraps the rebound triangle-multiplication calls); in-process: mosaic_opt.levers.install(\"fast\")",
                ("manifest", "levers.F6.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.trimul_layout",
                flag="--levers fast", origin="kit"),
    "F8": Lever("F8", "fused triangle-multiplication kernels ON TOP of F6's channel-major layout, served by the shared core's JAX-family kernel provider BY TIER WORD: "
                "F6's served call is given another body through F6's one extension point (trimul_layout.set_body) — the op mosaic.fast.trimul_provider asks "
                "opt_core.kernels.pallas (serve.triangle_multiplication) with the word the mode's tier row carries (`fast` on the fast row, `big` on the big row) "
                "and binds the provider's measured cell for (JAX line, card, activation dtype, module family, token count, call kind): a fused Pallas / FFI row "
                "(LayerNorm + projections x sigmoid gates x mask as channel-major planes, the cubic contraction as one batched GEMM, LayerNorm + output projection x "
                "gate; a fused backward where the row has one) where one measured ahead of XLA on this card, F6's own trimul_cmajor BY NAME (counted xla(cell)) where "
                "none did or no cell covers the call; the un-differentiated call (the refold, inference) and the differentiated call (the design step) are asked "
                "separately and bound behind one custom_vjp per call class; both activation dtypes (float32, and the bfloat16 calls P7 hands over: P7 installs no "
                "dtype route while F8 holds the body); the kit names no kernel, row, tile or size threshold; words: `fast` | `big` (the provider's tier words), "
                "`stock`/`off` (F6's XLA body); anything else refused unknown_spec; refuses by name (unknown_spec, needs_F6, not_installed, op_missing, "
                "probe_failed:<kind>, op_refused:<kind>); a call the provider refuses at trace time runs F6's body counted by the refusal's name — no silent "
                "fallback; fast numerics class (the served rows' tolerance class)",
                f"{KIT_FAST_DIR}/trimul_fused.py", "forward", "tier2",
                "driver flag --levers fast / --levers big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.trimul_fused.install() + "
                "configure(the row's word: modes.KIT_MODES[<mode>][\"specs\"][\"F8\"]) after F6's, before P5 on `big`; in-process: mosaic_opt.levers.install(\"fast\") / install(\"big\")",
                ("manifest", "levers.F8.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.trimul_fused", needs=("F6",),
                flag="--levers fast", origin="kit", requires="F6 installed and on in the same process; mosaic.fast.trimul_provider in the installed mosaic; an opt_core whose kernels.pallas provider carries the tier words and the trimul cells (the [tool.opt_core] floor); a GPU backend (the op's probe)"),
    "F9": Lever("F9", "fused pair TRANSITION kernels (LayerNorm + fc1|fc2 + silu(a)*b + fc3 of joltz.Transition) on P7's bf16 operands: two Pallas (Triton "
                "lowering) kernels, FORWARD and BACKWARD (custom_vjp; the backward recomputes the hidden activations per row tile, does the LayerNorm backward in "
                "registers and writes the bf16 planes of the three weight-gradient GEMMs XLA runs) — the [rows, 4c] intermediates never round-trip HBM (XLA's own "
                "chain for this sub-layer is at HBM roofline: bytes are the only lever); default setting `zres` = P7's transition LINE through P7's one extension "
                "point (halfpair.set_tz_body): the kernels read the block's float32 pair activation and write z + f32(bf16(transition)) — P7's casts and the residual "
                "add inside, P7's rounding points kept; `sub` = the named alternative (joltz.Transition.__call__ rebound: bf16 calls in / out, P7's casts and the "
                "residual by XLA); float32 calls (sequence transition, MSA module unless P7 `msa`) are joltz's own, counted f32_stock; P7 off or `tz` outside its "
                "spec = a named aside (aside=p7_tz_off), never a refusal; refuses by name (unknown_spec, not_installed, no_gpu, probe_failed:<kind>, joltz_missing); "
                "an envelope miss (fc bias, hidden/tile mismatch) is counted unserved and fails the gate — no silent fallback; fast numerics class",
                f"{KIT_FAST_DIR}/transition_fused.py", "forward", "tier2",
                "driver flag --levers fast / --levers big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.transition_fused.install() + "
                "configure(None = its default setting) after F8's, before P5 on `big` and before P7; in-process: mosaic_opt.levers.install(\"fast\") / install(\"big\")",
                ("manifest", "levers.F9.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.transition_fused",
                flag="--levers fast", origin="kit", requires="a GPU where the kernels' probe passes (Pallas-Triton: compute capability 8.0+); P7 with the tz extension point (mosaic_opt 0.3.20+) for `zres`; joltz's bias-free Transition (fc1/fc2 hidden = 4c, c % 16 == 0)"),
    "P7": Lever("P7", "the trunk pairformer's PAIR track in bfloat16, as upstream Boltz-2 runs its trunk (torch bf16-mixed autocast): each block's triangle "
                "multiplications, triangle attentions and pair transition (regions tm / ta / tz; `pf` = all three, the default setting) applied with bf16 operands "
                "(module parameters and the pair activation cast at the sub-layer boundary, float32 accumulation, output cast back before the residual add); LayerNorm "
                "statistics, the residual stream, the sequence track and everything outside the blocks stay float32; `carry` also carries the pair activation in bf16 "
                "between the 64 blocks (half-size checkpoints), `msa` extends the policy to the MSA module's pair blocks (the record on fast and big: `pf+msa`); composes "
                "with F6 (dtype-generic), F8 (an F8 word serving bfloat16 — `<word>/cd`, the record — takes the bf16 calls itself, trimul=f8_<word>; a float32-only F8 "
                "word gets a dtype route at F6's extension point: bf16 calls on F6's XLA arithmetic, float32 calls on F8's kernels, and `msa` is then refused by name, "
                "msa_starves_f8), K1 (cuDNN word native bf16; the Pallas word probed, "
                "else `ta` steps aside by name), P5 (installed after it) — fast numerics class by construction, never bitwise with stock",
                f"{KIT_FAST_DIR}/halfpair.py", "forward", "tier2",
                "driver flag --levers fast | big (tools/public_design_run.py → tools/recipe.py install_levers → mosaic_opt.levers): mosaic.fast.halfpair.install() + "
                "configure(None = its default setting) LAST of the row (after F6/F8/K1, and after P5 on `big`: it wraps the Pairformer2 call it finds); in-process: "
                "mosaic_opt.levers.install(\"fast\")",
                ("manifest", "levers.P7.state", "equals:on"), True, "CHANGES.md §Lever map", klass="fast", route="install", module="mosaic.fast.halfpair",
                flag="--levers fast", origin="kit", requires="F6/F8/K1/P5 installed BEFORE it in the same process when they are on (row order)"),
}
for _k, _v in LEVERS.items():                                                      # the table's own consistency, refused at import (a typo never reaches a row)
    assert _v.klass in KLASSES and _v.route in ROUTES, (_k, _v.klass, _v.route)
    assert (_v.tier == "bitwise") == (_v.klass in ("exact", "qol")), (_k, _v.tier, _v.klass)      # bitwise ⇔ exact or qol; tier2 ⇔ fast or big
    assert not (_v.wired and _v.route == "install" and not _v.module), _k                          # a wired per-step lever names its module
    assert all(n in LEVERS and n != _k and LEVERS[n].route == _v.route == "install" for n in _v.needs), (_k, _v.needs)   # a need names another per-step lever
    assert bool(_v.env) == (_v.route == "env"), (_k, _v.route, _v.env)                            # an env-route lever names the variables its row assignments carry; no other lever owns environment
WIRED: Tuple[str, ...] = tuple(k for k, v in LEVERS.items() if v.wired)          # the wired levers: every lever of the registry on this version
INSTALL: Tuple[str, ...] = tuple(k for k, v in LEVERS.items() if v.route == "install")             # the per-step levers (levers.py's domain), wired or not
IN_PROCESS: Tuple[str, ...] = tuple(k for k in WIRED if LEVERS[k].route in ("env", "install"))    # P1 and the per-step levers: what an import hook / enable() can apply without the caller's cooperation
CALL_SITE: Tuple[str, ...] = tuple(k for k in WIRED if LEVERS[k].route == "flag")                 # ("P2", "P3"): replaced at the call site by the driver's flags


def probe_key(name: str) -> Tuple[str, str, str]:
    """The probe of an install-route lever: the driver records `manifest["levers"][<name>]["state"] = "on"` from the installer's evidence."""
    return ("manifest", f"levers.{name}.state", "equals:on")
