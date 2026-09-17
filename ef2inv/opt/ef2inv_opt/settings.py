"""The shipped settings — the upstream cookbook's defaults, every deviation named.

Nothing here is passed into the loop: the cookbook file's own module constants (``STEPS``, ``LOSS_WEIGHTS``, ``LEARNING_RATE``, ...) run
as shipped; the values below are READ BACK from the imported module at run time and compared (``check_shipped``, from stock_design) so that a
changed constant is a refusal, never a silent deviation. The two knobs the design verb sets are the cookbook's public call arguments
(``ESMFold2Design.load(use_scaling_critics)`` and ``design(...)``) — listed under ``deviations`` with the shipped value they replace.

The design verb takes the binder length (``--binder-len``; absent = the cookbook's own ``--binder-name`` route) and passes its ``main()`` / ``design()`` knobs through
verbatim under their stock names (``--batch-size``, ``--is-antibody``, ``--epitope-contact-distance``, ``--binder-sequence``,
``--use-scaling-critics``; ``STOCK_KNOBS``); ``for_run`` builds the record (every knob passes through to the cookbook unchanged).
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# the cookbook file's module constants as shipped (stock/src/cookbook/tutorials/binder_design.py; line numbers there)
SHIPPED_CONSTANTS: Dict[str, object] = {
    "LOSS_WEIGHTS": {"intra_contact": 0.5, "inter_contact": 0.5, "glob": 0.2, "epitope": 0.5},   # l.96
    "STEPS": 150,                       # l.97  — K design steps
    "LOG_INTERVAL": 5,                  # l.98
    "LEARNING_RATE": 0.1,               # l.99  — SGD, lr = LEARNING_RATE * temperature per step (l.1101)
    "TEMPERATURE_MIN": 1e-2,            # l.100 — cosine schedule 1 -> 0.01 (l.1132-1134)
    "ESMC_MASK_FRACTION": 0.15,         # l.101 — pPPL loss masking
    "LM_LOSS_BATCH_SIZE": 128,          # l.102
    "LM_MASK_PASSES": 4,                # l.103
    "COMPILE": True,                    # l.105 — torch.compile of the inversion models' MSA encoder + pair-update blocks (l.1256-1268, 1300-1302)
    "REUSE_ESMC": True,                 # l.110 — the LM head's ESMC trunk shared with the inversion models (l.1313-1322)
}
# the cookbook's fixed per-call values (not settings: cited so callers can name them)
SHIPPED_CALLS = {
    "inversion_lm_dropout": 0.5,        # l.1296 configure_lm_dropout(0.5, force_lm_dropout_during_inference=True) (l.1246)
    "critic_lm_dropout": 0.25,          # l.1307-1308
    "design_step_num_loops": 1,         # l.1067
    "design_step_num_sampling_steps": "50 if calculate_confidence else 1",   # l.1068
    "design_step_calculate_confidence": "temperature < 0.05",                 # l.1139
    "critic_num_loops": 3, "critic_num_sampling_steps": 200, "critic_calculate_confidence": True,   # l.1168-1170
    "esmc_lm_dtype": "float32",         # l.1311 ESMCForMaskedLM.from_pretrained(..., torch_dtype=torch.float32)
    "replicate_choice": "random.seed(seed + step); random.randint over the 2 inversion models",   # l.1057-1059
    "plm_grad_weight": "0.15 (0.05 for antibodies)",   # l.1097
    "kernel_backend": "fused when TRITON_KERNELS_AVAILABLE else cuequivariance when CUE_AVAILABLE else None",   # l.1247-1252; the fork engages fused kernels in no-grad folds only (critics, confidence): the design steps run the reference pair stack under grad
    "chunk_size": "the fork's default pair-stack chunk (64): the file never calls set_chunk_size",              # the pair stack's triangle contraction chunk
}

# The two load-time model switches = upstream's own setters, `design --chunk-size none|N` -> `set_chunk_size(None|N)` and `--kernel-backend
# fused|cuequivariance|None` -> `set_kernel_backend(v)`; a flag absent = no call (the cookbook as shipped). `off` makes no call unless a flag says
# so (the stock arm's documented speed settings are `--chunk-size none --kernel-backend cuequivariance`); `exact`'s own values are chunk None +
# cuequivariance (modes.py) and a differing flag is a user override applied as on the stock arm, the one exact lever with nothing to attach to stepping
# aside by name (EXACT_SWITCH_NOTES, cli.switch_notes); `fast` / `big` make no call at load, a flag there is a user override the kit yields to by name.
# The design script applies the effective values right after `ESMFold2Design.load`, before the det recipe, the stock patch and the kit's enable:
# `set_chunk_size(v)` and `set_kernel_backend(v)` on EVERY loaded ESMFold2 model — the inversion models and the critics — each read back from
# the fork's own attributes before the arm goes on (stock_design.apply_model_switches; a switch that did not take effect exits 3 by name);
# no flag = untouched (the fork's chunk of 64; the cookbook's own backend call; recorded as `shipped`). The arm prints one `MODEL-SWITCH …` line per applied switch
# after the read-back, naming the models it covered, and records them in run.json `model_switches`; opt_manifest.json records the effective `model_switches`, names them
# in `deviations` (the mode's value or a user override), and lists user overrides under `overrides`; a user `--kernel-backend` also prints ONE
# NOTE line before the launch saying what the value reaches under grad. The mode's own composition runs after the switches at the kit's enable
# (`fast` / `big` set 64 on the inversion models).
CHUNK_SHIPPED = "shipped"

# The cookbook's own design knobs, passed through verbatim under their stock names (binder_design.py `main`, l.1443-1453 = `design`, l.1325-1335):
# flag -> (main()/design() keyword, stock default). `use_scaling_critics`: stock's `main()` default is True (l.1448; its `local=True` path — the
# one this tree runs — asserts False, l.1456-1460); this tree's base is False (USE_SCALING_CRITICS_BASE) because the 15 scaling-critic
# snapshots the cookbook names (l.1288-1300) are not among the six pinned weight snapshots (STOCK.md §Weights) — a NAMED
# difference, not a refusal: `--use-scaling-critics 1` passes through to `load(True)`, which reads those 15
# snapshots from HF_HOME like every other weight. `batch_size` B passes through to `design(batch_size=B)`: B trajectories in one process,
# exactly the cookbook's; design.fasta / design.pdb are trajectory 0's, trajectories 1..B-1 are in critics.json (batch_idx, designed_sequence,
# their critic structures critic_<name>_b<k>.pdb) and trajectory.jsonl (every loss term a list of B).
STOCK_KNOBS = {"--batch-size": ("batch_size", 1), "--is-antibody": ("is_antibody", None), "--epitope-contact-distance": ("epitope_contact_distance", 12.0),
               "--binder-sequence": ("binder_sequence", None), "--use-scaling-critics": ("use_scaling_critics", True), "--binder-name": ("binder_name", "minibinder")}
STOCK_BINDER_NAME = "minibinder"        # main(binder_name) in the cookbook's __main__ (l.1493): the registered factory `minibinder`, uniform length 60-200 per seed (l.137) — the binder route when --binder-len is absent
USE_SCALING_CRITICS_BASE = False

# The kernel backend values are the literals the fork's API takes (modeling_esmfold2_common.py:80-82: BACKEND_FUSED = "fused", BACKEND_CUEQ =
# "cuequivariance", _VALID_BACKENDS = (None, "fused", "cuequivariance")); `set_kernel_backend(v)` goes on every ESMFold2 model the cookbook loaded
# (the inversion models and the hero critics — the models its `_load_hf_model` set the backend on, l.1247-1252; the cookbook's own call picks
# fused when the fork's Triton kernels import). What each value reaches, from the fork's gates (the transformers pin,
# modeling_esmfold2_common.py): `fused` is gated by `_fused_active` (l.85-92: `not torch.is_grad_enabled()`) — it acts in the no-grad folds
# (hero critics, confidence, first folds) and the design steps run the reference pair stack under grad; `cuequivariance` is NOT grad-gated
# (`_cueq_active`, l.95-96): the triangle updates run cuequivariance_torch's `triangle_multiplicative_update` under grad too (l.2413-2434), so
# the design-step numerics differ from the reference pair stack (its pair-bias attention kernel engages only above 750 queries, l.1066-1074;
# transitions stay reference; a kernel failure inside the fork logs a warning and takes the chunked einsum, l.2435-2447); `None` = the reference
# modules in every fold. A user `--kernel-backend <v>` prints ONE `NOTE --kernel-backend <v> — …; proceeding` line with these words before the launch.
KERNEL_BACKEND_SHIPPED = "shipped"                      # internal word, never a flag value: the flag absent = no set_kernel_backend call (the cookbook's own choice stands)
KERNEL_BACKENDS = ("fused", "cuequivariance", None)      # the fork's _VALID_BACKENDS literals
KERNEL_BACKEND_WORDS = ("fused", "cuequivariance", "None", "none")   # every argv spelling of --kernel-backend: parse_kernel_backend's whole domain ("None" / "none" -> None)
KERNEL_BACKEND_NOTES = {
    "fused": "applies to the models' no-grad folds (hero critics, confidence, first folds); the design steps run under grad, where the fork's backend switch "
             "dispatches the reference pair stack (`_fused_active` requires no_grad, modeling_esmfold2_common.py:85-92); design-step numerics = the reference's",
    "cuequivariance": "NOT grad-gated (`_cueq_active`, modeling_esmfold2_common.py:95-96): the design steps' triangle updates run cuequivariance_torch's "
                      "triangle_multiplicative_update under grad (l.2413-2434) — design-step numerics differ from the reference pair stack; its pair-bias attention "
                      "kernel engages only above 750 queries (l.1066-1074); transitions stay reference; the no-grad folds take it too",
    None: "the reference modules in every fold, grad and no-grad (the cookbook's fused choice for the no-grad folds is replaced); the design steps are the "
          "reference pair stack either way",
}   # one entry per value parse_kernel_backend returns (KERNEL_BACKEND_WORDS): a user flag naming ANY legal word over a kit mode whose value differs is a NOTE, never a KeyError


# The one input `fast` / `big` refuse that stock accepts. The stock fused pair-bias kernel (the transformers fork's
# models/esmfold2/kernels/fused_attention_pair_bias.py `_pair_bias_kernel`: LN(z) @ pair_bias_proj + key mask for the diffusion transformer's
# AttentionPairBias, launched under the `fused` backend inside `ESMFold2.sample()` on every design step's fold and every critic fold) forms its
# element offsets in 32-bit integers over z [B, L, L, 256] and its [B, 16, L, L] output: a plane with B*L*L*max(256, 16) > 2**31 - 1 element
# offsets wraps them (an illegal address or a wrong bias, never an error message). B = design(batch_size), L = the folded length (target +
# binder): batch 1 holds up to 2,896 tokens, batch 2 up to 2,047, batch 13 up to 800. `fast` / `big` reach that kernel whenever their backend
# is the cookbook's own choice (`fused` when the fork's Triton kernels import, which the kit modes require) or `--kernel-backend fused`, so the
# design script refuses such a run by name before anything loads (`REFUSED pair_bias_int32_bound: …`, exit 3, opt_manifest.json `refusals`);
# `--kernel-backend cuequivariance|None` does not reach the kernel. `off` (stock exactly) and `exact` (cuEquivariance pinned) are untouched.
# The remedy upstream is 64-bit offsets in that kernel.
PAIR_BIAS_DIM_Z = 256                                   # the pair width the kernel reads per (q, k) (the model's pairwise state)
PAIR_BIAS_HEADS = 16                                    # token_num_heads: the bias planes it writes
PAIR_BIAS_OFFSET_MAX = 2 ** 31 - 1                      # the largest element offset one launch may form (int32)
PAIR_BIAS_GUARDED_MODES = ("big", "fast")                # by mode name: the two tier-2 modes
PAIR_BIAS_WORD = "pair_bias_int32_bound"


def pair_bias_max_batch(n_tokens: int) -> int:
    """The largest design(batch_size) whose pair-bias plane stays inside the kernel's int32 element offsets at ``n_tokens`` folded tokens (0 = none does)."""
    L = int(n_tokens)
    return PAIR_BIAS_OFFSET_MAX // (max(PAIR_BIAS_DIM_Z, PAIR_BIAS_HEADS) * L * L)


def pair_bias_max_tokens(batch_size: int) -> int:
    """The largest folded length whose pair-bias plane stays inside the kernel's int32 element offsets at ``batch_size`` (2,896 at batch 1)."""
    import math
    return math.isqrt(PAIR_BIAS_OFFSET_MAX // (max(PAIR_BIAS_DIM_Z, PAIR_BIAS_HEADS) * int(batch_size)))


def pair_bias_int32_problem(mode_name: str, kernel_backend, batch_size: int, n_tokens: int) -> Optional[str]:
    """The refusal words of a ``fast`` / ``big`` run whose pair-bias plane would cross the stock fused kernel's int32 offset bound
    (``B*L*L*max(256, 16) > 2**31 - 1`` on the ``fused`` route: the cookbook's own backend — no ``--kernel-backend`` flag — or ``--kernel-backend
    fused``), else None: inside the bound, another mode (``off`` is stock exactly, ``exact`` pins cuEquivariance), or a backend that does not
    reach the kernel. The design script prints them as ``REFUSED <words>`` and exits 3 before anything loads (stock_design.main)."""
    if mode_name not in PAIR_BIAS_GUARDED_MODES or kernel_backend not in (KERNEL_BACKEND_SHIPPED, "fused"):
        return None
    B, L = int(batch_size), int(n_tokens)
    offsets = B * L * L * max(PAIR_BIAS_DIM_Z, PAIR_BIAS_HEADS)
    if offsets <= PAIR_BIAS_OFFSET_MAX:
        return None
    fits = pair_bias_max_batch(L)
    remedy = f"lower --batch-size to <= {fits}" if fits >= 1 else f"no batch size fits at {L} tokens (batch 1 holds up to {pair_bias_max_tokens(1)} tokens)"
    return (f"{PAIR_BIAS_WORD}: batch {B} x {L}^2 tokens x {PAIR_BIAS_DIM_Z} = {offsets:,} element offsets exceed the int32 range (2**31-1) of the stock fused "
            f"pair-bias kernel (transformers models/esmfold2/kernels/fused_attention_pair_bias.py _pair_bias_kernel, the diffusion transformer's AttentionPairBias "
            f"under the 'fused' backend: past the bound its offsets wrap); {remedy}, or run with the upstream int64-offset patch of that kernel")


def parse_kernel_backend(value) -> object:
    """``--kernel-backend``: ``fused`` | ``cuequivariance`` | ``None`` — the literals the fork's ``set_kernel_backend`` takes (the flag = that
    call; absent = no call). Anything else raises ValueError by name (a usage error; the fork itself accepts nothing else: _VALID_BACKENDS)."""
    v = str(value).strip()
    if v not in KERNEL_BACKEND_WORDS:
        raise ValueError(f"--kernel-backend {value!r}: fused | cuequivariance | None (omit the flag to keep the cookbook's own call)")
    return None if v in ("None", "none") else v


def as_argparse_type(fn):
    """A parse_* function as an argparse ``type=``: its ValueError words become the usage error (argparse.ArgumentTypeError), so a refused value
    is named together with the legal words instead of argparse's generic 'invalid value'."""
    import argparse

    def conv(value):
        try:
            return fn(value)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e))
    conv.__name__ = fn.__name__
    return conv


def kernel_backend_word(value) -> str:
    """The argv / record spelling of a backend value (``None`` for None)."""
    return "None" if value is None else str(value)


def kernel_backend_note(value) -> tuple:
    """(what, consequence) of the one NOTE line printed when ``--kernel-backend`` is passed (report.note_line)."""
    return (f"--kernel-backend {kernel_backend_word(value)}", KERNEL_BACKEND_NOTES[value])


def parse_chunk_size(value) -> object:
    """``--chunk-size``: ``none`` (``set_chunk_size(None)``: the pair stack unchunked) | a positive integer (``set_chunk_size(N)``); the flag
    absent = no call (the fork's chunk of 64). Anything else raises ValueError by name (a usage error)."""
    v = str(value).strip().lower()
    if v == "none":
        return None
    if v.isdigit() and int(v) > 0:
        return int(v)
    raise ValueError(f"--chunk-size {value!r}: none | a positive integer (omit the flag to keep the fork's chunk of 64)")


def chunk_size_word(value) -> str:
    """The argv / record spelling of a chunk value (``none`` for None, else the integer)."""
    return "none" if value is None else str(value)


def user_switch_word(key: str, value) -> str:
    """The record spelling of one pair-stack switch value: ``chunk_size`` -> none|N, ``kernel_backend`` -> fused|cuequivariance|None."""
    return chunk_size_word(value) if key == "chunk_size" else kernel_backend_word(value)


def user_switches(mode, chunk_size=CHUNK_SHIPPED, kernel_backend=KERNEL_BACKEND_SHIPPED) -> Dict[str, object]:
    """The pair-stack switches THE USER set for this run on a kit mode: a flag that is present and differs from the mode's own value
    (``mode.chunk_size`` / ``mode.kernel_backend``; ``shipped`` = the flag absent). ``off`` has no kit value to override: always ``{}`` there
    (the flag is simply the stock arm's setting). The same rule the design verb records under ``overrides`` and the kit reads at enable: a
    user switch is every loaded model's (stock's setter, applied at load), and a kit lever whose contract needs another value steps aside for
    it by name instead of refusing the run."""
    if not getattr(mode, "is_kit", False):
        return {}
    out: Dict[str, object] = {}
    for key, val, mine in (("chunk_size", chunk_size, getattr(mode, "chunk_size", CHUNK_SHIPPED)), ("kernel_backend", kernel_backend, getattr(mode, "kernel_backend", KERNEL_BACKEND_SHIPPED))):
        if isinstance(val, str) and val == CHUNK_SHIPPED:
            continue
        if val != mine or type(val) is not type(mine):
            out[key] = val
    return out


# The design batch. The cookbook's `design(batch_size=B)` runs B trajectories in one process; the kit's memory plan and CUDA-graph pools (the
# trunk step graphs, the sampler / ESM-C / pseudo-perplexity graphs, the target-chain hoist) are sized for ONE trajectory per process, so a kit
# mode (`exact` / `fast` / `big`) accepts B = 1 only: `--batch-size` > 1 there is a usage error named by ONE sentence before anything loads
# (the launcher and the design script both check it at argument parse; exit 2). `--mode off` passes any B through untouched, as stock.
BATCH_GT1_SENTENCE = "--batch-size >1 is supported only with --mode off (stock)"


def batch_size_problem(mode, batch_size) -> Optional[str]:
    """The one sentence when a kit mode is asked for a design batch above one, else None (``off``: any B; B <= 1: none)."""
    return BATCH_GT1_SENTENCE if getattr(mode, "is_kit", False) and int(batch_size or 1) > 1 else None


# What `exact` does with a user pair-stack switch (one NOTE line per switch before the launch, report.note_line): the switch is applied as the
# stock arm applies it — stock's own setter on every loaded model — and every exact lever keeps its bitwise contract against THAT stock
# configuration except the one that has nothing to attach to, which steps aside by name on its LEVER line.
EXACT_SWITCH_NOTES = {
    "kernel_backend": "stock's set_kernel_backend({word}) on every loaded model, as on the stock arm with this flag; no model then runs the cuEquivariance "
                      "triangle multiplication, so the cuEquivariance tile table steps aside by name (LEVER name=cueq_tiles state=stepped_aside "
                      "reason=user_kernel_backend:{word}) and the memory plan prices the kernels that serve; every other exact lever unchanged",
    "chunk_size": "stock's set_chunk_size({word}) on every loaded model, as on the stock arm with this flag: the pair stack runs stock's chunked path at {word}; "
                  "every exact lever unchanged (bitwise against stock at that setting); the memory plan keeps its unchunked row (LEVER name=ef2_bwd_ckpt … "
                  "plan_row=<row>(unchunked-priced)) — an upper bound, never a new coefficient",
}


def exact_switch_note(key: str, value) -> tuple:
    """(what, consequence) of the NOTE line `exact` prints for a user pair-stack switch (report.note_line)."""
    flag = "--chunk-size" if key == "chunk_size" else "--kernel-backend"
    word = user_switch_word(key, value)
    return (f"--mode exact with {flag} {word}", EXACT_SWITCH_NOTES[key].format(word=word))


@dataclass(frozen=True)
class Settings:
    binder_len: Optional[int]           # the binder length: fixed (--binder-len, a PromptFactory with length_ranges (L, L)), the given --binder-sequence's, or the length the cookbook's own factory samples for the seed (None on the CLI side of that route)
    use_scaling_critics: bool = False   # load(use_scaling_critics): base False (USE_SCALING_CRITICS_BASE); --use-scaling-critics 1 loads the 15 scaling critics as the cookbook does
    batch_size: int = 1                 # design(batch_size): B trajectories per process, the cookbook's own (--batch-size)
    is_antibody: Optional[bool] = False # the fixed-length factory's value (l.137) = what upstream resolves ``None`` to on the factory route (l.1015-1019); --is-antibody sets it; None only on the --binder-sequence route with the flag unset (upstream then auto-detects, l.1025-1026)
    epitope_contact_distance: float = 12.0   # design() default (l.1335)
    deviations: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def design_kwargs(self) -> dict:
        return {"batch_size": self.batch_size, "is_antibody": self.is_antibody, "epitope_contact_distance": self.epitope_contact_distance}


def _deviations(binder_len, use_scaling_critics: bool = False, binder_name: Optional[str] = None) -> List[str]:
    scaling = [] if use_scaling_critics else [                       # --use-scaling-critics 1 = the shipped __main__'s value: no deviation to name
        "use_scaling_critics: False (the shipped __main__ passes True, l.1503; the 15 ESMFold2-Experimental-Fast-base*-step*k checkpoints are not in the "
        "weights cache)"]
    if binder_name is not None:                                       # the cookbook's own factory route (main(binder_name=<factory>)): the binder is the cookbook's, nothing of the kit's to name
        length = [f"binder: the cookbook's factory {binder_name!r} as registered (BINDER_PROMPT_FACTORIES; its length sampled per seed by the cookbook" + (f": {binder_len} for this seed)" if binder_len is not None else ")")]
    else:
        length = [f"binder_len: fixed {binder_len} (a PromptFactory with length_ranges ({binder_len}, {binder_len}); "
                  f"shipped default: the 'minibinder' factory, uniform 60-200 per seed, l.137)"]
    return [
        *length,
        *scaling,
        "target: the case's sequence (a built-in target name when it equals a TARGET_SEQUENCES entry, l.159-171, else target_name=<case id> + "
        "target_sequence=<sequence>, the shipped route l.997-1005 — target_name=<case id>:explicit when the case id itself names a TARGET_SEQUENCES entry of "
        "another sequence: the explicit target file takes precedence, stock_design.resolve_target); hotspots from the case when given",
    ]


def binder_word(binder_len, binder_name=None, binder_sequence=None) -> str:
    """The RUN line's binder_len field: the fixed length, else the given sequence's length, else the cookbook factory's name (its length is the cookbook's to sample)."""
    if binder_len is not None:
        return str(int(binder_len))
    if binder_sequence is not None:
        return str(len(binder_sequence))
    return str(binder_name or STOCK_BINDER_NAME)


def for_run(binder_len, *, binder_name: Optional[str] = None, batch_size: int = 1, is_antibody: Optional[int] = None, epitope_contact_distance: float = 12.0,
            use_scaling_critics: int = 0, binder_sequence: Optional[str] = None) -> Settings:
    """The design verb's settings for a binder length (``--binder-len`` L fixed; on the cookbook's own factory route, ``binder_name`` given, L is
    the length the cookbook samples for the seed — the caller passes it; on the CLI side, before any cookbook import, ``binder_len`` may be
    None there) and the cookbook's pass-through knobs (``STOCK_KNOBS``: every value reaches ``load()`` / ``design()`` unchanged; the flags
    absent = the base values). Usage errors (ValueError): a non-positive ``--binder-len`` / ``--batch-size``, a ``--binder-sequence`` whose
    length is not ``--binder-len``."""
    if binder_len is None and binder_sequence is not None:
        binder_len = len(binder_sequence)                                # the sequence route: the starting binder sets the length
    if binder_len is None:                                               # the cookbook's factory route, length not yet sampled (the CLI side): the knobs are still checked
        if int(batch_size) < 1:
            raise ValueError(f"--batch-size {batch_size!r}: a positive integer")
        name = binder_name or STOCK_BINDER_NAME
        return Settings(binder_len=None, use_scaling_critics=bool(int(use_scaling_critics)), batch_size=int(batch_size), is_antibody=(None if is_antibody is None else bool(int(is_antibody))),
                        epitope_contact_distance=float(epitope_contact_distance), deviations=_deviations(None, bool(int(use_scaling_critics)), name))
    L = int(binder_len)
    if L <= 0:
        raise ValueError(f"--binder-len {binder_len!r}: a positive integer")
    if int(batch_size) < 1:
        raise ValueError(f"--batch-size {batch_size!r}: a positive integer")
    if binder_sequence is not None and len(binder_sequence) != L:
        raise ValueError(f"--binder-sequence has {len(binder_sequence)} residues but --binder-len is {L}: they must agree (the sequence is the cookbook's starting binder, l.1030-1048)")
    critics = bool(int(use_scaling_critics))
    dev = _deviations(L, critics, binder_name)
    if binder_sequence is not None:
        dev[0] = (f"binder_len: fixed {L} = the given --binder-sequence (the cookbook's sequence route: binder_name not a factory, binder_sequence the "
                  f"starting sequence, l.1030-1048; shipped default: the 'minibinder' factory, uniform 60-200 per seed, l.137)")
    ab: Optional[bool] = (bool(int(is_antibody)) if is_antibody is not None else (None if (binder_sequence is not None or binder_name is not None) else False))   # None on the cookbook's own routes (sequence / its factory): upstream resolves it
    if is_antibody is not None:
        dev.append(f"is_antibody: {ab} (--is-antibody; shipped default None = the factory's value / auto-detected from a binder sequence, l.1015-1026)")
    if float(epitope_contact_distance) != 12.0:
        dev.append(f"epitope_contact_distance: {float(epitope_contact_distance)} (--epitope-contact-distance; design() default 12.0, l.1335)")
    return Settings(binder_len=L, use_scaling_critics=critics, batch_size=int(batch_size), is_antibody=ab, epitope_contact_distance=float(epitope_contact_distance), deviations=dev)


def check_shipped(module) -> List[str]:
    """Compare the imported cookbook module's constants with SHIPPED_CONSTANTS; returns one line per difference (empty = as shipped)."""
    bad = []
    for name, want in SHIPPED_CONSTANTS.items():
        got = getattr(module, name, "<absent>")
        if got != want:
            bad.append(f"{name}: module has {got!r}, shipped {want!r}")
    return bad
