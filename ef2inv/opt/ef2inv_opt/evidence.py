"""Evidence and the named fallbacks — what a kit arm proves about itself, and every quiet path in the carried kit turned into a
named refusal (a lever that did not install or confirm without a declared reason: exit 3, the reason in ``opt_manifest.json``
``refusals``), a named aside (a lever that stood aside BY ITS OWN DECLARED WORD, or yielded to a user pair-stack switch: its LEVER
record and the EVIDENCE line's ``fallback=`` name it, ``opt_manifest.json`` ``fallback`` carries the sentence, the run completes exit 0)
or a recorded event.

The carried kit is never edited in place. Every site where it would return to stock behaviour
without failing is intercepted here, from outside, at one of two times: BEFORE the loop (a condition probed at activation) or AFTER the loop
(the kit's own counters and log lines read back). The sites, under this module's own labels F1-F11 (the design kit's ``k/`` files and the
kit's cookbook-level integration ``opt/ef2inv_opt/fastkit.py``, line numbers of the carried bytes):

  F1  k/ef2_pppl_graph.py:63-67    pPPL graph capture failure -> "eager forever", a warning     -> after: disabled_reason / replay_fail = its declared word: stood aside by name (LEVER state / EVIDENCE fallback=, exit 0); else its bookkeeping against the run's own: captures == the step's LM input shapes (1), replays == steps x LM calls per step (B x LM_MASK_PASSES rows in chunks of LM_LOSS_BATCH_SIZE), eager == the ESMC forward graph's capture runs through the shared trunk ((n_warmup + 1) x its captures) — a count off that is refused
  F2  k/ef2_esmc_graph.py:32-34    a non-graphable ESMC call falls through to eager (stats.eager) -> after: eager > 0 = its declared word (counted): stood aside by name, exit 0; replays == 0 with no word refused
  F3  k/ef2_autograd_kernels.py:72-86  torch.bmm(out_dtype=) unsupported -> bf16-rounded product  -> before: torch.bmm(out_dtype=) probed (recorded); after: _HAS_BMM_OUT_DTYPE is True when the composition asks for fp32 out (agk3 = bf16: not exercised, an event)
  F4  k/ef2_autograd_kernels.py:640-648  libdevice import chain -> _EXP "tl.exp"                    -> before: _KD3_OK True (fast) else refused; an _EXP other than "libdevice…" is an event (named, the kernels engage)
  F5  k/ef2_bwd_ckpt.py enable()    the memory plan cannot be made (agk absent) / a policy given by hand / a plan whose rule is not the composition's    -> after: every inversion model carries a plan (m._bwd_ckpt) that is not `forced` and whose rule is the composition's (fastkit.COMPOSITION ckpt: big = floor -> policy block reason memory_floor; exact / fast = budget -> no reason word), else refused; the LEVER line names policy / [reason] / kept / budget
  F6  fastkit.py enable()           EF2_FAST_KIT_D59 / EF2_FAST_KIT_PPPL knobs                     -> before: stripped and proven absent (envproof); defaults recorded
  F7  fastkit.py enable()           a second complex size returns the pools and re-plans           -> one trajectory per process: app._fast_kit_tokens asserted None before the design script enables, == n_tokens after
  F8  k/ef2_stepgraph.py SegmentPool  a capture that fails disables the pool (eager from then on); a rebound trunk forward bypasses it; a new signature re-captures a slot -> after: at or below fastkit.POOL_MAX_TOKENS every inversion model holds an active pool (none installed / a rebound forward: refused); its declared words — disabled_reason, eager passes, a slot re-captured or never captured after a step — stand aside by name (the stock eager trunk served there), exit 0; above the size no pool
  F9  k/ef2_autograd_kernels.py:475-481 + k/ef2_trimul.py _tmu_forward  no-grad / non-bf16 calls stay on the stock path (design)  -> after: agk's transition entry point counted against the grad-mode bf16 block calls and ef2_trimul's own served counter against twice them (fast / big); any kernel call on exact refused; the no-grad calls recorded; K-D3's fused W12 + SwiGLU forward (ef2_kd3_gemmswiglu, where K-D3's own forward serves and the card has an entry: sm_80): live -> its launches counted against the transition calls, a card without an entry -> stepped_aside by name (an event), never asked where t16 serves (state=off reason=t16_serves)
  F11 k/ef2_loop_prep.py, k/ef2_loop_pppl.py, k/ef2_esmc_overlap.py, k/ef2_esmc_rope.py, k/ef2_sampler_graph.py, k/ef2_esmc_hoist.py  levers that switch themselves off by name (prep: anchor mismatch / signature / splice_off / pin_failed; overlap: early_failed; sampler graph: capture_failed / mismatch; hoist: anchor / capture_failed / structure, or never served) -> after: each such DECLARED word is an aside — the lever's LEVER record says state=stepped_aside reason=<word> (or state=on … aside=<word> where it served part of the run), the EVIDENCE line names it under fallback=, exit 0; a lever of the composition that is not installed at all (no word) is refused; by-design counters (no_reference, form words, the rotary's stock branch) are events
  F10 k/ef2_state_guard.py:42,51-53  swallowed exceptions while snapshotting                        -> after: the guard's summary recorded; a guard diff raises in the kit itself (mode "assert")

A stock arm (``off``) proves the converse: no kit line in its log (``stock_lever_lines``), no kit module loaded, no kit dir on the path.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

KIT_LINE_PATTERNS = (r"\[ef2_", r"EF2_FAST_KIT", r"fast kit:", r"\bagk\b", r"GraphedTrunk", r"pppl_graph", r"ef2_esmc_graph", r"checkpoint policy")
FASTKIT_LINE_RE = re.compile(r"fast kit: (?P<mode>\w+), checkpoint policy (?P<ckpt>[\w:]+) for (?P<tokens>\d+) tokens")
PINNED_EXP_PREFIX = "libdevice"


def stock_lever_lines(log_text: str) -> List[str]:
    """Lines of a stock arm's log that carry a kit marker (any = the arm is not stock: exit 3)."""
    pats = [re.compile(p) for p in KIT_LINE_PATTERNS]
    return [ln for ln in log_text.splitlines() if any(p.search(ln) for p in pats)]


def fastkit_lines(log_text: str) -> List[dict]:
    """The kit's own activation line(s) (fastkit.enable) parsed: mode, checkpoint policy, tokens."""
    return [m.groupdict() for m in FASTKIT_LINE_RE.finditer(log_text)]


class LogCapture(logging.Handler):
    """Every log record of the process (the cookbook logs through ``logging``), kept as text for run.log and the evidence parsers."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.lines: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))

    def emit(self, record):
        try:
            self.lines.append(self.format(record))
        except Exception:
            pass

    def text(self) -> str:
        return "\n".join(self.lines)


def probe_before(mode_name: str, agk=None, torch=None) -> Dict[str, Any]:
    """Conditions probed at activation (F3, F4, and the graph API F8 depends on), by the arm's COMPOSITION (fastkit.COMPOSITION for the mode's
    kit switch: its transition word asks for agk's kernels, its pool word for the graph API). Returns the facts; ``refusals`` lists what is refused."""
    from . import fastkit as FK
    from .modes import MODES
    facts: Dict[str, Any] = {"refusals": [], "events": []}
    m = MODES.get(mode_name)
    comp = FK.COMPOSITION.get(m.kit_switch) if (m is not None and m.kit_switch) else None   # None on off (nothing installed: nothing to refuse)
    if torch is not None:
        facts["cuda_graph_register_generator_state"] = hasattr(getattr(torch.cuda, "CUDAGraph", object), "register_generator_state")
        dev = "cuda" if getattr(getattr(torch, "cuda", None), "is_available", lambda: False)() else None
        facts["bmm_out_dtype_probe"] = dev or "none"
        if dev:                                   # the keyword is probed where the kit uses it (CUDA); a CPU attempt fails by construction and is not run
            try:
                a = torch.zeros(1, 2, 2, dtype=torch.bfloat16, device=dev); b = torch.zeros(1, 2, 2, dtype=torch.bfloat16, device=dev)
                torch.bmm(a, b, out_dtype=torch.float32)
                facts["bmm_out_dtype_supported"] = True
            except TypeError as e:
                facts["bmm_out_dtype_supported"] = False; facts["bmm_out_dtype_error"] = str(e)[:200]
            except RuntimeError as e:
                facts["bmm_out_dtype_supported"] = "TypeError" not in type(e).__name__; facts["bmm_out_dtype_error"] = str(e)[:200]
        else:
            facts["bmm_out_dtype_supported"] = None
    if agk is not None:
        facts["agk_KERNELS_OK"] = bool(getattr(agk, "KERNELS_OK", False))
        facts["agk_KD3_OK"] = bool(getattr(agk, "_KD3_OK", False))
        facts["agk_EXP"] = str(getattr(agk, "_EXP", ""))
        if comp is not None and comp["transition"]:                                # the fast-class trunk kernels (fast / big): agk's must import and engage
            if not facts["agk_KERNELS_OK"]:
                facts["refusals"].append(f"F3/F4 vendored Triton kernels unavailable ({getattr(agk, '_IMPORT_ERR', None)!r})")
            if not facts["agk_KD3_OK"]:
                facts["refusals"].append(f"F4 K-D3 kernels unavailable (_EXP={facts['agk_EXP']!r})")
            if not facts["agk_EXP"].startswith(PINNED_EXP_PREFIX):
                facts["events"].append(f"F4 Triton exp resolved to {facts['agk_EXP']!r} (the pinned stack's starts with {PINNED_EXP_PREFIX!r}): this Triton's intrinsic, the kernels engage on it")
            if facts.get("bmm_out_dtype_supported") is False:
                facts["events"].append("F3 torch.bmm(out_dtype=) unsupported on this torch (only an fp32 einsum_out_dtype composition would exercise it; agk3 = bf16)")
    if comp is not None and comp["pool"] and torch is not None and not facts.get("cuda_graph_register_generator_state", True):   # every kit mode carries the trunk graph pool (at or below fastkit.POOL_MAX_TOKENS)
        facts["refusals"].append("F8 torch.cuda.CUDAGraph.register_generator_state missing: the kit's trunk graphs would capture without RNG state")
    return facts


class KernelCounter:
    """F9: counts the kit's kernel entry points (module-level functions the kit's patched block forward looks up by name — rebinding
    them to a counting pass-through changes no numerics) against the pair-update block calls that qualify for them."""

    def __init__(self):
        self.calls = {"trimul_with_residual_frozen": 0, "transition_refround": 0}
        self.block_calls = {"grad_bf16": 0, "other": 0}
        self._handles = []

    def install(self, agk, models: List[Any], torch) -> None:
        for name in list(self.calls):
            orig = getattr(agk, name, None)
            if orig is None:
                continue

            def make(nm, fn):
                def wrapped(*a, **k):
                    self.calls[nm] += 1
                    return fn(*a, **k)
                wrapped.__wrapped__ = fn
                return wrapped
            setattr(agk, name, make(name, orig))
        try:
            from transformers.models.esmfold2 import modeling_esmfold2_common as C
        except Exception:
            return

        def pre(m, args, kwargs=None):
            pair = args[0] if args else (kwargs or {}).get("pair")
            ok = pair is not None and torch.is_grad_enabled() and getattr(pair, "dtype", None) == torch.bfloat16 and getattr(pair, "is_cuda", False)
            self.block_calls["grad_bf16" if ok else "other"] += 1
        for model in models:
            for m in model.modules():
                if isinstance(m, C.PairUpdateBlock):
                    self._handles.append(m.register_forward_pre_hook(pre))

    def summary(self, cfg_trimul: Optional[str], cfg_transition: Optional[str], trimul_handles=(), nosave_handles=()) -> Dict[str, Any]:
        """The kit's kernel entry points against the grad-mode bf16 block calls that qualify (counted over EVERY hooked model): agk's own two
        (counted by the rebinding above) and ef2_trimul's fused residual update (the sum of its per-model handles' own ``served`` counters:
        two triangle multiplications per block call, less the ones the no-save first pass of a checkpointed block served on the provider's
        forward-only row — ef2_trimul_nosave's own ``served`` counter; that pass is a grad-mode bf16 block call that records no graph)."""
        handles = {id(h): h for h in (trimul_handles or ()) if h is not None and getattr(h, "variant", None) == "fused"}
        fused = bool(handles)
        nosave = sum(int(h.stats.get("served", 0)) for h in {id(h): h for h in (nosave_handles or ()) if h is not None and getattr(h, "on", False)}.values())
        calls = dict(self.calls, ef2_trimul_fused=sum(int(h.stats.get("served", 0)) for h in handles.values()), ef2_trimul_nosave=nosave)
        exp = {"trimul_with_residual_frozen": 2 * self.block_calls["grad_bf16"] if cfg_trimul else 0,
               "transition_refround": self.block_calls["grad_bf16"] if cfg_transition else 0,
               "ef2_trimul_fused": (2 * self.block_calls["grad_bf16"] - nosave) if fused else 0}
        wanted = {k: v for k, v in exp.items() if v > 0}
        stock = cfg_trimul is None and cfg_transition is None and not fused
        return {"kernel_calls": calls, "block_calls": dict(self.block_calls), "expected": exp,
                "applied": stock or (bool(wanted) and all(calls[k] == v for k, v in wanted.items()))}


CHUNKED_SUBMODULES = ("tri_mul_out", "tri_mul_in", "pair_transition")          # what the fork's PairUpdateBlock.set_chunk_size fans out to (modeling_esmfold2_common.py)


def chunk_attrs(model) -> Dict[str, Any]:
    """Every chunk-size attribute the fold model exposes, by dotted name. The fork's `model.set_chunk_size(c)` writes `_chunk_size` on
    folding_trunk.blocks[i].{tri_mul_out, tri_mul_in, pair_transition} (or their `_engine`), default 64; those are read on block 0 (every
    block gets the same value), plus any `chunk`-named plain attribute on the model, its `folding_trunk` (+ `.config` / `.cfg`) and block 0
    itself. {} = none exposed (refused by name on a big arm — never guessed)."""
    found: Dict[str, Any] = {}
    trunk = getattr(model, "folding_trunk", None)
    blocks = getattr(trunk, "blocks", None)
    block0 = blocks[0] if blocks is not None and len(blocks) else None

    def scan(prefix, obj):
        if obj is None:
            return
        names = set(getattr(obj, "__dict__", {})) | {n for n in dir(type(obj)) if isinstance(getattr(type(obj), n, None), property)}
        for n in sorted(names):
            if "chunk" in n.lower() and not n.startswith("__"):
                try:
                    v = getattr(obj, n)
                except Exception:                                            # noqa: BLE001 — a property that raises is recorded as such
                    v = "<unreadable>"
                if v is None or (isinstance(v, (int, str, bool)) and not callable(v)):
                    found[f"{prefix}.{n}"] = v
    for prefix, obj in (("model", model), ("folding_trunk", trunk), ("folding_trunk.config", getattr(trunk, "config", None)),
                        ("folding_trunk.cfg", getattr(trunk, "cfg", None)), ("folding_trunk.blocks[0]", block0)):
        scan(prefix, obj)
    for sub in CHUNKED_SUBMODULES:                                           # the modules stock's set_chunk_size touches
        m = getattr(block0, sub, None) if block0 is not None else None
        scan(f"folding_trunk.blocks[0].{sub}", m)
        scan(f"folding_trunk.blocks[0].{sub}._engine", getattr(m, "_engine", None))
    return found


def chunk_value(chunks: Dict[str, Any]):
    """The chunk size the census reports, judged on the modules stock's set_chunk_size touches (CHUNKED_SUBMODULES on block 0) when any is
    exposed, else on every exposed attribute: None if ANY judged attribute is None (unchunked), else the first integer, else None."""
    judged = {k: v for k, v in chunks.items() if any(f".{sub}." in k + "." for sub in CHUNKED_SUBMODULES)} or chunks
    if not judged or any(v is None for v in judged.values()):
        return None
    ints = [v for v in judged.values() if isinstance(v, int) and not isinstance(v, bool)]
    return ints[0] if ints else None


def _fb_words(d: Optional[dict], keys=None) -> Dict[str, int]:
    """The non-zero fallback counters of a lever's stats: a nested ``fallback`` dict, or the flat ``fallback_*`` keys."""
    if not d:
        return {}
    out = {}
    fb = d.get("fallback")
    if isinstance(fb, dict):
        out.update({str(k): int(v) for k, v in fb.items() if v})
    elif isinstance(fb, int) and fb:
        out["stock_path"] = fb
    for k, v in d.items():
        if k.startswith("fallback_") and isinstance(v, int) and v and (keys is None or k in keys):
            out[k[len("fallback_"):]] = int(v)
    return out


ESMC_GRAPH_WARMUPS = 2                        # k/ef2_esmc_graph.py _GraphedESMC(n_warmup=2): eager runs per capture before the captured call (read from the wrapper when it says)


def lever_record(name: str, state: str, served=None, fallback: Optional[Dict[str, int]] = None, **extra) -> Dict[str, Any]:
    """One LEVER census record (report.lever_line renders it): name, state on|off|stepped_aside|user|ablated (stepped_aside = the lever
    stood aside BY NAME — at enable on this card / stack / user switch (``reason=``), or at run time by its own declared word and never served;
    a lever that served part of the run and then stood aside by its word reads ``state=on … aside=<word>``; user = a pair-stack switch the user
    set, which the kit's own choice yielded to), served (the lever's own engagement counter; None = a static install with no per-call notion),
    fallback (its named quiet-path counters, non-zero only), then lever-specific words."""
    return {"name": name, "state": state, "served": served, "fallback": dict(fallback or {}), "extra": {k: v for k, v in extra.items() if v is not None}}


def critic_census(app, expected: Optional[Dict[str, Any]], rep: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
    """The hero critics' two model switches read back AFTER the loop — one LEVER record ``critic_switches`` (state on when the arm put the
    critics on a value: fast / big's critic scope = the stock arm's none + cuequivariance, or an every-model value — exact's pins, a user flag;
    words chunk_size / kernel_backend / scope / models / readback). ``expected`` = modes.Mode.critic_switches for the arm ({chunk_size,
    kernel_backend, scope}; None = the mode's own line, no user flag); every critic model under the app's ``hf_critic_models`` is read through stock_design.read_back_switches (the fork's
    ``_chunk_size`` / ``_kernel_backend`` / ``_use_kernels``): a critic that does not read the value it was switched to at load, or no critic to read,
    is refused by name — a critic left on the loader's inert ``fused`` choice would fold on the reference pair stack, chunked, unannounced."""
    from . import stock_design as SD                                        # at call time: stock_design imports this module
    from .modes import CRITIC_MODELS_ATTR, MODES
    mode = MODES[mode_name]
    exp = dict(expected if expected is not None else mode.critic_switches(mode.chunk_size, mode.kernel_backend))   # absent = the mode's own line (its every-model values ride its argv: launch.argv_for)
    scope = exp.get("scope") if isinstance(exp.get("scope"), dict) else {}
    want_chunk = exp.get("chunk_size", "shipped"); want_backend = exp.get("kernel_backend", "shipped")
    state = "on" if (want_chunk != "shipped" or want_backend != "shipped") else "off"
    critics = SD.model_handles(app, attrs=(CRITIC_MODELS_ATTR,)) if app is not None else []
    readback = 0; bad: List[str] = []
    if state == "on":
        for label, m in critics:
            n, b = SD.read_back_switches(label, m, want_chunk, want_backend)
            readback += n; bad += b
            if n == 0:
                bad.append(f"{label} no module carries the fork's switch attributes")
        head = f"critic switches ({mode_name} arm: hero critics chunk_size={'none' if want_chunk is None else want_chunk} kernel_backend={want_backend})"
        if not critics:
            rep["refusals"].append(f"{head}: the loaded app holds no hero critic under {CRITIC_MODELS_ATTR} to read them back from")
        elif bad:
            rep["refusals"].append(f"{head} not in effect after the loop: " + "; ".join(bad[:6]) + (f" (+{len(bad) - 6} more)" if len(bad) > 6 else ""))
    scope_word = ",".join(f"{k}:{scope[k]}" for k in ("chunk_size", "kernel_backend") if scope.get(k) and scope[k] != "shipped") or None
    return lever_record("critic_switches", state, None, None, chunk_size=("none" if want_chunk is None else want_chunk) if state == "on" else None,
                        kernel_backend=str(want_backend) if state == "on" else None, scope=scope_word, models=len(critics), readback=readback if state == "on" else None)


def collect_after(mode_name: str, BD, app, counter: Optional[KernelCounter], log_text: str, agk=None, n_tokens: Optional[int] = None,
                  pppl_stats_fn=None, n_steps: Optional[int] = None, critic_switches: Optional[Dict[str, Any]] = None,
                  user_switches: Optional[Dict[str, Any]] = None, batch_size: int = 1) -> Dict[str, Any]:
    """The activation report of a kit arm after the loop: the kit's line, every lever's census (``levers``: one LEVER record each), the
    config per model, the hero critics' switches read back (critic_census), the guard; and the verdict. Three kinds of finding:
    ``refusals`` — a lever of the composition that did not install or confirm WITHOUT a declared reason (F1, F2, F3, F5, F7, F8, F9, F10, F11:
    not installed, a count off its bookkeeping, the composition not the kit's): the arm is NOT ACTIVE, exit 3; ``asides`` — a lever that stood
    aside at run time BY ITS OWN DECLARED WORD (a capture that failed and said so, an anchor that did not hold, pinned staging the driver
    refused: the stock path served from then on, counted and named): its LEVER record says ``state=stepped_aside reason=<word>`` (or ``state=on …
    aside=<word>`` where it served part of the run), the EVIDENCE line lists it under ``fallback=`` and the run completes, exit 0; ``events`` —
    by-design words. The user's pair-stack switches (``user_switches`` = settings.user_switches: a ``--chunk-size`` / ``--kernel-backend`` over the
    mode's value, on every model from load onward) are part of the expected composition: the kit's chunk of 64 yields to the user's (``LEVER name=chunk
    state=user``), the cuEquivariance tile table steps aside under another backend (``LEVER name=cueq_tiles state=stepped_aside
    reason=user_kernel_backend:<v>``) — never a refusal. ``batch_size`` = the design's B, a term of the pPPL graph's expected counts (a kit arm
    runs B = 1: the launcher accepts ``--batch-size`` > 1 on ``--mode off`` only, settings.batch_size_problem). The arm's composition is checked word for word against fastkit.COMPOSITION for its kit switch and the pool's size rule. A lever the
    ablation word switched off (``BD._LEVERS_OFF``, fastkit.resolve_levers_off) is not expected: its record reads ``state=ablated`` and its
    absence is no refusal (``ablated`` lists the names) — a lever that failed to engage without a word is still refused by name."""
    from . import fastkit as FK, settings as S
    from .modes import MODES, STOCK_KERNEL_BACKEND
    kit = MODES[mode_name].kit_switch
    comp = FK.COMPOSITION[kit]
    off = tuple(getattr(BD, "_LEVERS_OFF", None) or ())
    aside = dict(getattr(BD, "_KIT_ASIDE", None) or {})
    user = {k: v for k, v in dict(user_switches if user_switches is not None else (getattr(BD, "_USER_SWITCHES", None) or {})).items() if k in ("chunk_size", "kernel_backend")}
    user_chunk = user.get("chunk_size", S.CHUNK_SHIPPED)                     # the user's --chunk-size over the mode's value (on every model from load onward), else "shipped"
    user_backend = user.get("kernel_backend", S.KERNEL_BACKEND_SHIPPED)      # the user's --kernel-backend over the mode's value, else "shipped"
    cueq_user_off = user_backend != S.KERNEL_BACKEND_SHIPPED and user_backend != STOCK_KERNEL_BACKEND   # the user's backend is every model's: no triangle multiplication of the process runs the cuEquivariance kernels, the tile table has nothing to serve (fastkit._enable_cueq_tiles stood it aside by name)
    backend_word = f"user_kernel_backend:{S.kernel_backend_word(user_backend)}" if cueq_user_off else None

    moff = FK.mode_off(kit)                                                  # the levers the composition switches off by name (big's mode_off; empty on exact / fast): not expected, their lines read state=off reason=mode:<mode>

    def on(name: str) -> bool:
        return name not in off and name not in moff
    rep: Dict[str, Any] = {"mode": mode_name, "kit": kit, "refusals": [], "asides": [], "events": [], "levers": [], "ablated": list(off), "mode_off": list(moff), "aside": aside,
                           "user_switches": {k: S.user_switch_word(k, v) for k, v in user.items()}, "batch_size": int(batch_size or 1)}

    def stand_aside(name: str, code: str, reason: str, text: str) -> str:
        """A lever that stood aside at run time BY ITS OWN DECLARED WORD: an ``asides`` record and an event — the stock path served, counted and
        named; the run completes (never a refusal). Returns the reason word for the lever's LEVER record."""
        rep["asides"].append({"name": name, "code": code, "reason": reason, "text": text})
        rep["events"].append(f"{code} {name} stood aside by its declared word {reason}: {text}")
        return reason
    lines = fastkit_lines(log_text)
    rep["fastkit_lines"] = lines
    if not lines:
        rep["refusals"].append("missing: the kit printed no `fast kit: ...` activation line (fastkit.enable) — the lever set did not activate")
    toks = getattr(app, "_fast_kit_tokens", None)
    pool_wanted = FK.trunk_pool_wanted(kit, n_tokens if n_tokens is not None else (toks or 0)) and on("ef2_stepgraph")
    want_chunk = comp["chunk"] if on("chunk") else None                    # fast / big: 64; `chunk` ablated: the kit set the pair stack unchunked
    if user_chunk != S.CHUNK_SHIPPED:
        want_chunk = user_chunk                                             # a user --chunk-size is every model's (stock's setter at load); fast / big's 64 yielded to it at enable by name (LEVER chunk state=user)
    crit = dict(critic_switches) if critic_switches is not None else MODES[mode_name].critic_switches(MODES[mode_name].chunk_size, MODES[mode_name].kernel_backend)
    cueq_expected = on("cueq_tiles") and not cueq_user_off and (comp["trimul"] == "cueq_tiles" or crit.get("kernel_backend") == STOCK_KERNEL_BACKEND)   # exact: its own backend; fast / big: a hero critic folds on cuequivariance; a user backend other than cuequivariance: stood aside by name
    models = list(getattr(app, "inversion_models", {}).values())
    L = rep["levers"]

    def opt(modname):
        try:
            return __import__(modname)
        except Exception:                                                    # noqa: BLE001 — a census reader; a module that does not import has nothing installed
            return None
    sgm, trm, bcm, lzm, pbm, spm, rpm, ovm, lpm, ppm, flm, hsm, t16m, gsm = (opt(n) for n in ("ef2_stepgraph", "ef2_trimul", "ef2_bwd_ckpt", "ef2_lazy_structure", "ef2_pairbias_attn",
                                                                                      "ef2_sampler_graph", "ef2_esmc_rope", "ef2_esmc_overlap", "ef2_loop_prep", "ef2_loop_pppl", "ef2_fused_ln",
                                                                                      "ef2_esmc_hoist", "ef2_t16_transition", "ef2_kd3_gemmswiglu"))
    per_model = []
    for name, m in getattr(app, "inversion_models", {}).items():
        cfg = getattr(m, "_agk_cfg", None)
        eeg = getattr(getattr(m, "_esmc", None), "_eeg", None)
        pool = sgm.stats(m) if sgm is not None else None
        th = m.__dict__.get("_ef2_trimul_handle") if hasattr(m, "__dict__") else None
        plan = getattr(m, "_bwd_ckpt", None)
        chunks = chunk_attrs(m)
        head = getattr(m, "confidence_head", None)
        entry = {"model": name, "agk": (agk.describe(cfg) if agk is not None and cfg is not None else ("stock" if cfg is None else str(cfg))),
                 "trimul": ({"variant": th.variant, **dict(th.stats), "entries": len(getattr(th, "cueq_saved", {}) or {}), "reason": (getattr(th, "cueq_reason", "") or None),
                             "gemm": (getattr(th, "gemm", "") or None)} if th is not None else None),
                 "trimul_kernel": (th.variant if th is not None else (getattr(cfg, "trimul", None) or "stock")),
                 "nosave": (lambda nh: ({"on": bool(nh.on), "word": nh.word, "row": nh.row, "reason": nh.reason or None, **dict(nh.stats),
                                         "refused": (",".join("%s:%s" % kv for kv in nh.refused.items()) or None), "canary": nh.canary or None,
                                         "tier_fast": (getattr(nh, "tier_fast", None) or "unasked")} if nh is not None else None))(
                     m.__dict__.get("_ef2_nosave_handle") if hasattr(m, "__dict__") else None),
                 "skip_unused_confidence": hasattr(m, "_agk_orig_model_forward"),
                 "esmc_graph": dict(eeg.stats) if eeg is not None else None,
                 "trunk_graph_pool": pool, "bwd_ckpt": (dict(plan) if isinstance(plan, dict) else None),
                 "lazy_structure": (lzm.stats(m) if lzm is not None and hasattr(m, "_els_orig_forward") else None),
                 "bf16_confidence": bool(head is not None and hasattr(head, "_bc_orig_forward")),
                 "pairbias_attn": (dict(m._pba_state.stats, variant=m._pba_state.variant) if hasattr(m, "_pba_state") else None),
                 "sampler_graph": (dict(m._sg_state.stats, **({"last_failure": m._sg_state.last_failure} if getattr(m._sg_state, "last_failure", None) else {})) if hasattr(m, "_sg_state") else None),
                 "chunk_attrs": chunks, "chunk_size": chunk_value(chunks)}
        per_model.append(entry)
        # the composition, word for word
        want_tr = comp["transition"] if on("agk_transition") else None
        want_tm = comp["trimul"] if on("trimul") else "stock"                  # `trimul` ablated (fast / big): no fast-class kernel, the loader's backend under grad
        want_agk_tm = "bmm2" if want_tm == "bmm2" else None
        if cfg is None or cfg.trimul != want_agk_tm or cfg.transition != want_tr:
            rep["refusals"].append(f"{name}: {mode_name} arm trunk config is not the kit's (trimul={want_agk_tm} transition={want_tr}): {entry['agk']}")
        if want_tm == "fused":
            if th is None or th.variant != "fused":
                rep["refusals"].append(f"F9 {name}: {mode_name} arm without the fused triangle multiplication (ef2_trimul handle {entry['trimul']})")
        elif want_tm == "bmm2":
            if th is not None:
                rep["refusals"].append(f"{name}: {mode_name} arm carries ef2_trimul variant {th.variant!r} beside agk's K-A2 (the composition names bmm2 alone)")
        elif want_tm == "stock":
            if th is not None and th.variant != "cueq_tiles":
                rep["refusals"].append(f"{name}: {mode_name} arm with `trimul` ablated carries ef2_trimul variant {th.variant!r}")
        elif th is not None and th.variant != "cueq_tiles":
            rep["refusals"].append(f"{name}: exact arm carries ef2_trimul variant {th.variant!r} (only cueq_tiles is exact)")
        elif th is None and m is models[0] and on("cueq_tiles") and not cueq_user_off:   # under a user backend other than cuequivariance the table stood aside by name (LEVER cueq_tiles state=stepped_aside)
            rep["refusals"].append(f"{name}: exact arm without the cuEquivariance tile table (ef2_trimul cueq_tiles not installed)")
        if comp["chunk"] != FK.CHUNK_PINNED or user_chunk != S.CHUNK_SHIPPED:   # fast / big: the kit's chunk (or the user's it yielded to); exact: only a user --chunk-size is read back here (its own none is the argv's, LEVER critic_switches)
            whose = f"the user's --chunk-size {S.chunk_size_word(user_chunk)}" if user_chunk != S.CHUNK_SHIPPED else ("`chunk` ablated" if not on("chunk") else f"the mode's {want_chunk}")
            if want_chunk is None:
                if entry["chunk_size"] is not None:
                    rep["refusals"].append(f"{name}: {mode_name} arm runs the pair stack chunked ({entry['chunk_size']}) with {whose} (unchunked asked)")
            elif not chunks:
                rep["refusals"].append(f"{name}: {mode_name} arm exposes no chunk attribute (folding_trunk.blocks[0].tri_mul_out/tri_mul_in/pair_transition._chunk_size, model, trunk) — chunking unproven")
            elif entry["chunk_size"] != want_chunk:
                rep["refusals"].append(f"{name}: {mode_name} arm pair-stack chunk is {entry['chunk_size']!r}, not {whose} ({chunks})")
        if plan is None:
            if on("ef2_bwd_ckpt"):
                rep["refusals"].append(f"F5 {name}: no memory plan installed (ef2_bwd_ckpt) — the checkpoint policy is unplanned")
        elif plan.get("forced"):
            rep["refusals"].append(f"F5 {name}: the checkpoint policy was forced ({plan.get('policy')}), not planned")
        elif comp.get("ckpt") == "floor" and not (plan.get("policy") == "block" and plan.get("reason") == "memory_floor"):
            rep["refusals"].append(f"F5 {name}: {mode_name} arm's memory plan is policy={plan.get('policy')} reason={plan.get('reason')} kept_per_pass={plan.get('kept_per_pass')}, not the mode's floor (policy=block reason=memory_floor: every pair block checkpointed)")
        elif comp.get("ckpt") != "floor" and (plan.get("reason") == "memory_floor" or plan.get("floor")):
            rep["refusals"].append(f"F5 {name}: {mode_name} arm's memory plan was pinned to the floor (policy={plan.get('policy')} reason={plan.get('reason')}); the mode's plan fills the budget")
        entry["trunk_graph_aside"] = None
        if pool_wanted:
            if pool is None:
                rep["refusals"].append(f"F8 {name}: no trunk graph pool installed ({mode_name}, {toks} tokens <= {FK.POOL_MAX_TOKENS})")
            else:
                if not pool.get("active", True):                          # no word of the pool's own explains a rebound forward: refused
                    rep["refusals"].append(f"F8 {name}: trunk graph pool bypassed (the trunk forward was rebound after the pool installed)")
                words = []                                                # the pool's declared words: a capture that failed disabled it (disabled_reason), passes it ran eagerly and counted (eager), a slot re-captured / never captured after a step (recaptures, slots) — the trunk ran stock's eager path there, by name
                if pool.get("disabled_reason"):
                    words.append("disabled:" + str(pool["disabled_reason"]).split(":")[0].strip().replace(" ", "_"))
                if pool.get("eager", 0):
                    words.append(f"eager:{pool['eager']}")
                if pool.get("steps", 0) and (pool.get("slots") != pool.get("n_slots") or pool.get("recaptures", 0)):     # a model the loop never picked has nothing captured, by design
                    words.append(f"slots:{pool.get('slots')}/{pool.get('n_slots')}" + (f",recaptures:{pool['recaptures']}" if pool.get("recaptures", 0) else ""))
                if words:
                    entry["trunk_graph_aside"] = stand_aside("ef2_stepgraph", "F8", ",".join(words),
                                                             f"{name}: trunk graph pool {('disabled itself (' + str(pool['disabled_reason']) + '); ') if pool.get('disabled_reason') else ''}"
                                                             f"{pool.get('eager', 0)} grad-mode pass(es) ran the trunk eagerly, {pool.get('slots')}/{pool.get('n_slots')} slots after {pool.get('steps', 0)} step(s), "
                                                             f"recaptures {pool.get('recaptures', 0)} — the stock eager trunk served those passes")
        elif pool is not None:
            rep["refusals"].append(f"{name}: {mode_name} arm at {toks} tokens installed a trunk graph pool ({pool.get('slots')}/{pool.get('n_slots')} slots); the mode's rule is none{' above %d tokens' % FK.POOL_MAX_TOKENS if comp['pool'] else ''}")
        if entry["bf16_confidence"] != bool(comp["bf16_confidence"] and on("ef2_bf16_confidence")):
            rep["refusals"].append(f"{name}: bf16 confidence head {'installed' if entry['bf16_confidence'] else 'absent'} on a {mode_name} arm")
        if not entry["skip_unused_confidence"] and on("skip_unused_confidence"):
            rep["refusals"].append(f"{name}: skip-unused-confidence not installed")
        if entry["lazy_structure"] is None and on("ef2_lazy_structure"):
            rep["refusals"].append(f"{name}: ef2_lazy_structure not installed")
        sg = entry["sampler_graph"]
        if (sg is None and on("ef2_sampler_graph")) or (entry["pairbias_attn"] is None and on("ef2_pairbias_attn")):
            rep["refusals"].append(f"{name}: sampler levers not installed (sampler_graph {sg is not None}, pairbias_attn {entry['pairbias_attn'] is not None})")
        elif sg is not None and (sg.get("capture_failed", 0) or sg.get("mismatch", 0)):   # the sampler graph's declared words: a capture that failed (after its retry) / a replay that did not match — those samples ran the stock eager sampler, counted
            entry["sampler_aside"] = stand_aside("ef2_sampler_graph", "F11", ",".join(f"{k}:{sg[k]}" for k in ("capture_failed", "mismatch") if sg.get(k)),
                                                 f"{name}: sampler graph capture_failed={sg.get('capture_failed')} mismatch={sg.get('mismatch')} — the stock eager sampler served those samples"
                                                 + (f" (last_failure={sg['last_failure']})" if sg.get("last_failure") else ""))
        elif sg is not None and sg.get("capture_retries", 0):
            rep["events"].append(f"F11 {name}: sampler graph capture_retries={sg['capture_retries']} — a capture that ran out of device memory was retried after empty_cache and captured (k/ef2_sampler_graph.py)")
    rep["models"] = per_model
    m0 = per_model[0] if per_model else {}
    # ---- the LEVER census (one record per lever of the composition; the first inversion model speaks for the per-model levers)
    plan = m0.get("bwd_ckpt") or {}
    plan_row = (f"{plan.get('kernels')}(unchunked-priced)" if plan and comp["chunk"] == FK.CHUNK_PINNED and isinstance(user_chunk, int) and not isinstance(user_chunk, bool) else None)   # exact under a user --chunk-size N: the grad-mode pair stack runs stock's chunked path, the plan keeps its unchunked estimate (an upper bound; no coefficient invented)
    L.append(lever_record("ef2_bwd_ckpt", "on" if plan else "off", None, None, policy=plan.get("policy"), reason=plan.get("reason"), kept_per_pass=(f"{plan.get('kept_per_pass')}/{plan.get('n_blocks')}" if plan else None),   # reason: the floor plan's word (memory_floor, big); absent under the budget rule (None is not printed)
                          kernels=plan.get("kernels"), copies=plan.get("copies"), est_peak_gib=plan.get("est_peak_gib"), budget_gib=plan.get("budget_gib"), base_gib=plan.get("base_gib"), plan_row=plan_row))
    ks = None
    if counter is not None:
        cfg0 = getattr(models[0], "_agk_cfg", None) if models else None
        ks = counter.summary(getattr(cfg0, "trimul", None), getattr(cfg0, "transition", None), [m.__dict__.get("_ef2_trimul_handle") for m in models if hasattr(m, "__dict__")],
                             nosave_handles=[m.__dict__.get("_ef2_nosave_handle") for m in models if hasattr(m, "__dict__")])
        rep["kernels"] = ks
    th = m0.get("trimul")                      # the triangle multiplication: ef2_trimul's handle (cueq_tiles: a static install; fused: its own counter) or agk's K-A2 (the F9 counter)
    tm_kernel = m0.get("trimul_kernel", "stock")
    tm_served = (th.get("served") if th and th["variant"] == "fused" else ((ks["kernel_calls"].get("trimul_with_residual_frozen") if ks else None) if tm_kernel == "bmm2" else None))
    L.append(lever_record("trimul", "on" if tm_kernel != "stock" else "off", tm_served, ({"stock_path": th["fallback"]} if th and th.get("fallback") else None),
                          kernel=tm_kernel, entries=((th or {}).get("entries") if th and th["variant"] == "cueq_tiles" else None),
                          reason=((th or {}).get("reason") if th and th["variant"] == "cueq_tiles" else (backend_word if tm_kernel == "stock" and comp["trimul"] == "cueq_tiles" else None)),    # a card without a tile entry: entries=0 reason=cc_untuned:sm_NN (the library default runs; named, not refused); exact under a user backend: kernel=stock reason=user_kernel_backend:<v> (the user's backend serves the triangle multiplication, as on the stock arm with that flag)
                          gemm=((th or {}).get("gemm") if th and th["variant"] == "fused" else None)))             # fused: the gated-GEMM launch table in effect (sm_90) or vendored:cc_untuned:sm_NN — the vendored launches, named
    if comp.get("nosave_fwd"):                                             # fast / big: the checkpointed blocks' no-save first pass on the provider's forward-only row
        ns = m0.get("nosave")
        if ns is None:
            L.append(lever_record("ef2_trimul_nosave", "off", None, None, reason=(aside.get("ef2_trimul_nosave") or None)))
            if on("ef2_trimul_nosave"):
                rep["refusals"].append(f"ef2_trimul_nosave: {mode_name} arm without the no-save first pass installed on the inversion models (k/ef2_trimul_nosave.py enable never ran)")
        elif not ns["on"]:                                                   # stepped aside by name at enable (opt_core absent / too old, sm_80: cc_no_gain, no row served the canary): an event, the word on the line
            L.append(lever_record("ef2_trimul_nosave", "stepped_aside", None, None, word=ns["word"], reason=ns["reason"], refused=ns["refused"], tier_fast=ns.get("tier_fast")))
            rep["events"].append(f"F11 ef2_trimul_nosave stepped aside at enable: {ns['reason']} (torch's checkpoint runs both passes on the grad-path kernels)")
        else:
            L.append(lever_record("ef2_trimul_nosave", "on", ns.get("served"), ({"inner": ns["inner"]} if ns.get("inner") else None), word=ns["word"], row=ns["row"],
                                  first_pass=ns.get("first_pass"), recompute=ns.get("recompute"), refused_calls=(ns.get("refused_calls") or None), refused=ns["refused"], canary=ns["canary"],
                                  tier_fast=ns.get("tier_fast")))   # what the provider's tier word resolves to on this stack at the design's size, next to the row asked by name (k/ef2_trimul_nosave.py: The tier word)
            if ns.get("refused_calls"):
                rep["events"].append(f"F11 ef2_trimul_nosave: {ns['refused_calls']} no-save call(s) refused by the row at call time ({ns['refused']}) and served by the grad-path forward")
    # the process-wide tile table of the stock cuEquivariance kernels: the handle lives on the first fold model that runs the backend (exact: an
    # inversion model; fast / big: a hero critic) — expected on every kit arm whose critics (or pinned inversion models) fold on cuequivariance
    holders = [(f"inversion_models.{n}", m) for n, m in getattr(app, "inversion_models", {}).items()] + [(lbl, fm) for lbl, fm in FK.fold_models(app, labelled=True) if all(fm is not m for m in models)]
    cqs = [(lbl, h) for lbl, h in ((lbl, getattr(fm, "__dict__", {}).get("_ef2_trimul_handle")) for lbl, fm in holders) if h is not None and getattr(h, "variant", None) == "cueq_tiles"]
    cq_label, cq = cqs[0] if cqs else (None, None)
    cq_user = cq is None and cueq_user_off and on("cueq_tiles")             # the user's --kernel-backend left no model on cuequivariance: stood aside by name at enable (fastkit._enable_cueq_tiles), worded here
    L.append(lever_record("cueq_tiles", "on" if cq is not None else ("stepped_aside" if cq_user else "off"), None, None, entries=(len(getattr(cq, "cueq_saved", {}) or {}) if cq is not None else None), holder=cq_label,
                          models=(len(cqs) or None), reason=((backend_word if cq_user else aside.get("cueq_tiles")) if cq is None else (getattr(cq, "cueq_reason", "") or None))))   # a card without a tile entry: entries=0 reason=cc_untuned:sm_NN (named, not refused)
    if cq is None and cueq_expected:
        rep["refusals"].append(f"cueq_tiles: {mode_name} arm without the cuEquivariance tile table on any fold model ({aside.get('cueq_tiles') or 'ef2_trimul cueq_tiles not installed'})")
    elif cq_user:
        rep["events"].append(f"cueq_tiles stepped aside by name: the user's --kernel-backend {S.kernel_backend_word(user_backend)} is every model's, no triangle multiplication runs the cuequivariance kernels ({aside.get('cueq_tiles') or backend_word})")
    elif cq is None and on("cueq_tiles"):
        rep["events"].append(f"cueq_tiles stepped aside: no fold model runs the cuequivariance backend on this arm (hero critics kernel_backend={crit.get('kernel_backend')}): {aside.get('cueq_tiles') or 'not installed'}")
    if user_chunk != S.CHUNK_SHIPPED:                                        # the user's --chunk-size on every model (stock's setter at load): fast / big's 64 yielded to it by name at enable, exact carries it as the stock arm would — read back from the first inversion model
        L.append(lever_record("chunk", "user", None, None, size=S.chunk_size_word(m0.get("chunk_size")), models=len(models) or None,
                              yielded=(S.chunk_size_word(comp["chunk"] if on("chunk") else None) if comp["chunk"] != FK.CHUNK_PINNED else None)))   # LEVER name=chunk state=user size=<none|N> models=<n> [yielded=64]: the user's value read back, and the kit's value that yielded to it
        rep["events"].append(f"chunk: the user's --chunk-size {S.chunk_size_word(user_chunk)} is every model's" + (f"; the kit's {S.chunk_size_word(comp['chunk'] if on('chunk') else None)} yielded to it by name ({aside.get('chunk') or 'fastkit.enable'})" if comp["chunk"] != FK.CHUNK_PINNED else " (exact: as on the stock arm with that flag)"))
    elif comp["chunk"] != FK.CHUNK_PINNED:                                  # fast / big: the pair-stack chunk the kit set on the inversion models (exact: the mode table's switch, LEVER critic_switches / the argv)
        L.append(lever_record("chunk", "on" if m0.get("chunk_size") is not None else "off", None, None, size=m0.get("chunk_size"), models=len(models) or None))
    tr_kernel = comp["transition"] if on("agk_transition") else None
    L.append(lever_record("agk_transition", "on" if tr_kernel else "off", (ks["kernel_calls"].get("transition_refround") if ks else None), None, kernel=tr_kernel))
    # the carried one-kernel transition FORWARD K-D3 lean runs on (fast / big, compute capability 9.0): on = served this arm's grad-mode transitions;
    # stepped_aside = this card / stack cannot run it, K-D3's own forward served (named, not refused); off = not asked (exact; or ablated: the loop below says `ablated`)
    t16_asked = comp.get("transition_fwd") == "t16" and bool(tr_kernel) and on("ef2_t16_transition")
    t16 = None
    if t16m is not None and t16_asked:
        try:
            t16 = t16m.describe()
        except Exception:                                                    # noqa: BLE001 — a census reader
            t16 = None
    rep["t16_transition"] = dict(t16) if t16 else None
    if t16_asked:
        kw = dict((t16 or {}).get("kernel") or {})
        L.append(lever_record("ef2_t16_transition", (t16 or {}).get("state", "off"), (t16 or {}).get("served"), (t16 or {}).get("fallback") or None,
                              kernel=(t16 or {}).get("version"), reason=(t16 or {}).get("reason"), device=(t16 or {}).get("device"),
                              cubin=kw.get("cubin"), regs=kw.get("regs"), smem=kw.get("smem"), arch=kw.get("arch"), canary=kw.get("canary")))
        n_tr = (ks["kernel_calls"].get("transition_refround") if ks else None)
        if t16 is None:
            rep["refusals"].append("F9 ef2_t16_transition: the composition asks the t16 transition forward and the module is not importable / never engaged")
        elif t16.get("state") == "on":
            if n_tr is not None and int(t16.get("served") or 0) != int(n_tr):
                rep["refusals"].append(f"F9 ef2_t16_transition live but served {t16.get('served')} of {n_tr} grad-mode transition calls (fall-through {t16.get('fallback') or 'none'})")
        elif t16.get("state") == "stepped_aside":
            rep["events"].append(f"ef2_t16_transition stepped aside on this card / stack ({t16.get('reason')}: {t16.get('reason_text')}); K-D3's own forward served the pair transition")
        else:
            rep["refusals"].append("F9 ef2_t16_transition: the composition asks the t16 transition forward and it was never engaged (state off)")
    else:
        L.append(lever_record("ef2_t16_transition", "off"))
    # K-D3 lean's OWN forward with its W12 projection + SwiGLU as one kernel (fast / big, where that forward serves: a card with an entry, sm_80): on = served
    # this arm's transitions (counted against agk's entry point like t16); off reason=t16_serves = not asked on this card (the t16 kernel runs K-D3's forward:
    # sm_90); stepped_aside = no entry for this card / stack, K-D3's cuBLAS projection + SwiGLU kernel served (named, not refused); off = not asked (exact; ablated: `ablated`)
    gsw_asked = tr_kernel == "refround_lean" and on("ef2_kd3_gemmswiglu")
    gsw = None
    if gsm is not None and gsw_asked:
        try:
            gsw = gsm.describe()
        except Exception:                                                    # noqa: BLE001 — a census reader
            gsw = None
    rep["kd3_gemmswiglu"] = dict(gsw) if gsw else None
    t16_serves = bool(t16_asked and t16 and t16.get("state") == "on") or aside.get("ef2_kd3_gemmswiglu") == "t16_serves"   # the t16 kernel runs K-D3's forward on this card: the fused W12 + SwiGLU forward is not asked (named)
    if gsw_asked and t16_serves:
        L.append(lever_record("ef2_kd3_gemmswiglu", "off", None, None, reason="t16_serves"))
    elif gsw_asked:
        L.append(lever_record("ef2_kd3_gemmswiglu", (gsw or {}).get("state", "off"), (gsw or {}).get("served"), (gsw or {}).get("fallback") or None,
                              kernel=((gsw or {}).get("version") if (gsw or {}).get("state") == "on" else None), tile=(gsw or {}).get("tile"), reason=(gsw or {}).get("reason"), cc=(gsw or {}).get("cc")))
        n_tr = (ks["kernel_calls"].get("transition_refround") if ks else None)
        if gsw is None:
            rep["refusals"].append("F9 ef2_kd3_gemmswiglu: the composition asks K-D3's fused W12 + SwiGLU forward and the module is not importable / never engaged")
        elif gsw.get("state") == "on":
            if n_tr is not None and int(gsw.get("served") or 0) != int(n_tr):
                rep["refusals"].append(f"F9 ef2_kd3_gemmswiglu live but served {gsw.get('served')} of {n_tr} grad-mode transition calls (fall-through {gsw.get('fallback') or 'none'})")
        elif gsw.get("state") == "stepped_aside":
            rep["events"].append(f"ef2_kd3_gemmswiglu stepped aside on this card / stack ({gsw.get('reason')}: {gsw.get('reason_text')}); K-D3's cuBLAS W12 projection + SwiGLU kernel served")
        else:
            rep["refusals"].append("F9 ef2_kd3_gemmswiglu: the composition asks K-D3's fused W12 + SwiGLU forward and it was never engaged (state off)")
    else:
        L.append(lever_record("ef2_kd3_gemmswiglu", "off"))
    fl = flm.stats() if flm is not None else None
    L.append(lever_record("ef2_fused_ln", "on" if fl and fl.get("installed") else "off", (fl or {}).get("served"), _fb_words(fl), channel_major=(fl or {}).get("channel_major")))
    pool = m0.get("trunk_graph_pool"); pool_aside = m0.get("trunk_graph_aside")
    sg_words: Dict[str, Any] = {}                                            # a pool that disabled itself / ran passes eagerly BY ITS WORD: state=on … aside=<word> (it served the rest) or state=stepped_aside reason=<word> (it never replayed)
    if pool and pool_aside:
        sg_words["aside" if pool.get("replays_fwd") else "reason"] = pool_aside
    L.append(lever_record("ef2_stepgraph", ("off" if not pool else ("on" if (pool.get("replays_fwd") or not pool_aside) else "stepped_aside")), (pool or {}).get("replays_fwd"), ({"eager": pool["eager"]} if pool and pool.get("eager") else None),
                          slots=(f"{pool.get('slots')}/{pool.get('n_slots')}" if pool else None), captures=(pool or {}).get("captures"), pool_gib=(round(pool["pool_bytes"] / 2 ** 30, 2) if pool else None),
                          rule=f"<={FK.POOL_MAX_TOKENS}" if comp["pool"] else "never", **sg_words))
    L.append(lever_record("skip_unused_confidence", "on" if m0.get("skip_unused_confidence") else "off"))
    lz = m0.get("lazy_structure")
    L.append(lever_record("ef2_lazy_structure", "on" if lz is not None else "off", (lz or {}).get("deferred"), ({"eager": lz["eager"]} if lz and lz.get("eager") else None), materialized=(lz or {}).get("materialized")))
    L.append(lever_record("ef2_bf16_confidence", "on" if m0.get("bf16_confidence") else "off"))
    pb = m0.get("pairbias_attn")
    L.append(lever_record("ef2_pairbias_attn", "on" if pb else "off", (pb or {}).get("served"), _fb_words(pb), variant=(pb or {}).get("variant"), computed=(pb or {}).get("computed")))
    sg = m0.get("sampler_graph"); sg_aside = m0.get("sampler_aside")
    L.append(lever_record("ef2_sampler_graph", ("off" if not sg else ("on" if (sg.get("replays") or not sg_aside) else "stepped_aside")), (sg or {}).get("replays"), ({k: sg[k] for k in ("eager", "capture_failed", "capture_retries", "mismatch") if sg.get(k)} if sg else None),
                          captures=(sg or {}).get("captures"), samples=(sg or {}).get("samples"), **({("aside" if sg.get("replays") else "reason"): sg_aside} if sg and sg_aside else {})))
    # F2: the shared ESMC graph (one wrapper on the shared _esmc)
    eegs = [getattr(getattr(m, "_esmc", None), "_eeg", None) for m in models]
    eeg = next((g for g in eegs if g is not None), None)
    eeg_aside = None
    if eeg is None:
        if on("ef2_esmc_graph"):
            rep["refusals"].append("F2 ESMC-6B forward graph not installed")
    else:
        rep["esmc_graph"] = dict(eeg.stats)
        if eeg.stats.get("eager", 0) > 0:                                   # the graph's declared word: calls outside its graphable form ran the stock forward, counted (k/ef2_esmc_graph.py:32-34)
            eeg_aside = stand_aside("ef2_esmc_graph", "F2", f"eager:{eeg.stats['eager']}", f"the ESMC forward ran the stock eager path {eeg.stats['eager']} time(s) — calls outside the graphable form (k/ef2_esmc_graph.py:32-34)")
        if eeg.stats.get("replays", 0) == 0 and not eeg_aside:             # never replayed and no word says why: refused
            rep["refusals"].append("F2 ESMC forward graph never replayed")
    L.append(lever_record("ef2_esmc_graph", ("off" if eeg is None else ("on" if (eeg.stats.get("replays", 0) or not eeg_aside) else "stepped_aside")), (rep.get("esmc_graph") or {}).get("replays"),
                          ({"eager": rep["esmc_graph"]["eager"]} if eeg is not None and rep["esmc_graph"].get("eager") else None),
                          captures=(rep.get("esmc_graph") or {}).get("captures"), **({("aside" if eeg.stats.get("replays", 0) else "reason"): eeg_aside} if eeg_aside else {})))
    # F11: the target-chain hoist over the shared ESMC forward (served = hoisted feature passes, full = calls that ran every row: the first of a
    # design, a changed target); a failed anchor, a failed capture or a trunk form it does not compute switch it off by name -> refused
    hst = None
    if hsm is not None and models:
        try:
            hst = hsm.stats(models[0])
        except Exception:                                                    # noqa: BLE001 — a census reader
            hst = None
    rep["esmc_hoist"] = dict(hst) if hst else None
    hoist_aside = (hst or {}).get("stepped_aside")                           # a compute capability outside the lever's PROVEN_CC: not installed, by name, before the loop — a word, never a refusal
    hw = (lambda k: None) if hoist_aside else (lambda k: (hst or {}).get(k))   # stepped aside: no per-call words (nothing ran)
    run_aside = None                                                          # the hoist's run-time words: it switched itself off (a failed anchor / capture, a trunk form it does not compute: `off`), or never served — the full forward served, by name
    if not hst:
        if on("ef2_esmc_hoist"):
            rep["refusals"].append("F11 ef2_esmc_hoist not installed on the shared ESMC trunk")
    elif hoist_aside:
        rep["events"].append(f"F11 ef2_esmc_hoist stepped aside by name at enable: {hoist_aside} (the trunk's forward untouched; its reduced pass is proven bitwise on {', '.join('sm_%d%d' % c for c in sorted(getattr(hsm, 'PROVEN_CC', {}) or {})) or '?'} only)")
    elif hst.get("off"):
        run_aside = stand_aside("ef2_esmc_hoist", "F11", "off:" + str(hst["off"]).replace(" ", "_"), f"the hoist switched itself off ({hst['off']}; inner calls {_fb_words(hst) or 'none'}) — the trunk's full forward served from then on")
    elif n_steps is not None and n_steps >= 2 and not hst.get("served"):
        run_aside = stand_aside("ef2_esmc_hoist", "F11", "never_served", f"inert in a {n_steps}-step design (full={hst.get('full')}, inner calls {_fb_words(hst) or 'none'}) — the trunk's full forward served every call")
    else:
        forms = {k: v for k, v in _fb_words(hst).items() if k in ("grad", "form", "layout", "capturing")}
        if forms:
            rep["events"].append(f"F11 ef2_esmc_hoist left these calls to the inner forward (input form): {forms}")
    L.append(lever_record("ef2_esmc_hoist", "off" if not hst else ("stepped_aside" if (hoist_aside or (run_aside and not hst.get("served"))) else "on"), (hst or {}).get("served"), _fb_words(hst),
                          reason=(hoist_aside or (run_aside if run_aside and not hst.get("served") else None)), aside=(run_aside if run_aside and hst.get("served") else None),
                          full=hw("full"), anchored=hw("anchored"), captures=hw("captures"), live=hw("live"), cc=(hst or {}).get("cc")))
    # F1: the pPPL graph on the LM
    st = None; st_aside = None
    if pppl_stats_fn is not None:
        st = pppl_stats_fn(app.esmc_model)
        rep["pppl_graph"] = st
        if st is None:
            if on("ef2_pppl_graph"):
                rep["refusals"].append("F1 pPPL graph not installed (EF2_FAST_KIT_PPPL default '1' expected)")
        else:
            # the graph's bookkeeping against the run's own: replays == design steps x the loss's LM calls per step (the B x LM_MASK_PASSES masked
            # rows go through the stack in chunks of LM_LOSS_BATCH_SIZE, binder_design.py l.919-931: one call per step up to B = 32); captures ==
            # the distinct chunk shapes of one step (1; 2 when a last partial chunk exists); eager == the calls that reached the wrapped stack
            # OUTSIDE the loss — exactly the ESMC forward graph's own capture runs through the shared trunk (n_warmup + the captured call per
            # capture, k/ef2_esmc_graph.py:44-50; every later feature pass replays that graph and never re-enters the stack). More eager calls
            # than that = a loss call fell through without a word: refused. A graph that disabled itself SAID so (disabled_reason): it stood aside.
            B = max(int(batch_size or 1), 1)
            n_passes = int(getattr(BD, "LM_MASK_PASSES", 4) or 4); rows_per_call = int(getattr(BD, "LM_LOSS_BATCH_SIZE", 128) or 128)
            rows = B * n_passes
            calls_per_step = -(-rows // rows_per_call)
            shapes = 1 if (rows <= rows_per_call or rows % rows_per_call == 0) else 2
            exp_replays = n_steps * calls_per_step if n_steps is not None else None
            exp_eager = (int(getattr(eeg, "n_warmup", ESMC_GRAPH_WARMUPS)) + 1) * int(eeg.stats.get("captures", 0) or 0) if eeg is not None else None
            if st.get("disabled_reason"):                                  # the graph's declared word (k/ef2_pppl_graph.py:63-67: a capture that raised, a replay that did not match its eager run): the loss ran the stock eager stack from then on
                st_aside = stand_aside("ef2_pppl_graph", "F1", ("replay_fail:%d" % st["replay_fail"]) if st.get("replay_fail", 0) else "disabled:" + str(st["disabled_reason"]).split(":")[0].strip().replace(" ", "_"),
                                       f"pPPL graph disabled itself: {st['disabled_reason']} — the loss ran the stock eager stack from then on (captures={st.get('captures')} replays={st.get('replays')} eager={st.get('eager')})")
            else:
                if exp_replays is not None and st.get("replays", 0) != exp_replays:
                    rep["refusals"].append(f"F1 pPPL graph replays {st.get('replays')} != {exp_replays} ({n_steps} design steps x {calls_per_step} LM call(s) per step: B={B} x {n_passes} masked passes in chunks of {rows_per_call})")
                if st.get("captures", 0) != shapes:
                    rep["refusals"].append(f"F1 pPPL graph captures {st.get('captures')} != {shapes} (the LM input shape(s) of one step: B={B} x {n_passes} masked passes in chunks of {rows_per_call}; one trajectory batch per process)")
                if exp_eager is not None and st.get("eager", 0) != exp_eager:
                    rep["refusals"].append(f"F1 pPPL graph eager calls {st.get('eager')} != {exp_eager} (the ESMC forward graph's {eeg.stats.get('captures')} capture(s) x {int(getattr(eeg, 'n_warmup', ESMC_GRAPH_WARMUPS)) + 1} runs through the shared trunk; more = a loss call fell through)")
            rep["events"].append(f"F1 pPPL graph captures={st.get('captures')} replays={st.get('replays')} eager={st.get('eager')} (expected {shapes} / {exp_replays} / {exp_eager if exp_eager is not None else 'unjudged: no ESMC forward graph'}: B={B}, eager = the ESMC graph's capture runs by design)")
    L.append(lever_record("ef2_pppl_graph", ("off" if not st else ("on" if (st.get("replays") or not st_aside) else "stepped_aside")), (st or {}).get("replays"), None, captures=(st or {}).get("captures"),
                          **({("aside" if st.get("replays") else "reason"): st_aside} if st and st_aside else {})))
    # the loop-level levers (F11: a lever installed by the composition that could not apply is refused by name; by-design counters are words)
    rope = [getattr(rpm.handle(h), "__dict__", {}) for h in FK.esmc_trunks(app)] if rpm is not None else []
    rope = [r for r in rope if r.get("stats") is not None]
    r_served = sum(int(r["stats"].get("served", 0)) for r in rope); r_fb: Dict[str, int] = {}
    for r in rope:
        for k, v in _fb_words(r["stats"]).items():
            r_fb[k] = r_fb.get(k, 0) + v
    L.append(lever_record("ef2_esmc_rope", "on" if rope else "off", r_served, r_fb, trunks=len(rope) or None, modules=sum(len(r.get("modules") or []) for r in rope) or None))
    if not rope:
        if on("ef2_esmc_rope"):
            rep["refusals"].append("F11 ef2_esmc_rope not installed on any ESMC trunk")
    elif r_served == 0:                                                          # the stock flash-attn rotary bound (no upstream fix) or a module form the lever names: stock as shipped, said once
        rep["events"].append(f"F11 ef2_esmc_rope installed on {len(rope)} trunk(s) and inert: every call took the stock branch ({r_fb or 'no calls'})")
    prep = lpm.stats(BD) if lpm is not None else {}
    prep_words: List[str] = []                                                   # the lever's declared words: each names a part that switched ITSELF off and the stock path that served instead — asides, never quiet, never a refusal
    if not prep:
        if on("ef2_loop_prep"):
            rep["refusals"].append("F11 ef2_loop_prep not installed on the cookbook module")
    else:
        defects = {k: prep.get(k) for k in ("mismatch", "fallback_anchor", "fallback_signature") if prep.get(k)}
        if defects:                                                              # the lever's own hidden states / featurisation disagreed with the model's, or the cookbook's signature moved: it said so and the model's own LM path / featuriser served
            prep_words.append(stand_aside("ef2_loop_prep", "F11", ",".join(f"{k.replace('fallback_', '')}:{v}" for k, v in defects.items()), f"switched its feature pass off: {defects} — the model's own LM path and featuriser served from then on"))
        forms = {k: prep.get(k) for k in ("fallback_no_esmc", "fallback_lm_mask") if prep.get(k)}
        if forms:
            rep["events"].append(f"F11 ef2_loop_prep left these calls to the model's own LM path (input form): {forms}")
        if prep.get("splice_off", 0):
            prep_words.append(stand_aside("ef2_loop_prep", "F11", f"splice_off:{prep['splice_off']}", "the target featurisation splice switched itself off after a failed field comparison — the full featuriser served from then on"))
        if prep.get("pin_failed"):
            prep_words.append(stand_aside("ef2_loop_prep", "F11", "pin_failed", f"pinned staging refused by the driver ({prep['pin_failed']}): {prep.get('fallback_pageable')} pageable uploads instead"))
    L.append(lever_record("ef2_loop_prep", ("off" if not prep else ("on" if (prep.get("served") or not prep_words) else "stepped_aside")), prep.get("served"), _fb_words(prep), early=prep.get("early"), late=prep.get("late"), memo_hits=prep.get("memo_hits"), pinned=prep.get("pinned"),
                          anchored=prep.get("anchored"), mismatch=prep.get("mismatch"), spliced=prep.get("spliced"), splice_off=prep.get("splice_off"),
                          **({("aside" if prep.get("served") else "reason"): ",".join(prep_words)} if prep_words else {})))
    pq = ppm.stats(BD) if ppm is not None else {}
    pq_on = bool(pq) and getattr(BD, "_ef2_loop_pppl", None) is not None
    L.append(lever_record("ef2_loop_pppl", "on" if pq_on else "off", pq.get("served") if pq_on else None, None, checked=pq.get("checked") if pq_on else None, first_read=pq.get("first_read") if pq_on else None))
    if not pq_on:
        if on("ef2_loop_pppl"):
            rep["refusals"].append("F11 ef2_loop_pppl not installed on the cookbook module")
    ov = getattr(BD, "_ef2_esmc_overlap", None)
    ovs = dict(ov.stats) if ov is not None else {}
    ov_aside = None
    if ov is None:
        if on("ef2_esmc_overlap"):
            rep["refusals"].append("F11 ef2_esmc_overlap not installed on the cookbook module")
    elif any(str(k).startswith("early_failed") for k in (ovs.get("fallback") or {})):   # its declared word: the early launch raised, the loss ran in the loop's own order from then on
        early = {k: v for k, v in (ovs.get("fallback") or {}).items() if str(k).startswith("early_failed")}
        ov_aside = stand_aside("ef2_esmc_overlap", "F11", ",".join(f"{k}:{v}" for k, v in early.items()).replace(" ", "_"), f"the early launch of the pseudo-perplexity term failed ({early}) — the loss ran in the loop's own order from then on")
    L.append(lever_record("ef2_esmc_overlap", ("off" if ov is None else ("on" if (ovs.get("served") or not ov_aside) else "stepped_aside")), ovs.get("served"), _fb_words(ovs), computed=ovs.get("computed"),
                          **({("aside" if ovs.get("served") else "reason"): ov_aside} if ov_aside else {})))
    L.append(lever_record("featurisation_cache", "on" if getattr(BD, "_FEATURE_CACHE", None) is not None else "off", None, None, entries=len(getattr(BD, "_FEATURE_CACHE", {}) or {})))
    L.append(critic_census(app, critic_switches, rep, mode_name))          # the hero critics' chunk / kernel backend after the loop (fast / big: the stock arm's pair; refused by name if a critic reads otherwise)
    for rec in L:                                                           # the ablation word: a lever switched off by name reads state=ablated (never `off`: that word is a lever that did not engage)
        if rec["name"] in off:
            if rec["state"] == "on":
                rep["refusals"].append(f"{rec['name']}: ablated by {FK.LEVERS_OFF_VAR} yet engaged after the loop ({rec})")
            rec["state"] = "ablated"
        elif rec["name"] in moff:                                           # a lever the composition switches off by name (big's mode_off): state=off reason=mode:<mode>; engaged all the same = not the composition, refused by name
            if rec["state"] != "off":
                rep["refusals"].append(f"{rec['name']}: switched off by the {mode_name} composition (fastkit.COMPOSITION mode_off) yet {rec['state']} after the loop ({rec})")
            rec["state"] = "off"; rec["extra"] = {**(rec.get("extra") or {}), "reason": f"mode:{mode_name}"}
    # F3 after the loop (fast / big): the lazily-probed flag
    if agk is not None and comp["transition"] and on("agk_transition"):
        rep["bmm_out_dtype"] = getattr(agk, "_HAS_BMM_OUT_DTYPE", None)
        cfg0 = getattr(models[0], "_agk_cfg", None) if models else None
        eo = getattr(cfg0, "einsum_out_dtype", None)
        if eo == "fp32":                       # the only composition whose contraction asks for an fp32 out_dtype (k/ef2_autograd_kernels.py:482)
            if rep["bmm_out_dtype"] is not True:
                rep["refusals"].append(f"F3 _HAS_BMM_OUT_DTYPE is {rep['bmm_out_dtype']!r} after the loop (k/ef2_autograd_kernels.py:72-86)")
        else:
            rep["events"].append(f"F3 not exercised: einsum_out_dtype={eo!r} (the default bf16: _bmm never asks for an out_dtype); flag={rep['bmm_out_dtype']!r}")
    # F7: one complex size per process
    rep["fast_kit_tokens"] = toks
    if n_tokens is not None and toks != n_tokens:
        rep["refusals"].append(f"F7 app._fast_kit_tokens={toks!r} != n_tokens={n_tokens} (a second complex size in this process)")
    # F9: kernel accounting
    if ks is not None:
        if ((comp["transition"] and on("agk_transition")) or (comp["trimul"] in ("bmm2", "fused") and on("trimul"))) and not ks["applied"]:
            rep["refusals"].append(f"F9 kernel calls {ks['kernel_calls']} != expected {ks['expected']} (grad-mode bf16 block calls {ks['block_calls']['grad_bf16']})")
        if not comp["transition"] and comp["trimul"] == "cueq_tiles" and any(v for v in ks["kernel_calls"].values()):
            rep["refusals"].append(f"F9 exact arm engaged fast-class kernels: {ks['kernel_calls']}")
        rep["events"].append(f"F9 no-grad / non-bf16 block calls on the stock path by design: {ks['block_calls']['other']}")
    # F10: the guard
    g = getattr(BD, "_D59_GUARD", None)
    try:
        rep["d59_guard"] = g.summary() if g is not None else None
    except Exception as e:
        rep["d59_guard"] = f"summary failed: {e}"
    if g is None:
        rep["refusals"].append("F10 no D59 state guard was created (the kit did not enable)")
    names: List[str] = []
    for a_ in rep["asides"]:                                                 # one entry per lever that stood aside by its word, in census order (the EVIDENCE line's fallback= names them; opt_manifest.json `fallback` carries the sentences)
        if a_["name"] not in names:
            names.append(a_["name"])
    rep["fallback"] = ([r for r in rep["refusals"] if r.startswith(("F1 ", "F2 ", "F3 ", "F5 ", "F7 ", "F8 ", "F9 ", "F11 "))]
                       + [f"{n} stood aside by its declared word: " + "; ".join(f"{a_['reason']} ({a_['text']})" for a_ in rep["asides"] if a_["name"] == n) for n in names])
    rep["applied"] = not rep["refusals"]                                     # asides complete the run (exit 0, EVIDENCE levers=applied fallback=<names>); refusals do not (exit 3)
    return rep
