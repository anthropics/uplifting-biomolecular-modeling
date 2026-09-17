"""evo2_opt.gen.specdec — mode fast's generate(): speculative sampling for Evo 2 with the `evo2_1b_base` draft (Leviathan et al. 2023; Chen et
al. 2023). Per round the draft proposes k tokens with single-token cached steps; the target checks them in ONE cached (1, k+1) call — its Hyena
layers step the stock sequential update once per position, its dense and attention layers run over the k+1 positions at once — and p is the
target's transformed distribution (vortex's top_k / top_p / temperature transform) from THAT call; the rule of `rule.py` accepts drafts while
u·q < p, draws the first rejected position from norm(max(p − q, 0)) and the bonus token from the next row when all k pass. Every emitted token
is an exact sample from p at its position; the sequence for a seed is not the stock sampler's sequence. Greedy (top_k=1): drafts accepted while
they are the argmax, so the tokens are the argmax path of the target calls.

Installed on the target instance by `install(evo2_model)` (mode fast's last generation member, after hyenafuse and cudagraph): the draft is
located beside the target's checkpoint (`<dir of local_path>/evo2_1b_base.pt`, else `$EVO2_OPT_WEIGHTS/evo2_1b_base.pt`, else upstream's own
Hugging Face download when the target itself was constructed without a local_path), its sha256 checked against `stock/PINS.json
checkpoints.evo2_1b_base` (a mismatch is one UNPINNED line: the acceptance rate depends on the draft, the sampled distribution does not), built
through the stock constructor on the target's first device, and given the generation members hyenafuse and cudagraph for its decode step; the
multi-token hooks go on both models' HyenaCascade instances; `Evo2.generate` on the instance runs the loop (`core.generate`). A call the loop
does not take (`cached_generation=False`, `force_prompt_threshold` set) is named on one FALLBACK line and runs the stock loop with the exact
members. No draft checkpoint beside a local target: `specdec=refused: …` on the GENERATION line naming the install line — the model is constructed
and scores as exact; `generate()` raises `Evo2OptRefused` by name until the draft is installed. A stack on which upstream's constructor cannot
build the draft (no Transformer Engine for its FP8 projections): `specdec=n/a` by name — `generate()` stays exact's."""
import os
import sys
import time

from evo2_opt.gen._chain import chain_install, chain_next, chain_uninstall

from . import core, graph

TAG = core.TAG
LEVER = "specdec"
DRAFT_NAME = "evo2_1b_base"
K = 6                                            # drafted tokens per target call, every model and route
WEIGHTS_ENV = "EVO2_OPT_WEIGHTS"
__all__ = ["install", "arm", "remove", "status", "Handle", "SpecDecRefused", "locate_draft", "TAG", "LEVER", "DRAFT_NAME", "K", "core"]


class SpecDecRefused(RuntimeError):
    """No draft to arm with, by name (the install line included); `install()` then leaves the model scoring as exact and makes `generate()` raise
    `Evo2OptRefused` with this reason."""


def _say(word, reason, log=None):
    print(f"{TAG} {word}: {reason}", file=log or sys.stderr, flush=True)


def locate_draft(target_local_path, pins):
    """(path | None, how): the draft checkpoint beside the target's, else under $EVO2_OPT_WEIGHTS, else None = upstream's download (only when the
    target itself came from upstream's download); raises SpecDecRefused naming the install line when a local target has no draft beside it."""
    fn = ((pins.get("checkpoints") or {}).get(DRAFT_NAME) or {}).get("file", DRAFT_NAME + ".pt")
    cands = []
    if target_local_path:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(target_local_path)), fn))
    if os.environ.get(WEIGHTS_ENV):
        cands.append(os.path.join(os.environ[WEIGHTS_ENV], fn))
    for c in cands:
        if os.path.isfile(c):
            return c, "beside the target's checkpoint" if target_local_path and c == cands[0] else f"${WEIGHTS_ENV}"
    if not target_local_path:
        return None, "upstream's Hugging Face download (the target was constructed without a local_path)"
    raise SpecDecRefused(f"no draft checkpoint {fn} beside {target_local_path}" + (f" or under ${WEIGHTS_ENV}" if os.environ.get(WEIGHTS_ENV) else "")
                         + f" — bash run.sh install --weights {os.path.dirname(os.path.abspath(target_local_path))} --model_name {DRAFT_NAME}")


def _check_pin(path, pins, log):
    """sha256 of the draft against the pin; a mismatch or a missing pin is ONE UNPINNED line (the arm proceeds). Returns the record."""
    from evo2_opt.weights import sha256_file
    want = ((pins.get("checkpoints") or {}).get(DRAFT_NAME) or {}).get("sha256")
    got = sha256_file(path)
    rec = {"path": path, "sha256": got, "pinned": want, "matches": got == want}
    if want is None:
        _say("UNPINNED", f"{path} sha256={got}: stock/PINS.json has no checkpoints.{DRAFT_NAME} — running on an unpinned draft", log)
    elif got != want:
        _say("UNPINNED", f"{path} sha256={got} != the pin's {want} (stock/PINS.json checkpoints.{DRAFT_NAME}) — running on an unpinned draft "
                         f"(the sampled distribution is the target's whatever the draft; the acceptance rate depends on it)", log)
    return rec


class Handle:
    """`.record`, `.draft`, `.generate(prompt_seqs, **kw)` (the loop), `.uninstall()`."""
    name = LEVER

    def __init__(self, evo2_model, draft, record, log, refused=None):
        self.model, self.draft, self.record, self.log = evo2_model, draft, record, log
        self.refused = refused                       # the SpecDecRefused reason: generate() raises it by name; None when armed
        self.target_graph = None
        self._hook = self._prev = None

    def generate(self, prompt_seqs, **kw):
        """`Evo2.generate`'s keywords; k = the record's (K)."""
        out = core.generate(self.model, self.draft, prompt_seqs, k=self.record["k"], progress=self.record["progress"], **kw)
        self.record["calls"] += 1
        self.record["last_call"] = [{kk: r[kk] for kk in ("prompt_len", "n_tokens", "k", "greedy", "rounds", "target_calls", "acceptance_per_test",
                                     "alpha_expected", "accepted_per_proposed", "tokens_per_target_call", "decode_tok_s", "events")} for r in out.speculative]
        return out

    def _install_generate_hook(self):
        """`generate` on the Evo2 INSTANCE with upstream's signature (evo2/models.py:146): the loop; a call it refuses by name runs what was there
        before (the stock loop, exact's members); with no draft (refused at construction) every call raises Evo2OptRefused naming the install line."""
        def generate(prompt_seqs, n_tokens=500, temperature=1.0, top_k=4, top_p=1.0, batched=True, cached_generation=True, verbose=1, force_prompt_threshold=None):
            kwargs = dict(n_tokens=n_tokens, temperature=temperature, top_k=top_k, top_p=top_p, batched=batched, cached_generation=cached_generation,
                          verbose=verbose, force_prompt_threshold=force_prompt_threshold)
            if self.refused is not None:
                from evo2_opt.activation import Evo2OptRefused
                raise Evo2OptRefused(f"specdec: {self.refused}")
            try:
                return self.generate(prompt_seqs, **kwargs)
            except core.Refused as exc:
                self.record["fallbacks"] += 1
                _say("FALLBACK", f"generate({exc}): this call runs the stock loop with the exact members", self.log)
                return chain_next(self.model, self._prev, prompt_seqs, attr="generate", **kwargs)
        generate._evo2_gen_specdec = True
        self._hook = generate
        self._prev = chain_install(self.model, generate, attr="generate")

    def uninstall(self):
        remove(self.model, log=self.log)


def install(evo2_model, *, local_path=None, draft=None, log=None):
    """Arm speculative sampling on a constructed `Evo2` (mode fast). `local_path`: the target's checkpoint path as given to its constructor (the
    draft is located from it); `draft`: an already constructed Evo2 to use instead. Returns a Handle; idempotent per instance."""
    from evo2_opt import pins as P
    from evo2_opt import gen as G
    log = log or sys.stderr
    tsh = core._striped(evo2_model)
    existing = getattr(tsh, "_evo2_gen_specdec_handle", None)
    if existing is not None:
        return existing
    pins = P.load()
    record = {"lever": LEVER, "armed": False, "k": K, "calls": 0, "fallbacks": 0, "draft": None, "t_install": time.time(),
              "progress": {"prompt": 0, "emitted": 0, "rounds": 0}}          # the current generate() call's live state, updated per round (status(model)["progress"])
    t0 = time.perf_counter()
    if draft is None:
        try:
            path, how = locate_draft(local_path, pins)
        except SpecDecRefused as e:                  # no draft: the model scores as exact; generate() raises by name until the draft is installed
            record.update(refused=str(e))
            h = Handle(evo2_model, None, record, log, refused=str(e))
            tsh._evo2_gen_specdec_handle = h
            if hasattr(evo2_model, "generate"):
                h._install_generate_hook()
            _say("REFUSED", f"{e}; generate() raises Evo2OptRefused by name until then (scoring runs as exact)", log)
            return h
        record["draft"] = {"name": DRAFT_NAME, "path": path, "located": how}
        if path is not None:
            record["draft"]["pin"] = _check_pin(path, pins, log)
        try:
            draft = core.load_draft(DRAFT_NAME, path)
        except ImportError as e:                     # upstream's constructor names a missing requirement of the draft's config (its FP8 projections need Transformer Engine)
            why = " ".join(str(e).split("\n", 1)[0].split())   # upstream's first sentence (the requirement); its install advice follows on later lines
            raise G.NotApplicable(f"the draft {DRAFT_NAME} cannot be constructed on this stack ({why}) — generate() runs the stock loop with the exact members") from e
    dsh = core._striped(draft)
    record["draft"] = dict(record["draft"] or {"name": DRAFT_NAME}, devices=sorted({str(v) for v in dsh.block_idx_to_device.values()}), blocks=len(dsh.blocks),
                           use_fp8_input_projections=bool(dsh.config.get("use_fp8_input_projections", False)),
                           max_seqlen=int(dsh.config.get("max_seqlen", 8192)), load_s=round(time.perf_counter() - t0, 1))
    record["draft"]["members"] = G.arm_members(draft, G.MEMBERS_OF["exact"], log)     # hyenafuse + cudagraph on the draft's decode step
    n_t = core.install_hooks(tsh)
    n_d = core.install_hooks(dsh)
    h = Handle(evo2_model, draft, record, log)
    h.target_graph = graph.adopt(evo2_model)
    if h.target_graph is not None:
        tsh._evo2_gen_specdec_targetgraph = h.target_graph
    record.update(armed=True, hooks={"target_layers": n_t, "draft_layers": n_d}, target_graph=h.target_graph is not None)
    tsh._evo2_gen_specdec_handle = h
    if hasattr(evo2_model, "generate"):
        h._install_generate_hook()
    return h


arm = install


def remove(evo2_model, log=None):
    tsh = core._striped(evo2_model)
    h = getattr(tsh, "_evo2_gen_specdec_handle", None)
    if h is None:
        return {"lever": LEVER, "armed": False, "removed": False}
    if h._hook is not None:
        chain_uninstall(evo2_model, h._hook, h._prev, tag=TAG, attr="generate")
    if h.target_graph is not None:
        h.target_graph.restore()
        tsh.__dict__.pop("_evo2_gen_specdec_targetgraph", None)
    n_t = core.remove_hooks(tsh) if h.refused is None else 0
    n_d = core.remove_hooks(core._striped(h.draft)) if h.draft is not None else 0
    del tsh._evo2_gen_specdec_handle
    rec = dict(h.record, armed=False, removed=True, hooks_removed={"target_layers": n_t, "draft_layers": n_d})
    tsh._evo2_gen_specdec_record = rec
    h.draft = None
    return rec


def status(evo2_model):
    tsh = core._striped(evo2_model)
    h = getattr(tsh, "_evo2_gen_specdec_handle", None)
    return h.record if h is not None else getattr(tsh, "_evo2_gen_specdec_record", {"lever": LEVER, "armed": False})
