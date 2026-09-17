"""evo2_opt.gen — the generation members, armed once on every constructed ``Evo2`` (``arm(model, mode=...)``). Mode exact: ``hyenafuse``
(one Triton launch per Hyena block for the cached decode step's state update, the stock's arithmetic per element) and ``cudagraph`` (the
one-token decode step captured into a CUDA graph once per ``generate()`` call after two eager warm-up steps, then replayed for every remaining
token: the same kernels on the same buffers); both engage inside ``Evo2.generate`` -> vortex's cached generation loop; the prompt's prefill and
the first decode step run as the stock runs them; tokens and per-step logits are the stock's. Mode fast: the same two, then ``specdec`` —
``generate()`` becomes speculative sampling with the ``evo2_1b_base`` draft (``gen/specdec``): every emitted token an exact sample from the
target's transformed distribution as its (1, k+1) target call computes it; the sequence for a seed is not the stock sampler's. A member the
model gives nothing to act on says ``N/A`` by name; a member that cannot run raises by name (``GenerationRefused``) — the call never proceeds
silently on a subset."""
import sys

PREFIX = "[evo2-opt]"
MEMBERS_OF = {"exact": ("hyenafuse", "cudagraph"), "fast": ("hyenafuse", "cudagraph", "specdec")}


class NotApplicable(RuntimeError):
    """A member has nothing to act on in this model (named); nothing is patched."""


class GenerationRefused(RuntimeError):
    """A member cannot run on this model / stack (named)."""


def arm_members(model, members, log, local_path=None) -> dict:
    """Arm `members` in order on one model; {member: 'armed' | 'n/a: <why>' | 'refused: <why>'}; a member that cannot run raises GenerationRefused by name."""
    from evo2_opt.gen import hyenafuse, cudagraph
    out = {}
    for name in members:
        try:
            if name == "hyenafuse":
                hyenafuse.arm(model, selftest=False)
            elif name == "cudagraph":
                cudagraph.install(model)
            elif name == "specdec":
                from evo2_opt.gen import specdec
                h = specdec.install(model, local_path=local_path, log=log)
                if h.refused is not None:            # no draft checkpoint: the model is constructed and scores; generate() raises by name
                    out[name] = f"refused: {h.refused}; generate() raises Evo2OptRefused by name until then"
                    continue
            else:
                raise GenerationRefused(f"unknown generation member {name!r}")
            out[name] = "armed"
        except NotApplicable as e:
            out[name] = f"n/a: {e}"
        except (hyenafuse.HyenaFuseRefused, cudagraph.CaptureRefused) as e:
            raise GenerationRefused(f"{name}: {e}") from e
    return out


def arm(evo2_model, log=None, mode="exact", local_path=None) -> dict:
    """Arm the mode's members on the instance in order; one ``[evo2-opt] GENERATION …`` line. Returns {member: 'armed' | 'n/a: <why>'}."""
    from evo2_opt.gen import cudagraph
    log = log or sys.stderr
    out = arm_members(evo2_model, MEMBERS_OF[mode], log, local_path=local_path)
    words = "; ".join(f"{k}={v}" for k, v in out.items())
    tail = (f" — engaged inside generate(): the fused Hyena decode step on every token; the decode step captured once per call "
            f"after {cudagraph.WARMUP} eager steps (a fixed per-call cost) and replayed")
    if mode == "fast":
        from evo2_opt.gen import specdec
        rec = specdec.status(evo2_model)
        if out.get("specdec") == "armed":
            tail += (f"; generation=speculative(draft={specdec.DRAFT_NAME},k={rec.get('k', specdec.K)}): generate() drafts k tokens with "
                     f"{specdec.DRAFT_NAME} and scores them in one (1, k+1) target call per round — every token an exact sample from the target's "
                     f"transformed distribution of that call; the sequence for a seed is not the stock sampler's; scores and forwards are exact's")
        elif str(out.get("specdec", "")).startswith("refused"):
            tail += "; generation=refused (specdec, named above): generate() raises until the draft is installed; scores and forwards are exact's"
        else:
            tail += "; generation=exact's (specdec not armed, named above): tokens and per-step logits are the stock's"
    print(f"{PREFIX} GENERATION {words}{tail}", file=log, flush=True)
    return out
