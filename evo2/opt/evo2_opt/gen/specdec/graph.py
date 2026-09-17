"""The target's (1, k+1) target call replayed through the decode-step member (`evo2_opt.gen.cudagraph`): that member's runner captures "a
cached call of width `step_tokens`"; the target call is such a call with step_tokens = k+1. The runner the target already carries (armed by
`evo2_opt.gen.arm`) is ADOPTED: `runner.step_tokens = k+1` for the duration of each speculative generate call, the previous value afterwards,
so a stock-loop generate on the same instance keeps its width-1 replay. Capture happens by that member's protocol at the second width-(k+1)
call on a given inference-params object, i.e. once per generate call; the first target call of a call runs eager; a shorter tail call runs eager."""
TAG = "[evo2-gen specdec]"


class TargetGraph:
    def __init__(self, runner):
        self.runner = runner
        self.prev_step_tokens = int(getattr(runner, "step_tokens", 1))
        self.mode = getattr(runner, "mode", None)

    @property
    def record(self):
        return self.runner.record

    def set_width(self, width):
        self.runner.step_tokens = int(width)

    def restore(self):
        self.runner.step_tokens = self.prev_step_tokens

    def describe(self):
        r = self.runner.record
        return {"mode": self.mode, "step_tokens_now": self.runner.step_tokens, "prev_step_tokens": self.prev_step_tokens,
                **{k: r.get(k) for k in ("captures", "replays", "eager_decode_steps", "prefills", "refusals", "disabled", "segments")}}


def adopt(model):
    """The target's armed cudagraph runner as a TargetGraph, or None when that member is not armed on it (target calls then run eager)."""
    sh = getattr(model, "model", model)
    runner = getattr(sh, "_evo2_gen_cudagraph", None)
    if runner is None:
        return None
    if not hasattr(runner, "step_tokens"):
        raise RuntimeError(f"{TAG} the target's cudagraph runner has no step_tokens: target calls cannot be replayed through it")
    return TargetGraph(runner)
