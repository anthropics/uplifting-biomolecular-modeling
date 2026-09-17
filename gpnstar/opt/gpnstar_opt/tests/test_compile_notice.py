"""The COMPILE line: printed once when a forward of the patched model is entered from inside a torch.compile (dynamo) wrapper, never
otherwise. The dynamo wrapper is a stand-in frame compiled under torch's eval_frame file name (no torch here)."""
import types

from gpnstar_opt import report, stack

SRC = (
    "class _TorchDynamoContext:\n"
    "    def __call__(self, fn):\n"
    "        def _fn(*a, **k):\n"
    "            return fn(*a, **k)\n"
    "        return _fn\n"
    "class DisableContext:\n"
    "    def __call__(self, fn):\n"
    "        def _fn(*a, **k):\n"
    "            return fn(*a, **k)\n"
    "        return _fn\n"
)


def _standins():
    ns = {}
    exec(compile(SRC, "/site-packages/torch/_dynamo/eval_frame.py", "exec"), ns)  # noqa: S102 - a stand-in for the wrapper's frame
    return ns["_TorchDynamoContext"](), ns["DisableContext"]()


def test_detects_a_dynamo_wrapper_above_and_not_the_disable_wrapper():
    dyn, dis = _standins()
    assert stack._entered_through_dynamo() is False
    assert dyn(stack._entered_through_dynamo)() is True
    assert dis(stack._entered_through_dynamo)() is False            # the levers' own eager guard is not a compile request
    assert dyn(dis(stack._entered_through_dynamo))() is True        # compile outside, eager guard inside: still a compile request


def test_compile_line_once_and_only_under_a_wrapper(capsys):
    dyn, _ = _standins()
    stack._reset_for_tests()
    stack._compile_notice()
    assert capsys.readouterr().err == ""                            # plain eager call: nothing to say
    dyn(stack._compile_notice)(); dyn(stack._compile_notice)()
    err = capsys.readouterr().err.strip().splitlines()
    assert err == [report.compile_line("eager")] == ["[gpnstar-opt] COMPILE levers=eager(kit-disabled) rest=eager"]   # once
    assert report.RE_COMPILE.match(err[0]).group("rest") == "eager"
