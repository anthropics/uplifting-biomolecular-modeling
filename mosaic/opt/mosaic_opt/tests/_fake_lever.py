"""A per-step lever module with the uniform API (ENV_REQUIRED / configure / install / uninstall / describe), for tests/test_levers.py:
it patches nothing real — it records the calls. `BEHAVIOUR` lets a test make describe() incomplete or install() refuse."""
ENV_REQUIRED: dict = {}
CALLS: list = []
BEHAVIOUR: dict = {"describe": None, "install_raises": None, "gate_raises": None}
_SPEC = [None]
_ON = [False]


def configure(spec):
    CALLS.append(("configure", spec)); _SPEC[0] = spec


def install():
    CALLS.append(("install", _SPEC[0]))
    if BEHAVIOUR["install_raises"]:
        raise RuntimeError(BEHAVIOUR["install_raises"])
    _ON[0] = True


def uninstall():
    CALLS.append(("uninstall", None)); _ON[0] = False


def describe() -> dict:
    if BEHAVIOUR["describe"] is not None:
        return dict(BEHAVIOUR["describe"])
    return {"impl": "fake_lever@test", "origin": "kit", "chunk": _SPEC[0] or "of_record", "installed": _ON[0]}


def reset():
    CALLS.clear(); BEHAVIOUR.update({"describe": None, "install_raises": None, "gate_raises": None}); _SPEC[0] = None; _ON[0] = False; ENV_REQUIRED.clear()


def emit_line(tag=""):
    CALLS.append(("emit_line", tag))
    return f"[fake] LEVER name=P5.fake state=on served=3 fallback=0"


def gate():
    CALLS.append(("gate", None))
    if BEHAVIOUR["gate_raises"]:
        raise RuntimeError(BEHAVIOUR["gate_raises"])
