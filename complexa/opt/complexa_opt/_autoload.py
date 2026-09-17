"""The kit's autoload hook, run at interpreter start by ``complexa_opt_autoload.pth`` (``import complexa_opt._autoload``).

Import-free until COMPLEXA_OPT names a kit mode: the .pth runs in every interpreter of the environment the kit is installed into — the stock
route's ``complexa generate`` processes included — and those load nothing of the core (the stock proof records this module and the package's
``__init__`` as the declared inert pair: ``stack.PTH_MODULES``). A variable under the package prefix the package does not read (a mistyped
``COMPLEXA_OPT_LEVERS``, say) is refused here, in every process, with the NOT ACTIVE line and exit 3 — never ignored silently. With a kit mode
named, the core's finder (``opt_core.autoload``) is installed from this kit's ``AutoloadSpec``: it waits for the first import of the trigger
module and, right after that module's own body has executed, calls ``complexa_opt.enable(mode, strict=True, trigger=<name>)``; a mode whose
levers cannot be installed prints its NOT ACTIVE line and the process exits 3 (stock never runs silently under COMPLEXA_OPT); a word outside
the mode table is refused by the core at start with the same line form and exit 3. Until the trigger nothing else is imported (no torch, no
proteinfoundation); with COMPLEXA_OPT unset or ``off`` no finder is installed at all.

The trigger is ``proteinfoundation.proteina``: the module holding ``Proteina``, whose import pulls in every module the levers patch (the
transformer, the feature factories, the flow matchers) and which every route into the model passes through before a checkpoint loads —
``python -m proteinfoundation.generate`` (upstream's ``complexa generate`` launches exactly that, generate.py line 28) and library use alike.
``complexa generate`` itself (``proteinfoundation.cli.cli_runner``) imports no model code: the finder stays armed and idle there, and the
generation process it spawns — same interpreter, same environment — is where the trigger fires.
"""
import os
import sys

PACKAGE = "complexa_opt"
ENV = "COMPLEXA_OPT"
TAG = "complexa-opt"
TRIGGERS = ("proteinfoundation.proteina",)
DECLARED = ("COMPLEXA_OPT", "COMPLEXA_OPT_RECORD")            # the variables under the package prefix the package reads (modes.ENV_MODE, modes.ENV_RECORD)
FINDER = None

_value = os.environ.get(ENV, "").strip().lower()
_unread = sorted(k for k in os.environ if k.startswith(ENV) and k not in DECLARED)
if _unread:
    sys.stderr.write("[%s] NOT ACTIVE: reason=%s is not a variable this package reads (it reads %s: the mode, %s); unset it (mode=%s pid=%d)\n"
                     % (TAG, ",".join(_unread), ENV, "|".join("off exact fast big".split()), _value or "unset", os.getpid()))
    sys.stderr.flush()
    raise SystemExit(3)
if _value and _value != "off":
    from ._core_gate import gate as _core_gate

    _core_gate(__file__, tag=TAG)                             # the core pin gate ahead of the first core import: an absent or mismatched core is ITS line and exit 3, never an ImportError swallowed by site.py
    from opt_core import autoload as _autoload

    from .modes import MODES

    SPEC = _autoload.AutoloadSpec(env=ENV, package=PACKAGE, tag=TAG, triggers=TRIGGERS, modes=MODES)      # the table's words (off included: the core reads `off` as no selection)
    FINDER = _autoload.install(SPEC)
