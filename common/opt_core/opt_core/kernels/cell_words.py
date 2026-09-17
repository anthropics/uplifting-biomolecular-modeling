"""opt_core.kernels.cell_words — the words of a cell / config-table lever's decision for a key its table has no admitted row for (standard library only).

A lever's table lists the keys — (compute capability, triton, shape / dtype class) — it was measured on; every other key is one of exactly three named cases:

  * MEASURED OFF — a key the measured compositions run with the lever OFF (its table lists it: a ``named_off`` entry or a ``candidate`` row, a
    ``MEASURED_OFF`` cell): the caller's stock statements serve it, counted under the lever's own refusal word for the key with the qualifier
    ``+off(not-measured)`` appended (:func:`with_off`; ``+off(slower)`` where the table records the lever measured slower than stock there) —
    e.g. ``no-cell:fpf:prologue:64x4x32:9.0|3.7+off(not-measured)``.  The word's head is unchanged, so a caller that routes on the head
    (``no-cell:…``) routes exactly as for any other miss; the qualifier says the miss is a recorded decision, not an unknown.  Nothing engages.
  * UNKNOWN — a key no table lists: the lever ENGAGES its SAFE settings (``opt_core.kernels.safe_settings``: ``SafeNet.engage``,
    line ``safe settings served (no_cell:<key>, cc …, triton …)``, census ``settings=safe:no_cell:<key>``) or, for a lever with no safe-settings row, its
    capability-free default settings named ``unverified(<key>)`` (:func:`unverified_word`: ``cell=unverified(<key>)`` on the lever's line,
    ``:unverified(<key>)`` as the suffix of a served census event) — the environment's uncertainty is named, never a reason to disengage.
  * CANNOT RUN — the settings fail to build / launch even as SAFE settings, or the shape / dtype is outside what the kernel computes: the lever's own
    refusal (``safe_settings`` case iii; the kernel's ``Unsupported`` / ``Refusal`` words) — the mode that promised the lever refuses by name.
"""

OFF = "off"                              # the measured-off qualifier: <word>+off(<why>)
NOT_MEASURED = "not-measured"            # the measured compositions run this key with the lever off; no measurement says it may engage
SLOWER = "slower"                        # measured slower than the stock statements on this key (the table row says where)
UNVERIFIED = "unverified"                # engaged on a key without a measurement record (the lever's default settings), named


def off_suffix(why: str = NOT_MEASURED) -> str:
    """``+off(<why>)`` — the qualifier a measured-off key's word carries (``why``: :data:`NOT_MEASURED` | :data:`SLOWER` | a table's own word)."""
    return "+%s(%s)" % (OFF, why or NOT_MEASURED)


def with_off(word: str, why: str = NOT_MEASURED) -> str:
    """``<word>+off(<why>)`` — the lever's refusal word for the key, qualified as MEASURED OFF."""
    return str(word) + off_suffix(why)


def is_off_word(word) -> bool:
    """True when ``word`` carries the measured-off qualifier (``…+off(<why>)``)."""
    return isinstance(word, str) and ("+" + OFF + "(") in word


def off_why(word) -> str:
    """The ``<why>`` of a measured-off word (``""`` when ``word`` carries no qualifier)."""
    if not is_off_word(word):
        return ""
    tail = word.split("+" + OFF + "(", 1)[1]
    return tail[:-1] if tail.endswith(")") else tail.split(")", 1)[0]


def unverified_word(key: str) -> str:
    """``unverified(<key>)`` — names an engagement on a key with no measurement record (``cell=unverified(<key>)``; ``served:…:unverified(<key>)``)."""
    return "%s(%s)" % (UNVERIFIED, key)


def decide(*, pinned=None, candidate=None, allow_candidate: bool = False, measured_off: bool = False, why: str = NOT_MEASURED, miss_word: str = "",
           safe_lever: str = "", cc=None, dims=None, shape_word: str = ""):
    """The ONE decision statement of a pinned-configuration table for one key -> ``(kind, settings, word)``:

      * ``("pinned", settings, "")``              the table's row serves;
      * ``("candidate", settings, "candidate")``  a candidate row serves (only when the caller's qualification switch admits candidates);
      * ``("off", None, "<miss_word>+off(<why>)")``  MEASURED OFF: a candidate row not admitted, or a key the table names as measured off — the stock path BY NAME;
      * ``("safe", settings, "no_cell:<shape_word>")``  UNKNOWN: the SAFE row of ``safe_lever`` on ``cc`` admits the shape (``opt_core.kernels.safe_settings``)
        — the caller ENGAGES it (``SafeNet.engage(word, where)``: ONE line, census word);
      * ``("none", None, "<miss_word>[+no_safe(<refusal>)]")``  nothing admits the key (no safe settings on the capability, or its safe rows exclude the class by name).
    """
    if pinned is not None:
        return "pinned", dict(pinned), ""
    if candidate is not None and allow_candidate:
        return "candidate", dict(candidate), "candidate"
    if candidate is not None or measured_off:
        return "off", None, with_off(miss_word, why)
    if safe_lever and cc is not None:
        from .safe_settings import safe_settings_for
        srow, refusal = safe_settings_for(safe_lever, cc, dims)
        if srow is not None:
            return "safe", dict(srow["settings"]), "no_cell:" + (shape_word or str(miss_word))
        if refusal:
            return "none", None, "%s+no_safe(%s)" % (miss_word, refusal)
    return "none", None, str(miss_word)


DEFAULT_CC = "9.0"                       # the capability the core's capability-free default tables were measured on (H100): served them without a row, nothing to say


def name_default_row(net, table, cc, triton_mm, default_cc: str = DEFAULT_CC) -> None:
    """Say ONCE, through the lever's safety net (``opt_core.kernels.safe_settings.SafeNet.inform``), that a capability WITHOUT a row in the lever's
    cc-keyed ``table`` — other than ``default_cc``, whose measurements the default settings are — is served the lever's default settings:
    ``[opt_core/<lever>] default settings (no row for cc <cc>, cc <cc>, triton <mm>); tuned rows exist for cc …`` and ``net.note() == 'default:no_row'``
    (the ``cells_note=`` fact of the lever's line).  Nothing for ``cc`` None (no CUDA device), ``default_cc``, a capability with a row, or a second call."""
    from .safe_settings import cc_word, where_word, table_ccs, resolve_key
    w = cc_word(cc)
    if w is None or w == default_cc or net.note() is not None or resolve_key(table, cc, triton_mm) is not None:
        return
    net.inform("no row for cc %s" % w, where_word(cc, triton_mm), tuned=sorted(set([default_cc] + list(table_ccs(table))), key=lambda c: tuple(int(x) for x in c.split("."))))

