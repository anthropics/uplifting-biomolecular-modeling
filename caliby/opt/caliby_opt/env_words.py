"""Switch words read by the lever files at call time (`CALIBY_X_MULTISEQ`, `CALIBY_X_BG_CIF`, `CALIBY_X_CIF_WORKERS`): the kit modes
export them from the mode table's row (`modes.ROWS`); a caller that sets one by hand gets exactly what the word says or a usage
error — a malformed value never silently reads as "off".
"""
import os
from typing import Optional, Sequence


class UsageError(ValueError):
    """A malformed switch value: the message opens with the switch's bracket tag, e.g. ``[CALIBY_X_CIF_WORKERS] malformed value ...``."""


def int_word(name: str, default: int = 0, minimum: int = 0) -> int:
    """``int(os.environ[name])``; unset or empty -> ``default``; not an integer, or below ``minimum`` -> UsageError."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        raise UsageError(f"[{name}] malformed value {raw!r}: expected an integer >= {minimum} (unset = {default})") from None
    if value < minimum:
        raise UsageError(f"[{name}] malformed value {raw!r}: expected an integer >= {minimum} (unset = {default})")
    return value


def choice_word(name: str, allowed: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """``os.environ[name]`` when it is one of ``allowed``; unset or empty -> ``default``; anything else -> UsageError."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    if raw not in allowed:
        raise UsageError(f"[{name}] malformed value {raw!r}: expected one of {'|'.join(allowed)} (unset = {default})")
    return raw
