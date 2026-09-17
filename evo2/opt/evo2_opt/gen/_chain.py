"""Instance-`forward` chaining — the ONE discipline every generation arm uses when it hooks a module's INSTANCE attribute `forward`
(several parties may: the scoring kit's hooks on the vortex model, the cudagraph runner and its block hooks).

    prev = chain_install(obj, fn)          # set fn as obj.forward; prev = the instance hook found there (None: only the class method)
    chain_next(obj, prev, *args, **kw)     # call what was there before: prev, else type(obj).forward resolved AT CALL TIME (a class-level
                                           #   gate installed later — the scoring kit's `StripedHyena.forward` — is honoured)
    chain_uninstall(obj, fn, prev, tag)    # undo exactly: put prev back (or delete the attribute); RAISES by name when obj.forward is no
                                           #   longer fn — a later party sits on top and must uninstall first (LIFO)
The same three calls take `attr=` for another hooked instance attribute (the composer's per-call members hook `Evo2.generate`).
"""
__all__ = ["chain_install", "chain_next", "chain_uninstall", "is_same_hook"]


def chain_install(obj, fn, attr="forward"):
    """Set `fn` as obj's instance attribute `attr` (default `forward`), remembering the previous instance hook (None when only the class
    method exists). Returns the previous hook; pass it to `chain_next` / `chain_uninstall`. Never refuses because someone else's hook is
    present — it chains."""
    prev = vars(obj).get(attr)
    setattr(obj, attr, fn)
    return prev


def chain_next(obj, prev, *args, attr="forward", **kwargs):
    """Call what was there before us: the previous instance hook, else the CLASS method resolved now (a class-level gate installed after
    this arm is honoured)."""
    if prev is not None:
        return prev(*args, **kwargs)
    return getattr(type(obj), attr)(obj, *args, **kwargs)


def is_same_hook(cur, fn):
    """Sameness of two hooks: the same object, or two bound methods of the same function on the same instance (a bound method read twice
    from an instance dict compares its parts with `is`, not the wrapper)."""
    return cur is fn or (getattr(cur, "__self__", None) is getattr(fn, "__self__", object()) and getattr(cur, "__func__", None) is getattr(fn, "__func__", object()))


def chain_uninstall(obj, fn, prev, tag="[evo2-gen]", attr="forward"):
    """Undo chain_install exactly: re-set the previous hook, or delete the instance attribute when there was none. Raises (by name) when
    the instance hook is no longer `fn` — someone chained on top of this arm and must uninstall first (LIFO)."""
    cur = vars(obj).get(attr)
    if not is_same_hook(cur, fn):
        raise RuntimeError(f"{tag} uninstall: the model's instance {attr} is no longer this arm's ({cur!r}); uninstall the later hook first")
    if prev is None:
        del obj.__dict__[attr]
    else:
        obj.__dict__[attr] = prev
