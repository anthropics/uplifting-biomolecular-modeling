"""genie3_opt.jitdirs — where the compile caches `fast` fills may live: a directory this user alone can have written, never a
fixed name in the shared temporary directory.

A warm inductor cache (TORCHINDUCTOR_CACHE_DIR) is read back as code: its generated wrapper modules are imported and its graphs are
unpickled, so a directory another account could have planted or written is never used. `run.sh` names a cache root on every route
(MODEL_OPT_JIT_ROOT: the caller's own, the image's, or the per-user root it makes with mode 0700 and checks) and that root is used as
given. A process started without one — `genie3-opt`, `python -m genie3_opt`, upstream's command line with the kit's patches — gets the
per-user root `<tmp>/model_opt_jit-uid<uid>` held to the shared core's directory rule: created 0700, owned by this uid alone, no group
or other write bit, never a symbolic link; otherwise one stderr line names the directory, the reason and the fix, and this process
compiles again in a fresh private directory of its own. A TORCHINDUCTOR_CACHE_DIR the caller already set is the caller's word and is
kept (g3batch.py sets a default, never an override)."""
import os
import tempfile

PER_USER_ROOT = "model_opt_jit-uid%d"                      # the name run.sh gives the per-user root under ${TMPDIR:-/tmp} (its jit-cache block): one root per user


def jit_root() -> str:
    """MODEL_OPT_JIT_ROOT as given, else the per-user root under the temporary directory after the shared core's directory rule."""
    root = os.environ.get("MODEL_OPT_JIT_ROOT")
    if root:
        return root
    from opt_core.kernels.triattn_exact._paths import _private_cache_dir      # the shared core's one owner / mode / link rule for cache directories
    return _private_cache_dir(os.path.join(tempfile.gettempdir(), PER_USER_ROOT % os.getuid()), "genie3_opt_jit", own_only=True)


def inductor_cache_dir() -> str:
    """`<jit_root()>/inductor`: the default g3batch.py gives TORCHINDUCTOR_CACHE_DIR when the line compiles (run.sh keys it by stack first)."""
    return os.path.join(jit_root(), "inductor")
