"""``run.sh install --weights DIR``: fetch the ESM-IF1 checkpoint into DIR and check it against the pin — ``python -m esm_if1_opt.weights DIR``.

Upstream's loader (``esm.pretrained.esm_if1_gvp4_t16_142M_UR50`` → ``load_model_and_alphabet_hub`` → ``torch.hub.load_state_dict_from_url``)
downloads ``https://dl.fbaipublicfiles.com/fair-esm/models/<name>.pt`` into torch.hub's cache and deserializes it in the same call; it exposes no
transfer-only entry point. This module runs the transfer routine that loader itself uses — ``torch.hub.download_url_to_file`` (a temporary file
beside the target, moved into place once complete) — on the URL the pin card lists (``stock/PINS.json`` ``weights.source_url``, upstream's URL for
this model), straight into ``DIR/<file>``; a file already in DIR is never fetched again. Then the digest gate: the file's byte count and sha256 must
equal the pin card's ``weights.bytes`` / ``weights.sha256`` (the values ``stack.WEIGHTS_BYTES`` / ``stack.WEIGHTS_SHA256`` carry and ``run.sh check``
prints beside the file's own) — ``WEIGHTS OK: 1/1 files in DIR (<file>: <bytes> bytes, sha256 = the pin)`` plus the ``export ESM_IF1_WEIGHTS=DIR/<file>`` line
and 0, or ``WEIGHTS FAILED: …`` naming the file and 1, the file left in place for inspection (nothing is deleted here). DIR is then what
``ESM_IF1_WEIGHTS`` points into (configs/*.env, README.md 'Setup'); a DIR that is ``$TORCH_HOME/hub/checkpoints`` serves upstream's own cache
lookup equally. Exit 2 on a usage error.
"""
import os
import sys

from opt_core import gates

from . import stack
from .lines import PREFIX

PINS_REL = os.path.join("stock", "PINS.json")


def pin(tree=None):
    """The pin card's ``weights`` entry {file, source_url, bytes, sha256} (``stock/PINS.json`` of the tree ``stack.tree_home()`` names)."""
    return gates.load_pins(os.path.join(tree or stack.tree_home(), PINS_REL))["weights"]


def hub_transfer(url, dst):
    """The transfer upstream's loader performs (``torch.hub``), without the deserialization: ``url`` → ``dst`` through a temporary file beside it."""
    from torch import hub
    hub.download_url_to_file(url, dst, hash_prefix=None, progress=sys.stderr.isatty())


def digest_check(path, want):
    """(ok, sentence): the file at ``path`` against the pin's byte count and sha256 — the count first (cheap), then the digest."""
    if not os.path.isfile(path):
        return False, f"{path} is absent"
    n = os.path.getsize(path)
    if n != int(want["bytes"]):
        return False, f"{path} is {n} bytes; the pin (stock/PINS.json weights.bytes) is {want['bytes']}"
    digest = gates.sha256_file(path)
    if digest != want["sha256"]:
        return False, f"{path} sha256 {digest[:16]}… is not the pin {want['sha256'][:16]}… (stock/PINS.json weights.sha256)"
    return True, f"{n} bytes, sha256 = the pin"


def fetch(directory, want=None, transfer=hub_transfer):
    """Fetch (when absent) and digest-check the checkpoint in ``directory``; returns the exit code and prints the WEIGHTS OK / WEIGHTS FAILED line."""
    want = want or pin()
    directory = os.path.abspath(directory)
    dst = os.path.join(directory, want["file"])
    os.makedirs(directory, exist_ok=True)
    if os.path.isfile(dst):
        print(f"{PREFIX} weights: {want['file']} is already in {directory} — not fetched again", file=sys.stderr)
    else:
        print(f"{PREFIX} weights: fetching {want['source_url']} -> {dst}", file=sys.stderr)
        try:
            transfer(want["source_url"], dst)
        except BaseException as e:                                            # noqa: BLE001 — a partial transfer is named, whatever interrupted it
            print(f"{PREFIX} WEIGHTS FAILED: the download of {want['file']} into {directory} did not complete ({type(e).__name__}: {e}); run the step again", file=sys.stderr)
            return 1
    ok, sentence = digest_check(dst, want)
    if not ok:
        print(f"{PREFIX} WEIGHTS FAILED: {want['file']}: {sentence} — the file is left in place; remove it and run the step again to refetch", file=sys.stderr)
        return 1
    print(f"{PREFIX} WEIGHTS OK: 1/1 files in {directory} ({want['file']}: {sentence})")
    print(f"{PREFIX} weights: export ESM_IF1_WEIGHTS={dst}  (configs/*.env read it; or make {directory} your $TORCH_HOME/hub/checkpoints)")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m esm_if1_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return 2
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
