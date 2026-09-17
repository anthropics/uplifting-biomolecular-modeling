"""The image recipe's source checkout of a dependency fails closed: the step that clones NVIDIA/TransformerEngine and checks out the pinned
commit runs under an effective `set -e` (set before any command of the step, with the step's commands joined by `;` — `set -e` is void
inside an `if … fi` that is itself a non-final member of a `&&` list), and the commit that HEAD lands on is compared with the pin, so a
failed or wrong checkout stops the build instead of compiling whatever the default branch holds (environment/Dockerfile,
environment/build_wheels.sh; the pin is stock/PINS.json pinned_stack: transformer_engine 2.15.0+42b8400)."""
import os
import re

from .test_installer_pinned import _logical_lines

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
DOCKERFILE = os.path.join(KIT, "environment", "Dockerfile")
BUILD_WHEELS = os.path.join(KIT, "environment", "build_wheels.sh")
TE_COMMIT = "42b840051647eef89761a16dfdff87e82bb253ab"

_CLONE = re.compile(r"\bgit clone\b[^;&|]*TransformerEngine")
_HEAD_TEST = re.compile(r'test "\$\(git -C \S*TransformerEngine\S* rev-parse HEAD\)" = ' + TE_COMMIT)


def _checkout_step():
    """(line, text) of the one RUN step of the Dockerfile that clones TransformerEngine (continuations joined)."""
    steps = [(n, text) for n, text in _logical_lines(DOCKERFILE) if text.startswith("RUN ") and _CLONE.search(text)]
    assert len(steps) == 1, steps
    return steps[0]


def test_pin_agrees_with_pins_json():
    import json
    pins = json.load(open(os.path.join(KIT, "stock", "PINS.json")))
    assert pins["pinned_stack"]["transformer_engine"].endswith("+" + TE_COMMIT[:7]), pins["pinned_stack"]["transformer_engine"]


def test_checkout_step_runs_under_an_effective_set_e():
    n, text = _checkout_step()
    body = text[len("RUN "):].lstrip()
    assert body.startswith("set -e;") or body.startswith("set -eu;"), "environment/Dockerfile:%d: the step does not begin with `set -e`" % n
    assert not re.search(r"\bthen\s+set -e\b", body), "environment/Dockerfile:%d: `set -e` inside an if-branch of the step (void when the `if` is part of a `&&` list)" % n
    assert not re.search(r"\b(fi|done|esac)\s*&&", body), "environment/Dockerfile:%d: a compound command of the step is a non-final member of a `&&` list (its commands would not stop the build under `set -e`)" % n


def test_checkout_step_verifies_head_is_the_pinned_commit():
    n, text = _checkout_step()
    assert ("checkout -q " + TE_COMMIT) in text or ("checkout --quiet " + TE_COMMIT) in text, "environment/Dockerfile:%d: no checkout of the pinned commit" % n
    assert _HEAD_TEST.search(text), "environment/Dockerfile:%d: HEAD is not compared with the pinned commit after the checkout" % n
    clone_at = _CLONE.search(text).start()
    build_at = text.find("pip wheel", clone_at)
    assert clone_at < _HEAD_TEST.search(text).start() < build_at, "environment/Dockerfile:%d: the HEAD comparison must sit between the clone and the wheel build" % n


def test_build_wheels_script_fails_closed_too():
    text = open(BUILD_WHEELS, encoding="utf-8").read()
    assert re.search(r"^set -euo pipefail$", text, re.M), "environment/build_wheels.sh: no `set -euo pipefail` at top level"
    assert ("checkout --quiet " + TE_COMMIT) in text
    assert re.search(r'test "\$\(git -C \S*TransformerEngine\S* rev-parse --short HEAD\)" = ' + TE_COMMIT[:7], text)
