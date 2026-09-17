"""stock/PINS.json vs the stock files present in the tree; the pinned stack's four pins; the must-be-absent names are the kit-side
names and nothing of the caller's own; stack.stack_key()'s environment-fingerprint string (a missing component named, never a bare "unknown")."""
import os

from progen2_opt import modes, stack, stock_cli


def test_pins_match_stock_bytes(tree):
    p = stack.pins()
    assert p["upstream"]["progen"]["commit"] == "c27a419c234a0997923761e1fe7daffcebf0eaf5"
    sd = os.path.join(tree, "stock", "src", "progen2")
    files = stack.stock_file_pins(p)
    for rel in files:
        assert os.path.isfile(os.path.join(sd, rel)), rel
    assert set(stack.STOCK_FILES) <= set(files)
    ok, detail, why = stack.pins_gate(p)
    assert detail["files"] and all(v == "ok" for v in detail["files"].values()), detail["files"]
    assert ok or why.startswith("pinned stack not installed"), why           # this box is not the pinned stack: the gate says so by name (no override)


def test_pinned_stack_pins():
    st = stack.stack_pins(stack.pins())
    assert st["torch"] == "2.8.0+cu128" and st["transformers"] == "4.16.2" and st["tokenizers"] == "0.10.3" and str(st["python"]).startswith("3.9")


def test_weights_pinned_for_every_variant():
    p = stack.pins()
    for v in modes.VARIANTS:
        up = modes.UPSTREAM_NAME[v]
        assert len(stack.weight_pins(p, up)["pytorch_model.bin"]) == 64, up


def test_env_absent_spec_is_the_kit_side_names_only():
    spec = stock_cli.env_absent_spec(stack.pins())
    for name in ("PROGEN2_KIT", "PROGEN2_KIT_SOCKET", "PROGEN2_KIT_OFF", "PROGEN2_ANYTHING"):        # the whole PROGEN2_ family (the package's variables; the kits read none of their own)
        assert stock_cli._forbidden({name: "1"}, spec) == [name], name
    for name in ("CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "ATEN_CPU_CAPABILITY"):          # the caller's own settings are theirs: neither stripped nor added
        assert stock_cli._forbidden({name: "1"}, spec) == [], name
    assert stock_cli._forbidden({"PROGEN2_STOCK_DIR": "x"}, spec) == ["PROGEN2_STOCK_DIR"]      # the prefix catches it; clean_env keeps it via the allowed list
    allowed = stock_cli.env_allowed(stack.pins(), stack.DATA_ENV)
    assert {"PROGEN2_STOCK_DIR", "PROGEN2_WEIGHTS", "PROGEN2_RUN_DIR", "PROGEN2_PYTHON"} <= set(allowed)
    env = stock_cli.clean_env({"PROGEN2_KIT": "v0_ew", "PROGEN2_STOCK_DIR": "/s", "PROGEN2_WEIGHTS": "/w", "PATH": "/bin", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}, spec, allowed)
    assert env == {"PROGEN2_STOCK_DIR": "/s", "PROGEN2_WEIGHTS": "/w", "PATH": "/bin", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONDONTWRITEBYTECODE": "1"}
    assert stock_cli.CLEAN_RECORD == {"stripped": ["PROGEN2_KIT"], "kept": ["PROGEN2_STOCK_DIR", "PROGEN2_WEIGHTS"]}
    assert stock_cli.env_proof(spec, environ={"PROGEN2_PYTHON": "/o"}, modules={}, path=[], allowed=allowed)["forbidden_present"] == []
    import inspect
    assert "det" not in inspect.signature(stock_cli.clean_env).parameters and "det_level" not in inspect.signature(stock_cli.env_absent_spec).parameters


def test_stack_key_happy_path_unchanged():
    assert stack.stack_key({"cc": "9.0"}) == "torch2.8.0-cu128-sm90" if stack._dist_version("torch") == "2.8.0+cu128" \
        else stack.stack_key({"cc": "9.0"}).startswith("torch")   # the box's own torch version if this suite runs off-stack


def test_stack_key_no_gpu_is_named_not_bare_unknown():
    key = stack.stack_key({"cc": None})
    assert key.endswith("-smunknown:no_gpu") and "smunknown:no_gpu" in key and "smunknown-" not in key


def test_stack_key_no_torch_is_named_not_bare_unknown(monkeypatch):
    monkeypatch.setattr(stack, "_dist_version", lambda name: None)
    key = stack.stack_key({"cc": "9.0"})
    assert key == "torchunknown:no_torch-cuunknown:no_torch-sm90"
    assert "unknown-" not in key.split("-sm")[0].replace("unknown:no_torch", "")   # no bare 'unknown' token distinct from the named form


def test_stack_key_no_cuda_suffix_is_named_not_bare_unknown(monkeypatch):
    monkeypatch.setattr(stack, "_dist_version", lambda name: "2.8.0+cpu")   # a CPU-only torch build: no '+cuNNN' local version
    key = stack.stack_key({"cc": "9.0"})
    assert key == "torch2.8.0-cuunknown:no_cuda_suffix-sm90"
