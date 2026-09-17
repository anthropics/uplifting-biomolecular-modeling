"""RULE: every broad exception handler in the opt_core package (kernels/** — the carried cells and their serve layers — and every other served
module) that reroutes around a lever or kernel on the SERVED path asks the core's one out-of-memory classifier first —
`if is_oom(e): raise` (opt_core.oom; a carried cell imports it lazily inside the handler) — so an out-of-memory reaches the caller instead of a
fallback.  One handler recovers device state before re-raising (a failed CUDA-graph capture must be ended and its pool released first) and is
named in CLEANUP_THEN_RERAISE.  Every other broad handler is not a reroute (import and capability probes, table/file reads, report fields,
cleanup and synchronisation guards, worker supervision, memory instruments) and is named in NOT_REROUTES per file and enclosing function with
multiplicity; the census must match exactly, so a new broad handler either asks is_oom first or is added here with its reason in review.
Development scripts (bench / sweep / debug / selftest / timing probe / verify_batch / exp_* / tests/) are out of scope."""
import ast
import os
from collections import Counter

import opt_core

PKG = os.path.dirname(opt_core.__file__)
DEV = ("bench_", "sweep_", "debug_", "selftest", "microbench", "verify_batch", "exp_exact", "/tests/", "_selftest")

CLEANUP_THEN_RERAISE = {("capture/graphs.py", "_capture_body"): 1}      # the capture handler: recover_after_failed_capture(...) then `if is_oom(e): raise e`

NOT_REROUTES = {
    "kernels/triattn_xla/_batching.py": ["custom_vmap_available"],       # probes jax.custom_batching once; the reason text is what report()/refusals quote (no work rerouted)
    "kernels/triattn_xla/_vjp.py": ["bwd_import_error"],                 # probes the backward's Pallas modules once; the import error is quoted in the refusal by name
    "kernels/triattn_xla/_launch.py": ["load", "load", "load", "load"], "kernels/trimul_xla/__init__.py": ["_cublas_selfcheck"],                          # installs the CUDA libraries' entries at registration when they load; a failure is recorded for report() and the row is refused by name at trace time (no work rerouted)
    "attn/apb_core.py": ['kernel'],                                            # the carried-kernel import probe (a failed import is the named refusal `import:apb_attn`)
    "kernels/apb/l3a/fab_batched.py": ['<module>'],                           # carried: the triton import guard (no triton -> the reference path / KoptAttnUnavailable by name)
    "kernels/apb/fpf_apb/apb_triton.py": ['<module>'],                        # carried: the host-side TMA descriptor import probe (_HAS_TMA False -> plain loads, same arithmetic)
    "kernels/apb/fpf_apb/install.py": ['install_atom_attn', 'install_dit_attn', 'install_pf_attn'],   # carried for byte identity, never called by the face: annotate + re-raise as RuntimeError
    "kernels/apb/ef2/ef2_pairbias_attn.py": ['<module>'],                     # carried: the image's transformers-fork import probe (TRITON_OK False -> refused by name has_esm)
    "kernels/ln/ef2/ef2_fused_ln.py": ['<module>'],                           # carried: the same fork's kernel import probe (KERNELS_OK False -> the row refuses by name)
    "kernels/ln/exactln/cubind.py": ['_e', '_e', '_nvrtc_candidates'],        # ctypes bindings: error-NAME lookups for a failing driver / NVRTC call (the error itself is raised) and the libnvrtc path probe
    "attn/pair_fused.py": ['_carried', '_carried', '_impl_string', '_stack', 'describe', 'pack_triattn_weights'],
    "attn/sdpa_bias.py": ['available_backends'],
    "attn/shared_bias_attn.py": ['ledger', 'triton2_dot_shim', 'triton_major', 'triton_major'],   # capability probes: the census impl word, the triton import / version probes (a failed import is the named refusal `triton_lt_3`, not a served-path fallback)
    "autoload.py": ['_apply', '_before_body', '_fire', '_fire', '_real_spec', 'install'],
    "capture/graphs.py": ['_capture_and_answer', '_capture_body', '_capture_body', '_capture_body', '_capture_body', '_capture_body', '_capture_body', '_release', '_release', '_torch', 'allocator_health_probe', 'end_dangling_capture', 'end_dangling_capture', 'numerics_mode', 'release_failed_capture_pool', 'release_failed_capture_pool', 'release_failed_capture_pool', 'reset', 'reset', 'reset_generator_capture_state', 'reset_generator_capture_state', 'reset_generator_capture_state', 'rng_health_probe'],
    "capture/hoist.py": ['capturing'],
    "of3_sampler/rollout_memo.py": ['_capturing', '_enter_rollout', 'active', 'arm_boundary', 'drop_all', 'refuse_all'],   # the epoch attribute on a foreign instance, the grad-mode probe, marker copies, store cleanup, a user's refuse callback: bookkeeping guards
    "capture/xla_cache.py": ['_jax', '_ser', 'enable_persistent_cache', 'stack_identity', 'stack_identity'],
    "of3_sampler/apb_trunk.py": ['census_line'],                           # the pair-bias core entry's census as a report field
    "of3_sampler/dit_glue.py": ['census_line'],                            # the pair-bias core entry's census as a report field
    "of3_sampler/dit_rows.py": ['resolve_apb_core', "resolve_apb_core", "_stack_word"],                      # the pair-bias core entry's import / kernel-route probe (a failure is the named reason, the dtk path serves)  # + the provider-word door: a second import/construct probe of the shared provider (named reason, the engine steps aside by name) and the stack-word probe
    "diffusion_loop/rng.py": ['_is_default_generator', '_position'],
    "gates.py": ['core_pin_check', 'dist_version', 'nvidia_smi_probe', 'run_gates'],
    "host/coldstart.py": ['fast_exit'],
    "host/memo.py": ['_disk_read', '_disk_write', '_disk_write'],
    "host/outputs.py": ['__del__', '_exit_guard', '_guard', '_rebuild', '_run', '_wait', 'snapshot', 'submit', 'submit', 'submit'],
    "host/workers.py": ['__init__', '_fill', '_take', 'close'],
    "host_cache/bg_writer.py": ['__del__', '_drain', '_guard', 'submit', 'submit'],
    "host_cache/resident.py": ['run_items', 'run_items'],
    "instances.py": ['find_spec'],
    "jax_arch.py": ['jax_device_memory'],
    "jax_design/pcc.py": ['backend_initialized', 'in_force'],
    "kernels/flash_triattn.py": ['<module>', '<module>', '_launch_generic'],
    "kernels/flash_triattn_serve.py": ['_compute_dtype_name', 'kernel_impl', 'kernel_module', 'probe', 'probe', 'probe'],
    "kernels/fpf_glue_v2/__init__.py": ['_dump', '_triton_mm', 'install', 'resolve_cells'],
    "kernels/fpf_glue_v2/kernels.py": ['<module>', '<module>', '_range_supports'],
    "kernels/fpf_mkpf/__init__.py": ['_eligible_triatt_shape', '_triton_mm', 'install'],
    "kernels/fpf_mkpf/_engine_adapter.py": ['_rebind_levers_prologue'],
    "kernels/fpf_mkpf/kernels.py": ['<module>', '<module>', '<module>', '_range_supports'],
    "kernels/fpf_pallas_serve.py": ['compute_capability', 'jax_line', 'probe', 'probe', 'probe'],
    "kernels/fpf_transition/transition.py": ['<module>', '<module>', '<module>', '_weights'],
    "kernels/transition/__init__.py": ['_try_stack', 'carried_module', '_liger_silu_mul', '_census', '_cc_memo', '_note_stepped_aside', '_note_substituted', '_note_refused_in_capture'],   # the stack-word probe without torch / a GPU (the reference stack's numbers, flagged) and the carried-module import probe (a failed import is the named refusal `import:<module>`)
    "kernels/transition/esm/ef2_autograd_kernels.py": ['<module>', '<module>', '<module>', '<module>'],   # carried: the ESM-family kit's import-time capability probes (libdevice exp, div_rn, the vendored LN backward, the K-D3 kernel build) whose reasons its own levers report
    "kernels/transition/esm/ef2_pair_v2.py": ['_t15_bind_resolve'],     # carried: the ESM inference kit's own lever resolving its binding to this provider by word (never called by the face; a failure keeps the kit-local kernel, named on its LEVER line)
    "kernels/transition/flash_sm90a/__init__.py": ['install'],           # carried: the sealed unit's install() LN-class probe (never called by the face; the kit's lever calls it)
    "kernels/fpf_triatt_epi/epilogue.py": ['<module>', '<module>', '<module>'],
    "kernels/fpf_triatt_k2b/flash_triattn_k2.py": ['<module>', '<module>', '_launch_generic', '_pf_load_device_table'],
    "kernels/fpf_triatt_k2b/triatt_k2b.py": ['<module>', '<module>', '_launch_generic', '_pf_load_device_table', '_triton_ge'],
    "kernels/fpf_triatt_pro/triatt.py": ['<module>', '<module>'],
    "kernels/fpf_trimul/engines.py": ['<module>'],
    "kernels/fpf_trimul/trimul.py": ['<module>'],
    "kernels/fpf_trimul_v4/cells.py": ['triton_mm'],
    "kernels/fpf_trimul_v4/trimul.py": ['_report'],
    "kernels/lnl_fused.py": ['_cc_key', '_tile_table', 'autotune_report'],
    "kernels/pallas_attn/af2_flash_pallas.py": ['make_flash_attention'],
    "kernels/pallas/cd_layers/layers_ln.py": ['<module>', '<module>', '_require'],             # carried: the stack import guards (jax / Pallas absent -> the module stays importable, install refuses by name) and the install-time requirement probe
    "kernels/pallas/cd_layers/layers_opm.py": ['<module>', '_require'],                        # carried: the same import guard and requirement probe
    "kernels/pallas/cd_layers/layers_transition.py": ['_require'],                             # carried: the install-time requirement probe (not on the served call path)
    "kernels/pallas/rowshared_flash_pallas.py": ['_carried_kernel', 'make_rowshared_attention'],   # carried: the sibling-kernel import probe and the index-width guard -> its named RowAttnRefusal
    "kernels/pallas/serve.py": ['_attention_row_at', '_attention_row_at', '_tokamax', "_walk", "_walk"],              # the stock library's own argument refusal and two optional-import probes -> the row's named Refusal (the face then serves the next measured arm by name)
    "kernels/pallas_attn_serve.py": ['_load_kernel', 'kernel_impl', 'probe', 'probe', 'probe', 'probe', 'served_attention_call'],
    "kernels/rfd_layernorm.py": ['<module>'],
    "kernels/triattn/triattn_native/__init__.py": ['_module'],   # the payload import under a private name -> the named refusal import:<exception> (never a reroute inside)
    "kernels/triattn/triattn_native/pkg/v11/test_pkg.py": ['env_record', 'env_record', 'env_record', 'env_record', 'run_case', 'verify'],   # carried (generation 11): the sealed package's own probes / self-test report
    "kernels/triattn/triattn_native/pkg/v11/tools/build_prebuilt.py": ['_ver'],   # carried (generation 11): the producer's builder, a toolchain version probe
    "kernels/triattn/triattn_native/pkg/v11/triattn_pkg/_face.py": ['_has_toolkit', '_has_toolkit', 'prebuilt_status'],   # carried (generation 11): toolkit / prebuilt-status probes, as generation 10
    "kernels/triattn_exact/cuda_mma/__init__.py": ['<module>', 'build', 'device_class', 'supports', 'supports'],   # carried: an optional-import probe at module load, the compiler-version probe (-> the typed refusal), the device-name probe and two capability probes answering (False, reason) -- metadata checks at resolve / select time, no served work rerouted
    "kernels/triattn_exact/_prebuilt/__init__.py": ['source_fingerprint'],   # carried: an unreadable-source-file skip inside the source fingerprint (the fingerprint then differs -> the cell refuses by name)
    "kernels/triattn_exact/_prebuilt/cuda_mma.py": ['supports'],   # carried: the capability probe that answers (False, reason) instead of raising -- a refusal by name before any launch
    "kernels/triattn_exact/_prebuilt/driver.py": ['_chk', '_chk2'],   # carried: best-effort decoding of the driver's error NAME while raising the typed error for a failed driver call (the error is raised either way)
    "kernels/triattn_exact/face.py": ['_compile_opaque', '_compile_opaque', '_dispatch_order', '_route', '_selfcheck', '_selfcheck', 'select_routes', 'triangle_attention'],   # carried: two probes that make the entry point opaque to torch.compile when that API exists (absent -> plain eager function); the optional route-ORDER table absent -> cell-table order; a route module that cannot import -> refused by name (the stock op serves); the once-per-process self-check against the library (library not importable -> refuse by name; a route that raises during the check -> that route is poisoned by name and the next proven route is tried); the memoised route selection's optional order lookup; a served route raising the typed refusal mid-dispatch -> the next proven route or the refusal by name -- never an untyped error inside a served call
    "kernels/triattn/cuda_sm90a/__init__.py": ['install'],           # carried: the prebuilt extension's load attempt -> its named Refused (the row's fallback is named by the face)
    "kernels/triattn/headsplit/__init__.py": ['_install_prologue', '_narrow_head', '_write_report', 'install'],   # carried: import probes of the kit-side prologue, a tensor tag copy, the exit report write, the install record -- none on the served call path
    "kernels/trimul/__init__.py": ['cueq_version', 'preload', 'stock_preload_once', '_census', '_census_stepaside', '_native_token'],           # the distribution-metadata version read ('?' on any metadata fault), preload()'s and stock_preload_once()'s per-module 'error:<word>' (imports ahead of time; never a serving path)
    "_pystack.py": ['padded_call', 'guarded_import'],                     # guarded_import: rolls the failed import's sys.modules keys back and RE-RAISES (never swallows)                       # the trampoline cannot be built -> the call is made directly (nothing rerouted, nothing swallowed: fn's own exceptions propagate)
    "warm.py": ['warm_imports', 'warm_imports', 'warm_imports', 'warm_imports', '_unavailable', 'auto_warm'],   # per-library guarded import -> 'unavailable:<Exc>' word (never a serving path); a closed stderr; the serving-path belt                          # warm_imports(): a library whose import fails is the word 'error:<Type>' in the report (the row that needs it refuses by name at its call); the trimul preload part likewise -- imports ahead of time, never a serving path
    "kernels/trimul/tx_sm90a/trimul.py": ['_report'],                    # carried: the kit plug's exit-report writer (never imported by the face; not on a served path)
    "kernels/trimul/tx_sm90a/trimul_exact.py": ['_report'],              # carried: as above, the exact plug
    "kernels/trimul/esm_v61/__init__.py": ['_install'],                   # the sealed package's cell word read for the install report: a fact line, never a reroute
    "kernels/trimul/esm_v61/pkg/v6.1/python/face.py": ['describe'],       # carried: the sealed package's own describe() (cell fact or 'unresolved: ...')
    "kernels/trimul/esm_v61/pkg/v6.1/src/ef2_trimul_v6.py": ['_jit_dir'], # carried: the compile-cache directory name (never reached: prebuilt cubins only, no compile in a serving process)
    "kernels/trimul/native/__init__.py": ['_device_key', 'face', 'face', 'install', 'module', 'stamp_dirs', 'stamp_dirs', 'stamp_dirs', 'stamp_facts', 'stamp_facts'],   # _device_key: the current-device probe for the check memo key (no torch / no device -> key 'None'); the sealed package imported by path under a private name -> the named refusal import:<exception>; install: the payload's own driver / loader error classes -> driver_unavailable:<name> by name (else re-raised); never a reroute + verification stamp: the other provider's cache-root helper absent -> the generic cache root; no framework / device facts -> no stamp (the gate runs; recorded as verdict_stamp none)
    "kernels/trimul/native/pkg/v5/python/trimul_native/_driver.py": ['error_name', 'error_name'],   # carried (1.2.2): the driver binding's error-name lookup (a report string)
    "kernels/trimul/native/pkg/v5/python/trimul_native/build.py": ['_naming', '_naming', 'build_unit', 'compute_serves'],   # carried (1.2.2): the package's build tool (never imported in a serving process)
    "kernels/trimul/native/pkg/v5/python/trimul_native/face.py": ['_notice', '_notice', 'check', 'serve'],   # carried (1.2.2): the information-token logger and its sinks (never raise); check(): the device-facts record; serve(): the notice sink
    "kernels/trimul/native/pkg/v5/python/trimul_native/manifest.py": ['elf_facts'],   # carried (1.2.2): cubin ELF facts for the build record (a report field)
    "kernels/trimul/native/pkg/v5/python/trimul_native/vectors.py": ['_cueq_fn', 'make', 'replay'],   # carried (1.2.2): the test-vector tool: reference-library probe and per-case error records (report fields)
    "mem/allocator.py": ['jax_state'],
    "mem/budget.py": ['device_free_bytes'],
    "mem/jax_mem.py": ['_set_child', '_set_child', '_set_child'],
    "mem/offload.py": ['cudart', 'cudart', 'cudart'],
    "mem/peak.py": ['_after_fork_in_child', '_after_fork_in_child', '_at_exit', '_at_exit', '_jax_backend_ready', '_loop', '_register_at_fork', '_register_at_fork', '_torch_cuda_ready', 'boot', 'boot', 'install_pth', 'tick', 'tick'],
    "mem/registry.py": ['setting'],
    "mem/rowpair/__init__.py": ['visible_gpus'],
    "mem/rowpair/_torch.py": ['_load'],
    "mem/rowpair/census.py": ['_emit', '_emit', '_rank_world', '_top_storages', '_top_storages', 'mark', 'mark'],
    "mem/rowpair/ckpt.py": ['_save', 'body', 'list_complete', 'phase', 'rng_state', 'save_tag', 'set_rng_state'],
    "mem/rowpair/dist.py": ['_hard_exit', '_leave_at_exit', '_who', '_who', 'call', 'host_wait', 'watch'],   # the bounded leave / exit guard: a raising collective-library call is handed to the caller (`call`), the exit lines are best-effort before os._exit (`_hard_exit`, `watch`, `_who`)
    "mem/rowpair/evidence.py": ['_impl'],
    "mem/rowpair/launch.py": ['_pdeathsig', '_teardown', '_teardown', '_worker', '_worker', '_worker'],
    "mem/rowpair/msa_host.py": ['is_pinned', 'sync_host_features_', 'to_host'],
    "mem/rowpair/triatt.py": ['attention_core', 'attention_core'],           # the cueq import (a failed import is the named refusal `cueq_missing`) and its version probe for the census impl word
    "mem/rowpair/trimul_fused.py": ['_resolve', '_resolve', '_resolve_rows', '_resolve_rows', 'describe'],   # triton / carried-kernel import probes (named refusals) x2 units; describe: the optional rows-unit manifest entry names its own failure
    "mem/rowpair_jax/_lazy.py": ['haiku', 'jax', 'np', 'shard_map_impl', 'sharding'],
    "mem/rowpair_jax/alphafold.py": ['install'],
    "mem/rowpair_jax/alphafold_heads.py": ['install_heads'],
    "mem/rowpair_jax/haiku.py": ['_internal_state_cm', 'wrap_method'],
    "mem/rowpair_jax/mesh.py": ['build', 'device_facts'],
    "mem/torch_alloc.py": ['effective', 'torch_state'],
    "mem/rowpair/structure_first.py": ['capture', 'naming'],               # capability / field probes of the structure-first write (no writer, malformed batch fields -> the named `skipped:` word; the heads run as usual)
    "mem/torch_hostpair.py": ['_itemsize', 'alloc_counted', 'host_cache_trim', 'host_cached_bytes'],   # + the pinned-pool shrink's memory instruments: the host-allocator statistics probe (`cached-free n/a`) and the trim call (`TRIM UNAVAILABLE: <error>` on the [pool] line)
    "mem/torch_rowchunk.py": ['_storage_ptr', 'release', 'torch_module'],
    "kernels/triattn/__init__.py": ['_census'],   # the cell-coverage census hook: classification of an already-decided selection; never gates
    "kernels/apb/__init__.py": ['_census', '_rows_census'],   # the cell-coverage census hooks (square face, rows face): classification of an already-decided selection; never gates
    "kernels/ln/__init__.py": ['_census'],   # the cell-coverage census hook: classification of an already-decided selection; never gates
    "ops/_fallback.py": ["<module>"],
    "tools/graph_audit/audit.py": ["audit_capture_safety"],   # an offline audit tool: a capture that raises is reported as a finding, nothing served          # import probe: the add-on registry's exception class when importable, else a local class of the same shape (no work rerouted)
    "ops/msa_opm/__init__.py": ["_tma_available", "device_cc"],   # probes: triton's tensor-descriptor import / the device capability, cached; the launch row is chosen from the answer, nothing runs inside
    "ops/msa_pwa2/__init__.py": ["<module>"],  # import probe: libdevice's module path differs across triton versions
    "kernels/pallas/__init__.py": ['_census', 'census_stepaside'],   # the cell-coverage census hook: classification of an already-decided selection; never gates
    "kernels/triattn_xla/__init__.py": ['_census'],   # the cell-coverage census hook: classification of an already-decided selection; never gates
    "report.py": ['_print_exit_tally', '_print_exit_tally', 'forced_exit'],
    "cell_census.py": ['_arm_atexit', 'record', 'write', '_atexit_print'],           # the cell-coverage census: a record / exit hook / dump write counts and reports, it never gates or breaks a served call
    "seq/det_torch.py": ['_cuda_initialised'],
    "seq/hostio.py": ['_emit', '_loop', '_loop'],
    "seq/lut.py": ['_np'],
    "seq/warm_witness.py": ['install', 'install', 'uninstall'],
    "testing.py": ['body', 'body', 'run_ranks'],
    "trimul.py": ['_cache', '_resolved_from'],
}


def _handlers():
    out = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests")]
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f); rel = os.path.relpath(path, PKG)
            if any(g in rel for g in DEV):
                continue
            tree = ast.parse(open(path, encoding="utf-8").read())
            parents = {}
            for node in ast.walk(tree):
                for ch in ast.iter_child_nodes(node):
                    parents[ch] = node

            def encl(n):
                while n in parents:
                    n = parents[n]
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return n.name
                return "<module>"
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                t = node.type
                names = ["<bare>"] if t is None else [getattr(e, "id", getattr(e, "attr", "?")) for e in (t.elts if isinstance(t, ast.Tuple) else [t])]
                if any(n in ("<bare>", "Exception", "BaseException") for n in names):
                    out.append((rel, encl(node), node))
    return out


def _is_oom_raise(stmt, name):
    t = getattr(stmt, "test", None)
    return (isinstance(stmt, ast.If) and isinstance(t, ast.Call) and getattr(t.func, "id", None) == "is_oom" and len(t.args) == 1
            and getattr(t.args[0], "id", None) == name and len(stmt.body) == 1 and isinstance(stmt.body[0], ast.Raise))


def _lazy_oom_import(stmt):
    """`from opt_core.oom import is_oom` — absolute, or the package-relative spelling (`from ..oom import is_oom`) — as the handler's first statement."""
    return isinstance(stmt, ast.ImportFrom) and (stmt.module == "opt_core.oom" or (stmt.module == "oom" and (stmt.level or 0) > 0)) \
        and any(a.name == "is_oom" for a in stmt.names)


def _asks_first(handler):
    body = list(handler.body)
    if body and _lazy_oom_import(body[0]):
        body = body[1:]
    return bool(body) and _is_oom_raise(body[0], handler.name)


def _asks_after_cleanup(handler):
    return any(_is_oom_raise(s, handler.name) for s in handler.body)



def served_reroutes():
    """(relpath, function) of every handler that asks is_oom (first, or after cleanup where named) — the list CHANGES prints."""
    return sorted((rel, fn, node.lineno) for rel, fn, node in _handlers() if _asks_first(node) or ((rel, fn) in CLEANUP_THEN_RERAISE and _asks_after_cleanup(node)))


def test_every_broad_handler_asks_is_oom_first_or_is_a_named_non_reroute():
    seen = Counter(); cleanup = Counter(); asking = []
    for rel, fn, node in _handlers():
        if _asks_first(node):
            asking.append((rel, fn))
        elif (rel, fn) in CLEANUP_THEN_RERAISE and _asks_after_cleanup(node):
            cleanup[(rel, fn)] += 1; asking.append((rel, fn))
        else:
            seen[(rel, fn)] += 1
    expected = Counter()
    for rel, fns in NOT_REROUTES.items():
        for fn in fns:
            expected[(rel, fn)] += 1
    unexpected = sorted((k, n - expected.get(k, 0)) for k, n in seen.items() if n > expected.get(k, 0))
    stale = sorted((k, n - seen.get(k, 0)) for k, n in expected.items() if n > seen.get(k, 0))
    assert unexpected == [], "broad handlers that neither ask is_oom first nor are named non-reroutes: %r" % unexpected
    assert stale == [], "named non-reroute handlers absent from the tree (shrink the table): %r" % stale
    assert dict(cleanup) == CLEANUP_THEN_RERAISE, dict(cleanup)
    assert len(asking) == 76, asking          # + kernels/triattn.triangle_attention x1 (0.5.225.0: the triattn_exact member's install at first serve — is_oom first, re-raise; else refused_at_install by name) # + mem/rowpair/triatt._free_device_bytes x1 (0.5.220.5: the no_room guard's allocator-statistics read — is_oom first, re-raise; else the driver's number alone) # + mem/rowpair/triatt.StageSlot._plan x1 (0.5.220.4 the per-plane key plan — is_oom first, re-raise; else per-window compares BY NAME) # + mem/rowpair/diffusion.DitBias._ln_proj / .ln_proj_into (0.5.218.1 the rows producer's packer / launcher guards: is_oom first, re-raise; else the engine statement by name) # + mem/rowpair/triatt x6 + mem/rowpair/pairstack x1 (0.5.218.0 the stage-once glue's _native_member x2 / _tier_sel / _big_policy and attention_core's native-whole-window + staged-window step-asides, and pair_stack_'s opt-in block log — is_oom first, re-raise; else per call / qblocks / print BY NAME) + kernels/apb.pair_bias_attention_rows x2 (0.5.216.0: the rows face's apb_attn / sba launch guards — is_oom first, re-raise; else a refusal by name) # kernels/trimul.triangle_multiplication (the inherited-row launch guard: is_oom first, re-raise; else the row steps aside by name) + kernels/ln.layer_norm + kernels/apb.pair_bias_attention (the inherited-row launch guard: is_oom first, re-raise; else retire the arm and serve the next row) + kernels/ln.layer_norm + kernels/apb.pair_bias_attention (the inherited-row launch guard: is_oom first, re-raise; else retire the arm and serve the next row) + kernels/ln.layer_norm + kernels/apb.pair_bias_attention (the inherited-row launch guard: is_oom first, re-raise; else retire the arm and serve the next row) + mem/rowpair/triatt.attention_core (the tier door: a provider refusal asks is_oom before stepping aside to the flash path) + of3_sampler/dit_rows.pair_bias_attention (a pre-warm failure asks is_oom before it becomes a census token) + attn/pair_fused._plan_triattn (engage-cell facts unreadable: asks is_oom, else no cell applies) + kernels/ln/ext_loader.load (an ABI/arch load error asks is_oom before naming itself Unavailable); kernels/triattn._install_at_resolve: a sealed row's resolve-time load error asks is_oom first (re-raise) before naming the row refused; kernels/triattn.warm: the activation hook's catch-all asks is_oom first (re-raise) before reporting a row not-ready by name; kernels/trimul/esm_v61: install()'s catch-all asks is_oom first (re-raise) before naming a load failure a refusal; kernels/apb: pair_bias_attention's shared-bias row and pair_bias_planes' lnl row (the module's named refusal re-raised as this face's word after is_oom; anything else re-raised) x2; kernels/transition: transition()'s lnl branch and _serve_v2 (a tile / launch word that fails to BUILD is refused by name after is_oom; anything else re-raised) x2; kernels/trimul/tx_sm90a plugs fn x2 (carried: is_oom first, else the stock forward by name); of3_trunk tuner_guard install (the engine's tuner class import) + castcache (source guard, install) + apb_hoist (permute import probe, install) + trunk_graph (nmax parse, failed capture -> eager by name) and of3_sampler/post_release (token count / free-memory reads, the two release-point installs) x10: an OOM re-raised before their by-name refusals; trimul Lever.serve; of3_sampler/atom_window plan + fused block (-> the stock block, counted) + construction hook (naming / JIT warm-up) x3, an OOM re-raised; of3_sampler/apb_trunk forward (plan + served path: an OOM is the caller's, anything else runs the stock forward counted); of3_sampler/rollout_memo _enter_rollout (a user's boundary callback) + _on_first_sampler_call (graphs cooperation arming: a failure refuses the memo levers by name, an OOM is re-raised); of3_sampler/atom_hoist get_atom_reps (a namespace fill: fatal by name under a live graph, else counted) / block-utils key building; of3_sampler/token_agg served path (-> the stock aggregation counted, fatal by name once a graph holds the kernel) (-> the stock statement) and of3_sampler/dit_glue plan + schedule (-> the stock block, counted; an OOM is re-raised) x7 together; attn/shared_bias_attn attention; mem/rowpair/triatt attention_core's cueq-reject reroute (a launch / compile failure is the named refusal `launch:<type>`, an OOM is re-raised); graphs warm-up + capture; xla_cache x4; flash_triattn x2; fpf_triatt_k2b x4; fpf_trimul_v4 fn; precision/probe IdentityProbe.cell (a raising candidate is a failed cell, an OOM is re-raised); kernels/rfd_layernorm _launch (the carried fast-launch guard asks is_oom before falling to the JIT launch); of3_trunk/templ_embed _make_forward (a feature dict / module outside the checked layout is refused by name, an OOM re-raised) + of3_sampler/dit_glue (the class-wide conditioned-transition / AdaLN rows: is_oom first, else the stock method by name, x2)


def test_no_second_out_of_memory_classifier_in_the_package():
    hits = []
    for root, dirs, files in os.walk(PKG):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "tests")]
        for f in files:
            if f.endswith(".py") and os.path.join(root, f) != os.path.join(PKG, "oom.py"):
                if "def is_oom" in open(os.path.join(root, f), encoding="utf-8").read():
                    hits.append(os.path.relpath(os.path.join(root, f), PKG))
    assert hits == [], hits
