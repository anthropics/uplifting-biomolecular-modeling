"""The evidence lines of a design pass — route ``kit`` (``--mode fast``, the batched driver: every line below) or route ``stock`` (``--mode off``,
upstream's script in the proven child: PEAK and KERNELS only, printed after the script returns) — clocks, memory counters and version strings in ONE grammar.

This module is standard-library only and imports nothing of the package or of the shared core, so the proven stock child holds no core
module beyond the proof machinery when it runs (``stock_design.py``). ``torch`` and ``esm`` are read through ``sys.modules`` when the
process has imported them; nothing here moves a tensor, seeds or reads an RNG, or touches a model — every function reads clocks, memory
counters and version strings and prints. CUDA is synchronized before a model clock starts and before it stops so queued kernels are inside
the span (a synchronize changes no numerics).

Lines (stderr, flushed; ``[esm_if1-opt]`` prefix; ``<r>`` = ``kit`` on the driver's lines, ``stock`` on the two the stock child prints; the ``*_RE``
patterns below match the ``route=stock`` spelling)::

    [esm_if1-opt] PREPASS route=<r> pid=<pid> t_start=<unix s> t_end=<unix s> wall_s=<s> n=<backbones parsed>
    [esm_if1-opt] STARTUP route=<r> pid=<pid> done_ts=<unix s> load_s=<s> esm=<version> torch=<version> device=<cuda|cpu>
    [esm_if1-opt] CALL route=<r> pid=<pid> name=<str> i=<k> t_start=<unix s> t_end=<unix s> wall_s=<s>
    [esm_if1-opt] ITEM route=<r> pid=<pid> unit=batch name=<str> n_backbones=<int> n_seq=<int> length=<int> t_start=<unix s> t_end=<unix s> wall_s=<s> model_s=<s> call_s=<s,s,...> alloc_gib=<GiB> reserved_gib=<GiB>
    [esm_if1-opt] OUTPUTS_WRITTEN route=<r> pid=<pid> t=<unix s> n_files=<int> n_records=<int>
    [esm_if1-opt] PEAK route=<r> pid=<pid> item=<last unit name|pass> alloc_gib=<GiB> reserved_gib=<GiB> device=<name|none>
    [esm_if1-opt] KERNELS route=<r> pid=<pid> esm=<version> esm_tree=<sha256[:16]> upstream=<commit8> torch=<v> cuda=<v> cudnn=<int> python=<v> pyg=<v> torch_scatter=<importable@version|absent> device=<name|none> cc=<smNN|none> batch=<B> tf32_matmul=<bool> inference_mode=<bool> verdict=report

Semantics. All ``t_*`` / ``done_ts`` / ``t`` stamps are ``time.time()`` (unix seconds, comparable across processes and with a launcher's
own stamps); durations come from ``time.perf_counter()``. PREPASS = structure parsing (upstream ``load_coords``: file -> backbone coordinates
+ native sequence): the driver parses every input BEFORE the model load (``n`` = all backbones). STARTUP ``load_s`` = checkpoint read +
module construction + ``.eval()`` + the move to the device + the seed; ``done_ts`` is the model-ready instant. CALL = ONE model invocation
(one batched forward over B rows) with a synchronize on each side: ``t_start``/``t_end`` are its unix stamps, ``i`` its 1-based index within
the current unit, so an external device sampler can be integrated over exactly the in-model windows. ITEM = one timed unit, ``unit=batch``
(one forward batch of B (backbone, sample) rows: ``t_start`` = its forward begins, ``t_end`` = its records written; ``n_backbones`` = distinct
backbones in the batch, ``length`` = their common length); ``model_s`` = the sum of the unit's CALL spans and ``call_s`` lists them in order,
so ``wall_s - model_s`` is the CPU work around the model. ``alloc_gib``/``reserved_gib`` on ITEM are the allocator's running high-water marks when
the line prints (the process's peak so far; nothing resets them) — diagnostics only. PEAK is the allocator's whole-process high-water mark (``max_memory_allocated`` /
``max_memory_reserved``), printed once at the end with ``item=pass``. OUTPUTS_WRITTEN stamps the instant the last FASTA is closed. KERNELS is
the census of what actually ran: the ``esm`` version and the digest of the imported ``esm`` package's ``*.py`` tree (which installed copy of
upstream ran), the torch/CUDA build, the ``torch_geometric`` version and whether ``torch_scatter`` imports (``importable@version`` |
``absent``), the device, the batch size, and the numerics switches as READ (never set) at exit (``tf32_matmul``, ``inference_mode``).

``ESM_IF1_TIMING_JSONL=<path>``: when set, every line is also appended there as one JSON object (``{"kind": ..., <fields>}``) — the pass's
``timing.jsonl``; the manifest embeds these records.
"""
import hashlib
import json
import os
import re
import sys
import time

TAG = "esm_if1-opt"
PREFIX = "[" + TAG + "]"
ROUTES = ("stock", "kit")                        # the route word a line carries (modes.STOCK: upstream's script under off; modes.KIT: the batched driver under fast; held equal by the tests)
UNITS = ("batch",)
TIMING_ENV = "ESM_IF1_TIMING_JSONL"                     # a data path: where the JSON copy of each line goes (unset: lines only)
UPSTREAM_COMMIT = "2b369911bb5b4b0dda914521b9475cad1656b2ac"   # upstream facebookresearch/esm commit of the carried archive (stack.UPSTREAM_COMMIT; held equal by the tests); printed as upstream=<commit8>

PREPASS_FMT = PREFIX + " PREPASS route=%s pid=%d t_start=%.3f t_end=%.3f wall_s=%.3f n=%d"
OUTPUTS_WRITTEN_FMT = PREFIX + " OUTPUTS_WRITTEN route=%s pid=%d t=%.3f n_files=%d n_records=%d"
STARTUP_FMT = PREFIX + " STARTUP route=%s pid=%d done_ts=%.3f load_s=%.3f esm=%s torch=%s device=%s"
CALL_FMT = PREFIX + " CALL route=%s pid=%d name=%s i=%d t_start=%.3f t_end=%.3f wall_s=%.3f"
ITEM_FMT = (PREFIX + " ITEM route=%s pid=%d unit=%s name=%s n_backbones=%d n_seq=%d length=%d t_start=%.3f t_end=%.3f wall_s=%.3f"
            " model_s=%.3f call_s=%s alloc_gib=%.3f reserved_gib=%.3f")
PEAK_FMT = PREFIX + " PEAK route=%s pid=%d item=%s alloc_gib=%.3f reserved_gib=%.3f device=%s"
KERNELS_FMT = (PREFIX + " KERNELS route=%s pid=%d esm=%s esm_tree=%s upstream=%s torch=%s cuda=%s cudnn=%s python=%s pyg=%s torch_scatter=%s"
               " device=%s cc=%s batch=%d tf32_matmul=%s inference_mode=%s verdict=report")

_P = re.escape(PREFIX)
PREPASS_RE = _P + r" PREPASS route=(?P<route>stock) pid=(?P<pid>\d+) t_start=(?P<t_start>[\d.]+) t_end=(?P<t_end>[\d.]+) wall_s=(?P<wall_s>[\d.]+) n=(?P<n>\d+)"
OUTPUTS_WRITTEN_RE = _P + r" OUTPUTS_WRITTEN route=(?P<route>stock) pid=(?P<pid>\d+) t=(?P<t>[\d.]+) n_files=(?P<n_files>\d+) n_records=(?P<n_records>\d+)"
STARTUP_RE = _P + r" STARTUP route=(?P<route>stock) pid=(?P<pid>\d+) done_ts=(?P<done_ts>[\d.]+) load_s=(?P<load_s>[\d.]+) esm=(?P<esm>\S+) torch=(?P<torch>\S+) device=(?P<device>\S+)"
CALL_RE = _P + r" CALL route=(?P<route>stock) pid=(?P<pid>\d+) name=(?P<name>\S+) i=(?P<i>\d+) t_start=(?P<t_start>[\d.]+) t_end=(?P<t_end>[\d.]+) wall_s=(?P<wall_s>[\d.]+)"
ITEM_RE = (_P + r" ITEM route=(?P<route>stock) pid=(?P<pid>\d+) unit=(?P<unit>batch) name=(?P<name>\S+) n_backbones=(?P<n_backbones>\d+)"
           r" n_seq=(?P<n_seq>\d+) length=(?P<length>\d+) t_start=(?P<t_start>[\d.]+) t_end=(?P<t_end>[\d.]+) wall_s=(?P<wall_s>[\d.]+)"
           r" model_s=(?P<model_s>[\d.]+) call_s=(?P<call_s>[\d.]+(?:,[\d.]+)*|none) alloc_gib=(?P<alloc_gib>[\d.]+) reserved_gib=(?P<reserved_gib>[\d.]+)")
PEAK_RE = _P + r" PEAK route=(?P<route>stock) pid=(?P<pid>\d+) item=(?P<item>\S+) alloc_gib=(?P<alloc_gib>[\d.]+) reserved_gib=(?P<reserved_gib>[\d.]+) device=(?P<device>\S+)"
KERNELS_RE = (_P + r" KERNELS route=(?P<route>stock) pid=(?P<pid>\d+) esm=(?P<esm>\S+) esm_tree=(?P<esm_tree>[0-9a-f]{16}|none) upstream=(?P<upstream>[0-9a-f]{8})"
              r" torch=(?P<torch>\S+) cuda=(?P<cuda>\S+) cudnn=(?P<cudnn>\S+) python=(?P<python>\S+) pyg=(?P<pyg>\S+) torch_scatter=(?P<torch_scatter>\S+)"
              r" device=(?P<device>\S+) cc=(?P<cc>\S+) batch=(?P<batch>\d+) tf32_matmul=(?P<tf32_matmul>True|False) inference_mode=(?P<inference_mode>True|False) verdict=report")


# ------------------------------------------------------------------------------------------------------------------ primitives


def _torch():
    """The ``torch`` module when the process has imported it, else None (this module never imports a framework)."""
    return sys.modules.get("torch")


def cuda_sync():
    """``torch.cuda.synchronize()`` when torch is imported and CUDA is initialised and available; a no-op otherwise."""
    t = _torch()
    if t is not None and t.cuda.is_available() and t.cuda.is_initialized():
        t.cuda.synchronize()


def mem_gib():
    """(max_memory_allocated, max_memory_reserved) in GiB of the current device — (0.0, 0.0) without CUDA."""
    t = _torch()
    if t is None or not t.cuda.is_available() or not t.cuda.is_initialized():
        return 0.0, 0.0
    return t.cuda.max_memory_allocated() / 2 ** 30, t.cuda.max_memory_reserved() / 2 ** 30


def device_label():
    """``<GPU_name>`` of device 0 (spaces spelled ``_``: every line value is one token) when torch sees CUDA, else ``none``."""
    t = _torch()
    if t is None or not t.cuda.is_available():
        return "none"
    return "_".join(str(t.cuda.get_device_name(0)).split()) or "none"


def device_cc():
    """``smNN`` of device 0, else ``none``."""
    t = _torch()
    if t is None or not t.cuda.is_available():
        return "none"
    major, minor = t.cuda.get_device_capability(0)
    return "sm%d%d" % (major, minor)


def tree_digest(pkg_dir):
    """(sha256 hex, n_files): a digest of a package directory's Python sources — sha256 over ``<relpath>\\0<file sha256>\\n`` for every ``*.py``
    under ``pkg_dir`` in sorted relpath order (``__pycache__`` skipped). The KERNELS census prints its first 16 hex digits for the ``esm``
    package the process imported."""
    entries = []
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith(".py"):
                p = os.path.join(root, f)
                h = hashlib.sha256()
                with open(p, "rb") as fh:
                    for b in iter(lambda: fh.read(1 << 20), b""):
                        h.update(b)
                entries.append((os.path.relpath(p, pkg_dir).replace(os.sep, "/"), h.hexdigest()))
    entries.sort()
    top = hashlib.sha256()
    for rel, digest in entries:
        top.update(rel.encode() + b"\0" + digest.encode() + b"\n")
    return top.hexdigest(), len(entries)


def esm_facts():
    """``{"esm": version, "esm_dir": dir, "esm_tree": sha256, "esm_tree_files": n}`` of the ``esm`` package THIS process imported (``none`` values
    before the import)."""
    esm = sys.modules.get("esm")
    if esm is None or not getattr(esm, "__file__", None):
        return {"esm": "none", "esm_dir": None, "esm_tree": "none", "esm_tree_files": 0}
    d = os.path.dirname(os.path.abspath(esm.__file__))
    version = getattr(sys.modules.get("esm.version"), "version", None) or getattr(esm, "__version__", "unknown")
    digest, n = tree_digest(d)
    return {"esm": str(version), "esm_dir": d, "esm_tree": digest, "esm_tree_files": n}


def _dist_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "absent"


def scatter_label():
    """``importable@<version>`` when ``torch_scatter`` imports in this process, ``absent`` otherwise (ESM-IF1's forward never calls it:
    gvp_modules.py:438-441). Importing it here loads a compiled extension only; no numerics."""
    try:
        import importlib
        m = importlib.import_module("torch_scatter")
        return "importable@" + str(getattr(m, "__version__", _dist_version("torch_scatter")))
    except Exception:
        return "absent"


def emit(text):
    sys.stderr.write(text + "\n")
    sys.stderr.flush()
    return text


def _record(kind, fields):
    path = os.environ.get(TIMING_ENV)
    if not path:
        return
    rec = dict(kind=kind, **fields)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------------------------------ the clocks of one pass

class Clocks:
    """The per-process state behind the lines: the load clock (STARTUP), the current unit's first-sample stamp and model seconds (ITEM)."""

    def __init__(self, route):
        if route not in ROUTES:
            raise ValueError("route %r is not one of %s" % (route, ROUTES))
        self.route = route
        self.pid = os.getpid()
        self.load_t0 = None
        self.unit_t_start = None          # wall stamp of the current unit's first model clock start
        self.model_s = 0.0
        self._m0 = None
        self._m0_wall = None
        self.n_calls = 0
        self.call_s = []
        self.length = 0
        self.name = None
        self.prepass_t0 = None

    # PREPASS
    def prepass_begin(self):
        """Before the structure parsing (the loop over all inputs)."""
        self.prepass_t0 = time.time()

    def prepass_done(self, n):
        """After it: print the PREPASS line (``n`` structures parsed)."""
        t_end = time.time()
        t_start = self.prepass_t0 if self.prepass_t0 is not None else t_end
        _record("PREPASS", dict(route=self.route, pid=self.pid, t_start=round(t_start, 3), t_end=round(t_end, 3), wall_s=round(t_end - t_start, 3), n=int(n)))
        self.prepass_t0 = None
        return emit(PREPASS_FMT % (self.route, self.pid, t_start, t_end, t_end - t_start, int(n)))

    # STARTUP
    def load_begin(self):
        self.load_t0 = time.time()

    def load_done(self, device):
        now = time.time()
        load_s = now - self.load_t0 if self.load_t0 is not None else 0.0
        t = _torch()
        f = esm_facts()
        fields = dict(route=self.route, pid=self.pid, done_ts=round(now, 3), load_s=round(load_s, 3), esm=f["esm"],
                      torch=(t.__version__ if t is not None else "none"), device=device)
        _record("STARTUP", fields)
        return emit(STARTUP_FMT % (self.route, self.pid, now, load_s, fields["esm"], fields["torch"], device))

    # ITEM
    def begin_unit(self, name, length):
        """A unit is named (the batch's rows are chosen); its t_start is the first model span's start."""
        self.name, self.length, self.model_s, self._m0, self.n_calls, self.call_s = str(name).replace(" ", "_"), int(length), 0.0, None, 0, []
        self.unit_t_start = None

    def model_begin(self):
        """Before a model span (the batch's encoder + decode loop). The first one of a unit is its t_start."""
        cuda_sync()
        self._m0, self._m0_wall = time.perf_counter(), time.time()
        if self.unit_t_start is None:
            self.unit_t_start = self._m0_wall

    def model_end(self):
        """After a model span: accumulate model_s and print the span's CALL line."""
        cuda_sync()
        if self._m0 is not None:
            span = time.perf_counter() - self._m0
            self.model_s += span
            self.call_s.append(span)
            t_end = time.time()
            self.n_calls += 1
            fields = dict(route=self.route, pid=self.pid, name=str(self.name), i=self.n_calls, t_start=round(self._m0_wall, 3), t_end=round(t_end, 3),
                          wall_s=round(t_end - self._m0_wall, 3))
            _record("CALL", fields)
            emit(CALL_FMT % (self.route, self.pid, str(self.name), self.n_calls, self._m0_wall, t_end, t_end - self._m0_wall))
            self._m0 = self._m0_wall = None

    def item_done(self, unit, n_backbones, n_seq, name=None):
        """The unit's outputs are written: print the ITEM line."""
        if unit not in UNITS:
            raise ValueError("unit %r is not one of %s" % (unit, UNITS))
        t_end = time.time()
        t_start = self.unit_t_start if self.unit_t_start is not None else t_end
        alloc, reserved = mem_gib()
        nm = str(name if name is not None else self.name).replace(" ", "_")
        call_s = ",".join("%.3f" % c for c in self.call_s) or "none"
        fields = dict(route=self.route, pid=self.pid, unit=unit, name=nm, n_backbones=int(n_backbones), n_seq=int(n_seq), length=int(self.length),
                      t_start=round(t_start, 3), t_end=round(t_end, 3), wall_s=round(t_end - t_start, 3), model_s=round(self.model_s, 3),
                      call_s=[round(c, 3) for c in self.call_s], alloc_gib=round(alloc, 3), reserved_gib=round(reserved, 3))
        _record("ITEM", fields)
        line = emit(ITEM_FMT % (self.route, self.pid, unit, nm, int(n_backbones), int(n_seq), int(self.length), t_start, t_end, t_end - t_start,
                                self.model_s, call_s, alloc, reserved))
        self.unit_t_start, self.model_s, self._m0, self.n_calls, self.call_s = None, 0.0, None, 0, []
        return line

    # OUTPUTS_WRITTEN
    def outputs_written(self, n_files, n_records):
        """The pass's last FASTA is written and closed."""
        t = time.time()
        _record("OUTPUTS_WRITTEN", dict(route=self.route, pid=self.pid, t=round(t, 3), n_files=int(n_files), n_records=int(n_records)))
        return emit(OUTPUTS_WRITTEN_FMT % (self.route, self.pid, t, int(n_files), int(n_records)))

    # PEAK + KERNELS
    def finish(self, batch, inference_mode):
        """The pass's last two lines: PEAK (``item=pass``: the whole-process high-water mark) then KERNELS."""
        alloc, reserved = mem_gib()
        dev = device_label()
        item = "pass"
        _record("PEAK", dict(route=self.route, pid=self.pid, item=item, alloc_gib=round(alloc, 3), reserved_gib=round(reserved, 3), device=dev))
        emit(PEAK_FMT % (self.route, self.pid, item, alloc, reserved, dev))
        t = _torch()
        f = esm_facts()
        fields = dict(route=self.route, pid=self.pid, esm=f["esm"], esm_tree=(f["esm_tree"][:16] if f["esm_tree"] != "none" else "none"),
                      esm_tree_sha256=f["esm_tree"], esm_tree_files=f["esm_tree_files"], esm_dir=f["esm_dir"], upstream=UPSTREAM_COMMIT[:8],
                      torch=(t.__version__ if t is not None else "none"), cuda=(str(t.version.cuda) if t is not None else "none"),
                      cudnn=(str(t.backends.cudnn.version()) if t is not None else "none"), python=sys.version.split()[0],
                      pyg=_dist_version("torch_geometric"), torch_scatter=scatter_label(), device=dev, cc=device_cc(), batch=int(batch),
                      tf32_matmul=(bool(t.backends.cuda.matmul.allow_tf32) if t is not None else False), inference_mode=bool(inference_mode))
        _record("KERNELS", fields)
        return emit(KERNELS_FMT % (self.route, self.pid, fields["esm"], fields["esm_tree"], fields["upstream"], fields["torch"], fields["cuda"],
                                   fields["cudnn"], fields["python"], fields["pyg"], fields["torch_scatter"], dev, fields["cc"], int(batch),
                                   fields["tf32_matmul"], fields["inference_mode"]))


def parse(line):
    """``(kind, fields)`` for a PREPASS / STARTUP / CALL / ITEM / OUTPUTS_WRITTEN / PEAK / KERNELS line (numbers as float/int), or ``(None, None)``."""
    for kind, rx in (("PREPASS", PREPASS_RE), ("STARTUP", STARTUP_RE), ("CALL", CALL_RE), ("ITEM", ITEM_RE), ("OUTPUTS_WRITTEN", OUTPUTS_WRITTEN_RE),
                     ("PEAK", PEAK_RE), ("KERNELS", KERNELS_RE)):
        m = re.search(rx, line)
        if m:
            out = {}
            for k, v in m.groupdict().items():
                if k in ("pid", "n_backbones", "n_seq", "length", "batch", "cudnn", "i", "n", "n_files", "n_records") and v.isdigit():
                    out[k] = int(v)
                elif k in ("done_ts", "load_s", "t_start", "t_end", "wall_s", "model_s", "alloc_gib", "reserved_gib", "t"):
                    out[k] = float(v)
                elif k == "call_s":
                    out[k] = [] if v == "none" else [float(x) for x in v.split(",")]
                else:
                    out[k] = v
            return kind, out
    return None, None
