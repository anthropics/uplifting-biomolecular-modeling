"""The compiled step's ahead-of-time packages vs the HOST CPU's instruction set — stdlib only, no torch import (the kit's
ACTIVE-line probe in ``chai1_opt.modes`` loads this file before any model exists; ``aoti.Package`` calls it at load with torch's own word).

Why: AOTInductor compiles a package's C++ launcher (``<hash>.wrapper.so`` inside the ``.pt2``) with ``-march=native`` on the BUILD host
(torch/_inductor/cpp_builder.py ``_get_optimization_cflags``), so the launcher carries that host's instruction set: a launcher built on an AVX-512
host holds AVX-512 encodings (``vcvtusi2sd`` / ``kmovq`` in ``AOTInductorModelContainerCreate``,
``…UpdateUserManagedConstantBufferPairs``, ``load_constants`` — the calls ``aoti_load_package`` and ``load_constants`` make). On a host without
AVX-512 (e.g. an AVX2-only x86 CPU) the first such instruction ends the process with SIGILL inside the load — no Python traceback, no
EXIT line (the CLI then reports the fold's rc 1). torch (>= 2.8) records the build host's ISA word in the archive
(``<hash>.wrapper_metadata.json`` ``AOTI_CPU_ISA`` = ``str(torch._inductor.cpu_vec_isa.pick_vec_isa()).upper()``, codecache.get_device_information)
and only WARNS on a difference at load (torch/export/pt2_archive/_package.py ``_load_aoti``: "Device information mismatch for AOTI_CPU_ISA: <host>
vs <package>"). This module reads both words and gives the verdict the loader acts on BY NAME (``aoti.Refused(<package>, "cpu_isa_mismatch")``
→ the eager hoisted step serves the crop, exactly as a ``no_layout_meta`` package is served).

Rule: the package is loadable here iff the CPU features its recorded word implies are all implied by the host's word (torch's x86 words:
``AVX2`` < ``AVX512`` < ``AVX512 AVX512_VNNI`` < ``AVX512 AVX512_VNNI AMX_TILE``; a word outside that vocabulary falls back to torch's own test,
literal equality). A package that records no word (built by a torch that did not write one) is not judged (None: loaded as torch loads it).
``CHAI1_OPT_AOTI_HOST_ISA=<word>`` declares the host's word by hand (a host whose probe is wrong, or to exercise
the refusal on an AVX-512 host: ``CHAI1_OPT_AOTI_HOST_ISA=AVX2``).

Surface: ``WORD`` ``ENV_HOST_ISA`` ``package_isa`` ``cpuinfo_isa`` ``host_isa`` ``features`` ``loadable`` ``verdict``."""
from __future__ import annotations

import json, os, zipfile
from typing import Dict, Optional, Tuple

WORD = "cpu_isa_mismatch"                       # the refusal / step-aside word (aoti.Refused(...).word; EXIT aoti=0/n(refused:cpu_isa_mismatch:n) compiled=stepped_aside:cpu_isa_mismatch; ACTIVE compile=off:cpu_isa_mismatch)
ENV_HOST_ISA = "CHAI1_OPT_AOTI_HOST_ISA"        # the host's ISA word declared by hand (overrides the probe): torch's spelling, e.g. "AVX2", "AVX512 AVX512_VNNI"
META_KEY = "AOTI_CPU_ISA"                       # torch's key in <hash>.wrapper_metadata.json / <hash>.kernel_metadata.json of the pt2 archive
_ORDER = ("avx512 avx512_vnni amx_tile", "avx512 avx512_vnni", "avx512", "avx2")   # torch._inductor.cpu_vec_isa.supported_vec_isa_list (x86), best first — str(VecAMX()), str(VecAVX512VNNI()), …
_IMPLIES = {"avx2": {"avx2"}, "avx512": {"avx2", "avx512"}, "avx512_vnni": {"avx2", "avx512", "avx512_vnni"},
            "amx_tile": {"avx2", "avx512", "avx512_vnni", "amx_tile"}}             # VecAMX's arch flags = AVX-512 F/DQ/VL/BW + VNNI + AMX; VecAVX512VNNI's = AVX-512 + VNNI; AVX-512 hosts run AVX2 code
_AVX512_FLAGS = ("avx512f", "avx512dq", "avx512vl", "avx512bw")                     # torch.cpu._is_avx512_supported (cpuinfo: F and BW and VL and DQ)


def package_isa(path: str) -> Optional[str]:
    """The ``AOTI_CPU_ISA`` word recorded in the pt2 archive at `path` (the launcher's ``*.wrapper_metadata.json`` first, else any
    ``*_metadata.json`` under ``data/aotinductor/``), upper-cased as torch writes it; None when the archive records none or is unreadable."""
    try:
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if "data/aotinductor/" in n and n.endswith("_metadata.json")]
            names.sort(key=lambda n: (0 if n.endswith(".wrapper_metadata.json") else 1, n))
            for n in names:
                try:
                    meta = json.loads(z.read(n).decode("utf-8"))
                except Exception:  # noqa: BLE001 — a member that is not JSON: the next one
                    continue
                w = meta.get(META_KEY) if isinstance(meta, dict) else None
                if isinstance(w, str) and w.strip():
                    return " ".join(w.split()).upper()
    except Exception:  # noqa: BLE001 — not a zip / unreadable: no recorded word (the loader names what it cannot open itself)
        return None
    return None


def cpuinfo_flags(text: Optional[str] = None) -> set:
    """The ``flags`` of the first processor entry of /proc/cpuinfo (or of `text`), as a set of lower-case words; empty when unreadable."""
    if text is None:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except Exception:  # noqa: BLE001
            return set()
    for line in text.splitlines():
        if line.lower().startswith("flags") and ":" in line:
            return {w.strip().lower() for w in line.split(":", 1)[1].split() if w.strip()}
    return set()


def cpuinfo_isa(flags: Optional[set] = None) -> str:
    """torch's ISA word for this host computed from /proc/cpuinfo flags WITHOUT torch (torch.cpu._is_*_supported read the same CPUID bits through
    cpuinfo; torch additionally test-compiles each candidate, which can only lower the pick where the image's compiler lacks the ISA): the first of
    ``AVX512 AVX512_VNNI AMX_TILE`` | ``AVX512 AVX512_VNNI`` | ``AVX512`` | ``AVX2`` the flags support, else ``INVALID_VEC_ISA`` (torch's word for none)."""
    f = cpuinfo_flags() if flags is None else {str(x).lower() for x in flags}
    have = set()
    if "avx2" in f: have.add("avx2")
    if all(x in f for x in _AVX512_FLAGS): have.add("avx512")
    if "avx512" in have and ("avx512_vnni" in f or "avx512vnni" in f): have.add("avx512_vnni")
    if "amx_tile" in f: have.add("amx_tile")
    for cand in _ORDER:
        if all(w in have for w in cand.split()):
            return cand.upper()
    return "INVALID_VEC_ISA"


def torch_isa() -> Optional[str]:
    """torch's own word for this host — ``str(torch._inductor.cpu_vec_isa.pick_vec_isa()).upper()``, the expression codecache.get_device_information
    records at the build and _load_aoti compares at the load — when torch is ALREADY imported in this process (never imported here); None otherwise
    or when that private surface is absent."""
    import sys
    t = sys.modules.get("torch")
    if t is None:
        return None
    try:
        from torch._inductor import cpu_vec_isa as V  # noqa: PLC0415 — torch is already imported: its submodule only
        w = " ".join(str(V.pick_vec_isa()).split()).upper()
    except Exception:  # noqa: BLE001
        try:
            w = str(t.backends.cpu.get_cpu_capability()).upper()          # "AVX512" | "AVX2" | "DEFAULT" … (coarser: no VNNI / AMX word)
        except Exception:  # noqa: BLE001
            w = ""
    return w if w and w not in ("INVALID_VEC_ISA", "DEFAULT", "NO AVX") else None   # torch found no vector ISA it could BUILD for (toolchain) or was capped to none: the CPU's own flags decide (host_isa → cpuinfo)


_HOST: Dict[str, Tuple[str, str]] = {}


def host_isa(environ=None) -> Tuple[str, str]:
    """(word, source): ``CHAI1_OPT_AOTI_HOST_ISA`` when set (source ``env``), else torch's word when torch is imported in this process and names
    one (``torch`` — the word torch itself compares at the load), else the /proc/cpuinfo word (``cpuinfo``: before torch exists, or when torch could
    build for no ISA here — the CPU's features are what the launcher needs, not the compiler's). Cached per source."""
    env = os.environ if environ is None else environ
    ov = str(env.get(ENV_HOST_ISA) or "").strip()
    if ov:
        return " ".join(ov.split()).upper(), "env"
    import sys
    key = "torch" if "torch" in sys.modules else "cpuinfo"
    if key not in _HOST:
        w = torch_isa() if key == "torch" else None
        _HOST[key] = (w, "torch") if w else (cpuinfo_isa(), "cpuinfo")
    return _HOST[key]


def features(word: Optional[str]) -> Optional[set]:
    """The CPU features a torch ISA word implies ({avx2, avx512, avx512_vnni, amx_tile} subsets); None for a word outside torch's x86 vocabulary."""
    if word is None:
        return None
    toks = [t.lower() for t in str(word).split()]
    if not toks or any(t not in _IMPLIES for t in toks):
        return None
    out = set()
    for t in toks:
        out |= _IMPLIES[t]
    return out


def loadable(package_word: Optional[str], host_word: Optional[str]) -> Optional[bool]:
    """True when everything the package's launcher was built for is present on the host (package features <= host features), False when the host
    lacks one, None when the package records no word (not judged). Words outside the vocabulary: torch's own rule, literal equality."""
    if not package_word:
        return None
    p, h = features(package_word), features(host_word)
    if p is None or h is None:
        return " ".join(str(package_word).split()).upper() == " ".join(str(host_word or "").split()).upper()
    return p <= h


def verdict(path: str, environ=None) -> Dict[str, object]:
    """{ok: True|False|None, package: <word>|None, host: <word>, source: env|torch|cpuinfo, word: WORD} for the archive at `path` on this host."""
    pw = package_isa(path); hw, src = host_isa(environ)
    return {"ok": loadable(pw, hw), "package": pw, "host": hw, "source": src, "word": WORD}
