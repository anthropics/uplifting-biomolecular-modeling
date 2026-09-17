
"""
recipe.py — the design recipe as the ONE module every caller imports: the kit driver (tools/public_design_run.py: one design in one
fresh process) and the package's accessor (opt/mosaic_opt/recipe.py). The notebook's literals
(escalante-bio/mosaic examples/boltz_notebook.py @ 70fec525) live here once, each with its source line; every function below is the notebook's
own call through mosaic's public API. Nothing in this file is a lever: the two call-site levers the driver's flags select (P2 `--weights fastinit`,
P3 `--features-in/--features-sha`) are the two branches of `load_model` and `features`, named by the flag values.

Importing this module loads nothing of jax / mosaic / torch (they are imported inside the functions that need them), so the recipe's constants can
be read on a CPU box without the stack.

  target    a protein sequence (the public example target barstar 1BRS:D when no FASTA is given), `copies` chains of it, single-sequence
            (`TargetChain(use_msa=False)`, boltz `msa: empty`) unless a precomputed .a3m is given per target chain (`msa: <path>`; upstream's
            MSA server fetch never runs here: a chain that asks for an MSA without a path is refused by name)
  binder    poly-X of length L                                             notebook :101,107 (`binder_length = 75`; the driver's default 80, --binder-length)
  loss      2·BinderTargetContact + WithinBinderContact + 5.0·InverseFoldingSequenceRecovery(ProteinMPNN v_48_020, temp 0.01)   notebook :119-122
            → model.build_loss → Boltz2Loss(recycling_steps=1, sampling_steps=25, deterministic=True)                          models/boltz2.py:143-151
  x0        softmax(0.5 · gumbel(key(seed), (L, 20)))                     notebook :139-142 (seeded: the notebook draws np.random / key=None; the explicit key
                                                                            is the driver's one deviation, for replayability)
  stage1    simplex_APGM(n_steps=75, stepsize=0.1, momentum=0.0)           notebook :136-146, key = fold_in(key(seed), 1)
  stage2    simplex_APGM(n_steps=50, stepsize=0.5, scale=1.5, momentum=0.0) from stage 1's best x   notebook :154-160, key = fold_in(key(seed), 2)
  refold    model.model_output(PSSM=x2, features, key=key(0))              notebook :167-168 (predict, key(0))
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from fetch_public_inputs import BARSTAR_1BRS_D, BARSTAR_SHA256, PDB_ID   # noqa: E402  the public target's constants live in the fetch tool, once

# ---------------------------------------------------------------------------------------------------------------- the notebook's literals
LOSS_TERMS = (("BinderTargetContact", 2), ("WithinBinderContact", None), ("InverseFoldingSequenceRecovery", 5.0))   # notebook :119-122 — `2 * BinderTargetContact() + WithinBinderContact() + 5.0 * InverseFoldingSequenceRecovery(...)`: the scalar literals AS WRITTEN (LossTerm.__rmul__ stores jnp.array([scalar]), common.py:17-21 — an int and a float are different arrays); None = the bare term
MPNN_TEMP = 0.01                                                          # InverseFoldingSequenceRecovery(mpnn, temp=jax.numpy.array(0.01))  notebook :122
MPNN_WEIGHTS = "v_48_020"                                                 # ProteinMPNN.from_pretrained()'s default weights  (proteinmpnn/mpnn.py)
BOLTZ2_LOSS = {"recycling_steps": 1, "sampling_steps": 25, "deterministic": True}   # model.build_loss's Boltz2Loss (models/boltz2.py:143-151); asserted after the build
X0_GUMBEL_SCALE = 0.5                                                     # notebook :140
STAGES = ({"phase": "stage1", "n_steps": 75, "stepsize": 0.1, "scale": 1.0, "momentum": 0.0, "key_index": 1, "x": "x0"},          # notebook :136-146 (scale 1.0 and momentum 0.0 = simplex_APGM's defaults, optimizers.py:265-277; the driver passes momentum explicitly)
          {"phase": "stage2", "n_steps": 50, "stepsize": 0.5, "scale": 1.5, "momentum": 0.0, "key_index": 2, "x": "best1"})       # notebook :154-160 (x = stage 1's best x)
REFOLD_KEY = 0                                                            # notebook :167-168 (predict, key=jax.random.key(0))
REFOLD = {"phase": "refold", "kind": "forward", "key": REFOLD_KEY, "call": "model_output"}   # the recipe's one forward-only phase: model.model_output(PSSM=x2, features, key(0)), once per design, its own executable
PHASES = tuple(s["phase"] for s in STAGES) + (REFOLD["phase"],)           # the phase words every reader uses: stage1 · stage2 · refold
STAGE_CALL = "simplex_APGM"                                               # mosaic.optimizers.simplex_APGM: both stages, one jitted executable (_____eval_loss_and_grad, optimizers.py:89-91)
STEP_KIND = "grad"                                                        # every step of both stages is one loss+gradient evaluation
KIND_OF_PHASE = {**{s["phase"]: STEP_KIND for s in STAGES}, REFOLD["phase"]: REFOLD["kind"]}   # stage1/stage2 = grad, refold = forward
PUBLIC_TARGET = {"name": "barstar", "pdb": PDB_ID, "chain": "D", "sequence": BARSTAR_1BRS_D, "seq_sha256": BARSTAR_SHA256}   # the public example target (fetch_public_inputs.py)
TOKENS_PER_RESIDUE = 1                                                    # Boltz-2 tokenizes a standard protein residue as one token: tokens = target residues × copies + L
WEIGHTS = ("torch", "fastinit")                                           # --weights: stock Boltz2() | P2 fastload.load_stock_fast_init()
MSA_INDENT = "        "                                                   # chain_yaml's own indentation of its `msa: empty` line (models/boltz2.py:46-54)
MSA_SUFFIXES = (".csv", ".a3m")                                            # --msa: boltz's processed MSA `.csv` (key,sequence — the file boltz's own server route writes for the chain and parses, main.py compute_msa :520 / parse_csv :621-622: byte-for-byte what use_msa=True would have parsed) or a raw `.a3m` (parse_a3m :615-619)


class RecipeError(RuntimeError):
    """A recipe input refused by name (an MSA asked for without a staged file, a FASTA without a protein record, a frozen-features digest mismatch)."""


# ---------------------------------------------------------------------------------------------------------------- the kit's helper modules
KIT_HELPERS = ("fastload", "numstate")                                    # the two helper modules the driver uses on every arm: installed into the mosaic package as mosaic.fast.<name> (run.sh install), else loaded from the kit's mosaic_fast/ (kit_module)


KIT_FAST_PACKAGE = "mosaic.fast"                                          # the sub-package the kit adds to the installed mosaic (run.sh install copies mosaic_fast/ into it)
KIT_FAST_SRC = os.path.normpath(os.path.join(HERE, "..", "mosaic_fast"))   # the kit's own home of those files
_KIT_MODULES = {}                                                         # stem -> (module, record): ONE module object per process, whichever caller asks first


def kit_module(stem, package=KIT_FAST_PACKAGE):
    """(module, record): the kit's `mosaic/fast/<stem>.py` — the ONE door through which the driver, the package (`mosaic_opt.levers`,
    `mosaic_opt.recipe`) and any tool obtain a kit module, so every caller in a process holds the SAME module object (a per-step lever's
    install()/uninstall() state lives in its module: a second copy under another name would be a second, unpatched state). The module comes
    from the installed `<package>.<stem>` when the installed mosaic CARRIES the file (decided by find_spec before any import: a carried file
    that fails to import is an error, never a switch of home), else from the kit's own files
    `mosaic_fast/<stem>.py` — loaded under the module name `<stem>`. Whichever of the two homes supplies the
    module is trusted as-is: this repo's own commit names the bytes, and that home is already sourced from it, so there is no live byte
    check against a second copy. record = {stem, source: '<package>' | 'kit_src', file}. Unknown stem: `kit_module_unknown`."""
    import importlib
    import importlib.util
    key = (package, stem)
    if key in _KIT_MODULES:
        return _KIT_MODULES[key]
    home = os.path.join(KIT_FAST_SRC, stem + ".py")
    if not os.path.isfile(home):
        raise RecipeError(f"kit_module_unknown: {stem!r} is not a kit module ({KIT_FAST_SRC}/<stem>.py)")
    try:
        spec = importlib.util.find_spec(f"{package}.{stem}")
    except (ModuleNotFoundError, ValueError):
        spec = None
    if spec is not None and spec.origin:                                   # the installed package CARRIES the file
        source, file = package, os.path.abspath(spec.origin)
    else:
        source, file = "kit_src", os.path.abspath(home)
    if source == package:
        mod = importlib.import_module(f"{package}.{stem}")
    else:
        sp = importlib.util.spec_from_file_location(stem, file)
        mod = importlib.util.module_from_spec(sp); sys.modules[stem] = mod; sp.loader.exec_module(mod)
    rec = {"stem": stem, "source": source, "file": file}
    _KIT_MODULES[key] = (mod, rec)
    return mod, rec


def kit_modules(package=KIT_FAST_PACKAGE):
    """(fastload, numstate, record): the two helpers the driver uses on every arm, through `kit_module` (the one door). record = {source:
    '<package>' | 'kit_src', files}: which home loaded is NAMED on the driver's `[run] kit_modules` line. The two helpers must
    load from the SAME home (`kit_modules_split` otherwise: one from the installed package, one from the kit files) — whichever home supplies
    them is trusted as-is, with no byte check against another copy."""
    got = {n: kit_module(n, package) for n in KIT_HELPERS}
    sources = {rec["source"] for _, rec in got.values()}
    if len(sources) != 1:                                                   # the helpers are one install: both from the package or both from the kit files
        raise RecipeError(f"kit_modules_split: the helpers loaded from different homes { {n: rec['source'] for n, (_, rec) in got.items()} }")
    files = {n: rec["file"] for n, (_, rec) in got.items()}
    return got["fastload"][0], got["numstate"][0], {"source": sources.pop(), "files": files}


# ---------------------------------------------------------------------------------------------------------------- target
def read_fasta(path, first_record=False):
    """(record id, sequence) of the FIRST record of a FASTA file: id = the header's first whitespace-delimited token, sequence upper-cased with
    whitespace removed; refused by name when the file holds no record, the sequence is not plain amino-acid letters, or the file holds MORE than
    one record (`fasta_multi_record`: the target is one protein chain × `--target-copies`) — unless `first_record` (the driver's `--first-record`)
    asks for record 1 of a multi-record file: the count and the record used then ride in the target record (`fasta_records`,
    `fasta_record_used`, `first_record`) and on the driver's `INPUT target_fasta records=<n> used=1 (--first-record)` line (input_lines)."""
    name, seq = None, []
    n_records = fasta_records(path)
    if n_records > 1 and not first_record:
        raise RecipeError(f"fasta_multi_record: {path} holds {n_records} FASTA records; the target is ONE protein chain (repeated by --target-copies) — "
                          f"give a single-record FASTA, or pass --first-record to design against record 1 only (the other {n_records - 1} record(s) are then "
                          f"named and counted: INPUT target_fasta records={n_records} used=1)")
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if is_fasta_header(line):
                if name is not None:
                    break
                name = (line[1:].split() or [""])[0]
                continue
            if name is None:
                raise RecipeError(f"fasta_no_header: {path} has sequence text before any '>' header line")
            seq.append(line)
    s = re.sub(r"\s+", "", "".join(seq)).upper()
    if name is None or not s:
        raise RecipeError(f"fasta_no_record: {path} holds no FASTA record")
    if not re.fullmatch(r"[A-Z]+", s):
        raise RecipeError(f"fasta_not_protein: the first record of {path} ({name}) is not a plain amino-acid sequence: {s[:24]!r}")
    return name or "target", s


def is_fasta_header(line):
    """The ONE header predicate of the reader and the record census: a `>` line, leading blanks tolerated (read_fasta and fasta_records agree on
    what a record is, so a header the reader stops at is a record the census counts)."""
    return line.lstrip().startswith(">")


def fasta_records(path):
    """The number of header lines (is_fasta_header) in a FASTA file (the target is the first record; the count is carried in the target record as `fasta_records`)."""
    with open(path, "r", encoding="utf-8") as fh:
        return sum(1 for line in fh if is_fasta_header(line))


# ---------------------------------------------------------------------------------------------------------------- epitope (an input option, every mode)
EPITOPE_CONVENTION = "1-based residue positions along the target sequence as given (record 1 of --target-fasta, or the public example target), applied on every target copy"
EPITOPE_ITEM = re.compile(r"(\d+)(?:-(\d+))?")                            # one --epitope item: `12` or `40-48` (inclusive)


def parse_epitope(spec):
    """The residue positions an `--epitope` value names, sorted and unique: comma-separated items, each a position `12` or an inclusive range `40-48`
    (1-based along the target sequence). Refused by name (`epitope_format`) on an empty value, an item that is not digits / digits-digits, a
    position 0, or a range whose end precedes its start."""
    text = re.sub(r"\s+", "", str(spec or ""))
    if not text:
        raise RecipeError("epitope_format: --epitope is empty; give residue positions such as 12,15,40-48 (1-based along the target sequence)")
    out = set()
    for item in text.split(","):
        m = EPITOPE_ITEM.fullmatch(item)
        if not m:
            raise RecipeError(f"epitope_format: --epitope item {item!r} in {text!r} is not a position `12` or a range `40-48` (1-based along the target sequence)")
        lo = int(m.group(1)); hi = int(m.group(2)) if m.group(2) else lo
        if lo < 1 or hi < lo:
            raise RecipeError(f"epitope_format: --epitope item {item!r} in {text!r}: positions start at 1 and a range runs low-high")
        out.update(range(lo, hi + 1))
    return sorted(out)


def epitope_spec(residues):
    """The canonical spelling of a residue set: sorted, runs collapsed to ranges (`[12, 15, 40, 41, 42]` → `12,15,40-42`) — the word the shape key,
    the INPUT line and the manifest carry, whatever spelling the user gave."""
    res = sorted(set(int(r) for r in residues))
    parts, i = [], 0
    while i < len(res):
        j = i
        while j + 1 < len(res) and res[j + 1] == res[j] + 1:
            j += 1
        parts.append(str(res[i]) if j == i else f"{res[i]}-{res[j]}")
        i = j + 1
    return ",".join(parts)


def epitope(spec, target_rec):
    """The epitope record of an `--epitope` value on a target: {"spec": canonical spelling, "residues": [1-based positions], "target_length", "copies",
    "idx": upstream's `BinderTargetContact(epitope_idx=…)` list, "convention"}. upstream's index space (stock `mosaic/losses/structure_prediction.py`
    BinderTargetContact.__call__: `log_contact_inter[:, epitope_idx]` over `distogram_logits[:binder_len, binder_len:]`) is 0-based over the TARGET
    tokens that follow the binder — the target chains in order, `copies` × `target_length` residues (one token per standard residue) — so position p
    maps to (p − 1) + c·target_length for every copy c (EPITOPE_CONVENTION: the epitope is the same residues on every copy of the chain). A position
    outside 1..target_length is refused by name (`epitope_out_of_range`)."""
    residues = parse_epitope(spec)
    n, copies = int(target_rec["length"]), int(target_rec["copies"])
    bad = [p for p in residues if p > n]
    if bad:
        raise RecipeError(f"epitope_out_of_range: --epitope position(s) {epitope_spec(bad)} lie outside 1..{n} (the target {target_rec.get('id') or target_rec.get('name')} has {n} residues; "
                          f"positions are 1-based along the target sequence as given)")
    idx = [(p - 1) + c * n for c in range(copies) for p in residues]                     # one token per standard residue (TOKENS_PER_RESIDUE): residue p of copy c is target token (p-1) + c·n
    return {"spec": epitope_spec(residues), "residues": residues, "target_length": n, "copies": copies, "idx": idx, "n_idx": len(idx), "convention": EPITOPE_CONVENTION}


INPUT_LINE_HEAD = "INPUT "                                                  # the driver's input-option lines (report-beside: the package relays them under its tag, as `[run]` and `PEAK`)


def input_lines(target_rec, epitope_rec=None):
    """The driver's `INPUT …` lines — one per input option that narrows or reshapes what the recipe reads, printed once after the inputs resolve
    (never silent): `INPUT target_fasta records=<n> used=1 (--first-record)` when `--first-record` picked record 1 of the FASTA, and
    `INPUT epitope residues=<spec> positions=<k> target_length=<n> copies=<c> epitope_idx=<k·c> (--epitope: 1-based target residue positions,
    applied on every target copy)` when an epitope is given. [] when neither applies."""
    out = []
    if target_rec.get("first_record"):
        out.append(f"{INPUT_LINE_HEAD}target_fasta records={target_rec.get('fasta_records')} used={target_rec.get('fasta_record_used')} (--first-record)")
    if epitope_rec:
        out.append(f"{INPUT_LINE_HEAD}epitope residues={epitope_rec['spec']} positions={len(epitope_rec['residues'])} target_length={epitope_rec['target_length']} "
                   f"copies={epitope_rec['copies']} epitope_idx={epitope_rec['n_idx']} (--epitope: 1-based target residue positions, applied on every target copy)")
    return out


def target_id(name, sequence):
    """The target's id for file and directory names: the FASTA record id reduced to [A-Za-z0-9_-] plus the first 8 hex of sha256(sequence) — two
    targets that share a record id never share a features file or a P1 directory."""
    tid = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "target"
    return f"{tid}-{hashlib.sha256(sequence.encode()).hexdigest()[:8]}"


def target(fasta=None, sequence=None, name=None, copies=1, msa=None, first_record=False):
    """The target record every caller carries: {name, id, sequence, length, seq_sha256, copies, tokens_target, use_msa, msa, msa_sha256, source,
    fasta, fasta_sha256, fasta_records, fasta_record_used, first_record, pdb, chain}. No FASTA and no sequence = the public target (barstar 1BRS:D,
    single-sequence). `msa` = a precomputed alignment (MSA_SUFFIXES) applied to every copy of the target chain (use_msa semantics with a staged file;
    never a server fetch). `first_record` = the driver's `--first-record`: record 1 of a multi-record FASTA is the target (read_fasta refuses such a
    file without it); it names a FASTA's record, so it is refused without a FASTA (`first_record_without_fasta`)."""
    if fasta and sequence:
        raise RecipeError("target: give a FASTA or a sequence, not both")
    if first_record and not fasta:
        raise RecipeError("first_record_without_fasta: --first-record picks record 1 of a --target-fasta file; the public example target (no --target-fasta) is one sequence")
    if fasta:
        rid, seq = read_fasta(fasta, first_record=bool(first_record))
        rec = {"name": name or rid, "sequence": seq, "source": "fasta", "fasta": os.path.abspath(fasta), "fasta_sha256": sha256_file(fasta), "fasta_records": fasta_records(fasta),
               "fasta_record_used": 1, "first_record": bool(first_record), "pdb": None, "chain": None}
    elif sequence:
        seq = re.sub(r"\s+", "", sequence).upper()
        if not re.fullmatch(r"[A-Z]+", seq):
            raise RecipeError(f"target: the sequence is not plain amino-acid letters: {seq[:24]!r}")
        rec = {"name": name or "target", "sequence": seq, "source": "sequence", "fasta": None, "fasta_sha256": None, "fasta_records": None, "fasta_record_used": None,
               "first_record": False, "pdb": None, "chain": None}
    else:
        rec = {"name": PUBLIC_TARGET["name"], "sequence": PUBLIC_TARGET["sequence"], "source": "public", "fasta": None, "fasta_sha256": None, "fasta_records": None,
               "fasta_record_used": None, "first_record": False, "pdb": PUBLIC_TARGET["pdb"], "chain": PUBLIC_TARGET["chain"]}
    if int(copies) < 1:
        raise RecipeError(f"target: copies must be >= 1 (got {copies})")
    if msa is not None and rec["source"] == "public":
        raise RecipeError("msa_public_target: the public example target runs single-sequence; an --msa names the alignment of a --target-fasta target")
    msa_rows = None
    if msa is not None:
        if not os.path.isfile(msa):
            raise RecipeError(f"msa_not_found: --msa {msa} does not exist")
        if not str(msa).endswith(MSA_SUFFIXES):
            raise RecipeError(f"msa_format: {msa} is not one of {MSA_SUFFIXES} (boltz process_inputs parses those two: its processed .csv, the server route's own file, or a raw .a3m)")
        query, msa_rows = msa_query(msa)
        if query != rec["sequence"]:
            raise RecipeError(f"msa_query_mismatch: the alignment's query row ({len(query)} aa, {query[:16]}…) is not the target chain's sequence ({len(rec['sequence'])} aa, {rec['sequence'][:16]}…) — {msa} is another target's MSA")
    rec.update({"id": target_id(rec["name"], rec["sequence"]) if rec["source"] != "public" else rec["name"], "length": len(rec["sequence"]),
                "seq_sha256": hashlib.sha256(rec["sequence"].encode()).hexdigest(), "copies": int(copies),
                "tokens_target": len(rec["sequence"]) * int(copies) * TOKENS_PER_RESIDUE, "use_msa": msa is not None,
                "msa": os.path.abspath(msa) if msa else None, "msa_sha256": sha256_file(msa) if msa else None, "msa_rows": msa_rows})
    return rec


def msa_query(path):
    """(query sequence, n alignment rows) of a staged alignment: the first data row of boltz's processed `.csv` (key,sequence) or the first
    record of an `.a3m`, gaps `-` removed and lowercase insertion letters dropped (the a3m convention) — the row boltz aligns the chain to."""
    rows = []
    if str(path).endswith(".csv"):
        import csv
        with open(path, newline="") as fh:
            rd = csv.DictReader(fh)
            if "sequence" not in (rd.fieldnames or []):
                raise RecipeError(f"msa_format: {path} has no `sequence` column (boltz's processed MSA csv is key,sequence)")
            rows = [r["sequence"] for r in rd]
    else:
        cur = None
        for line in open(path):
            if line.startswith("#"): continue
            if line.startswith(">"):
                if cur is not None: rows.append(cur)
                cur = ""
            elif cur is not None: cur += line.strip()
        if cur is not None: rows.append(cur)
    if not rows:
        raise RecipeError(f"msa_format: {path} holds no alignment row")
    return re.sub(r"[a-z\-\.]", "", rows[0]), len(rows)


def tokens(target_rec, binder_length):
    """Token count of the designed complex: target residues × copies + L (one token per standard residue)."""
    return int(target_rec["tokens_target"]) + int(binder_length) * TOKENS_PER_RESIDUE


def chains(target_rec):
    """The notebook's `chains=[TargetChain(...)]` of a SINGLE-SEQUENCE target (use_msa=False → `msa: empty` in every chain's yaml). A target with a
    staged alignment is refused here by name (`msa_chains_outside_yaml`): its chains carry use_msa=True, and upstream's chain_yaml writes no msa line
    for those — handed to binder_features / target_only_features they would make boltz FETCH an MSA; featurize() (chains_yaml) is the only door."""
    if target_rec.get("use_msa"):
        raise RecipeError("msa_chains_outside_yaml: a target with a staged alignment is featurized through featurize() / chains_yaml() only (its msa line is written there); chains() serves single-sequence targets")
    return _chains(target_rec)


def _chains(target_rec):
    """The notebook's `chains=[TargetChain(...)]` for the target record: `copies` chains of the sequence; use_msa=True exactly when an alignment is
    staged (features() writes its path into the chain's yaml), else single-sequence (use_msa=False → `msa: empty`)."""
    from mosaic.structure_prediction import TargetChain
    return [TargetChain(sequence=target_rec["sequence"], use_msa=bool(target_rec["use_msa"])) for _ in range(int(target_rec["copies"]))]


# ---------------------------------------------------------------------------------------------------------------- model, features, loss
def import_stack():
    """Import the stack the design runs on (jax, equinox, mosaic's optimizers / losses / Boltz-2 model / MPNN — and through them boltz, torch,
    joltz) and return {name: module}. The driver calls it BEFORE its numeric-state snapshot, so a switch an import flips is inside the
    observation window (numeric_state_unchanged compares the snapshot after the design against this point)."""
    import importlib
    names = ("jax", "equinox", "mosaic.optimizers", "mosaic.common", "mosaic.losses.structure_prediction", "mosaic.models.boltz2", "mosaic.structure_prediction",
             "mosaic.proteinmpnn.mpnn", "mosaic.losses.protein_mpnn", "mosaic.losses.boltz2")
    return {n: importlib.import_module(n) for n in names}


LEVERS_OFF_WORD = "off"                                                    # the stock word: the stock arm OMITS --levers (nothing of the package is imported under stock)


def install_levers(spec):
    """The driver's `--levers WORD[+ID[=SPEC][,ID[=SPEC]...]]`: WORD is a mode word the package serves (`mosaic_opt.levers.MODES` — `fast`,
    `exact`; `big`, the memory tier), `+…` names per-step levers beyond the word's own set and per-lever settings (a development request, labelled
    `WORD+ID…` by the installer, never a tier word). The call is `mosaic_opt.levers.install(WORD, plus=…)` — the
    package's ONE installer — after `import_stack()` and before the numeric-state snapshot, any trace and the model load. "" / None = stock's
    step: nothing of the package is imported, nothing installed, returns None; the word `off` is refused by name (`lever_word_off`: the stock arm
    omits the flag). Returns `levers.installed()`; a refusal is `RecipeError` naming the lever / word and the reason (the driver's named result)."""
    text = (spec or "").strip()
    if not text:
        return None
    word, _, plus = text.partition("+")
    word = word.strip().lower()
    if word == LEVERS_OFF_WORD:
        raise RecipeError(f"lever_word_off: --levers {text!r}: the stock arm omits --levers (nothing of the package is imported under stock)")
    from mosaic_opt import levers                                         # the kit package (opt/mosaic_opt), importable in every kit arm; its core pin gate runs first
    try:
        return levers.install(word, **({"plus": plus} if plus.strip() else {}))
    except levers.LeverError as e:
        raise RecipeError(str(e) if str(e).startswith(("lever_", "unknown mode")) else f"lever_refused: {e}") from None


def levers_finalize(tag=""):
    """After every phase: the per-step levers' census lines, fail-closed gates and final facts (`mosaic_opt.levers.finalize`); {} when no lever is
    installed. A gate that fails is `RecipeError("lever_gate: …")` — the arm's named result."""
    import sys
    lv = sys.modules.get("mosaic_opt.levers")
    if lv is None:
        return {}
    try:
        return lv.finalize(tag)
    except lv.LeverError as e:
        raise RecipeError(str(e)) from None


def levers_record():
    """The driver's `manifest["levers"]`: {<id>: {"state": "on", "spec", …describe() read NOW}} for the per-step levers live in this process
    ({} when none) — recorded once after install and AGAIN when results.json is finalised (a lever's run-time facts: served / fallback counts)."""
    import sys
    lv = sys.modules.get("mosaic_opt.levers")
    return lv.manifest_record() if lv is not None else {}


def load_model(weights="torch"):
    """`Boltz2()` — the stock load (torch checkpoint → joltz.from_torch, losses/boltz2.py:49-79) — or, under --weights fastinit, the kit's P2
    `mosaic.fast.fastload.load_stock_fast_init()` (identical parameters, random-init fills skipped). Returns (model, load_path words)."""
    if weights not in WEIGHTS:
        raise RecipeError(f"--weights {weights!r}: one of {WEIGHTS}")
    import_stack()
    if weights == "fastinit":
        fastload, _numstate, rec = kit_modules()                              # the ONE resolution of the kit's helpers (mosaic.fast when installed, else the kit's mosaic_fast/ — named, trusted as-is once resolved)
        return fastload.load_stock_fast_init(), f"P2 load_stock_fast_init (torch ckpt -> joltz, random init skipped; fastload from {rec['source']})"
    from mosaic.models.boltz2 import Boltz2
    return Boltz2(), "stock Boltz2() (torch ckpt -> joltz.from_torch)"


def load_mpnn():
    """`ProteinMPNN.from_pretrained()` — the v_48_020 weights (notebook :117)."""
    from mosaic.proteinmpnn.mpnn import ProteinMPNN
    return ProteinMPNN.from_pretrained()


def chains_yaml(binder_length, target_rec):
    """The boltz input yaml the notebook's `binder_features` composes (models/boltz2.py:57-75,125-126: `_prefix()` + one `chain_yaml` per chain, the
    binder first as poly-X with `msa: empty`), with ONE addition when an alignment is staged: the target chains' `msa: <path>` line (boltz
    process_inputs' per-chain MSA file, boltz main.py:615-622). Every protein entry carries an `msa:` line — `empty` or a path — so boltz has no MSA
    left to generate and its server route cannot run (asserted). Returns (yaml text, chain objects incl. the binder)."""
    from mosaic.models.boltz2 import _prefix, chain_yaml
    from mosaic.structure_prediction import TargetChain
    binder = TargetChain(sequence="X" * int(binder_length), use_msa=False)
    objs = [binder] + _chains(target_rec)
    parts = []
    for cid, ch in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", objs):
        y = chain_yaml(cid, ch)
        if ch.use_msa:
            if not target_rec.get("msa"):
                raise RecipeError(f"msa_required_no_path: chain {cid} asks for an MSA (use_msa=True) and no --msa file is staged; the notebook's server "
                                  f"fetch (api.colabfold.com) never runs in this kit — stage a precomputed .a3m for the target")
            y = y + "\n" + MSA_INDENT + f"msa: '{target_rec['msa']}'"        # chain_yaml's own continuation form, the path a quoted scalar (models/boltz2.py:50-52: "\n        msa: empty")
        parts.append(y)
    text = "\n".join([_prefix()] + parts)                                  # upstream's composition verbatim (models/boltz2.py:58-64 target_only_features)
    check_chains_yaml(text, len(objs), expect_msa=target_rec["msa"] if target_rec.get("use_msa") else "empty")
    return text, objs


def check_chains_yaml(text, n_chains, expect_msa=None):
    """The composed yaml PARSES (yaml.safe_load — boltz parses the same text) into `sequences`: n_chains protein entries, every one with an `msa`
    key (`empty` or a staged file), the first (the binder) `empty` — or RecipeError('yaml_msa_lines: …'). A yaml without an msa key on a
    protein entry would make boltz fetch an MSA; a text that does not parse would die inside boltz. Both refused here, by name."""
    import yaml
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise RecipeError(f"yaml_msa_lines: the composed input yaml does not parse ({type(e).__name__}: {str(e)[:120]})")
    ents = [s.get("protein") for s in (doc or {}).get("sequences") or []]
    msas = [e.get("msa") for e in ents if isinstance(e, dict)]
    if len(ents) != n_chains or any(e is None for e in ents) or len(msas) != n_chains or any(m in (None, "") for m in msas) or msas[0] != "empty":
        raise RecipeError(f"yaml_msa_lines: {len(ents)} protein entries with msa keys {msas} for {n_chains} chains — every chain names `msa: empty` or a staged file "
                          f"(the binder `empty`), or boltz would fetch an MSA")
    if expect_msa is not None and [str(m) for m in msas[1:]] != [str(expect_msa)] * (n_chains - 1):
        raise RecipeError(f"yaml_msa_lines: the target chains' msa keys parse to {msas[1:]}, not the staged file {expect_msa} × {n_chains - 1}")
    return doc


def featurize(model, binder_length, target_rec):
    """The featurized complex, (features, writer): the notebook's `model.binder_features(binder_length=L, chains=chains)` UNCHANGED when no alignment
    is staged (single-sequence); with a staged .a3m, the same composition through upstream's own pieces with the chains' `msa:` lines
    (chains_yaml → build_template_yaml → load_features_and_structure_writer). boltz re-draws `ref_pos` at every featurization: freeze the result
    (save_features) when two processes must see the same inputs."""
    if not target_rec.get("use_msa"):
        return model.binder_features(binder_length=int(binder_length), chains=chains(target_rec))
    from mosaic.losses.boltz2 import load_features_and_structure_writer
    from mosaic.models.boltz2 import build_template_yaml
    text, objs = chains_yaml(binder_length, target_rec)
    tf, template_yaml = build_template_yaml("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(objs)], objs)   # (None, None): TargetChain templates are not part of the recipe; kept for parity with target_only_features
    if template_yaml:
        text = text + template_yaml
    return load_features_and_structure_writer(text)


def as_jax(features):
    """The features as the design loop holds them: every value a fresh jnp array (the driver's conversion after featurization)."""
    import numpy as np
    import jax.numpy as jnp
    return {k: jnp.array(np.asarray(v)) for k, v in features.items()}


def save_features(path, features):
    """Freeze the features dict to an npz (row A_stock1's --features-out; np.savez appends `.npz` to a path without it). Returns (the file written, its sha256)."""
    import numpy as np
    np.savez(path, **{k: np.asarray(v) for k, v in features.items()})
    written = path if str(path).endswith(".npz") else str(path) + ".npz"
    return written, sha256_file(written)


def target_words(target_rec):
    """`the public target` | `target <id> (<length> aa x <copies>)` — the manifest's words for what was featurized."""
    if target_rec["source"] == "public":
        return "the public target" + (f" x {target_rec['copies']}" if target_rec["copies"] != 1 else "")
    return f"target {target_rec['id']} ({target_rec['length']} aa x {target_rec['copies']})"


def msa_words(target_rec):
    """`msa: staged csv|a3m <sha256[:12]>` | `msa: empty -> single-sequence` — the manifest's words for the target chains' MSA line."""
    return f"msa: staged {os.path.splitext(target_rec['msa'])[1].lstrip('.')} {target_rec['msa_sha256'][:12]}" if target_rec.get("use_msa") else "msa: empty -> single-sequence"


def load_frozen_features(path, sha=None):
    """The frozen features (rows C / D's --features-in): the npz loaded as jnp arrays after its sha256 check (the in-line equivalent of the kit's
    frozen.load_features). Returns (features, sha256)."""
    import numpy as np
    import jax.numpy as jnp
    got = sha256_file(path)
    if sha is not None and got != sha:
        raise RecipeError(f"features_sha_mismatch: {path} has sha256 {got}, expected {sha}")
    d = np.load(path)
    return {k: jnp.asarray(d[k]) for k in d.files}, got


def feature_facts(features):
    """{n_tokens, msa_rows, atoms} read from the features (res_type: [1, N, …]; msa: [1, rows, N]; ref_pos: [1, atoms, 3]) — the executable's shape."""
    def dim(k, i):
        v = features.get(k)
        return int(v.shape[i]) if v is not None and len(getattr(v, "shape", ())) > i else None
    return {"n_tokens": dim("res_type", 1), "msa_rows": dim("msa", 1), "atoms": dim("ref_pos", 1)}


def loss_expression(mpnn, epitope_idx=None):
    """The notebook's loss: 2·BinderTargetContact() + WithinBinderContact() + 5·InverseFoldingSequenceRecovery(mpnn, temp=0.01) (LOSS_TERMS, in order).
    `epitope_idx` (the `--epitope` input option, every mode: `epitope()`'s "idx") is handed to upstream's own field
    `BinderTargetContact(epitope_idx=<list>)` (stock `mosaic/losses/structure_prediction.py`: the binder→target contact log-probabilities restricted
    to those target tokens), same weight; without it the term is constructed bare, exactly as the notebook writes it."""
    import jax.numpy as jnp
    import mosaic.losses.structure_prediction as sp
    from mosaic.losses.protein_mpnn import InverseFoldingSequenceRecovery
    contact = (lambda: sp.BinderTargetContact()) if epitope_idx is None else (lambda: sp.BinderTargetContact(epitope_idx=[int(i) for i in epitope_idx]))
    ctor = {"BinderTargetContact": contact, "WithinBinderContact": lambda: sp.WithinBinderContact(),
            "InverseFoldingSequenceRecovery": lambda: InverseFoldingSequenceRecovery(mpnn, temp=jnp.array(MPNN_TEMP))}
    expr = None
    for name, w in LOSS_TERMS:
        term = ctor[name]() if w is None else w * ctor[name]()
        expr = term if expr is None else expr + term
    return expr


def build_loss(model, mpnn, features, epitope_idx=None):
    """`model.build_loss(loss=<the notebook's expression>, features=features)` with the Boltz2Loss settings asserted (BOLTZ2_LOSS); `epitope_idx` as loss_expression's."""
    loss = model.build_loss(loss=loss_expression(mpnn, epitope_idx=epitope_idx), features=features)
    got = {k: getattr(loss, k) for k in BOLTZ2_LOSS}
    if got != BOLTZ2_LOSS:
        raise RecipeError(f"boltz2_loss_settings: build_loss returned {got}, the recipe's settings are {BOLTZ2_LOSS}")
    return loss


# ---------------------------------------------------------------------------------------------------------------- state, stages, refold
def x0(seed, binder_length):
    """The initial soft sequence: softmax(0.5 · gumbel(key(seed), (L, 20)))."""
    import jax
    return jax.nn.softmax(X0_GUMBEL_SCALE * jax.random.gumbel(key=jax.random.key(int(seed)), shape=(int(binder_length), 20)))


def stages(steps1=None, steps2=None):
    """The two stages with the driver's step counts applied (--steps1 / --steps2; None = the notebook's 75 / 50)."""
    out = [dict(s) for s in STAGES]
    for s, n in zip(out, (steps1, steps2)):
        if n is not None:
            s["n_steps"] = int(n)
    return out


def stage_key(seed, stage):
    """A stage's key: fold_in(key(seed), key_index)."""
    import jax
    return jax.random.fold_in(jax.random.key(int(seed)), int(stage["key_index"]))


def step_key(seed, stage, k):
    """The key simplex_APGM holds at step k of a stage: the stage key folded in with 0, k times (optimizers.py:326) — the key of a fixed state."""
    import jax
    key = stage_key(seed, stage)
    for _ in range(int(k)):
        key = jax.random.fold_in(key, 0)
    return key


def run_stage(loss, x, stage, key, trajectory_fn=None, n_steps=None):
    """One stage: `simplex_APGM(loss_function=loss, n_steps=…, x=x, stepsize=…, scale=…, momentum=…, key=key, trajectory_fn=…)` with the stage's
    settings (n_steps overridden when given). Returns simplex_APGM's own tuple: (x, best_x[, trajectory])."""
    from mosaic.optimizers import simplex_APGM
    return simplex_APGM(loss_function=loss, n_steps=int(stage["n_steps"] if n_steps is None else n_steps), x=x, stepsize=stage["stepsize"], scale=stage["scale"],
                        momentum=stage["momentum"], key=key, trajectory_fn=trajectory_fn)


def design(loss, x_start, seed, stage_list=STAGES, trajectory_fn=None, on_stage=None):
    """The notebook's design method, chained ONCE here: stage 1 from `x_start` (the notebook's x0), every later stage from the previous stage's
    BEST sequence (`simplex_APGM` returns (x, best_x); notebook :136-146 passes `best_x` on), each at its own key (`stage_key`). Returns
    [{"stage": <stage dict>, "x", "best", "trajectory": [trajectory_fn records], "wall_s"} per stage] — the driver and every other
    caller read the chaining from this function, never re-type it. `on_stage(i, record)` is called after each stage."""
    import time as _time
    out = []
    x_in = x_start
    for i, stage in enumerate(stage_list):
        t0 = _time.time()
        x, best, traj = run_stage(loss, x_in, stage, stage_key(seed, stage), trajectory_fn=trajectory_fn)
        rec = {"stage": stage, "x": x, "best": best, "trajectory": traj, "wall_s": _time.time() - t0}
        out.append(rec)
        if on_stage is not None:
            on_stage(i, rec)
        x_in = best                                                         # the next stage starts from this stage's best_x (notebook :143)
    return out


def eval_step(loss, x, key):
    """ONE loss+gradient evaluation at a fixed state — `_eval_loss_and_grad(loss, x=projection_simplex(x), key)` (optimizers.py:63-86: the jitted
    executable both stages call, then nan_to_num and the row-centred gradient): ((value, aux), g) exactly as the optimizer consumes them."""
    from mosaic.optimizers import _eval_loss_and_grad, projection_simplex
    return _eval_loss_and_grad(loss, x=projection_simplex(x), key=key)


def loss_terms_of(aux, prefix=""):
    """{name: float} — the loss's aux flattened: LinearCombination returns one entry per term (common.py:40-48), each the term's own aux (named
    scalars or arrays, possibly nested); arrays are reduced by their mean, list entries indexed. Report-beside: the weighted loss is the scalar reported."""
    import numpy as np
    out = {}
    if isinstance(aux, dict):
        for k, v in aux.items(): out.update(loss_terms_of(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(aux, (list, tuple)):
        for i, v in enumerate(aux): out.update(loss_terms_of(v, f"{prefix}[{i}]" if prefix else f"[{i}]"))
    else:
        try:
            a = np.asarray(aux)
            if a.dtype.kind in "fiub" and a.size: out[prefix or "value"] = float(a.astype(np.float64).mean())
        except Exception:  # noqa: BLE001 — a non-numeric leaf is not a term
            pass
    return out


def trajectory_record(aux, x):
    """The driver's trajectory_fn: the step's loss, upstream's own per-step wall (`aux['time']`, optimizers.py:351) and nnz."""
    return {"loss": float(aux["loss"]), "time": float(aux["time"]), "nnz": float(aux["nnz"])}


def refold(model, x, features, key=None):
    """The notebook's refold of a designed PSSM: model.model_output(PSSM=x, features=<a copy>, key=key(REFOLD_KEY)). `key`: an integer seed for a
    reader that refolds at other keys (one result per key); None = the recipe's REFOLD_KEY — the design route never passes it."""
    import jax
    import jax.numpy as jnp
    return model.model_output(PSSM=jnp.asarray(x), features=dict(features), key=jax.random.key(REFOLD_KEY if key is None else int(key)))


def peak_memory():
    """The process's device-memory peak as jax reports it: {peak_bytes_in_use, bytes_in_use, bytes_limit, source} (None fields + the reason when the
    backend has no memory_stats)."""
    try:
        import jax
        ms = jax.devices()[0].memory_stats() or {}
    except Exception as e:  # noqa: BLE001
        return {"peak_bytes_in_use": None, "bytes_in_use": None, "bytes_limit": None, "source": f"unavailable: {e!r}"[:200]}
    if "peak_bytes_in_use" not in ms:
        return {"peak_bytes_in_use": None, "bytes_in_use": None, "bytes_limit": None, "source": f"unavailable: memory_stats() has no peak_bytes_in_use (keys {sorted(ms)[:8]})"}
    return {"peak_bytes_in_use": int(ms["peak_bytes_in_use"]), "bytes_in_use": int(ms.get("bytes_in_use", 0)), "bytes_limit": int(ms.get("bytes_limit", 0)),
            "source": "jax.devices()[0].memory_stats()"}


def peak_line(item, peak):
    """`PEAK item=<id> peak_bytes_in_use_gib=<x> bytes_limit_gib=<y> source=<words>` — the report-only peak line (the package relays it under its tag)."""
    gib = lambda b: "none" if b is None else f"{b / (1 << 30):.3f}"
    return f"PEAK item={item} peak_bytes_in_use_gib={gib(peak.get('peak_bytes_in_use'))} bytes_limit_gib={gib(peak.get('bytes_limit'))} source={peak.get('source')}"


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(block), b""):
            h.update(b)
    return h.hexdigest()

