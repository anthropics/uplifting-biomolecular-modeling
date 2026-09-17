"""The batched driver — this kit's ``fast`` tier (route word ``kit``; lever ``batched_sampling``): UNMODIFIED upstream modules driven over B
(backbone, sample) rows per forward pass where upstream's example script (``--mode off``) designs one sequence per call.

Upstream's ``GVPTransformerModel.sample()`` (``esm/inverse_folding/gvp_transformer.py:88-140``) designs one backbone per call with a batch
of one hard-wired (``:104``, ``:109-110``, ``:132``, ``:137``) and the example script calls it once per requested sequence
(``examples/inverse_folding/sample_sequences.py:32-34``). This driver runs the same statements over B rows per forward: rows are
(backbone, sample) pairs in design order with each backbone's S samples adjacent, cut into batches of at most ``--batch_size`` rows of one
backbone length (consecutive backbones of equal length share a batch; a length change closes it), every row's backbone encoded in its
batch (no encode-once reuse across a backbone's samples), sampling under ``torch.inference_mode()``. Per batch: ``CoordBatchConverter``
over the rows' coordinates (``:102-105`` with B tuples), the ``<cath>``-prefixed all-``<mask>`` token matrix of shape (B, 1+L) (``:108-110``),
one encoder call (``:119``), then L decoder steps with the incremental state (``:126-131``), the last position's logits divided by the
temperature, softmax, ``torch.multinomial(probs, 1)`` per row (``:132-136`` — ``logits.transpose(1, 2)[:, -1]`` is ``logits[0].transpose(0, 1)``
row for row), tokens 1..L decoded through the dictionary (``:137-140``). At B = 1 the statements are upstream's and the sampled sequence
equals ``model.sample()``'s at the same generator state. No ``confidence`` input (upstream's optional argument; the script passes none).
``--multichain-backbone`` is upstream's second route, batched the same way: the whole structure is read (``load_structure``, every chain's
backbone, ``multichain_util.extract_coords_from_complex``) and a row is what ``multichain_util.sample_sequence_in_complex`` gives
``model.sample()`` (``multichain_util.py:93-103``): the chains concatenated with the designed chain first and 10-residue NaN pads between
them (upstream's ``_concatenate_coords``), a partial sequence of ``<mask>`` over the designed chain's positions and ``<pad>`` after
(``sample_complex_batch``; every row of a batch carries the same pattern — rows share a batch only at one (concatenated length, designed
length) — so upstream's statement ``if sampled_tokens[0, i] == mask_idx`` (``:135``) is the batch's ``all()``); the decode loop ends at the
designed chain's last position, where upstream's returned string ends (``sampled[:target_chain_len]``, ``:103``: the later positions are
``<pad>`` inputs whose logits it computes and never samples), so the random stream is consumed exactly as upstream consumes it; ``--chain``
names the designed chain and is required there. ``--nogpu`` keeps the model on the CPU exactly as the script's flag reads (``:21``, ``:45``);
without it the model goes to the GPU when torch sees one (``torch.cuda.is_available()``, the script's own test) and stays on the CPU
otherwise — the STARTUP and KERNELS lines name the device either way. The model is loaded by the script's own two statements (``sample_sequences.py:114-115``:
``esm.pretrained.esm_if1_gvp4_t16_142M_UR50()``, ``model.eval()``; the weights come through ``torch.hub``'s cache under ``$TORCH_HOME``) and
moved to the GPU as the script does (``:21-22``); the seed (``--seed``, an input of every pass) is applied once with ``torch.manual_seed``
after the load, before the first batch — upstream has no seed argument; with it the sampled records repeat run to run at a fixed ``--batch_size``
and input order, no deterministic-algorithm switch needed or set. Outputs: one FASTA per structure with records ``>sampled_seq_<i>`` for
i = 1..S, the script's own format (``:37-38``), at ``<out>/seqs/<stem>.fasta`` — or, for one structure, at upstream's ``--outpath`` (default
``output/sampled_seqs.fasta``; ``outputs.plan``) — written when a backbone's S rows are all sampled.

Every input file is parsed once at the start (``load_coords``, before the model load: the PREPASS line, n = N), so no parsing is inside a
unit window. Evidence lines (``lines.py``): PREPASS, STARTUP after load + eval + device move + seed, one CALL and one ITEM per batch
(``unit=batch``: t_start = its forward begins, t_end = its records written, model_s = the CUDA-synchronized encoder + decode loop; upstream's
per-step host synchronisation in the sampling loop is kept as is), OUTPUTS_WRITTEN after the last FASTA is closed, PEAK and KERNELS at the
end. ``run`` returns 0 when every backbone's FASTA holds S records, 1 otherwise (a CUDA out-of-memory propagates as the interpreter's own
failure, its text on stderr). ``driver_arguments`` is the argument table (the ``design`` command line adds ``--mode`` / ``--det`` to it);
``main`` runs one pass from those arguments in this process — what the kit child (``kit_design.py``) calls.
This module imports nothing of the shared core and, before ``run``/``main`` execute, nothing of torch or esm.
"""
import os
import sys
from pathlib import Path

DEFAULT_SEED = 37                                                          # det.DEFAULT_SEED (held equal by the tests; this module stays free of package imports at load)
DEFAULT_BATCH = 64                                                         # rows per forward when --batch_size is not given
DEFAULT_CHAIN = None                                                       # sample_sequences.py --chain default: the file's chains as the loader takes them
DEFAULT_OUTPATH = "output/sampled_seqs.fasta"                                # sample_sequences.py --outpath default: applies when ONE file is given with neither --out nor --outpath


def driver_arguments(p):
    """Add the driver's arguments to an ``argparse`` parser: upstream sample_sequences.py's own arguments with its names and defaults (the
    positional ``pdbfile``, ``--chain``, ``--temperature`` 1.0, ``--outpath``, ``--num-samples`` 1, ``--multichain-backbone`` /
    ``--singlechain-backbone``, ``--nogpu``) plus the driver's ``--input`` (a directory or one file), ``--out`` (an output directory),
    ``--seed`` (an input of every pass) and ``--batch_size`` (``--batch`` is accepted as the same flag)."""
    p.add_argument("pdbfile", nargs="?", default=None, help="input filepath, either .pdb or .cif (or give --input)")
    p.add_argument("--input", default=None, help="a directory of *.pdb / *.cif files (sorted name order = design order) or one .pdb / .cif file")
    p.add_argument("--out", default=None, help="the output directory: <out>/seqs/<stem>.fasta per structure and the run record")
    p.add_argument("--outpath", type=str, default=None,
                   help=f"output filepath for saving sampled sequences — one input structure (default when one file is given without --out: {DEFAULT_OUTPATH})")
    p.add_argument("--chain", type=str, help="chain id for the chain of interest", default=DEFAULT_CHAIN)
    p.add_argument("--temperature", type=float, help="temperature for sampling, higher for more diversity", default=1.)
    p.add_argument("--num-samples", type=int, help="number of sequences to sample", default=1)
    p.set_defaults(multichain_backbone=False)
    p.add_argument("--multichain-backbone", action="store_true",
                   help="use the backbones of all chains in the input for conditioning (upstream's multichain route, batched like the single-chain one: --batch_size rows per forward)")
    p.add_argument("--singlechain-backbone", dest="multichain_backbone", action="store_false", help="use the backbone of only target chain in the input for conditioning")
    p.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="N", help=f"torch.manual_seed(N) once after the model load, every pass (default {DEFAULT_SEED})")
    p.add_argument("--batch_size", "--batch", dest="batch", type=int, default=DEFAULT_BATCH, metavar="B",
                   help=f"(backbone, sample) rows per forward pass, in design order (default {DEFAULT_BATCH})")
    return p


def batches(keys, num_samples, batch):
    """The forward batches: lists of (backbone index, sample index) rows — design order, each backbone's samples adjacent, at most ``batch``
    rows, one key per batch (``keys[i]``: backbone i's length on the single-chain route; its (concatenated length, designed-chain length)
    pair on the multichain route — a change of key closes the batch)."""
    cur = []
    for i, key in enumerate(keys):
        for s in range(num_samples):
            if cur and (len(cur) == batch or keys[cur[0][0]] != key):
                yield cur
                cur = []
            cur.append((i, s))
    if cur:
        yield cur


def complex_pattern(total_length, designed_length):
    """upstream ``sample_sequence_in_complex``'s partial sequence (multichain_util.py:97-100): ``<mask>`` over the designed chain's positions
    (it is first in the concatenation), ``<pad>`` over every later position (the pads and the other chains: given, never sampled)."""
    if not 0 <= int(designed_length) <= int(total_length):
        raise ValueError("complex_pattern: the designed chain (%d) cannot be longer than the concatenation (%d)" % (designed_length, total_length))
    return ['<mask>'] * int(designed_length) + ['<pad>'] * (int(total_length) - int(designed_length))


def sample_batch(model, coords_list, temperature, device, partial_seq=None, steps=None):
    """upstream ``GVPTransformerModel.sample`` (gvp_transformer.py:100-140) over B equal-length backbones: B sampled sequences (strings —
    every position's token joined, :137-140). ``partial_seq`` (upstream's argument, :111-113) is ONE token pattern laid on every row; ``steps``
    ends the decode loop after that many positions (default all L): the positions past it keep their ``partial_seq`` token, as they do in
    upstream's loop, which never samples a position whose token is not ``<mask>`` — the multichain route (``sample_complex_batch``)."""
    import torch
    import torch.nn.functional as F
    from esm.inverse_folding.util import CoordBatchConverter

    B = len(coords_list)
    L = len(coords_list[0])
    if any(len(c) != L for c in coords_list):
        raise ValueError("sample_batch: the rows of one batch share one backbone length")
    n_steps = L if steps is None else int(steps)
    if not 0 <= n_steps <= L:
        raise ValueError("sample_batch: steps (%d) must lie within the backbone length (%d)" % (n_steps, L))
    batch_converter = CoordBatchConverter(model.decoder.dictionary)                                   # :102
    batch_coords, confidence, _, _, padding_mask = batch_converter([(c, None, None) for c in coords_list], device=device)   # :103-105, B tuples
    mask_idx = model.decoder.dictionary.get_idx('<mask>')                                              # :108
    sampled_tokens = torch.full((B, 1 + L), mask_idx, dtype=int)                                      # :109, B rows
    sampled_tokens[:, 0] = model.decoder.dictionary.get_idx('<cath>')                                  # :110
    if partial_seq is not None:                                                                        # :111-113, the one pattern on every row
        if len(partial_seq) != L:
            raise ValueError("sample_batch: partial_seq (%d tokens) must cover the backbone (%d)" % (len(partial_seq), L))
        sampled_tokens[:, 1:] = torch.tensor([model.decoder.dictionary.get_idx(c) for c in partial_seq], dtype=sampled_tokens.dtype)
    incremental_state = dict()                                                                         # :116
    encoder_out = model.encoder(batch_coords, padding_mask, confidence)                                # :119 — every row encoded
    if device:                                                                                         # :122-123
        sampled_tokens = sampled_tokens.to(device)
    for i in range(1, n_steps + 1):                                                                    # :126-136
        logits, _ = model.decoder(sampled_tokens[:, :i], encoder_out, incremental_state=incremental_state)
        logits = logits.transpose(1, 2)[:, -1]                                                         # (B, V): row b is logits[b].transpose(0, 1)[-1] (:132 at B = 1)
        logits /= temperature                                                                          # :133
        probs = F.softmax(logits, dim=-1)                                                              # :134
        if bool((sampled_tokens[:, i] == mask_idx).all()):                                             # :135 — the rows carry one pattern, so the row-0 test is the batch's
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)                             # :136
    out = []
    for b in range(B):                                                                                 # :137-140 per row
        sampled_seq = sampled_tokens[b, 1:]
        out.append(''.join([model.decoder.dictionary.get_tok(a) for a in sampled_seq]))
    return out


def sample_complex_batch(model, complexes, chain, temperature, device):
    """upstream ``multichain_util.sample_sequence_in_complex`` (multichain_util.py:80-104) over B rows: each row is one complex (a dict chain
    id -> L x 3 x 3 coordinates) whose chain ``chain`` is designed conditioned on every chain's backbone — the chains concatenated with the
    designed one first and 10-residue pads between them (upstream's ``_concatenate_coords``), the ``<mask>``/``<pad>`` pattern of
    ``complex_pattern``, then ``sample_batch`` through the designed chain's positions. The rows of one batch share the concatenated length
    and the designed length (``run`` batches them so). Returns B designed-chain sequences: each row's string cut as upstream cuts it,
    ``sampled[:target_chain_len]`` (:103)."""
    from esm.inverse_folding.multichain_util import _concatenate_coords

    rows = [_concatenate_coords(coords, chain) for coords in complexes]                                # :94, per row
    designed = len(complexes[0][chain])                                                                # :93
    total = len(rows[0])
    if any(len(r) != total for r in rows) or any(len(coords[chain]) != designed for coords in complexes):
        raise ValueError("sample_complex_batch: the rows of one batch share the concatenated length and the designed chain's length")
    return [s[:designed] for s in sample_batch(model, rows, temperature, device, partial_seq=complex_pattern(total, designed), steps=designed)]   # :97-103


def write_fasta(path, seqs):
    """The script's own record format (sample_sequences.py:37-38)."""
    with open(path, 'w') as f:
        for i, sampled_seq in enumerate(seqs):
            f.write(f'>sampled_seq_{i+1}\n')
            f.write(sampled_seq + '\n')


def run(route, files, stems, out_paths, chain, temperature, num_samples, batch, seed, nogpu=False, multichain=False):
    """One pass of the driver over ``files`` (design order), seeded with ``seed``, ``batch`` rows per forward; ``multichain`` = upstream's
    multichain route (``chain`` designed, every chain conditioning: ``sample_complex_batch``), else the single-chain route (``sample_batch``).
    Returns the exit status."""
    import torch
    import esm
    import esm.inverse_folding
    import esm.inverse_folding.multichain_util
    from . import lines as L

    clocks = L.Clocks(route)
    coords, lengths, keys = [], [], []                                     # keys: what the rows of one batch must share (batches)
    clocks.prepass_begin()
    for f in files:                                                        # every input parsed at the start, before STARTUP and any batch (PREPASS n=N)
        if multichain:                                                     # sample_sequences.py:48-51: the whole structure, coordinates per chain
            c, native_seqs = esm.inverse_folding.multichain_util.extract_coords_from_complex(esm.inverse_folding.util.load_structure(f))
            if chain not in c:
                sys.stderr.write("[esm_if1-opt] batched: %s: no chain %r in the file (chains: %s)\n" % (f, chain, ", ".join(sorted(c))))
                return 1
            coords.append(c)
            lengths.append(sum(len(v) for v in c.values()) + 10 * (len(c) - 1))   # what the encoder processes: the chains concatenated with 10-residue pads (multichain_util._concatenate_coords)
            keys.append((lengths[-1], len(c[chain])))                     # one <mask>/<pad> pattern per batch: the concatenated length and the designed chain's length
        else:                                                              # sample_sequences.py:24 once per backbone
            c, native_seq = esm.inverse_folding.util.load_coords(f, chain)
            coords.append(c)
            lengths.append(len(native_seq))
            keys.append(lengths[-1])
    clocks.prepass_done(len(files))
    clocks.load_begin()
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()        # sample_sequences.py:114
    model = model.eval()                                                   # sample_sequences.py:115
    use_cuda = torch.cuda.is_available() and not nogpu                    # sample_sequences.py:21-22 / :45-46
    if use_cuda:
        model = model.cuda()
    device = torch.device('cuda') if use_cuda else None                   # the script samples with device=torch.device('cuda') (:34)
    torch.manual_seed(int(seed))                                           # --seed: once, after the load, before the first batch
    clocks.load_done("cuda" if use_cuda else "cpu")

    for p in {os.path.dirname(os.path.abspath(p)) for p in out_paths}:
        Path(p).mkdir(parents=True, exist_ok=True)                         # sample_sequences.py:30
    S = int(num_samples)
    results = [[None] * S for _ in files]
    written = set()
    k = 0
    rows_per_forward = int(batch)                                          # both routes: --batch_size (backbone, sample) rows per forward
    with torch.inference_mode():
        for rows in batches(keys, S, rows_per_forward):
            k += 1
            backbones = sorted({i for i, _ in rows})
            clocks.begin_unit("batch%04d" % k, lengths[rows[0][0]])
            clocks.model_begin()
            if multichain:
                seqs = sample_complex_batch(model, [coords[i] for i, _ in rows], chain, temperature, device)
            else:
                seqs = sample_batch(model, [coords[i] for i, _ in rows], temperature, device)
            clocks.model_end()
            for (i, s), seq in zip(rows, seqs):
                results[i][s] = seq
            for i in backbones:
                if i not in written and all(x is not None for x in results[i]):
                    write_fasta(out_paths[i], results[i])
                    written.add(i)
            clocks.item_done("batch", n_backbones=len(backbones), n_seq=len(rows))
    for i in range(len(files)):                                            # a backbone whose rows are all present but not yet flushed (S = 0: the script writes an empty file)
        if i not in written and all(x is not None for x in results[i]):
            write_fasta(out_paths[i], results[i])
            written.add(i)
    clocks.outputs_written(len(written), len(written) * S)                # the pass's last FASTA is written and closed
    clocks.finish(batch=rows_per_forward, inference_mode=True)
    missing = [stems[i] for i in range(len(files)) if i not in written]
    if missing:
        sys.stderr.write("[esm_if1-opt] batched: %d of %d backbones without a complete FASTA: %s\n" % (len(missing), len(files), ", ".join(missing[:10])))
        return 1
    return 0


def main(argv, route="kit"):
    """Parse ``driver_arguments`` from ``argv`` and run one pass in this process; returns the exit status (2 on a usage error)."""
    import argparse
    from . import inputs, outputs
    p = driver_arguments(argparse.ArgumentParser(prog="python -m esm_if1_opt.kit_design", description="the batched driver (--mode fast)"))
    try:
        ns = p.parse_args(list(argv))
        if int(ns.batch) < 1:
            p.error("--batch_size must be >= 1")
        if int(ns.num_samples) < 1:
            p.error("--num-samples must be >= 1")
        if ns.multichain_backbone and not ns.chain:
            p.error("--multichain-backbone needs --chain: the chain to design")
        files, stems = inputs.resolve(inputs.input_arg(ns))
        _, paths = outputs.plan(ns.out, ns.outpath, files, stems)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2
    except (inputs.InputError, outputs.OutputError) as e:
        sys.stderr.write("[esm_if1-opt] usage: %s\n" % e)
        return 2
    return run(route, files, stems, [paths[s] for s in stems], ns.chain, ns.temperature, ns.num_samples, int(ns.batch), int(ns.seed),
               nogpu=bool(ns.nogpu), multichain=bool(ns.multichain_backbone))
