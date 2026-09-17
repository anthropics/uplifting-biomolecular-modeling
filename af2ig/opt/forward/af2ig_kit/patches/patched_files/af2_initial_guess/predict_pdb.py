#!/usr/bin/env python
"""
predict_pdb.py -- PyRosetta-free PDB-directory front end for af2_initial_guess/predict.py
(dl_binder_design, Bennett et al. 2023).

The AlphaFold2 "initial guess + target template" computation is the one of predict.py, unchanged:
same vendored alphafold package (this directory), model_1_ptm, num_ensemble 1, max_msa_clusters 5,
max_extra_msa 5, num_recycle (default 3), template features for the target residues
(af2_util.generate_template_features), all-atom initial guess of the whole complex
(af2_util.parse_initial_guess), +200 residue-index offset at every chain break detected by
af2_util.check_residue_distances, TF feature pipeline with random_seed=0, model call with
jax.random.PRNGKey(0), scores exactly as predict.py computes them (pLDDT / PAE slices, Ca RMSDs).

What differs from predict.py (I/O only):
  * structures are read from PDB files with a small numpy parser instead of PyRosetta poses
    (chain 1 = binder, all following chains = target; a one-chain file is a monomer);
    residues are taken in file order (predict.py renumbers poses with non-unique numbering, which
    gives the same order); atoms not present in the file are simply absent (PyRosetta would add
    e.g. terminal OXT atoms when it builds a pose -- this front end does not invent coordinates);
  * the predicted model is written with a plain PDB writer (heavy atoms in alphafold atom order,
    per-residue pLDDT in the B-factor column, binder = chain A, target = chain B) instead of a
    Rosetta pose dump; silent files are not supported (no Rosetta license needed or used).
The score file (-scorefilename) and checkpoint file have predict.py's exact format.
"""

import os
import sys
import time

T_PROC0 = time.time()

import json
import glob
import uuid
import argparse
import collections
import threading

import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
# The vendored alphafold package (with the initial-guess modification) must win over any
# pip-installed alphafold: put this directory first on sys.path.
if sys.path[0] != script_dir:
    sys.path.insert(0, script_dir)

from timeit import default_timer as timer

import jax

from alphafold.common import residue_constants
from alphafold.common import protein
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model

import af2_util

T_IMPORTS = time.time()

assert os.path.dirname(os.path.realpath(model.__file__)).startswith(script_dir), \
    f'wrong alphafold package imported: {model.__file__} (expected the vendored copy under {script_dir})'


def range1(size): return range(1, size+1)

#################################
# Parse Arguments
#################################

parser = argparse.ArgumentParser()

# I/O Arguments (same names and defaults as predict.py where they exist)
parser.add_argument( "-pdbdir", type=str, default="", help='The name of a directory of pdbs to run through the model' )
parser.add_argument( "-outpdbdir", type=str, default="outputs", help='The directory to which the output PDB files will be written' )
parser.add_argument( "-runlist", type=str, default='', help="The path of a list of pdb tags to run (default: ''; Run all PDBs)" )
parser.add_argument( "-checkpoint_name", type=str, default='check.point', help="The name of a file where tags which have finished will be written (default: check.point)" )
parser.add_argument( "-scorefilename", type=str, default='out.sc', help="The name of a file where scores will be written (default: out.sc)" )
parser.add_argument( "-maintain_res_numbering", action="store_true", default=False, help='Accepted for command-line compatibility; this front end always uses file order (see module docstring)' )

parser.add_argument( "-debug", action="store_true", default=False, help='When active, errors will cause the script to crash and the error message to be printed out (default: False)')

# AF2-Specific Arguments
parser.add_argument( "-max_amide_dist", type=float, default=3.0, help='The maximum distance between an amide bond\'s carbon and nitrogen (default: 3.0)' )
parser.add_argument( "-recycle", type=int, default=3, help='The number of AF2 recycles to perform (default: 3)' )
parser.add_argument( "-no_initial_guess", action="store_true", default=False, help='When active, the model will not use an initial guess (default: False)' )
parser.add_argument( "-force_monomer", action="store_true", default=False, help='When active, the model will predict the structure of a monomer (default: False)' )

# Front-end arguments (new)
parser.add_argument( "-af2_dir", type=str, default=os.path.join(script_dir, 'model_weights'),
                     help='Directory that contains params/params_model_1_ptm.npz (default: <this dir>/model_weights, as predict.py)' )
parser.add_argument( "-timers", type=str, default='', help='Optional path of a JSONL file receiving per-design wall-clock timers (does not change any computation)' )
parser.add_argument( "-sort_by_length", action="store_true", default=False, help='Process inputs sorted by total length instead of by file name' )

# Fast-inference kit levers (all opt-in; unset = the computation and scheduling of predict.py)
parser.add_argument( "-device_params", action="store_true", default=False, help='L1: put the model parameters on the GPU once at start-up instead of passing host numpy arrays to every model call (stock re-uploads ~370 MB of weights per design)' )
parser.add_argument( "-host_outputs", action="store_true", default=False, help='L6: copy the model outputs to host NumPy arrays in one transfer (jax.device_get) before post-processing, instead of letting numpy/scipy pull each device array lazily op by op; the arrays (float32 values) are the same' )
parser.add_argument( "-precompile", type=int, default=0, help='L7: before the prediction loop, featurize one input per distinct length and trigger the XLA compilations of the jitted model for all of them from N host threads in parallel (0 = off). The compiled executables are the ones the loop would have compiled serially; outputs unchanged' )
parser.add_argument( "-program_cache", type=str, default='', help='L13 (exact): DIR — keep every model program this process compiles (one per distinct residue count and lever set) under DIR as a serialized XLA executable and LOAD it in later processes instead of tracing, lowering and compiling the model again (af2ig_opt.programs: the key names the code, the levers, the stack and the device; a file that does not load is named and the program is traced). The executable is the one this stack compiled: bitwise-identical outputs. Empty = off (stock: every process traces and compiles each new length)' )
parser.add_argument( "-flash_attn", action="store_true", default=False, help='L8 (changes numerics slightly, see CHANGES.md): the attention core (logits + bias, softmax, weighted sum) of every eligible alphafold Attention call computed by the kit adapter af2ig_opt.flash_attn over the shared Pallas flash-attention kernel of the model-opt tree (opt_core.kernels.pallas_attn) instead of the three XLA ops; ineligible calls run the stock ops and are counted in the flash_attn timer record' )
parser.add_argument( "-fused_triattn", action="store_true", default=False, help='L10 (changes numerics slightly, see CHANGES.md): every TriangleAttention module (starting / ending node; Evoformer and template pair stack) computed by the model-opt tree\'s fused triangle-attention block (opt_core.kernels.fpf_pallas_serve.tri_attn_block via af2ig_opt.triattn: LayerNorm + pair bias + q|k|v|gate projections + flash core + gating + output projection in one Pallas program, float32 with TF32 products) instead of the stock body; calls the block refuses run the stock body and are counted by reason' )
parser.add_argument( "-fused_trimul", action="store_true", default=False, help='L11 (changes numerics slightly, see CHANGES.md): every TriangleMultiplication module (outgoing / incoming; Evoformer and template pair stack) computed by the model-opt tree\'s fused triangle-multiplication block (opt_core.kernels.fpf_pallas_serve.trimul_block via af2ig_opt.fused_trimul: input LayerNorm + left|right projections and gates + mask in a Pallas prologue, one batched TF32 GEMM, centre LayerNorm + output projection + gate in a Pallas epilogue, float32) instead of the stock body; calls the block refuses run the stock body and are counted by reason' )
parser.add_argument( "-opm_reassoc", action="store_true", default=False, help='L12 (changes numerics slightly, see CHANGES.md): every OuterProductMean module (Evoformer and extra-MSA stacks) computed re-associated by the kit adapter af2ig_opt.opm: the output weight folded into the right projection, then one GEMM over (MSA row, channel) — no [N, N, 32x32] outer-product intermediate and ~6.6x fewer FLOPs at this model\'s 5-row MSA (pure JAX, float32, XLA default-precision products as stock\'s einsums); an input the body does not expect runs the stock body, counted by reason' )
parser.add_argument( "-subbatch", type=int, default=0, help='L9: the row sub-batch of every Attention module and every Transition (alphafold global_config.subbatch_size, run through mapping.inference_subbatch; stock 4 rows, sized for 16 GB cards) set to N rows: the same per-row math in N-row chunks, N/rows scan iterations of large GEMMs instead of many small ones; the decision record is the model-opt tree\'s shared sub-batch policy (opt_core.jax_design.subbatch_policy via af2ig_opt.subbatch). 0 = stock' )
parser.add_argument( "-tmpl_pointwise_sub", type=int, default=0, help='L18 (changes numerics slightly, see CHANGES.md): the row sub-batch of the template point-wise attention (alphafold template.subbatch_size; stock 128 rows of the N*N-row attention over the templates); 0 = stock' )
parser.add_argument( "-prefetch", type=int, default=0, help='L15 (exact): N > 0 — the inputs of the next N designs are read and featurised (PDB parse, template / single-sequence MSA features, chain-break detection, the TensorFlow feature pipeline: host CPU work) on one background worker thread while the current design runs on the GPU, and taken by the loop when their turn comes (af2ig_opt.prefetch). The featurisation, its one-at-a-time order and its results are unchanged — only when it runs moves — so the outputs, the scores and the order of every record are those of N = 0; a featurisation error surfaces at the design\'s turn as before. Host memory: at most N featurised inputs queued. 0 = off (stock: read, featurise, predict, write, serially per design)' )
parser.add_argument( "-overlap_output", type=int, default=0, help='L16 (exact): N > 0 — each design\'s output step (confidence metrics, RMSDs, the PDB file, the score line, then the checkpoint line) runs on one background writer thread, in the loop\'s order, while the next design\'s forward runs; at most N outputs wait behind the writer (af2ig_opt.prefetch.OutputWriter). The step\'s code and inputs are unchanged, so every file, score and line order is that of N = 0; a failing step is reported as the loop reports it and the design is still checkpointed. Host memory: up to N+1 designs\' outputs alive. 0 = off (stock: the loop writes each design\'s outputs before it reads the next)' )
parser.add_argument( "-fast", action="store_true", default=False, help='Recommended exact stack: -host_outputs -device_params -sort_by_length (all exact: the stock output bytes)' )
parser.add_argument( "-trimul_chunk", type=str, default='', help='M1 memory lever (changes numerics slightly, see CHANGES.md): "<rows>:<min residues>" — TriangleMultiplication evaluated in row chunks of <rows> pair rows once the compiled length reaches <min residues>, through the model-opt tree\'s shared row-chunk producer (opt_core.mem.rowpair_jax.rowchunk via af2ig_opt.pairstack); shorter lengths run the stock body' )

args = parser.parse_args()
if args.fast:
    args.device_params = True; args.host_outputs = True; args.sort_by_length = True
SUBBATCH = None
if args.subbatch:
    # L9: the inference sub-batch (rows per Attention / Transition chunk), decided and recorded through the tree's shared policy.
    from af2ig_opt import subbatch as SUBBATCH
    _sb = SUBBATCH.setup(args.subbatch)
    print(f"subbatch: {_sb['rows']} rows per Attention/Transition chunk (stock {_sb['stock_rows']}; {_sb['policy']})")

FLASH_ATTN = None
if args.flash_attn:
    # L8: the kit's adapter over the model-opt tree's shared Pallas flash-attention kernel. The kernel is
    # resolved here, before any design: a core without it, a jax below its floor or a CPU backend stops the
    # run with the reason named instead of falling back to the stock ops silently.
    from af2ig_opt import flash_attn as FLASH_ATTN
    _probe = FLASH_ATTN.setup()
    print(f"flash_attn: kernel {FLASH_ATTN.census()['impl']} origin={FLASH_ATTN.census()['origin']} jax={_probe.get('jax')} tested={_probe.get('tested')} min_tokens={_probe.get('min_tokens')} scope={'all' if _probe.get('all_calls') else 'pair_bias'}")
FUSED_TRIATTN = None
if args.fused_triattn:
    # L10: the kit's adapter over the model-opt tree's fused triangle-attention block. Resolved here, before any design (a core without the
    # block, a jax below its floor, a CPU backend or a GPU without a tile table stops the run with the reason named), then installed as
    # TriangleAttention's __call__: refused calls run the stock body and are counted by reason, never silently.
    from af2ig_opt import triattn as FUSED_TRIATTN
    _tprobe = FUSED_TRIATTN.setup()
    print(f"fused_triattn: block {FUSED_TRIATTN.IMPL} origin=core jax={_tprobe.get('jax')} cc={_tprobe.get('cc')} tiles={_tprobe.get('tiles')} precision={_tprobe.get('precision')} key_mask={_tprobe.get('key_mask')}")
FUSED_TRIMUL = None
if args.fused_trimul:
    # L11: the kit's adapter over the model-opt tree's fused triangle-multiplication block, resolved before any design (named refusal otherwise)
    # and installed as TriangleMultiplication's __call__: refused calls run the stock body and are counted by reason, never silently.
    from af2ig_opt import fused_trimul as FUSED_TRIMUL
    _mprobe = FUSED_TRIMUL.setup()
    print(f"fused_trimul: block {FUSED_TRIMUL.IMPL} origin=core jax={_mprobe.get('jax')} cc={_mprobe.get('cc')} tiles={_mprobe.get('tiles')} precision={_mprobe.get('precision')}")
OPM = None
if args.opm_reassoc:
    # L12: the kit's re-associated OuterProductMean (pure JAX: no kernel, no card rule), installed as OuterProductMean's __call__ before any design;
    # a call on an input the body does not expect runs the stock body and is counted by reason, never silently.
    from af2ig_opt import opm as OPM
    _oprobe = OPM.setup()
    print(f"opm_reassoc: impl {OPM.IMPL} origin=kit jax={_oprobe.get('jax')} backend={_oprobe.get('backend')} matmul_precision={_oprobe.get('matmul_precision')} dtype={_oprobe.get('dtype')}")
PAIRSTACK = None
if args.trimul_chunk:
    # M1: the memory line's pair-stack lever, installed ONCE here, before the model runner is built: the kit's
    # adapter rebinds the stock alphafold TriangleMultiplication class in this process to its row-chunked form.
    # A core without the shared producer stops the run with the reason named instead of running the stock body silently.
    from af2ig_opt import pairstack as PAIRSTACK
    _ps = PAIRSTACK.install(trimul=PAIRSTACK.parse_trimul(args.trimul_chunk))
    print(f"pairstack: trimul_chunk={_ps['trimul_chunk']}")

#################################
# Timers (pure instrumentation)
#################################

PREFETCHLIB = None   # af2ig_opt.prefetch: the inputs of the next designs featurised ahead of the loop on a worker thread (L15 -prefetch)
PREFETCH = None      # L15: the Prefetcher of this process (built after the precompile pass, before the loop)
WRITER = None        # L16: the OutputWriter of this process (built before the loop)
if args.prefetch > 0 or args.overlap_output > 0:
    from af2ig_opt import prefetch as PREFETCHLIB
PRECOMP = None       # L7 (0.7.7): the streaming precompile scheduler of this process (af2ig_opt.precompile.Streaming) — the loop waits on its own length only
PROGLIB = None       # af2ig_opt.programs: the model program prepared ahead of its call (L7 -precompile) and kept across processes (L13 -program_cache)
PROGRAMS = None      # L13: the serialized-program store (af2ig_opt.programs.Store)
if args.precompile > 0 or args.program_cache:
    from af2ig_opt import programs as PROGLIB
PRECOMPLIB = None    # af2ig_opt.precompile: the streaming scheduler (L7, 0.7.7)
if args.precompile > 0:
    from af2ig_opt import precompile as PRECOMPLIB

def _census_snapshot():
    # the trace-time censuses of the kernel levers of this process (cumulative), read around a model trace (af2ig_opt.programs census_delta)
    snap = {}
    if FLASH_ATTN is not None: snap['flash_attn'] = FLASH_ATTN.census()
    if FUSED_TRIATTN is not None: snap['fused_triattn'] = FUSED_TRIATTN.census()
    if FUSED_TRIMUL is not None: snap['fused_trimul'] = FUSED_TRIMUL.census()
    if OPM is not None: snap['opm_reassoc'] = OPM.census()             # L12's trace-time census travels with the program too
    return snap

def _census_replay(census, L, key):
    # L13: a LOADED program was traced in an earlier process — replay the kernel calls its trace served / left on the stock ops into this
    # process's ledgers (shape key 'replayed:L<n>'), so the run's census names the kernels inside the executables it ran (never invented:
    # the counts are the ones recorded around that trace, stored with the program)
    adapters = {'flash_attn': FLASH_ATTN, 'fused_triattn': FUSED_TRIATTN, 'fused_trimul': FUSED_TRIMUL, 'opm_reassoc': OPM}
    for kind, c in (census or {}).items():
        ad = adapters.get(kind)
        if ad is None: continue
        led = ad._ledger()
        for reason, n in (c.get('fallback_by') or {}).items():
            for _ in range(int(n)): led.fallback(reason)
        for _ in range(int(c.get('served') or 0)): led.serve(f'replayed:L{L}')
    print(f'program_cache: program {key} (L={L}) loaded; trace-time kernel census replayed: ' + ', '.join(f"{k}: served {v.get('served')} fallback {v.get('fallback')}" for k, v in (census or {}).items()))

if args.program_cache:
    # L13: the store is opened before any design; its key part that does not depend on the inputs (code, levers, stack, device, flags) is computed once here
    _plocal = None                                               # 0.7.0: with L10's bridge provider on, this process's programs launch triangle attention through FFI specs registered here at trace time — on a core older than 0.5.39.0 (per-process launch specs) not runnable elsewhere: then the store traces, never loads / stores them (named); on a self-describing core they travel (0.7.5)
    if FUSED_TRIATTN is not None and getattr(FUSED_TRIATTN, 'bridge_on', lambda: False)():
        _plocal = (getattr(FUSED_TRIATTN, 'process_local_word', None) or (lambda: getattr(FUSED_TRIATTN, 'PROCESS_LOCAL', 'triattn_xla_ffi_specs')))()   # 0.7.5: None on a core whose bridge launches are self-describing (opt_core >= 0.5.39.0): stored / loaded like any program
    PROGRAMS = PROGLIB.Store(args.program_cache, vars(args), script_dir, census_snapshot=_census_snapshot, census_replay=_census_replay, process_local=_plocal)
    if _plocal: print(f'program_cache: process-local programs ({_plocal}): the L10 bridge launches through FFI specs registered at trace time in this process — programs are traced here, not loaded or stored')
    _pp = PROGRAMS.probe()
    print(f"program_cache: dir {_pp['dir']} key {_pp['static_key'][:16]} (code files {_pp['n_code_files']}, jax {_pp['jax']}, device {_pp['device']}, keyed in {_pp['t_key']} s) entries={_pp['entries']}")

class Timers:
    def __init__(self, path):
        self.path = path
        if path:
            d = os.path.dirname(os.path.abspath(path))
            os.makedirs(d, exist_ok=True)
    _lock = threading.Lock()     # L16: the writer thread appends records too
    def emit(self, kind, **kw):
        if not self.path: return
        kw.update(kind=kind, t=time.time(), pid=os.getpid())
        with Timers._lock, open(self.path, 'a') as f:
            f.write(json.dumps(kw, default=str) + '\n')

timers = Timers(args.timers)

#################################
# PDB reading / writing (replaces the PyRosetta pose I/O of predict.py)
#################################

def read_pdb_structure(fn):
    '''
    Parse the ATOM records of a PDB file.

    Returns (seq, chain_lengths, all_positions [L,37,3] float64, all_positions_mask [L,37] int64).
    Residues are taken in file order; a residue is identified by (chain id, residue number, insertion code).
    Atom parsing mirrors af2_util.af2_get_atom_positions (atom37 names; first altloc kept).
    '''
    residues = collections.OrderedDict()
    with open(fn) as fh:
        for l in fh:
            if l[:4] != "ATOM":
                continue
            altloc = l[16]
            if altloc not in (' ', 'A'):
                continue
            key = (l[21], l[22:26], l[26])
            if key not in residues:
                residues[key] = [l[17:20], []]
            residues[key][1].append((l[12:16].strip(), [float(l[30:38]), float(l[38:46]), float(l[46:54])]))

    num_res = len(residues)
    all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
    all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num], dtype=np.int64)

    seq = []
    chain_lengths = collections.OrderedDict()
    for idx, (key, (resname, atoms)) in enumerate(residues.items()):
        chain_lengths[key[0]] = chain_lengths.get(key[0], 0) + 1
        seq.append(residue_constants.restype_3to1.get(resname, 'X'))

        # same dtype path as af2_util.af2_get_atom_positions: float32 staging arrays copied into float64
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        for atom_name, xyz in atoms:
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = xyz
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name.upper() == 'SE' and resname == 'MSE':
                # Put the coordinates of the selenium atom in the sulphur column.
                pos[residue_constants.atom_order['SD']] = xyz
                mask[residue_constants.atom_order['SD']] = 1.0
        all_positions[idx] = pos
        all_positions_mask[idx] = mask

    return ''.join(seq), list(chain_lengths.values()), all_positions, all_positions_mask


def write_pdb(fn, aatype, atom_positions, atom_mask, binderlen, plddt):
    '''
    Minimal PDB writer (format of alphafold.common.protein.to_pdb): heavy atoms in atom37 order,
    B-factor = per-residue pLDDT, chain A = binder (or the monomer), chain B = target,
    residues numbered 1..L continuously (as the Rosetta pose dump of predict.py numbers them).
    '''
    restypes = residue_constants.restypes + ['X']
    res_1to3 = lambda r: residue_constants.restype_1to3.get(restypes[r], 'UNK')
    atom_types = residue_constants.atom_types

    pdb_lines = []
    atom_index = 1
    L = aatype.shape[0]
    chain_of = lambda i: 'A' if (binderlen < 0 or i < binderlen) else 'B'
    for i in range(L):
        res_name_3 = res_1to3(aatype[i])
        chain_id = chain_of(i)
        for atom_name, pos, mask in zip(atom_types, atom_positions[i], atom_mask[i]):
            if mask < 0.5:
                continue
            record_type = 'ATOM'
            name = atom_name if len(atom_name) == 4 else f' {atom_name}'
            alt_loc = ''
            insertion_code = ''
            occupancy = 1.00
            element = atom_name[0]  # Protein supports only C, N, O, S, this works.
            charge = ''
            atom_line = (f'{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}'
                         f'{res_name_3:>3} {chain_id:>1}'
                         f'{i + 1:>4}{insertion_code:>1}   '
                         f'{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}'
                         f'{occupancy:>6.2f}{plddt[i]:>6.2f}          '
                         f'{element:>2}{charge:>2}')
            pdb_lines.append(atom_line)
            atom_index += 1
        if i == L - 1 or chain_of(i + 1) != chain_id:
            # Close the chain.
            chain_end = 'TER'
            chain_termination_line = (
                f'{chain_end:<6}{atom_index:>5}      {res_1to3(aatype[i]):>3} '
                f'{chain_id:>1}{i + 1:>4}')
            pdb_lines.append(chain_termination_line)
            atom_index += 1
    pdb_lines.append('END')
    pdb_lines.append('')
    with open(fn, 'w') as f:
        f.write('\n'.join(pdb_lines))


class FeatureHolder():
    '''
    This is a struct which holds the features for a single structure being run through the model
    '''

    def __init__(self, seq, all_atom_positions, all_atom_masks, monomer, binderlen, tag):
        self.tag    = tag
        self.outtag = self.tag + '_af2pred'

        self.seq       = seq
        self.binderlen = binderlen
        self.monomer   = monomer

        # Pre model features
        self.initial_all_atom_positions = all_atom_positions
        self.initial_all_atom_masks = all_atom_masks

        # Post model features
        self.plddt_array = None
        self.score_dict  = None


class AF2_runner():
    '''
    This class handles generating features, running the model, and parsing outputs
    (predict.py's AF2_runner; only the pose I/O calls are replaced)
    '''

    def __init__(self, args, struct_manager):

        self.max_amide_dist = args.max_amide_dist

        # For timing
        self.t0 = None

        self.struct_manager = struct_manager

        # Other models may be run but their weights will also need to be downloaded
        self.model_name = "model_1_ptm"

        model_config = config.model_config(self.model_name)
        model_config.data.eval.num_ensemble = 1

        model_config.data.common.num_recycle = args.recycle
        model_config.model.num_recycle = args.recycle

        model_config.model.embeddings_and_evoformer.initial_guess = False if args.no_initial_guess else True

        model_config.data.common.max_extra_msa = 5
        model_config.data.eval.max_msa_clusters = 5

        params_dir = args.af2_dir

        t = time.time()
        model_params = data.get_model_haiku_params(model_name=self.model_name, data_dir=params_dir)
        timers.emit('params_loaded', dt=time.time() - t, params_dir=params_dir)

        if SUBBATCH is not None:
            # L9: rows per chunk of every inference_subbatch call (Attention modules, Transitions): mapping.inference_subbatch reads global_config.subbatch_size at trace time
            model_config.model.global_config.subbatch_size = SUBBATCH.rows()
        if args.tmpl_pointwise_sub:
            # L18: rows per chunk of the template point-wise attention (TemplateEmbedding, template.subbatch_size; stock 128 over N*N rows)
            model_config.model.embeddings_and_evoformer.template.subbatch_size = int(args.tmpl_pointwise_sub)
            print(f"tmpl_pointwise_sub: template.subbatch_size={int(args.tmpl_pointwise_sub)} (stock 128)")
            timers.emit('tmpl_pointwise_sub', value=int(args.tmpl_pointwise_sub), stock_value=128)   # the lever's record (af2ig_opt evidence: value == the mode's rows)
        if FLASH_ATTN is not None:
            # L8: the attention core of eligible Attention calls on the flash-attention kernel (modules.py Attention, patch 05)
            model_config.model.global_config.use_flash_attention = True
        self.model_config = model_config

        self.model_runner = model.RunModel(model_config, model_params)
        if args.device_params:
            # L1: one host->device transfer of the weights for the whole campaign instead of one per design.
            # jax.device_put of a numpy-backed tree yields arrays with the same values/dtypes; the jitted
            # program is the same (bitwise-identical outputs).
            t = time.time()
            self.model_runner.params = jax.device_put(self.model_runner.params)
            timers.emit('params_on_device', dt=time.time() - t)

        self.seen_lengths = set()
        self.prepared = {}                   # L7: inputs featurized by the precompile pass, taken by process_struct (each input is featurized once)
        self.programs = {}                   # L7/L13: input signature -> the compiled model program (af2ig_opt.programs), for the rest of the process
        self.programs_lock = threading.Lock()

    def model_program(self, model_args, L):
        # L7/L13: the compiled program for these arguments — prepared ahead of the call by jax's AOT path (trace -> lower -> compile: the
        # executable jit itself would run, so the same output bytes), one per input signature for the rest of the process; with
        # -program_cache DIR loaded from / stored to DIR (af2ig_opt.programs.Store). Returns (callable, record-or-None for a signature
        # already prepared in this process).
        if PRECOMP is not None and L in PRECOMP and threading.current_thread().name.split('_')[0] != 'precompile':   # 0.7.7: a design waits on ITS OWN length's background prepare only (never on the other lengths'); a failed prepare raises here = this design's named failure
            _, waited = PRECOMP.wait(L)
            _precompile_emit()                                                          # records of the programs that became ready meanwhile (main thread)
            if waited >= 0.5:
                print(f'precompile: waited {waited:.1f} s for length {L}')
        sig = PROGLIB.signature(model_args)
        with self.programs_lock:
            fn = self.programs.get(sig)
        if fn is not None:
            return fn, None
        if PROGRAMS is not None:
            fn, rec = PROGRAMS.program(self.model_runner.apply, model_args, sig, L)
        else:
            fn, rec = PROGLIB.aot_compile(self.model_runner.apply, model_args)
        with self.programs_lock:
            fn = self.programs.setdefault(sig, fn)
        return fn, rec

    def featurize(self, feat_holder):

        initial_guess = af2_util.parse_initial_guess(feat_holder.initial_all_atom_positions)

        # Determine which residues to template
        if feat_holder.monomer:
            # For monomers predict all residues
            feat_holder.residue_mask = [False for i in range(len(feat_holder.seq))]
        else:
            # For interfaces fix the target and predict the binder
            feat_holder.residue_mask = [int(i) > feat_holder.binderlen for i in range(len(feat_holder.seq))]

        template_dict = af2_util.generate_template_features(
                                                            feat_holder.seq,
                                                            feat_holder.initial_all_atom_positions,
                                                            feat_holder.initial_all_atom_masks,
                                                            feat_holder.residue_mask
                                                           )
        # Gather features
        feature_dict = {
            **pipeline.make_sequence_features(sequence=feat_holder.seq,
                                            description="none",
                                            num_res=len(feat_holder.seq)),
            **pipeline.make_msa_features(msas=[[feat_holder.seq]],
                                        deletion_matrices=[[[0]*len(feat_holder.seq)]]),
            **template_dict
        }

        if feat_holder.monomer:
            breaks = []
        else:
            breaks = af2_util.check_residue_distances(
                            feat_holder.initial_all_atom_positions,
                            feat_holder.initial_all_atom_masks,
                            self.max_amide_dist
                        )

        feature_dict['residue_index'] = af2_util.insert_truncations(feature_dict['residue_index'], breaks)

        t = time.time()
        feature_dict = self.model_runner.process_features(feature_dict, random_seed=0)
        feat_holder.t_tf = time.time() - t

        return feature_dict, initial_guess

    def generate_scoredict(self, feat_holder, confidences, rmsds) -> None:
        '''
        Collect the confidence values, slicing them to the binder and target regions
        then add the parsed scores to the score_dict
        '''

        binderlen = feat_holder.binderlen

        plddt_array = confidences['plddt']
        plddt = np.mean( plddt_array )

        if feat_holder.monomer:
            plddt_binder = np.mean( plddt_array )
            plddt_target = float('nan')
        else:
            plddt_binder = np.mean( plddt_array[:binderlen] )
            plddt_target = np.mean( plddt_array[binderlen:] )

        pae = confidences['predicted_aligned_error']

        if feat_holder.monomer:
            pae_binder = np.mean( pae )
            pae_target = float('nan')
            pae_interaction_total = float('nan')
        else:
            pae_interaction1 = np.mean( pae[:binderlen,binderlen:] )
            pae_interaction2 = np.mean( pae[binderlen:,:binderlen] )
            pae_binder = np.mean( pae[:binderlen,:binderlen] )
            pae_target = np.mean( pae[binderlen:,binderlen:] )
            pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

        time_ = timer() - self.t0

        score_dict = {
                "plddt_total" : plddt,
                "plddt_binder" : plddt_binder,
                "plddt_target" : plddt_target,
                "pae_binder" : pae_binder,
                "pae_target" : pae_target,
                "pae_interaction" : pae_interaction_total,
                "binder_aligned_rmsd": rmsds['binder_aligned_rmsd'],
                "target_aligned_rmsd": rmsds['target_aligned_rmsd'],
                "time" : time_
        }

        # Store this in the feature holder for later use
        feat_holder.score_dict = score_dict

        # If we ever want to write strings to the score file we can do it here
        string_dict = None

        self.struct_manager.record_scores(feat_holder.outtag, score_dict, string_dict)

        print(score_dict)
        print(f"Tag: {feat_holder.outtag} reported success in {time_} seconds")

    def process_output(self, feat_holder, feature_dict, prediction_result) -> None:
        '''
        Take the AF2 output, parse the confidence scores from this and register the scores in the score file
        Also write out the structure
        '''

        # First extract the structure and confidence scores from the prediction result

        structure_module = prediction_result['structure_module']
        this_protein = protein.Protein(
            aatype=feature_dict['aatype'][0],
            atom_positions=structure_module['final_atom_positions'][...],
            atom_mask=structure_module['final_atom_mask'][...],
            residue_index=feature_dict['residue_index'][0] + 1,
            b_factors=np.zeros_like(structure_module['final_atom_mask'][...]) )

        confidences = {}
        confidences['distogram'] = prediction_result['distogram']
        confidences['plddt'] = confidence.compute_plddt(
                prediction_result['predicted_lddt']['logits'][...])
        if 'predicted_aligned_error' in prediction_result:
            confidences.update(confidence.compute_predicted_aligned_error(
                prediction_result['predicted_aligned_error']['logits'][...],
                prediction_result['predicted_aligned_error']['breaks'][...]))

        feat_holder.plddt_array = confidences['plddt']

        # Calculate the RMSDs
        if feat_holder.monomer:
            # predict.py computes these with binderlen = -1 for monomers; keep the same call
            target_mask = np.zeros(len(feat_holder.seq), dtype=bool)
            target_mask[feat_holder.binderlen:] = True
        else:
            target_mask = np.zeros(len(feat_holder.seq), dtype=bool)
            target_mask[feat_holder.binderlen:] = True

        # stock hands the device (jax) array to calculate_rmsds -> float32 SVD through jax; keep that path
        pred_for_rmsd = structure_module.get('final_atom_positions_device', this_protein.atom_positions)
        rmsds = af2_util.calculate_rmsds(
            feat_holder.initial_all_atom_positions,
            pred_for_rmsd,
            target_mask
        )

        # Now we can finally write the scores and the predicted structure to disk
        self.generate_scoredict(feat_holder, confidences, rmsds)
        self.struct_manager.dump_structure(feat_holder, this_protein, confidences)

    def prepare_struct(self, tag):
        '''CPU part of process_struct: read the PDB file and build the model inputs (may run on a worker thread)'''
        t_start = time.time()
        seq, all_atom_positions, all_atom_masks, monomer, binderlen, usetag = self.struct_manager.load_structure(tag)
        t_loaded = time.time()
        feat_holder = FeatureHolder(seq, all_atom_positions, all_atom_masks, monomer, binderlen, usetag)
        print(f'Processing struct with tag: {feat_holder.tag}')
        feature_dict, initial_guess = self.featurize(feat_holder)
        t_feat = time.time()
        feat_holder.t_load = t_loaded - t_start
        feat_holder.t_feat = t_feat - t_loaded
        return feat_holder, feature_dict, initial_guess

    def process_struct(self, tag) -> None:

        # Start the timer
        self.t0 = timer()
        t_start = time.time()

        pre = self.prepared.pop(tag, None)
        t_prefetch_wait = None
        if pre is not None:
            # L7: featurized by the precompile pass — the same inputs, built once
            feat_holder, feature_dict, initial_guess = pre
        elif PREFETCH is not None:
            # L15: featurized ahead of its turn by the prefetch worker (the same prepare_struct, one input at a time, in this loop's order);
            # blocks only if the worker is still on it; a featurization error re-raises here, where stock's would surface
            pre, t_prefetch_wait = PREFETCH.take(tag)
            if pre is None:
                # not scheduled on the worker (or featurized by the precompile pass meanwhile): prepared here, as stock
                pre = self.prepared.pop(tag, None) or self.prepare_struct(tag)
            feat_holder, feature_dict, initial_guess = pre
        else:
            feat_holder, feature_dict, initial_guess = self.prepare_struct(tag)
        feat_holder.t_prefetch_wait = t_prefetch_wait
        usetag, binderlen, monomer = feat_holder.tag, feat_holder.binderlen, feat_holder.monomer
        t_feat = time.time()

        L = len(feat_holder.seq)
        new_shape = L not in self.seen_lengths
        self.seen_lengths.add(L)

        # Run model
        start = timer()
        print(f'Running {self.model_name}')

        model_args = (self.model_runner.params, jax.random.PRNGKey(0), feature_dict, initial_guess)
        prog_rec = None
        if PROGLIB is not None:
            # L7/L13: the program prepared ahead of the call (loaded, or traced+lowered+compiled once per signature), then run — the
            # executable and the arguments are the ones apply() would use: the same output bytes
            prog_fn, prog_rec = self.model_program(model_args, L)
            prediction_result = prog_fn(*model_args)
        else:
            prediction_result = self.model_runner.apply(*model_args)
        # block so that the model timer is meaningful (process_output pulls these arrays to the host next anyway)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), prediction_result)
        if args.host_outputs:
            # L6: one device->host transfer of the whole output tree instead of many implicit per-op transfers
            # inside numpy/scipy post-processing. np.asarray of a jax Array yields the same float32 values.
            # EXCEPTION kept on purpose: predict.py feeds the *device* array of final atom positions to
            # af2_util.calculate_rmsds, whose numpy Kabsch/SVD then executes through jax in float32; with a host
            # float32 array numpy would promote to float64 (the reference coordinates are float64) and the two
            # RMSD columns would change in the 3rd decimal. The device array is therefore kept for that one call
            # (see process_output) so that every written number stays bitwise identical to stock.
            dev_final_atom_positions = prediction_result['structure_module']['final_atom_positions']
            prediction_result = jax.device_get(prediction_result)
            prediction_result['structure_module']['final_atom_positions_device'] = dev_final_atom_positions
        t_model = time.time()

        print(f'Tag: {feat_holder.tag} finished AF2 prediction in {timer() - start} seconds')

        if WRITER is not None:
            # L16: the output step below (and this design's records, then its checkpoint line) runs on the writer thread, in order, while the
            # loop moves on; the closure holds this design's outputs (host memory) until it has run
            t_submit = time.time()
            def _output_job():
                self._finish_struct(feat_holder, feature_dict, prediction_result, prog_rec, new_shape, L, usetag, binderlen, monomer, t_start, t_feat, t_model, t_submit)
            WRITER.submit(usetag, _output_job, always=lambda: self.struct_manager.record_checkpoint(usetag))
            return
        self._finish_struct(feat_holder, feature_dict, prediction_result, prog_rec, new_shape, L, usetag, binderlen, monomer, t_start, t_feat, t_model, None)

    def _finish_struct(self, feat_holder, feature_dict, prediction_result, prog_rec, new_shape, L, usetag, binderlen, monomer, t_start, t_feat, t_model, t_submit):
        '''The output step of process_struct and the design's records (runs on the loop's thread, or on the L16 writer thread)'''
        t_out0 = time.time()
        # Process outputs
        self.process_output(feat_holder, feature_dict, prediction_result)
        t_out = time.time()

        if prog_rec is not None:
            # L7/L13 evidence per new signature: where the first call's seconds went (trace / lower / compile-or-cache-load / store, or load)
            timers.emit('program', tag=usetag, **prog_rec)
        if FLASH_ATTN is not None and new_shape:
            # L8 evidence: the adapter's trace-time census of Attention calls routed to the kernel / left on the stock ops (cumulative over the shapes compiled so far)
            timers.emit('flash_attn', L_compiled=L, **FLASH_ATTN.census())
        if FUSED_TRIATTN is not None and new_shape:
            # L10 evidence: the adapter's trace-time census of TriangleAttention calls served by the fused block / left on the stock body, by reason
            timers.emit('fused_triattn', L_compiled=L, **FUSED_TRIATTN.census())
        if FUSED_TRIMUL is not None and new_shape:
            # L11 evidence: the same census for the TriangleMultiplication calls
            timers.emit('fused_trimul', L_compiled=L, **FUSED_TRIMUL.census())
        if OPM is not None and new_shape:
            # L12 evidence: the same census for the OuterProductMean calls (served shapes name the MSA depth per stack)
            timers.emit('opm_reassoc', L_compiled=L, **OPM.census())
        if PAIRSTACK is not None and new_shape:
            # M1 evidence: the pair-stack install's census (trimul_chunk traces engaged / disengaged)
            timers.emit('pairstack', L_compiled=L, **PAIRSTACK.census())
        if SUBBATCH is not None and new_shape:
            # L9 evidence: the shared policy's decision record for this compiled length (rows, source, stock value)
            timers.emit('subbatch', L_compiled=L, **SUBBATCH.record(L))
        timers.emit('design', tag=usetag, L=L, L_compiled=L, binderlen=binderlen, monomer=monomer, new_shape=new_shape,
                    t_load=feat_holder.t_load, t_feat=feat_holder.t_feat, t_tf=feat_holder.t_tf,
                    t_wait_inputs=t_feat - t_start, t_model=t_model - t_feat, t_output=t_out - t_out0, t_total=t_out - t_start,
                    t_prefetch_wait=getattr(feat_holder, 't_prefetch_wait', None),
                    t_main=(t_submit or t_out) - t_start, t_output_lag=(t_out0 - t_submit) if t_submit else None, output_overlapped=t_submit is not None,
                    pae_interaction=float(feat_holder.score_dict['pae_interaction']),
                    plddt_total=float(feat_holder.score_dict['plddt_total']), peak_bytes_in_use=_peak_bytes_in_use())


class StructManager():
    '''
    This class handles all of the input and output for the AF2 model: a directory of PDB files in,
    PDB files + score file out, with predict.py's checkpointing.
    '''

    def __init__(self, args):
        self.args = args

        self.score_fn = args.scorefilename
        self.force_monomer = args.force_monomer

        assert args.pdbdir != '', 'predict_pdb.py needs -pdbdir (silent files require PyRosetta and are not supported by this front end)'

        self.pdbdir    = args.pdbdir
        self.outpdbdir = args.outpdbdir

        self.struct_iterator = sorted(glob.glob(os.path.join(args.pdbdir, '*.pdb')))

        # Parse the runlist and determine which structures to process
        if args.runlist != '':
            with open(args.runlist, 'r') as f:
                self.runlist = set([line.strip() for line in f])

                # Filter the struct iterator to only include those in the runlist
                self.struct_iterator = [struct for struct in self.struct_iterator if '.'.join(os.path.basename(struct).split('.')[:-1]) in self.runlist]

                print(f'After filtering by runlist, {len(self.struct_iterator)} structures remain')

        if args.sort_by_length:
            def nres(fn):
                n = 0
                with open(fn) as fh:
                    for l in fh:
                        if l[:4] == 'ATOM' and l[12:16].strip() == 'CA': n += 1
                return n
            self.struct_iterator = sorted(self.struct_iterator, key=lambda fn: (nres(fn), fn))

        # Setup checkpointing
        self.chkfn = args.checkpoint_name
        self.finished_structs = set()

        if os.path.isfile(self.chkfn):
            with open(self.chkfn, 'r') as f:
                for line in f:
                    self.finished_structs.add(line.strip())

    def record_checkpoint(self, tag):
        '''
        Record the fact that this tag has been processed.
        Write this tag to the list of finished structs
        '''
        with open(self.chkfn, 'a') as f:
            f.write(f'{tag}\n')

    def iterate(self):
        '''
        Iterate over the pdb directory and run the model on each structure
        '''

        # Iterate over the structs and for each, check that the struct has not already been processed
        for struct in self.struct_iterator:
            tag = '.'.join(os.path.basename(struct).split('.')[:-1])

            if tag in self.finished_structs:
                print(f'{tag} has already been processed. Skipping')
                continue

            yield struct

    def record_scores(self, tag, score_dict, string_dict):
        '''
        Record the scores for this structure to the score file.
        '''

        # Check whether the score file exists
        write_header = False
        if not os.path.isfile(self.score_fn):
            write_header = True

        af2_util.add2scorefile(tag, self.score_fn, write_header, score_dict, string_dict)

    def dump_structure(self, feat_holder, this_protein, confidences):
        '''
        Write the predicted structure as a PDB file (binder = chain A, target = chain B, pLDDT as B-factor)
        '''
        if not os.path.exists(self.outpdbdir):
            os.makedirs(self.outpdbdir, exist_ok=True)

        pdbfile = os.path.join(self.outpdbdir, feat_holder.outtag + '.pdb')
        write_pdb(pdbfile, this_protein.aatype, this_protein.atom_positions, this_protein.atom_mask,
                  -1 if feat_holder.monomer else feat_holder.binderlen, feat_holder.plddt_array)


    def load_structure(self, fn):
        '''
        Load a structure from a PDB file. Chain 1 is the binder, every following chain is target.
        '''
        usetag = '.'.join(os.path.basename(fn).split('.')[:-1])

        seq, chain_lengths, all_atom_positions, all_atom_masks = read_pdb_structure(fn)

        if len(chain_lengths) < 1:
            raise Exception( f"Structure {fn} is empty. This is not supported by this script." )

        elif len(chain_lengths) == 1:
            monomer   = True
            binderlen = -1

        else:
            if len(chain_lengths) > 2:
                print( f"Structure {usetag} has {len(chain_lengths)} chains: chain 1 ({chain_lengths[0]} residues) is the binder, "
                       f"the remaining {len(chain_lengths)-1} chains are treated as the target (chain breaks are detected from amide distances, as predict.py does)" )
            if self.force_monomer:
                print( "/" * 60 )
                print( f"Structure {usetag} has several chains. But force_monomer is set to True. Treating as monomer.")
                print( f"I am going to assume that the first chain is the binder and that is the chain I will predict")
                print( "/" * 60 )

                monomer   = True
                n = chain_lengths[0]
                seq = seq[:n]
                all_atom_positions = all_atom_positions[:n]
                all_atom_masks = all_atom_masks[:n]
                binderlen = -1
            else:
                monomer   = False
                binderlen = chain_lengths[0]

        return seq, all_atom_positions, all_atom_masks, monomer, binderlen, usetag

####################
####### Main #######
####################

def _peak_bytes_in_use():
    """Report only (the `design` record's peak_bytes_in_use): the JAX allocator's peak bytes in use on the first local device,
    read after the design's outputs are written; process-cumulative (JAX has no peak reset). None where the backend keeps no
    memory statistics (CPU). nvidia-smi shows the preallocated pool (XLA_PYTHON_CLIENT_MEM_FRACTION), not this."""
    try:
        stats = jax.local_devices()[0].memory_stats() or {}
    except Exception:
        return None
    v = stats.get('peak_bytes_in_use')
    return None if v is None else int(v)

device = jax.default_backend()
if device == 'gpu':
    print('/' * 60)
    print('/' * 60)
    print('Found GPU and will use it to run AF2')
    print('/' * 60)
    print('/' * 60)
    print('\n')
else:
    print('/' * 60)
    print('/' * 60)
    print('WARNING! No GPU detected running AF2 on CPU')
    print('/' * 60)
    print('/' * 60)
    print('\n')

timers.emit('proc_start', t_proc0=T_PROC0, t_imports=T_IMPORTS - T_PROC0, argv=sys.argv, jax=jax.__version__,
            backend=device, devices=[str(d) for d in jax.devices()], alphafold_file=model.__file__,
            env={k: os.environ.get(k) for k in ('XLA_FLAGS', 'XLA_PYTHON_CLIENT_MEM_FRACTION', 'XLA_PYTHON_CLIENT_PREALLOCATE',
                                                'TF_FORCE_UNIFIED_MEMORY', 'JAX_PLATFORMS', 'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE')})

t = time.time()
struct_manager = StructManager(args)
af2_runner     = AF2_runner(args, struct_manager)
timers.emit('runner_ready', dt=time.time() - t, t_since_proc0=time.time() - T_PROC0, n_inputs=len(struct_manager.struct_iterator))

def precompile_shapes(paths, num_threads):
    '''
    L7: XLA compiles one program per host thread; stock compiles each new input length lazily inside the loop
    (~130-165 s each on H100 for this model). Featurize one representative input per distinct compiled length and
    prepare its program (af2ig_opt.programs: jax AOT trace -> lower -> compile, or a load from -program_cache) from a
    thread pool so the compilations overlap; no forward is run and the featurized inputs are kept for the loop, which
    calls the prepared executables (AF2_runner.model_program).
    '''
    import concurrent.futures
    from opt_core.oom import is_oom                                  # an out-of-memory is never a skipped input: it propagates (the model-opt tree's one OOM classifier)
    t0 = time.time()
    reps = {}
    for pth in paths:
        try:
            seq, chl, _, _ = read_pdb_structure(pth)
        except Exception as e:
            if is_oom(e): raise
            print(f'precompile: skipping {pth}: {e!r}'); continue
        L = len(seq) if (len(chl) > 1 and not args.force_monomer) else chl[0]
        if L not in reps: reps[L] = pth
    todo = list(reps.items())                                        # 0.7.7: the LOOP's order (first appearance; U1 sorts the inputs by length), not sorted() — the first design's length is prepared first
    print(f'precompile: {len(todo)} distinct compiled lengths over {len(paths)} inputs: {[l for l, _ in todo]} with {num_threads} threads')
    prepared = []
    def _featurize(L, pth):
        try:
            item = af2_runner.prepare_struct(pth)
            af2_runner.prepared[pth] = item                          # kept for the loop: the input is featurized once
            prepared.append((L, pth, item)); return item
        except Exception as e:
            if is_oom(e): raise
            print(f'precompile: featurization failed for {pth}: {e!r}'); return None
    def _compile(item):
        L, (pth, (fh, feat, ig)) = item
        t = time.time()
        # AOT: the program is traced, lowered and compiled (or loaded, -program_cache) WITHOUT running a forward — nothing is discarded and no
        # activation memory is taken here; the loop runs the prepared executable
        _, rec = af2_runner.model_program((af2_runner.model_runner.params, jax.random.PRNGKey(0), feat, ig), L)
        af2_runner.seen_lengths.add(L)
        return rec
    # 0.7.7 — STREAMING: the first length of the loop's order is prepared first and ALONE, the others on the N threads once it is ready; the loop
    # starts now and each design waits only on its own length (AF2_runner.model_program). Records are emitted as programs become ready
    # (_precompile_emit, main thread) and the precompile_done record at exit (precompile_finish).
    global PRECOMP
    bg = PRECOMPLIB.background_threads(num_threads)                   # the background concurrency for the lengths after the first, sized to the host (min(N, (cpus-2)//4)); the first is always alone
    PRECOMP = PRECOMPLIB.Streaming(_compile, threads=bg)
    PRECOMP.t_enter = t0; PRECOMP.requested = num_threads; PRECOMP.host_cpus = PRECOMPLIB.host_cpus()
    started = False
    for L, pth in todo:                                              # 0.7.7: the FIRST length is featurized and its compile started before the other lengths are even featurized
        item = _featurize(L, pth)
        if item is None: continue
        if not started:
            PRECOMP.start([(L, (pth, item))]); started = True
        else:
            PRECOMP.add_rest([(L, (pth, item))])
    if not started:
        PRECOMP.start([])
    PRECOMP.close()
    print(f'precompile: streaming — length {prepared[0][0] if prepared else None} first and alone, then {[l for l, _, _ in prepared[1:]]} on {bg} background thread(s) (requested {num_threads}, host cpus {PRECOMP.host_cpus}); the loop starts now (each design waits on its own length only)')

_PRECOMP_EMIT_LOCK = threading.Lock()
def _precompile_emit():
    # the records of every program that became ready since the last call (same kinds as before 0.7.7: precompile_shape, program, the kernel censuses)
    if PRECOMP is None: return
    with _PRECOMP_EMIT_LOCK:
        for L, dt, rec, exc in PRECOMP.drain():
            if exc is not None:
                print(f'precompile: length {L} program FAILED after {dt:.1f} s: {exc!r} (its designs fail by name in the loop)')
                timers.emit('precompile_shape', L_compiled=L, dt=dt, source='failed', err=repr(exc)); continue
            print(f"precompile: length {L} program ready in {dt:.1f} s ({(rec or {}).get('source', 'in memory')}), {PRECOMP.ready_at.get(L, 0):.1f} s after the pass began")
            timers.emit('precompile_shape', L_compiled=L, dt=dt, source=(rec or {}).get('source', 'memory'), ready_at=PRECOMP.ready_at.get(L), began_at=PRECOMP.began_at.get(L))
            if rec is not None:
                timers.emit('program', tag=None, **rec)
            if FLASH_ATTN is not None:
                timers.emit('flash_attn', L_compiled=L, **FLASH_ATTN.census())
            if FUSED_TRIATTN is not None:
                timers.emit('fused_triattn', L_compiled=L, **FUSED_TRIATTN.census())
            if FUSED_TRIMUL is not None:
                timers.emit('fused_trimul', L_compiled=L, **FUSED_TRIMUL.census())
            if OPM is not None:
                timers.emit('opm_reassoc', L_compiled=L, **OPM.census())
            if PAIRSTACK is not None:
                timers.emit('pairstack', L_compiled=L, **PAIRSTACK.census())
            if SUBBATCH is not None:
                timers.emit('subbatch', L_compiled=L, **SUBBATCH.record(L))

def precompile_finish():
    # at exit (or before the exit records): wait for the stragglers, emit their records and the precompile_done record (n_shapes, threads, dt = pass
    # start -> last program ready; dt_first = -> first program ready; streaming=1)
    if PRECOMP is None: return
    summ = PRECOMP.finish()
    _precompile_emit()
    dt_all = (summ.get('dt_all') or 0.0) + (PRECOMP.t0 - PRECOMP.t_enter if PRECOMP.t0 else 0.0)
    timers.emit('precompile_done', n_shapes=summ.get('n'), threads=summ.get('threads'), requested=getattr(PRECOMP, 'requested', None), host_cpus=getattr(PRECOMP, 'host_cpus', None), dt=dt_all, dt_first=summ.get('dt_first'), hold_s=summ.get('hold_s'), streaming=1, order=summ.get('order'), failed=summ.get('failed'))
    print(f"precompile: done — {summ.get('n')} program(s), first ready {summ.get('dt_first') or 0:.1f} s / all {dt_all:.1f} s after the pass began (streaming; failed: {summ.get('failed') or 'none'})")

if args.precompile > 0:
    precompile_shapes(list(struct_manager.iterate()), args.precompile)

if PREFETCHLIB is not None:
    # L15: one worker thread featurizes the next -prefetch N inputs of the loop's order ahead of their turn (inputs the precompile pass
    # featurized are skipped, not featurized twice); built here, after the precompile pass, so no two featurizations ever run at once
    _todo = list(struct_manager.iterate())
    try:
        # the worker's featurization keeps OFF the GPU: its only device work is af2_util.parse_initial_guess's jax.numpy array construction
        # (zeros that are discarded, then one array built from host coordinates: a dtype conversion and a copy, no arithmetic), which on the
        # GPU device would queue behind the running program (PjRt admits a bounded number of computations in flight per device and parks the
        # thread that exceeds it until the program ends — the worker would wait for the very forward it is meant to overlap). On jax's CPU
        # device the same calls run at once; the initial guess is handed over as host memory holding the same float32 values, and the model
        # call puts it on the GPU as it does any host input (the feature dict already is host memory). Bytes out are unchanged.
        _PREFETCH_CPU = jax.devices('cpu')[0]
    except Exception:      # no CPU backend in this process (JAX_PLATFORMS without cpu): the worker featurizes as the loop would, on the default device
        _PREFETCH_CPU = None
    def _prepare_ahead(tag):
        if _PREFETCH_CPU is None:
            return af2_runner.prepare_struct(tag)
        with jax.default_device(_PREFETCH_CPU):
            feat_holder, feature_dict, initial_guess = af2_runner.prepare_struct(tag)
        return feat_holder, feature_dict, np.asarray(initial_guess)
    PREFETCH = PREFETCHLIB.Prefetcher(_prepare_ahead, _todo, depth=args.prefetch, skip=lambda p: p in af2_runner.prepared,
                                      info={'offdevice': _PREFETCH_CPU is not None})
    print(f"prefetch: {PREFETCHLIB.IMPL} depth={args.prefetch} workers=1 offdevice={_PREFETCH_CPU is not None} inputs={len(_todo)} (already featurized by the precompile pass: {len(af2_runner.prepared)})")

if PREFETCHLIB is not None and args.overlap_output > 0:
    def _output_failed(tag, exc):
        print( "Struct with tag %s failed in its output step with error: %s"%( tag, type(exc) ) )
        timers.emit('failed', tag=tag, err=repr(exc))
    WRITER = PREFETCHLIB.OutputWriter(depth=args.overlap_output, on_error=_output_failed)
    print(f"overlap_output: thread_writer depth={args.overlap_output}")

n_done = 0

def _run_one(pdb):
    if args.debug: af2_runner.process_struct(pdb)

    else: # When not in debug mode the script will continue to run even when some structures fail
        t0 = timer()

        try: af2_runner.process_struct(pdb)

        except KeyboardInterrupt: sys.exit( "Script killed by Control+C, exiting" )

        except:
            if WRITER is not None:
                WRITER.drain()      # L16: earlier designs' outputs and checkpoint lines land before this failure is recorded, as in the serial loop
            seconds = int(timer() - t0)
            print( "Struct with tag %s failed in %i seconds with error: %s"%( pdb, seconds, sys.exc_info()[0] ) )
            timers.emit('failed', tag=pdb, err=repr(sys.exc_info()[1]))

    # We are done with one pdb, record that we finished
    tagbase = '.'.join(os.path.basename(pdb).split('.')[:-1])
    if WRITER is not None and WRITER.submitted(tagbase):
        return              # L16: the writer records the checkpoint right after it writes this design's outputs (order kept)
    struct_manager.record_checkpoint(tagbase)

for pdb in struct_manager.iterate():
    _run_one(pdb)
    n_done += 1
    if n_done == 1 and PRECOMP is not None:      # L7 (0.7.7): the first design is written (its outputs drained from the writer thread first) — only now may the other lengths' compiles start
        if WRITER is not None:
            WRITER.drain()
        PRECOMP.release()
        print(f'precompile: first design done {time.time() - T_PROC0:.1f} s after process start — background compiles released')

precompile_finish()      # L7 (0.7.7): stragglers + the precompile_done record before the exit records

if WRITER is not None:
    # L16: every queued output is written before the exit records; then the writer's census as the output_writer record and its LEVER line
    _ow = WRITER.close()
    timers.emit('output_writer', **_ow)
    print(PREFETCHLIB.writer_line(_ow), file=sys.stderr, flush=True)
if PREFETCH is not None:
    # L15 evidence at exit: the prefetcher's census (queued / taken / ready-without-waiting / wait and worker seconds) as the prefetch record, and its LEVER line on stderr
    _pf = PREFETCH.close()
    timers.emit('prefetch', **_pf)
    print(PREFETCHLIB.line(_pf), file=sys.stderr, flush=True)

if FLASH_ATTN is not None:
    # L8 evidence at exit: the whole run's census as the last flash_attn record, and the tree's per-lever line on stderr
    timers.emit('flash_attn', L_compiled=None, final=True, **FLASH_ATTN.census())
    FLASH_ATTN.emit()
if FUSED_TRIATTN is not None:
    # L10 evidence at exit: the whole run's census as the last fused_triattn record, and the tree's per-lever line on stderr
    timers.emit('fused_triattn', L_compiled=None, final=True, **FUSED_TRIATTN.census())
    FUSED_TRIATTN.emit()
if FUSED_TRIMUL is not None:
    # L11 evidence at exit, likewise
    timers.emit('fused_trimul', L_compiled=None, final=True, **FUSED_TRIMUL.census())
    FUSED_TRIMUL.emit()
if OPM is not None:
    # L12 evidence at exit, likewise
    timers.emit('opm_reassoc', L_compiled=None, final=True, **OPM.census())
    OPM.emit()
if PAIRSTACK is not None:
    # M1 evidence at exit: the whole run's pair-stack census as the last pairstack record, and the per-lever lines on stderr
    timers.emit('pairstack', L_compiled=None, final=True, **PAIRSTACK.census())
    PAIRSTACK.print_lines()
if PROGRAMS is not None:
    # L13 evidence at exit: the store's census as the last program_store record, and its per-lever line on stderr
    timers.emit('program_store', **PROGRAMS.census())
    print(PROGLIB.line(PROGRAMS), file=sys.stderr, flush=True)
timers.emit('proc_end', n_done=n_done, t_since_proc0=time.time() - T_PROC0)
