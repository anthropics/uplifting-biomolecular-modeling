"""Convert OpenFold3 (PyTorch) weights to AlphaFold3 (JAX) format.

Usage:
    python convert_of3_weights.py \\
        --of3_checkpoint /path/to/of3-p2-155k.pt \\
        --output_dir /path/to/af3_params/

    # Download OF3 weights from public AWS bucket first (requires AWS CLI):
    aws s3 cp s3://openfold/staging/of3-p2-155k.pt ./of3-p2-155k.pt --no-sign-request

Output:
    /path/to/af3_params/of3_ported_weights.bin.zst  (~1.4 GB)

The converted directory can then be passed to run_alphafold.py via --model_dir,
together with --of3_weights to enable the OF3-compatible model configuration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Convert OpenFold3 (PyTorch) weights to AlphaFold3 (JAX) format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--of3_checkpoint', type=Path, required=True,
                   help='Path to OF3 .pt checkpoint file')
    p.add_argument('--output_dir', type=Path, required=True,
                   help='Directory to write converted AF3 parameter file')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--use_ema', dest='use_ema', action='store_true', default=True,
                   help='Use EMA weights (recommended for inference)')
    g.add_argument('--no_ema', dest='use_ema', action='store_false',
                   help='Use raw training weights instead of EMA')
    p.add_argument('--n_pairformer_blocks', type=int, default=48)
    p.add_argument('--n_msa_blocks', type=int, default=4)
    p.add_argument('--n_diff_blocks', type=int, default=24,
                   help='Total diffusion transformer blocks (default 24 = 6 super × 4 sub)')
    p.add_argument('--n_super_blocks', type=int, default=6)
    p.add_argument('--n_atom_blocks', type=int, default=3)
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.of3_checkpoint.exists():
        print(f'ERROR: checkpoint not found: {args.of3_checkpoint}', file=sys.stderr)
        return 1

    from alphafold3.model.of3_weight_converter import (
        load_of3_checkpoint,
        map_of3_to_af3,
        save_af3_params,
    )

    print(f'Loading {args.of3_checkpoint}  (use_ema={args.use_ema}) ...')
    t0 = time.perf_counter()
    sd = load_of3_checkpoint(args.of3_checkpoint, use_ema=args.use_ema)
    print(f'  {len(sd)} tensors loaded  ({time.perf_counter()-t0:.1f}s)')

    print('Converting ...')
    t0 = time.perf_counter()
    af3_params = map_of3_to_af3(
        sd,
        n_pairformer_blocks=args.n_pairformer_blocks,
        n_msa_blocks=args.n_msa_blocks,
        n_diff_blocks=args.n_diff_blocks,
        n_super_blocks=args.n_super_blocks,
        n_atom_blocks=args.n_atom_blocks,
    )
    n_params = sum(v.size for scope in af3_params.values() for v in scope.values())
    print(f'  {len(af3_params)} scopes, {n_params:,} elements  ({time.perf_counter()-t0:.1f}s)')

    print(f'Saving to {args.output_dir} ...')
    t0 = time.perf_counter()
    out = save_af3_params(af3_params, args.output_dir)
    print(f'  {out}  ({out.stat().st_size/1e6:.0f} MB, {time.perf_counter()-t0:.1f}s)')
    print(f'\nDone. Use with:\n'
          f'  python run_alphafold.py --model_dir {args.output_dir} --of3_weights ...')
    return 0


def run():
    sys.exit(main())


if __name__ == '__main__':
    sys.exit(main())
