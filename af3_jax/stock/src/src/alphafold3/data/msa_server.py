"""ColabFold/MMseqs2 MSA server client for AlphaFold3.

Queries https://api.colabfold.com to generate unpaired and paired MSAs for
protein chains. RNA chains receive a query-sequence-only stub (ColabFold
does not have RNA databases).

Adapted from the ColabFold run_mmseqs2 implementation and OpenFold3's
colabfold_msa_server.py.
"""

from __future__ import annotations

import dataclasses
import logging
import random
import tarfile
import tempfile
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _query_server(
    sequences: list[str],
    *,
    use_pairing: bool = False,
    use_env: bool = True,
    host_url: str = 'https://api.colabfold.com',
    user_agent: str = 'alphafold3/1.0',
) -> list[str]:
    """Submit protein sequences to ColabFold and return a3m strings (one per input)."""
    import io
    import requests

    host_url = host_url.rstrip('/')
    headers = {'User-Agent': user_agent} if user_agent else {}

    if use_pairing:
        mode = 'pairgreedy-env' if use_env else 'pairgreedy'
        endpoint = 'ticket/pair'
    else:
        mode = 'env' if use_env else 'all'
        endpoint = 'ticket/msa'

    # Deduplicate while preserving order; track ColabFold M-indices (start at 101)
    seen: dict[str, int] = {}
    unique_seqs: list[str] = []
    for seq in sequences:
        if seq not in seen:
            seen[seq] = 101 + len(unique_seqs)
            unique_seqs.append(seq)
    m_ids = [seen[seq] for seq in sequences]

    query = ''.join(f'>{101 + i}\n{seq}\n' for i, seq in enumerate(unique_seqs))

    def _post():
        for attempt in range(6):
            try:
                r = requests.post(
                    f'{host_url}/{endpoint}',
                    data={'q': query, 'mode': mode},
                    timeout=6.02,
                    headers=headers,
                )
                return r.json()
            except requests.exceptions.Timeout:
                logger.warning('MSA server timeout on submit, retrying...')
            except Exception as e:
                logger.warning(f'MSA server submit error ({attempt}/5): {e}')
                time.sleep(5)
        raise RuntimeError('MSA server submit failed after 5 retries')

    out = _post()
    while out.get('status') in ('UNKNOWN', 'RATELIMIT'):
        t = 5 + random.randint(0, 5)
        logger.info(f'MSA server: {out["status"]}, sleeping {t}s')
        time.sleep(t)
        out = _post()

    if out.get('status') == 'ERROR':
        raise RuntimeError('MSA server returned ERROR. Check your sequences.')
    if out.get('status') == 'MAINTENANCE':
        raise RuntimeError('MSA server is under maintenance, try later.')

    job_id = out['id']
    logger.info(f'MSA job submitted: {job_id}')

    while out.get('status') in ('UNKNOWN', 'RUNNING', 'PENDING'):
        t = 5 + random.randint(0, 5)
        logger.info(f'MSA server status: {out["status"]}, waiting {t}s...')
        time.sleep(t)
        try:
            out = requests.get(
                f'{host_url}/ticket/{job_id}', timeout=6.02, headers=headers
            ).json()
        except Exception as e:
            logger.warning(f'MSA server poll error: {e}')

    if out.get('status') != 'COMPLETE':
        raise RuntimeError(f'MSA server job ended with status: {out.get("status")}')

    logger.info('Downloading MSA results...')
    for attempt in range(6):
        try:
            r = requests.get(
                f'{host_url}/result/download/{job_id}', timeout=60, headers=headers
            )
            break
        except Exception as e:
            logger.warning(f'MSA download error ({attempt}/5): {e}')
            time.sleep(5)
    else:
        raise RuntimeError('MSA result download failed after 5 retries')

    # Extract into a temp dir and parse a3m blocks
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(r.content)) as tf:
            tf.extractall(tmp)

        if use_pairing:
            a3m_files = [tmp / 'pair.a3m']
        else:
            a3m_files = [tmp / 'uniref.a3m']
            if use_env:
                env_f = tmp / 'bfd.mgnify30.metaeuk30.smag30.a3m'
                if env_f.exists():
                    a3m_files.append(env_f)

        blocks: dict[int, list[str]] = {}
        for a3m_file in a3m_files:
            if not a3m_file.exists():
                continue
            # ColabFold separates per-query blocks with \x00 bytes.
            # Split on \x00 first so hits from one query never bleed into another.
            for raw_block in a3m_file.read_text().split('\x00'):
                raw_block = raw_block.strip()
                if not raw_block:
                    continue
                # Append \n so the last line always terminates; without this,
                # the last sequence in the block has no trailing newline and
                # the next block's header gets concatenated onto it when we
                # join blocks from multiple files.
                lines = (raw_block + '\n').splitlines(keepends=True)
                first = lines[0].strip() if lines else ''
                if not first.startswith('>'):
                    continue
                try:
                    m = int(first[1:])
                except ValueError:
                    continue
                if m not in blocks:
                    blocks[m] = list(lines)
                else:
                    # Merge hits from additional files (e.g. bfd after uniref).
                    # Skip the ">M" header (lines[0]) AND the query sequence
                    # (lines[1]) — a bare sequence line with no preceding header
                    # would be concatenated onto the previous hit by AF3's parser.
                    blocks[m].extend(lines[2:])

    return [''.join(blocks.get(m, [])) for m in m_ids]


def fill_missing_msas(
    fold_input,
    *,
    host_url: str = 'https://api.colabfold.com',
    user_agent: str = 'alphafold3/1.0',
) -> object:
    """Fill missing MSAs in a FoldInput via the ColabFold server.

    Protein chains missing `unpaired_msa` are queried in a single batch.
    RNA chains get a query-sequence-only stub (ColabFold is protein-only).
    Chains that already have an MSA set are left unchanged.

    Call `save_msas(fold_input, output_dir)` afterwards to write the a3m
    files alongside the other output files.

    Args:
        fold_input: A FoldInput object.
        host_url: ColabFold server URL.
        user_agent: HTTP User-Agent string.

    Returns:
        A new FoldInput with MSA fields populated.
    """
    from alphafold3.common import folding_input

    protein_chains = fold_input.protein_chains
    rna_chains = fold_input.rna_chains

    protein_seqs_needing_msa: list[str] = []
    for chain in protein_chains:
        if chain.unpaired_msa is None and chain.sequence not in protein_seqs_needing_msa:
            protein_seqs_needing_msa.append(chain.sequence)

    seq_to_a3m: dict[str, str] = {}
    if protein_seqs_needing_msa:
        print(
            f'Querying ColabFold MSA server for'
            f' {len(protein_seqs_needing_msa)} unique protein sequence(s)...'
        )
        a3m_results = _query_server(
            protein_seqs_needing_msa, host_url=host_url, user_agent=user_agent
        )
        seq_to_a3m = dict(zip(protein_seqs_needing_msa, a3m_results))
        print('MSA query complete.')

    # Paired MSA for multi-protein complexes
    unique_protein_seqs = list(dict.fromkeys(
        c.sequence for c in protein_chains if c.paired_msa is None
    ))
    paired_seq_to_a3m: dict[str, str] = {}
    if len(unique_protein_seqs) > 1:
        print(
            f'Querying ColabFold for paired MSA'
            f' ({len(unique_protein_seqs)} unique protein chains)...'
        )
        paired_a3ms = _query_server(
            unique_protein_seqs,
            use_pairing=True,
            host_url=host_url,
            user_agent=user_agent,
        )
        paired_seq_to_a3m = dict(zip(unique_protein_seqs, paired_a3ms))

    rna_needing_stub = [c for c in rna_chains if c.unpaired_msa is None]
    if rna_needing_stub:
        print(
            f'Using query-sequence stub for {len(rna_needing_stub)} RNA chain(s)'
            ' (ColabFold does not support RNA).'
        )

    if not protein_seqs_needing_msa and not rna_needing_stub:
        return fold_input

    new_chains = []
    for chain in fold_input.chains:
        if isinstance(chain, folding_input.ProteinChain):
            unpaired = chain.unpaired_msa
            paired = chain.paired_msa
            if unpaired is None:
                unpaired = seq_to_a3m.get(chain.sequence, f'>query\n{chain.sequence}\n')
            if paired is None:
                paired = paired_seq_to_a3m.get(chain.sequence, '')
            chain = folding_input.ProteinChain(
                id=chain.id,
                sequence=chain.sequence,
                ptms=chain.ptms,
                description=chain.description,
                unpaired_msa=unpaired,
                paired_msa=paired,
                templates=list(chain.templates) if chain.templates is not None else None,
            )
        elif isinstance(chain, folding_input.RnaChain):
            unpaired = chain.unpaired_msa
            if unpaired is None:
                unpaired = f'>query\n{chain.sequence}\n'
            chain = folding_input.RnaChain(
                id=chain.id,
                sequence=chain.sequence,
                modifications=list(chain.modifications),
                description=chain.description,
                unpaired_msa=unpaired,
            )
        new_chains.append(chain)

    return dataclasses.replace(fold_input, chains=new_chains)


def save_msas(fold_input, output_dir: str | Path) -> None:
    """Save MSA a3m files from a filled FoldInput to `output_dir/msas/`.

    Call this after `fill_missing_msas` and once the final output directory
    is known, so the files land alongside the other output files.
    """
    from alphafold3.common import folding_input

    msa_dir = Path(output_dir) / 'msas'
    msa_dir.mkdir(parents=True, exist_ok=True)
    for chain in fold_input.chains:
        if isinstance(chain, (folding_input.ProteinChain, folding_input.RnaChain)):
            if chain.unpaired_msa is not None:
                (msa_dir / f'{chain.id}_unpaired.a3m').write_text(chain.unpaired_msa)
        if isinstance(chain, folding_input.ProteinChain):
            if chain.paired_msa:
                (msa_dir / f'{chain.id}_paired.a3m').write_text(chain.paired_msa)
