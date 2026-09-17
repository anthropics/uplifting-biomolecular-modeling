"""The multi-GPU line's rank launcher command: torchrun with an EXPLICIT c10d rendezvous on the loopback address. The carried unit's
launcher (``ptx_tp.launch``) builds its torchrun argv through ``ptx_tp.launch_core.torchrun_cmd``; ``tp_route.install`` rebinds that
name to :func:`torchrun_cmd` here in the launcher process, so the rendezvous store, the address the elastic agent advertises and the
address every rank connects to are all ``127.0.0.1`` — the node's hostname is never resolved (a box without DNS or an ``/etc/hosts``
entry for its hostname, e.g. a network-blocked container, starts the ranks all the same). One node, P ranks, no restarts; nothing else
about the unit's launcher changes. Named ``launcher=torchrun_loopback`` on the ACTIVE line.

    torchrun_cmd(nproc, rank_args, python=None, extra=None, port=None, run_id=None) -> argv
        P > 1:  python -m torch.distributed.run --nnodes=1 --nproc_per_node=P --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:<port>
                --rdzv-id=<run_id> --local-addr=127.0.0.1 --max-restarts=0 [extra] -m ptx_tp.launch_core --rank-side <rank_args>
                (<port>: a free loopback port of this box unless given; <run_id>: a fresh id unless given)
        P <= 1: python -m ptx_tp.launch_core --rank-side <rank_args>   (no torchrun)
"""
from __future__ import annotations

import sys
from typing import List, Optional

__all__ = ["torchrun_cmd", "free_port", "LOCAL_ADDR", "RANK_MODULE", "NAME", "EVIDENCE"]

LOCAL_ADDR = "127.0.0.1"                      # the rendezvous endpoint host AND torchrun --local-addr (the address the agent advertises to the ranks; else the node's FQDN)
RANK_MODULE = "ptx_tp.launch_core"            # the unit's rank-side entry (process group init, hooks, the stock CLI entry)
NAME = "torchrun_loopback"                    # the launcher= word of the ACTIVE line (tp.LAUNCHER)
EVIDENCE = f"TP-LAUNCH {NAME} rdzv=c10d@{LOCAL_ADDR} local_addr={LOCAL_ADDR}"   # the launcher process prints '[protenix-opt] <EVIDENCE> port=<port>' once


def free_port(host: str = LOCAL_ADDR) -> int:
    """A currently free TCP port on ``host`` (bind to port 0, read it back, release it)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
        sk.bind((host, 0))
        return int(sk.getsockname()[1])


def torchrun_cmd(nproc: int, rank_args: List[str], python: Optional[str] = None, extra: Optional[List[str]] = None,
                 port: Optional[int] = None, run_id: Optional[str] = None) -> List[str]:
    python = python or sys.executable
    if int(nproc) <= 1:
        return [python, "-m", RANK_MODULE, "--rank-side"] + list(rank_args)
    import uuid
    port = int(port or free_port())
    run_id = run_id or f"ptx2-{uuid.uuid4().hex[:12]}"
    sys.stderr.write(f"[protenix-opt] {EVIDENCE} port={port} run_id={run_id} nproc={int(nproc)}\n"); sys.stderr.flush()
    return ([python, "-m", "torch.distributed.run", "--nnodes=1", f"--nproc_per_node={int(nproc)}", "--rdzv-backend=c10d",
             f"--rdzv-endpoint={LOCAL_ADDR}:{port}", f"--rdzv-id={run_id}", f"--local-addr={LOCAL_ADDR}", "--max-restarts=0"]
            + list(extra or []) + ["-m", RANK_MODULE, "--rank-side"] + list(rank_args))
