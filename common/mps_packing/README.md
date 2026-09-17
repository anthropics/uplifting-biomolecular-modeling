# common/mps_packing — the CUDA MPS packing launcher

`mps_workers.sh` runs K copies of one serving-worker command on one GPU under uncapped CUDA MPS: it starts a fresh MPS
control daemon (pipe directory under `$MPS_DIR_ROOT/run_<ts>_<pid>/`), prints a memory NOTE when `K x WORKER_GB +
HEADROOM_GB` exceeds the card (an estimate, never a gate), launches the K workers (`{W}` in the command = the worker
index), relays every worker's exit code (`[mps_workers] worker <W> rc=<rc>`), stops the daemon, and scans the per-worker
logs `$LOG_DIR/mps_worker_<W>.log` for out-of-memory text (rc 6). Exit codes: 4 = `nvidia-cuda-mps-control` not on PATH,
5 = the daemon failed to start, 6 = OOM text in a worker log, else the last non-zero worker rc (0 when all succeeded).

    K=3 WORKER_GB=<per-worker GB> [HEADROOM_GB=4] [LOG_DIR=<dir>] [MPS_DIR_ROOT=<dir>] bash mps_workers.sh '<worker command with {W}>'

Kits launch their served lines through this file by path (`<release tree>/common/mps_packing/mps_workers.sh`;
rfdiffusion1 `run.sh design --pack K`). It needs `bash`, `nvidia-smi` and `nvidia-cuda-mps-control` on PATH.
