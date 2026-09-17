#!/bin/bash
# mps_workers.sh — runs K copies of ONE serving-worker command on ONE GPU under uncapped CUDA MPS: one MPS control daemon per launch with fresh pipe / log
# directories, the K workers in the background with one log file each, every worker's exit code relayed, and an out-of-memory scan of those logs at the end. Apache-2.0.
# Usage:  K=<workers> WORKER_GB=<per-worker GB> bash mps_workers.sh '<worker command; every {W} in it is replaced by the worker index 0..K-1>'
#   e.g.  K=3 WORKER_GB=10 LOG_DIR=out bash mps_workers.sh 'python worker.py --out out/w{W} --shard {W}'   (illustrative command: each worker needs its own output location,
#         and identical requests produce identical outputs in every worker — give each worker a different seed or request slice). Optional: HEADROOM_GB, LOG_DIR (default .), MPS_DIR_ROOT, TMPDIR.
# The MPS geometry is fixed; only under it do a worker's outputs stay byte-identical to the same command run alone: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is unset (thread-capped
# clients do not reproduce the solo run's bytes), CUDA_MPS_PINNED_DEVICE_MEM_LIMIT and PYTORCH_CUDA_ALLOC_CONF are unset (no per-client memory cap; the stock caching allocator),
# the GPU is expected in compute mode Default, the daemon's pipe / log directories are created fresh under MPS_DIR_ROOT (else a private mktemp directory) and exported BEFORE
# the workers initialise CUDA, and the daemon is sent `quit` in teardown. Worker W's stdout and stderr go to $LOG_DIR/mps_worker_<W>.log.
# Memory estimate: WORKER_GB is the per-worker peak device-memory footprint in GB (the caller supplies it); when K x WORKER_GB + HEADROOM_GB (default 5 % of the card's total)
# exceeds the card's total memory the launch prints a NOTE and proceeds — the estimate never refuses a launch. A launch that does run out of memory is caught by the OOM scan
# (co-tenants that survive at the memory edge can silently change numerics, so that launch's outputs are not exact-class). Exit codes: 4 nvidia-cuda-mps-control missing, 5 the MPS directory or daemon could not be set up, 6 OOM text in a worker log, else the last non-zero worker exit code (0 when every worker succeeded).
set -uo pipefail
K=${K:?number of workers}; CMD=${1:?worker command template}; WORKER_GB=${WORKER_GB:?per-worker device footprint GB}; HEADROOM_GB=${HEADROOM_GB:-}
TOTAL_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%.1f", $1/1024}')
[ -z "$HEADROOM_GB" ] && HEADROOM_GB=$(awk -v t="$TOTAL_GB" 'BEGIN{printf "%.1f", 0.05*t}')   # default head-room: 5 % of the card's total memory kept free at the workers' peak
NEED=$(awk -v k="$K" -v w="$WORKER_GB" -v h="$HEADROOM_GB" 'BEGIN{printf "%.1f", k*w+h}')
if awk -v n="$NEED" -v t="$TOTAL_GB" 'BEGIN{exit !(n>t)}'; then echo "[mps_workers] NOTE: K=$K x ${WORKER_GB} GB + ${HEADROOM_GB} GB headroom = ${NEED} GB > card ${TOTAL_GB} GB — requested worker footprint exceeds the card by estimate; proceeding — may OOM (memory-edge identity hazard: the OOM scan below marks a launch that ran out of memory)"; fi
command -v nvidia-cuda-mps-control > /dev/null || { echo "[mps_workers] nvidia-cuda-mps-control not found: cannot start the workers under the uncapped CUDA MPS geometry this launcher uses (plain time-slicing gives no throughput gain; the geometry and the memory-estimate NOTE are described in this file's header)"; exit 4; }
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_PINNED_DEVICE_MEM_LIMIT PYTORCH_CUDA_ALLOC_CONF
R=${MPS_DIR_ROOT:-}; [ -n "$R" ] || R=$(mktemp -d "${TMPDIR:-/tmp}/mps_XXXXXX") || { echo "[mps_workers] cannot create a private MPS directory under ${TMPDIR:-/tmp}"; exit 5; }   # no MPS_DIR_ROOT given (run by hand): a private root, mode 0700
D=$R/run_$(date +%s)_$$; mkdir -p "$D/pipe" "$D/log"
export CUDA_MPS_PIPE_DIRECTORY=$D/pipe CUDA_MPS_LOG_DIRECTORY=$D/log
nvidia-cuda-mps-control -d || { echo "[mps_workers] MPS control daemon failed to start"; exit 5; }
sleep 1; echo "[mps_workers] MPS daemon up (pipe $D/pipe); launching K=$K workers; card ${TOTAL_GB} GB, budget ${NEED} GB"
pids=(); logs=(); t0=$(date +%s)
for W in $(seq 0 $((K - 1))); do
  c=${CMD//\{W\}/$W}; log="${LOG_DIR:-.}/mps_worker_$W.log"; echo "[mps_workers] worker $W: $c"
  bash -c "$c" > "$log" 2>&1 & pids+=($!); logs+=("$log")
done
rc_all=0
for i in "${!pids[@]}"; do wait "${pids[$i]}"; rc=$?; echo "[mps_workers] worker $i rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc; done
echo "[mps_workers] all workers done in $(( $(date +%s) - t0 )) s; stopping MPS daemon"
echo quit | timeout 30 nvidia-cuda-mps-control > /dev/null 2>&1 || true
if grep -l -i -E "out of memory|OutOfMemoryError" "${logs[@]}" > /dev/null 2>&1; then echo "[mps_workers] WARNING: OOM text found in a worker log -> outputs of THIS launch must not be treated as exact-class (memory-edge hazard); reduce K"; rc_all=6; fi   # the K logs THIS launch wrote: an earlier, larger launch's mps_worker_<W>.log left in the same directory is not read
exit $rc_all
