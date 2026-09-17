# opt/kit_ho — the kit's entry script and host-side package

`tf/pred_bw_fast.py -cm <model.h5> -r <regions.bed> -g <genome.fa> -c <chrom.sizes> -op <prefix> [-os <stats>] [-bw <observed.bigWig>] [--items FILE]`
takes the stock `chrombpnet pred_bw` arguments and writes the same output files; the package's `exact` and `fast` modes run exactly this
line as a child process (`opt/chrombpnet_opt/stack.py` `kit_home`) and print one `[chrombpnet-opt]` mode line before it. No kit environment
variable is part of the interface. The torch side (`opt/kit`) must sit beside this directory: the entry script reads the Triton forward, its
caches and tables from there (`_BASE_KIT`).

Contents of `tf/`:
- `pred_bw_fast.py` — the job: route resolution, the model(s), the pipelined forward with featurisation prefetch, the writer and PNG helper
  processes spawned at start (fresh interpreters, before any CUDA context), the `--items` loop, the run record handed back to the package.
- `chrombpnet_fastkit/` — the host side: class tables and route resolution (`fastdefault.py`), the numpy bigWig writer and reader
  (`bigwig_numpy.py`, `bigwig_reader.py`), the HDF5 writer (`h5_fast.py`), the `-bw` finish stage as three concurrent tasks
  (`metrics_overlap.py`), the PNG helpers (`png_workers.py`), CPU sizing from the cgroup (`hostres.py`), allocator tunables (`malloc_env.py`),
  the out-of-memory classifier (`_oom.py`).
- `det_subprocess/sitecustomize.py` — the seed hook behind TensorFlow's determinism settings (`--mode off --det 1`, `--mode exact`): first on
  `PYTHONPATH`, it seeds TensorFlow, enables op determinism and turns TF32 off at interpreter start when `CHROMBPNET_DET_SUBPROCESS=1`.
- `test_*_cpu.py` — CPU tests beside the code (e.g. the spawned helpers' PNG bytes against the serial calls, the writer's bigWig + stats bytes
  across two items in sequence); the package's `opt/chrombpnet_opt/tests/test_fast_kit.py` and `test_multi.py` cover the same surface from outside.

The mode line's grammar: `[chrombpnet-opt] ACTIVE mode=<exact|fast> route=<k1|tf_function|keras_predict_fileorder> kit=<version> gpu=<class> det=<0|1> precision=<fp32|tf32>[ multi=<N>]`;
under `--items` each item also prints `[pred_bw_fast] multi: item <i>/<n> <prefix> wall=<s>s rc=0` after `[pred_bw_fast] multi: ready load=<s>s items=<n> route=<route>`.
