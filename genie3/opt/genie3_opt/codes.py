"""Exit codes and the line tag — the one table every entry point of the package reads (cli.py, design.py, _autoload.py, stock_cli.py,
_core.py). A leaf module with no imports: the .pth start-up line and the stock caller read it at no cost.

  0  EXIT_OK          ok
  1  EXIT_FAIL        the run failed (a process error, a forbidden driver-log line, a numerics readback off the line), or the PDB count differs
                      from the request (``incomplete: <found>/<expected>``)
  2  EXIT_USAGE       usage
  3  EXIT_NOT_ACTIVE  not active / refused before anything ran (a deployment fact — no CUDA device, no checkout, a weight file missing, the
                      shared core absent; a failed stock proof; on a kit mode a request naming a computation the kit line does not run — beam
                      search, the sidechain stage, more than one device: ``NOT ACTIVE: mode=<m> cannot serve …``; a shard index outside
                      0 <= K < M; the env route at an importer that is not ``genie3 generate``: ``NOT ACTIVE: env-route: …``; a batch the
                      driver's memory model refuses on this card; or, after the pass, a planned lever that could not run here — a mode is all
                      of its levers, so the mode refuses by name, ``levers_missing``)
"""
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3
TAG = "genie3-opt"                     # the tag of every line the package prints (``[genie3-opt] …``): report.py, _autoload.py, the core pin gate, the .pth guard
