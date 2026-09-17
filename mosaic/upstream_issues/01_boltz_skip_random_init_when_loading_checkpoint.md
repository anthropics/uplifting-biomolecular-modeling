# boltz skip random init when loading checkpoint

See README item 1. Repro: `python -c "import cProfile; cProfile.run('from mosaic.losses.boltz2 import load_boltz2; load_boltz2()', sort='cumtime')"` (472 calls to boltz/model/layers/initialize.py:trunc_normal_init_ = 21.1 s of 23.7 s construction). The kit's form of the change: `mosaic.fast.fastload.no_random_init()`, a context manager around model construction (the `exact` mode's P2).


## PR text (draft)
Title: Skip random parameter initialisation when constructing Boltz2 for checkpoint loading
`Boltz2.__init__` runs `initialize.trunc_normal_init_` (scipy.stats.truncnorm) and other init fills for every Linear; when the module is built by `load_from_checkpoint` these values are immediately overwritten by `load_state_dict(strict=True)`. On a 24-core host this costs ~21 s of the ~37-53 s model load (cProfile: 472 calls to boltz/model/layers/initialize.py:trunc_normal_init_ = 21.1 s). Proposal: guard the initialisers with a module-level flag / contextmanager (`boltz.model.layers.initialize.SKIP_INIT`) that `load_from_checkpoint` sets, or construct under `torch.nn.utils.skip_init` semantics. Correctness: parameters after load are bit-identical (sha256 over all tensors). A monkeypatch implementation is in mosaic_fast/fastload.py::no_random_init().
