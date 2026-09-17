# esmfold2 kit — prebuilt sm_90a kernels (driver/prebuilt/sm_90a/)

No kit module ships a cubin in this tree: `manifest.json` records an empty `"cubins"` set, and the pair TriMul and the transitions are served by
the shared core's provider rows under the modes' tier words. `driver/ef2_nvjit.py cubins` (run by `run.sh install`) checks this directory against
the tree's kernel specs (`ef2_nvjit.PREBUILD_MODULES`, empty); `driver/prebuilt/build_prebuilt.py` is the build route that writes `<stem>.cubin`,
the manifest entry, the `SHA256SUMS` line and this file for a driver module that declares `nvrtc_sources()`; the loader
(`ef2_nvjit.shipped_cubin`) re-hashes a shipped cubin against its `SHA256SUMS` line (`opt_core.gates.binary_refusal`) and refuses by name one
that is absent from it or altered — `SHA256SUMS` here lists none.
