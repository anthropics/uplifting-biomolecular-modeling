"""stand-in colabdesign.af.model (tests only): the tests' recording design surface (colabdesign_opt.tests.recipe DesignSurface / ModelSurface)
installed on the stand-in `_af_design` — where ColabDesign defines those methods — so the observer (units.py) and the nosub lever's `_prep_model`
replacement apply to the same classes as on the real stack. STUB_EXAMPLE_PDB (the structure save_pdb slices), STUB_PLDDT (the mean pLDDT
BindCraft's gates read), STUB_CALLS_JSON (dump every model's call transcript there at exit) are the tests' knobs."""
import atexit
import json
import os

from colabdesign.af.design import _af_design
from colabdesign.af.prep import _af_prep
from colabdesign_opt.tests.recipe import DesignSurface, ModelSurface, install_design_methods

CALLS_JSON = os.environ.get("STUB_CALLS_JSON")
_ALL_CALLS = []                                            # one transcript list per model instance (the lists, not the instances: nothing keeps a model alive)

install_design_methods(_af_design)
_af_design.EXAMPLE_PDB = os.environ.get("STUB_EXAMPLE_PDB")                     # the tests' knobs, on the class the methods live on
_af_design.PLDDT = float(os.environ.get("STUB_PLDDT", DesignSurface.PLDDT))


class mk_af_model(ModelSurface, _af_prep, _af_design):
    def __init__(self, *args, **kwargs):
        ModelSurface.__init__(self, *args, **kwargs)
        _ALL_CALLS.append(self.calls)
        import jax
        for program in ("fn", "grad_fn"):                     # where ColabDesign's _get_model jit-compiles its two executables: the cache events a real compile raises
            jax._stub_compile(program)



def _dump_calls():
    if CALLS_JSON and _ALL_CALLS:
        with open(CALLS_JSON, "w") as fh:
            json.dump(_ALL_CALLS, fh, default=str)


atexit.register(_dump_calls)
