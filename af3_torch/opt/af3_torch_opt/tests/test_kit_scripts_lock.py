"""Static locks on the model process script the package launches: forward.py's sys.path recipe (the one line it cannot import from the
package under the torch venv) equals stack.kit_sys_path, and its DTK swap keeps its three statements (the DiffusionTransformer.forward
rebinding, the whole-step graph drop, the synchronize)."""
import os

from af3_torch_opt import stack
from af3_torch_opt import forward as fwd

OPT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORWARD = os.path.join(OPT, "forward.py")


def test_forward_sys_path_recipe_equals_the_packages():
    assert fwd.kit_sys_path("/k", "/d") == stack.kit_sys_path("/k", "/d")


def test_dtk_swap_statements():
    ours = open(FORWARD, encoding="utf-8").read()
    for stmt in ('DT.DiffusionTransformer.forward = lambda self, act, mask, single_cond, pair_cond, pair_logits=None: dtk.forward(act, mask, single_cond, pair_cond, pair_logits)',
                 'st.pop("graph", None)', 'torch.cuda.synchronize()'):
        assert stmt in ours, stmt
