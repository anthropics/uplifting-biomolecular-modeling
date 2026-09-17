"""big's diff_free at the trunk boundary: af3_kernels.release_face_workspaces empties (1) the native TriMul package's shared per-geometry
payload caches of THIS device through the package's public surface (shared_geometries / payload_cache) and (2) every module's provider-face
dict, and reports (MiB, geometries, modules); forward_impl records it on the item; the ITEM line carries diff_freed_trimul_mib."""
import sys, types, importlib, pathlib
import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve()
KIT = HERE.parents[3]                                                  # .../af3_torch
sys.path.insert(0, str(KIT / "opt" / "forward" / "af3t" / "kernels"))                  # the model process imports it as a top-level module (af3_torch_api: import af3_kernels)


def _kernels():
    return importlib.import_module("af3_kernels")


def test_release_empties_shared_payload_caches_and_face_dicts(monkeypatch):
    K = _kernels()
    dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    shared = {(dev if dev is not None else 0, 448, 128, 128): {"ab": torch.zeros(4), "x": torch.zeros(2)},
              (99, 448, 128, 128): {"ab": torch.zeros(4)}}                  # another device's geometry: untouched when a device is current
    fake = types.SimpleNamespace(shared_geometries=lambda: [(k[0], k[1], k[2], k[3], 0) for k in shared],
                                 payload_cache=lambda d, n, cz, ch: shared[(d, n, cz, ch)])
    import opt_core.kernels.trimul as TP
    monkeypatch.setattr(TP, "native", fake, raising=False)
    monkeypatch.setitem(sys.modules, "opt_core.kernels.trimul.native", fake)
    m1, m2, m3 = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    object.__setattr__(m1, "_af3k", {"trimul_face": {"_last": object(), "desc": torch.zeros(3)}})
    object.__setattr__(m2, "_af3k", {"trimul_w10": {}})                     # no face dict yet: not counted
    model = torch.nn.Sequential(m1, m2, m3)
    mib, n_geo, n_mod = K.release_face_workspaces(model)
    assert n_mod == 1 and m1._af3k["trimul_face"] == {}
    mine = (dev if dev is not None else 0, 448, 128, 128)
    assert shared[mine] == {} and n_geo >= 1
    if dev is not None:
        assert shared[(99, 448, 128, 128)] != {}                            # only the current device's geometries
    assert mib == 0.0                                                       # CPU tensors carry no device bytes (CUDA storages are what is counted)


def test_release_without_native_package_is_by_name_zero(monkeypatch):
    K = _kernels()
    monkeypatch.setitem(sys.modules, "opt_core.kernels.trimul.native", None)   # import fails -> geometries 0, modules still emptied
    m = torch.nn.Linear(2, 2); object.__setattr__(m, "_af3k", {"trimul_face": {"k": 1}})
    mib, n_geo, n_mod = K.release_face_workspaces(torch.nn.Sequential(m))
    assert (n_geo, n_mod) == (0, 1) and m._af3k["trimul_face"] == {}


def test_forward_impl_records_and_item_line_carries_the_word():
    src = (KIT / "opt" / "af3_torch_opt" / "forward_impl.py").read_text()
    assert 'item["diff_freed_trimul_mib"]' in src and "release_face_workspaces(model)" in src
    i = src.index('_census(torch, phases, "trunk", t_wall)'); j = src.index("t_sampler = _clock(torch)")
    assert i < src.index("release_face_workspaces(model)") < j              # at the trunk -> sampler boundary, under diff_free
    assert "if diff_free:" in src[i:j]
    cli = (KIT / "opt" / "af3_torch_opt" / "cli.py").read_text()
    assert '"diff_freed_trimul_mib"' in cli
