"""0.1.7: pae_stream_m (AtlasFold-M per-sample PAE/PDE reduction) and conf_transition_chunk (row-blocked confidence-head pair Transition)
reproduce the stock values on CPU (torch.equal; NaN-aware for interface_iptm's empty diagonal)."""
import importlib, os
import torch


def _eq(a, b):
    if a.is_floating_point():
        return torch.equal(torch.nan_to_num(a, nan=-7.0), torch.nan_to_num(b, nan=-7.0))
    return torch.equal(a, b)


def _multimer_case(L=22, N=3, B=1, asym=None):
    from atlasfold.model.network.confidence_head import ConfidenceHead_Multimer
    torch.manual_seed(0)
    head = ConfidenceHead_Multimer(channel_s=384, channel_z=128, num_blocks=1).eval()
    for p in head.parameters():
        torch.nn.init.normal_(p, std=0.05)
    asym = torch.tensor([asym or ([0] * 9 + [1] * 8 + [2] * 5)] * B)
    batch = {"aatype": torch.nn.functional.one_hot(torch.randint(0, 21, (B, L)), 21).float(), "seq_mask": torch.ones(B, L, dtype=torch.bool),
             "pseudo_beta": torch.randint(0, 5, (B, L)), "asym_id": asym}
    batch["seq_mask"][:, -2:] = False
    s = torch.randn(B, L, 384); z = torch.randn(B, L, L, 128); x = torch.randn(B, N, L, 14, 3) * 6
    return head, batch, s, z, x


def _consume(out, batch, CM):
    """== model_multimer.inference L393-425 (the names are looked up on the module, as the model does)."""
    mask = batch["seq_mask"].unsqueeze(1)
    r = {"plddt": CM.compute_plddt(**out["plddt"], mask=mask), "pde": CM.compute_pde(**out["pde"], mask=mask)}
    pae_logits = out["pae"]["logits"]; centers = out["pae"]["bin_centers"]
    probs = torch.softmax(pae_logits, dim=-1)
    r["pae"] = CM.compute_pae_from_probs(probs, centers, mask)
    r["ptm"] = CM.compute_ptm_from_probs(probs, centers, mask)
    r["iptm"] = CM.compute_iptm_from_probs(probs, centers, batch["asym_id"], mask)
    r["chain_ptm"], r["interface_iptm"] = CM.compute_chain_tm_scores_from_probs(probs, centers, batch["asym_id"], mask)
    r["exp"] = out["experimentally_resolved"]["logits"]
    return r


def test_pae_stream_m_equals_stock_cpu():
    from atlasfold.model.network import confidence_head as CH
    from atlasfold.model.utils import confidence_metrics as CM
    head, batch, s, z, x = _multimer_case()
    with torch.no_grad():
        ref = _consume(head(batch, s, z, x, "torch"), batch, CM)
        from atlasfold_opt.hooks import pae_multimer as P
        ins = P.install("exact", "atlasfold-opt", {})
        try:
            assert ins.applied
            out = head(batch, s, z, x, "torch")
            assert type(out["pae"]["logits"]).__name__ == "StreamedM"          # no [B,N,L,L,64] stack was built
            got = _consume(out, batch, CM)
            line = ins.lines[0]()
        finally:
            ins.facts["restore"]()
    assert "served=1" in line, line
    for k in ref:
        assert _eq(ref[k], got[k]), (k, float((torch.nan_to_num(ref[k]) - torch.nan_to_num(got[k])).abs().max()))
    assert ref["interface_iptm"].shape == got["interface_iptm"].shape == (1, 3, 3, 3) and ref["chain_ptm"].shape == (1, 3, 3)


def test_pae_stream_m_single_sample_and_stock_restored():
    from atlasfold.model.network import confidence_head as CH
    from atlasfold.model.utils import confidence_metrics as CM
    head, batch, s, z, x = _multimer_case(L=22, N=1)
    with torch.no_grad():
        ref = _consume(head(batch, s, z, x, "torch"), batch, CM)
        from atlasfold_opt.hooks import pae_multimer as P
        ins = P.install("exact", "atlasfold-opt", {})
        try:
            got = _consume(head(batch, s, z, x, "torch"), batch, CM)
        finally:
            ins.facts["restore"]()
        assert CH.ConfidenceHead_Multimer.forward is not None and not hasattr(CH.ConfidenceHead_Multimer.forward, "__wrapped_stock__")
        again = _consume(head(batch, s, z, x, "torch"), batch, CM)                 # stock after restore
    for k in ref:
        assert _eq(ref[k], got[k]) and _eq(ref[k], again[k]), k


def test_conf_transition_chunk_equals_unchunked_cpu():
    """Row-blocked Transition == the stock call on CPU.  Two comparisons: (a) whole-tensor stock vs row-blocked, computed with ONE intra-op
    thread (torch.set_num_threads(1), restored after) — with several MKL/OpenMP threads a CPU GEMM may split its reduction differently for
    different M, so 'row-blocked == whole' is not a multi-threaded CPU-BLAS invariant (an 8-thread box showed max|d| 5.8e-11); (b) block vs
    block at any thread count: the stock module applied to the same row block equals the lever's block.  The BITWISE CLAIM OF THIS LEVER IS
    CARRIED BY THE GPU LEG (CHANGES.md 0.1.7/0.1.8: exact vs stock --det 1 byte-identical with conf_transition_chunk served), not by CPU BLAS."""
    from atlasfold.model.network.primitives.transition import Transition
    os.environ["AFO_TRANSITION_CHUNK_MIB"] = "0"                                  # every pair input is row-blocked, one row per block
    from atlasfold_opt.hooks import transition_chunk as T
    torch.manual_seed(1)
    tr = Transition(128, 2).eval()
    for p in tr.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 19, 19, 128)
    threads = torch.get_num_threads()
    with torch.no_grad():
        ins = T.install("exact", "atlasfold-opt", {})
        try:
            assert ins.applied
            # (b) block vs block, at the box's thread count: stock Sequential on row block i == the lever's output rows i
            T._IN_HEAD.depth = 1
            got_mt = tr(x)
            T._IN_HEAD.depth = 0
            stock_tr = T.__dict__.get("stock_tr") or type(tr).forward.__wrapped_stock__
            for i in range(x.shape[1]):
                blk = stock_tr(tr, x[:, i:i + 1])
                assert torch.equal(blk, got_mt[:, i:i + 1]), (i, float((blk - got_mt[:, i:i + 1]).abs().max()))
            # (a) whole vs row-blocked with one intra-op thread
            torch.set_num_threads(1)
            outside = tr(x)                                                     # not under a confidence head: stock path, counted
            ref = stock_tr(tr, x)
            T._IN_HEAD.depth = 1
            got = tr(x)
            T._IN_HEAD.depth = 0
            line = ins.lines[0](); ok = ins.gates[0]().ok
        finally:
            torch.set_num_threads(threads)
            T._IN_HEAD.depth = 0
            ins.facts["restore"](); os.environ.pop("AFO_TRANSITION_CHUNK_MIB", None)
    assert torch.equal(ref, outside)
    assert torch.equal(ref, got), float((ref - got).abs().max())
    assert "served=2" in line and "chunks=19" in line and "outside_conf_head" in line and ok, line


def test_conf_transition_chunk_rows_arithmetic():
    from atlasfold_opt.hooks.transition_chunk import chunk_rows
    # L=3,072, c_z 128 -> SwiGLU Linear out 512 channels fp32: 512 MiB blocks -> 85 rows per block, 37 blocks; whole tensor would be 18 GiB
    assert chunk_rows(3072, 1, 512, 4, 512 << 20) == 85
    assert -(-3072 // 85) == 37
    assert chunk_rows(2048, 1, 512, 4, 512 << 20) == 128 and chunk_rows(100, 1, 512, 4, 512 << 20) == 100


def test_pae_stream_m_slot_blocks_five_samples_odd_chains_cpu():
    from atlasfold.model.utils import confidence_metrics as CM
    head, batch, s, z, x = _multimer_case(L=24, N=5, asym=[0] * 7 + [1] * 11 + [2] * 6)
    with torch.no_grad():
        ref = _consume(head(batch, s, z, x, "torch"), batch, CM)
        from atlasfold_opt.hooks import pae_multimer as P
        ins = P.install("exact", "atlasfold-opt", {})
        try:
            got = _consume(head(batch, s, z, x, "torch"), batch, CM)
        finally:
            ins.facts["restore"]()
    for k in ref:
        assert _eq(ref[k], got[k]), k
    assert got["chain_ptm"].shape == (1, 5, 3) and got["interface_iptm"].shape == (1, 5, 3, 3)


def test_pair_transition_chunk_equals_stock_cpu():
    """0.1.9 pair_transition_chunk: a pair Transition OUTSIDE the confidence heads (trunk / diffusion conditioning scope) in row blocks equals the
    stock call — whole-vs-blocked under one intra-op thread, block-vs-block at the box's thread count (see the conf test's note: the bitwise claim on
    the card is carried by the GPU leg); the confidence lever's accounting is unchanged and AFO_PAIR_TRANSITION_CHUNK=0 restores the stock path."""
    from atlasfold.model.network.primitives.transition import Transition
    os.environ["AFO_TRANSITION_CHUNK_MIB"] = "0"
    from atlasfold_opt.hooks import transition_chunk as T
    torch.manual_seed(2)
    tr = Transition(128, 4).eval()                                             # the trunk's pair_transition_factor
    for p in tr.parameters():
        torch.nn.init.normal_(p, std=0.05)
    x = torch.randn(1, 17, 17, 128)
    threads = torch.get_num_threads()
    with torch.no_grad():
        ins_c = T.install("exact", "atlasfold-opt", {}); ins_p = T.install_pair("exact", "atlasfold-opt", {})
        try:
            assert ins_c.applied and ins_p.applied
            stock_tr = T._STATE["stock"]
            got_mt = tr(x)                                                      # outside a head: the pair lever serves
            for i in range(x.shape[1]):
                assert torch.equal(stock_tr(tr, x[:, i:i + 1]), got_mt[:, i:i + 1]), i
            torch.set_num_threads(1)
            ref = stock_tr(tr, x); got = tr(x)
            os.environ["AFO_PAIR_TRANSITION_CHUNK"] = "0"
            off = tr(x)                                                         # switch off: stock path, counted as disabled / outside_conf_head
            os.environ.pop("AFO_PAIR_TRANSITION_CHUNK")
            line_p = ins_p.lines[0](); line_c = ins_c.lines[0](); ok = ins_p.gates[0]().ok
        finally:
            torch.set_num_threads(threads)
            ins_p.facts["restore"](); ins_c.facts["restore"](); os.environ.pop("AFO_TRANSITION_CHUNK_MIB", None); os.environ.pop("AFO_PAIR_TRANSITION_CHUNK", None)
    assert not hasattr(Transition.forward, "__wrapped_stock__")                # both restored -> stock forward back
    assert torch.equal(ref, got) and torch.equal(ref, off), (float((ref - got).abs().max()), float((ref - off).abs().max()))
    assert "served=2" in line_p and "chunks=17" in line_p and "disabled:1" in line_p and ok, line_p
    assert "outside_conf_head:1" in line_c and "served=0" in line_c, line_c


def test_pair_transition_chunk_rows_arithmetic_trunk():
    from atlasfold_opt.hooks.transition_chunk import chunk_rows
    # trunk at L=4,096, bf16 autocast, factor 4 -> SwiGLU Linear out 1,024 channels x 2 B: 512 MiB blocks -> 64 rows x 64 blocks (whole: 32 GiB)
    assert chunk_rows(4096, 1, 1024, 2, 512 << 20) == 64 and -(-4096 // 64) == 64
    assert chunk_rows(2048, 1, 1024, 2, 512 << 20) == 128 and chunk_rows(3072, 1, 1024, 2, 512 << 20) == 85
    # diffusion pair conditioning at 4,096, fp32, factor 2 -> 512 channels x 4 B: 64 rows
    assert chunk_rows(4096, 1, 512, 4, 512 << 20) == 64


def test_pair_transition_chunk_off_switch_installs_and_counts_disabled():
    """AFO_PAIR_TRANSITION_CHUNK=0 must not refuse the mode: the lever installs, every call runs stock and is counted `disabled` (gate ok)."""
    from atlasfold.model.network.primitives.transition import Transition
    os.environ["AFO_TRANSITION_CHUNK_MIB"] = "0"; os.environ["AFO_PAIR_TRANSITION_CHUNK"] = "0"
    from atlasfold_opt.hooks import transition_chunk as T
    torch.manual_seed(3); tr = Transition(128, 4).eval(); x = torch.randn(1, 9, 9, 128)
    with torch.no_grad():
        ins = T.install_pair("exact", "atlasfold-opt", {})
        try:
            assert ins.applied
            ref = T._STATE["stock"](tr, x); got = tr(x)
            line = ins.lines[0](); ok = ins.gates[0]().ok
        finally:
            ins.facts["restore"](); os.environ.pop("AFO_TRANSITION_CHUNK_MIB", None); os.environ.pop("AFO_PAIR_TRANSITION_CHUNK", None)
    assert torch.equal(ref, got) and "served=0" in line and "disabled:1" in line and ok, line
