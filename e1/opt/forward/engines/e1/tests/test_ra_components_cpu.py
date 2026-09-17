"""CPU checks of the three retrieval-augmented components against the REAL upstream objects (E1 importable on CPU; torch >= 2.5):
  multiseq.docmask   the block-statistics BlockMask == torch's create_block_mask on the stock's document_mask closure, tensor for tensor,
                     over layouts with several sequences per row, padding suffixes, lengths off the block size, all-padding tails
  kvcache            DynamicCache views: batch_repeat_interleave / batch_select_indices give the stock's VALUES, as views; the row layout
                     (cu_seqlens / max / indices) equals the stock's _get_unpad_data on the stock's k_sequence_ids
  tokenmemo          prepare_multiseq with the memo == the stock's, field for field (tensors bitwise, strings equal), incl. the errors
The GPU paths (flash / flex kernels) are not exercised here.
"""
import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")


def _ids_layout(lengths_per_row, L):
    """sequence_ids rows: sequences of the given lengths (ids 0..n-1), then -1 padding to L."""
    rows = []
    for lens in lengths_per_row:
        r = []
        for i, n in enumerate(lens):
            r += [i] * n
        assert len(r) <= L
        r += [-1] * (L - len(r))
        rows.append(r)
    return torch.tensor(rows, dtype=torch.int64)


LAYOUTS = [
    ([[300, 200, 140], [640], [128, 128, 128, 100]], 640),           # multi-seq rows, one dense row, padding suffixes
    ([[397]], 397),                                                   # one sequence, L off the block size
    ([[130, 7], [1, 136]], 200),                                      # tiny sequences straddling block borders + long padding
    ([[256, 256], [512]], 512),                                       # block-aligned boundaries (full blocks on the diagonal band)
    ([[5, 5, 5, 5, 5, 100, 3]], 129),
]


@pytest.mark.parametrize("lengths,L", LAYOUTS)
def test_docmask_equals_torch_create_block_mask(lengths, L):
    fa = pytest.importorskip("torch.nn.attention.flex_attention")
    MS = importlib.import_module("engines.e1.kits.multiseq")
    ids = _ids_layout(lengths, L)
    assert MS.ids_sorted_with_pad_suffix(ids)

    def document_mask(b, h, q_idx, kv_idx):
        return (ids[b, q_idx] >= ids[b, kv_idx]) & (ids[b, q_idx] != -1) & (ids[b, kv_idx] != -1)
    ref = fa.create_block_mask(document_mask, ids.shape[0], 1, L, L, device="cpu")     # what E1.model.flex_attention builds
    got = MS.document_block_mask(ids)
    for f in ("kv_num_blocks", "kv_indices", "full_kv_num_blocks", "full_kv_indices", "q_num_blocks", "q_indices", "full_q_num_blocks", "full_q_indices"):
        a, b = getattr(ref, f), getattr(got, f)
        assert (a is None) == (b is None), f
        if a is not None:
            assert torch.equal(a, b), f
    assert ref.BLOCK_SIZE == got.BLOCK_SIZE and tuple(ref.seq_lengths) == tuple(got.seq_lengths)
    # the mask_mod is the same predicate: evaluate both on the full grid
    b = torch.arange(ids.shape[0])[:, None, None]; q = torch.arange(L)[None, :, None]; k = torch.arange(L)[None, None, :]
    assert torch.equal(ref.mask_mod(b, 0, q, k), got.mask_mod(b, 0, q, k))


def test_docmask_refuses_unsorted_ids():
    MS = importlib.import_module("engines.e1.kits.multiseq")
    assert not MS.ids_sorted_with_pad_suffix(torch.tensor([[0, 0, 1, 0, -1]]))
    assert not MS.ids_sorted_with_pad_suffix(torch.tensor([[0, -1, 1, 1, -1]]))
    assert MS.ids_sorted_with_pad_suffix(torch.tensor([[0, 0, 1, 1, -1], [3, 3, 3, -1, -1]]))


def test_kvcache_views_give_the_stock_values():
    DC = pytest.importorskip("E1.dynamic_cache")
    KV = importlib.import_module("engines.e1.kits.kvcache")
    c_stock, c_kit = DC.DynamicCache(), DC.DynamicCache()
    k = torch.randn(1, 37, 4, 8); v = torch.randn(1, 37, 4, 8)
    for c in (c_stock, c_kit):
        c.update(k.clone(), v.clone(), 0)
        c.update(k.clone() * 2, v.clone() * 2, 2)                                        # layer 1 left empty (the stock fills skipped layers with empty tensors)
    c_stock.batch_repeat_interleave(5)
    KV._batch_repeat_interleave(c_kit, 5)
    for i in (0, 2):
        assert torch.equal(c_stock.key_cache[i], c_kit.key_cache[i]) and torch.equal(c_stock.value_cache[i], c_kit.value_cache[i])
        assert c_kit.key_cache[i].stride(0) == 0 and c_kit.key_cache[i].shape[0] == 5          # a view, no copy
        assert c_kit.get_seq_length(i) == c_stock.get_seq_length(i) == 37
    c_stock.batch_select_indices([0]); KV._batch_select_indices(c_kit, [0])
    for i in (0, 2):
        assert torch.equal(c_stock.key_cache[i], c_kit.key_cache[i]) and c_kit.key_cache[i].shape[0] == 1
    # a non-expanded tensor (the prefill's crop) takes the stock call: a copy that does not alias the big storage
    big = torch.randn(3, 50, 4, 8)
    c2 = DC.DynamicCache(); c2.update(big, big.clone(), 0); c2.crop(37)
    KV._batch_select_indices(c2, [0])
    assert c2.key_cache[0].shape == (1, 37, 4, 8) and c2.key_cache[0].untyped_storage().data_ptr() != big.untyped_storage().data_ptr()
    # batch 1: a compact row is kept as it is (the stock's copy of it has the same values); a row that is a small slice of a big
    # storage is copied as the stock copies it (the view would pin the storage)
    one = torch.randn(1, 37, 4, 8)
    c3 = DC.DynamicCache(); c3.update(one, one.clone(), 0)
    KV._batch_select_indices(c3, [0])
    assert c3.key_cache[0].data_ptr() == one.data_ptr() and torch.equal(c3.key_cache[0], one)
    huge = torch.randn(1, 4000, 4, 8)
    c4 = DC.DynamicCache(); c4.update(huge, huge.clone(), 0); c4.crop(37)
    KV._batch_select_indices(c4, [0])
    assert c4.key_cache[0].untyped_storage().data_ptr() != huge.untyped_storage().data_ptr() and torch.equal(c4.key_cache[0], huge[:, :37])


def test_kvcache_layout_equals_the_stock_unpad_data():
    U = pytest.importorskip("E1.model.flash_attention_utils")
    KV = importlib.import_module("engines.e1.kits.kvcache")
    C = 50
    for qlens, Lq in (([9, 9, 9], 9), ([9, 4, 7, 1], 9)):
        sid = torch.full((len(qlens), Lq), -1, dtype=torch.int64)
        for b, n in enumerate(qlens):
            sid[b, :n] = 7                                                           # the query's sequence id (any constant per row)
        lay = KV._Layout(sid, C)
        first = sid[:, :1]
        k_ids = torch.cat([first.expand(len(qlens), C), sid], dim=-1)               # the stock's k_sequence_ids (Attention._flash_attn)
        idx_k, cu_k, mx_k = U._get_unpad_data(k_ids)
        idx_q, cu_q, mx_q = U._get_unpad_data(sid)
        assert torch.equal(cu_k.to(torch.int64), lay.cu_k.to(torch.int64)) and torch.equal(cu_q.to(torch.int64), lay.cu_q.to(torch.int64))
        assert lay.cu_k.dtype == torch.int32 == cu_k.dtype
        assert int(mx_k) == C + lay.max_q and int(mx_q) == lay.max_q
        assert lay.ident == all(n == Lq for n in qlens)
        if not lay.ident:
            assert torch.equal(idx_q, lay.idx_q)
        # the packed K the stock gathers == the direct write
        B = len(qlens); nkv, hd = 2, 4
        ck = torch.randn(1, C, nkv, hd); newk = torch.randn(B, Lq, nkv, hd)
        Kf = torch.cat([ck.repeat_interleave(B, dim=0), newk], dim=1)
        packed_stock = U.index_first_axis(Kf.reshape(B * (C + Lq), nkv, hd), idx_k)
        Kp = torch.empty(B * C + lay.total_q, nkv, hd)
        s = 0
        for b in range(B):
            Kp[s:s + C] = ck[0]; Kp[s + C:s + C + qlens[b]] = newk[b, :qlens[b]]; s += C + qlens[b]
        assert torch.equal(packed_stock, Kp)


def _rows():
    ctx = "MKTAYIAKQR,MKTAYLAKQRG,MKSAYIAK"
    return ["MKTAYIAKQRQISFVK", ctx + ",MKTA?IAKQR", ctx + ",MKTAYIA?QR", "MKTAYLAKQRG,AKQ?", ctx + ",MKTAYIAKQ?", "A,C?D", ctx + ",MKTA?IAKQR"]


@pytest.mark.parametrize("preserve", [False, True])
def test_tokenmemo_equals_the_stock_prepare_multiseq(preserve):
    BP = pytest.importorskip("E1.batch_preparer")
    TM = importlib.import_module("engines.e1.kits.tokenmemo")
    stock = BP.E1BatchPreparer(preserve_context_labels=preserve, device=torch.device("cpu"))
    TM.install()
    try:
        kit = BP.E1BatchPreparer(preserve_context_labels=preserve, device=torch.device("cpu"))
        assert BP.E1BatchPreparer.prepare_multiseq is TM._prepare_multiseq_memo
        for row in _rows() * 2:                                                          # second pass: all hits
            a = TM._R["orig"](stock, row)
            b = kit.prepare_multiseq(row)
            assert set(a) == set(b)
            for key in a:
                if torch.is_tensor(a[key]):
                    assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key]), (row, key)
                else:
                    assert a[key] == b[key], (row, key)
        c = TM.counters()
        assert c["rows_stock"] == 2 and c["ctx_hit"] > 0 and c["query_hit"] > 0 and c["rows_memo"] == 2 * (len(_rows()) - 1)
        # the batch path (pad_encodings) is identical too
        A = BP.E1BatchPreparer.get_batch_kwargs
        ka = A(kit, _rows()); TM.uninstall(); sa = A(stock, _rows())
        for key in sa:
            if torch.is_tensor(sa[key]):
                assert torch.equal(sa[key], ka[key]), key
            else:
                assert sa[key] == ka[key], key
    finally:
        TM.uninstall()
    assert TM.is_pristine()["prepare_multiseq"]


def test_tokenmemo_raises_what_the_stock_raises():
    BP = pytest.importorskip("E1.batch_preparer")
    TM = importlib.import_module("engines.e1.kits.tokenmemo")
    p = BP.E1BatchPreparer(device=torch.device("cpu"))
    TM.install()
    try:
        for bad in ("MKTA,mkta?", "MKT1A,MKTA", "MKTA,MK TA"):
            with pytest.raises(ValueError) as e_kit:
                p.prepare_multiseq(bad)
            with pytest.raises(ValueError) as e_stock:
                TM._R["orig"](p, bad)
            assert str(e_kit.value) == str(e_stock.value)
        p.data_prep_config.max_num_sequences = 2
        with pytest.raises(ValueError) as e_kit:
            p.prepare_multiseq("A,C,D")
        with pytest.raises(ValueError) as e_stock:
            TM._R["orig"](p, "A,C,D")
        assert str(e_kit.value) == str(e_stock.value)
    finally:
        TM.uninstall()
