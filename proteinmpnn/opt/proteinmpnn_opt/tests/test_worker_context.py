"""The exact worker's decode loop (opt/forward/mpnn_exact_worker/addon/mpnn_worker2.py, read as source: the worker imports torch, the tests
do not) assembles the decoder's encoder context for the ONE position a step decodes and never materialises upstream's all-positions
``h_EXV_encoder_fw`` / ``h_EX_encoder`` / ``h_EXV_encoder`` tensors ([N, L, K, 2H..3H]: at 16 backbones x 8 sequences and L = 500 they were
12.5 GiB per process). The stock statements that turn the decoding order into the attention masks live in ``Worker.order_masks``; a batch's
encoder outputs are released once its scores are written."""
import ast
import os
import re
import unittest

from proteinmpnn_opt import modes, stack

WORKER = os.path.join(stack.kit_home(), modes.WORKER_DIR, "addon", "mpnn_worker2.py")


def _methods():
    src = open(WORKER, encoding="utf-8").read()
    tree = ast.parse(src)
    worker, = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Worker"]
    def code(n):                                                             # the method's statements: comment lines dropped
        return "\n".join(l for l in ast.get_source_segment(src, n).split("\n") if not l.lstrip().startswith("#"))
    return {n.name: code(n) for n in worker.body if isinstance(n, ast.FunctionDef)}, src


class TestDecodeContext(unittest.TestCase):
    def test_sampling_never_builds_the_all_positions_context(self):
        m, _ = _methods()
        for name in ("sample", "sample_v2"):
            body = m[name]
            for dense in ("h_EXV_encoder_fw", "h_EX_encoder", "h_EXV_encoder"):
                self.assertIsNone(re.search(r"(?m)^\s*%s\s*=" % dense, body), f"Worker.{name} assigns {dense!r} (the all-positions context)")
            self.assertNotIn("h_EXV_encoder_fw[", body)
        self.assertIn("mask_bw, mask_fw = self.order_masks(decoding_order, E_idx, mask)", m["sample"])
        v2 = m["sample_v2"]
        self.assertIn('"zeros_EX": torch.zeros((N, 1, Kn, H)', v2)                                   # one position, not L (a decode-slot tensor)
        self.assertIn("h_EXV_encoder_t = mask_fw_t * cat_neighbors_nodes(h_V_enc, torch.cat([h_E_t, zeros_EX], -1), E_idx_t)", v2)

    def test_order_masks_carries_the_three_stock_statements(self):
        m, _ = _methods()
        om = m["order_masks"]
        for stmt in ('permutation_matrix_reverse = F.one_hot(decoding_order, num_classes=mask_size).float()',
                     'torch.einsum("ij, biq, bjp->bqp"', 'mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)',
                     'mask_bw = mask_1D * mask_attend', 'mask_fw = mask_1D * (1. - mask_attend)', 'return mask_bw, mask_fw'):
            self.assertIn(stmt, om)

    def test_the_batch_context_owns_its_encoder_outputs(self):
        """The batch's stock-shape encoder outputs are computed once, in the native-score stage, kept in the batch's context (its three readers — the
        native scoring, the sampling, the re-scoring — enter the batch with them) and dropped with it in finalize; nothing of self keeps them."""
        m, _ = _methods()
        nat, fin, ent = m["native"], m["finalize"], m["_enter"]
        self.assertIn("self.encode(X, mask, residue_idx, chain_encoding_all, B, K)", nat); self.assertIn('c["enc_key"], c["enc"] = self._enc_cache_key, self._enc_cache_val', nat)
        self.assertLess(nat.index("self.encode(X, mask"), nat.index("log_probs_nat = self.forward_scores("))     # once, before the native scoring reads it
        self.assertIn('self._enc_cache_key, self._enc_cache_val = c.get("enc_key"), c.get("enc")', ent)
        for stage_ in ("native", "advance", "rescore"): self.assertIn("self._enter(c)", m[stage_])   # decode's turns run in advance()
        self.assertIn('if self._enc_cache_key == c.get("enc_key"): self._enc_cache_key = self._enc_cache_val = None', fin)
        self.assertIn("if self.scores_per_backbone(K):", m["forward_scores"])                   # one predicate routes the scoring forwards
        self.assertIn("h_V_all, h_E_all, E_idx_all = self.encode(X, mask, residue_idx, chain_encoding_all, B, K)", m["forward_scores"])   # the per-backbone scoring reads the batch's encoder outputs
        self.assertIn("torch.cuda.empty_cache()", m["release_step_graphs"]); self.assertNotIn("empty_cache", m["sample_v2"])

    def test_stages_are_ordered_by_their_own_events(self):
        """Three streams, ordered per batch by events — the decode waits for ITS batch's inputs and encoder outputs (not the native forwards queued behind
        them), the re-scoring for ITS batch's sampling and native forward (not a later batch's decode) — and the host waits once per batch, in finalize."""
        m, src = _methods()
        self.assertIn('sD.wait_event(c["ev_enc"])', m["_decode_steps"]); self.assertNotIn("wait_stream", m["_decode_steps"])
        self.assertIn('self.sR.wait_event(c["ev_sample"][1]); self.sR.wait_event(c["ev_native"][1])', m["rescore"]); self.assertNotIn("wait_stream", m["rescore"])
        self.assertIn('c["ev_rescore"][1].synchronize()', m["finalize"])
        for stage_ in ("prep", "native", "decode", "_decode_steps", "advance", "rescore"):
            self.assertNotIn("synchronize()", m[stage_]); self.assertNotIn(".cpu()", m[stage_]); self.assertNotIn(".item()", m[stage_])   # no host stall while the batches are in flight
        self.assertIn("torch.cuda.Stream(device=device, priority=hi) for _ in range(self.lanes)", src)   # one prioritised decode stream per lane

    def test_the_pipelined_loop_and_the_first_batch_alone(self):
        m, src = _methods()
        rp = m["run_pipelined"]
        self.assertIn("yield self.run_batch(batches[0], chain_id_dict, out_dir, B)", rp)         # the first batch (the captures) runs alone
        self.assertIn("while todo and len(live) < self.lanes:", rp)                             # one decode in flight per lane
        self.assertIn("live.append(self.decode(self.native(self.prep(todo.pop(0), chain_id_dict, B, lane)), keep=True))", rp)
        self.assertIn("if not self.advance(c):", rp); self.assertIn("waiting.append(self.rescore(c))", rp)   # turns per lane; re-scoring issued behind a fully enqueued decode
        self.assertIn('while waiting and (not live or waiting[0]["ev_rescore"][1].query()):', rp)               # files in batch order, no idle wait while a lane has steps to enqueue
        self.assertIn("if self.sN is None:", rp)                                                # CPU: one batch after the other
        self.assertIn("while self.advance(c): pass", m["run_batch"])
        self.assertIn('for fin in w.run_pipelined([[proteins[j] for j in bi] for bi in batches], chain_id_dict, args.out_folder, args.batch_size):', src)
        self.assertIn("    w.release_step_graphs()\n", src)

    def test_decode_slot_and_scoring_graphs_are_reused_by_shape(self):
        m, _ = _methods()
        v2, sf, rel = m["sample_v2"], m["scoring_forward"], m["release_step_graphs"]
        self.assertIn('key = (K, B, Lmax)', v2); self.assertIn('if slot is None or slot["key"] != key:', v2)
        self.assertIn('slot[nm].copy_(src)', v2); self.assertIn('slot["all_probs"].zero_(); slot["h_S"].zero_(); slot["S"].zero_()', v2)   # a reused slot starts from stock's zeros
        self.assertIn('t_static, step_idx, graphs = slot["t_static"], slot["step_idx"], slot["graphs"]', v2)
        self.assertIn('self.release_step_graphs((K, B, Lmax), lane=c["lane"])', m["_decode_steps"]); self.assertIn('if slot is None or (key is not None and slot["key"] == key):', rel)
        self.assertIn("slot = self._slots[self._lane]", v2)                                                     # one decode slot per lane
        self.assertIn("if enq % self.DECODE_CHUNK == 0: yield", v2)                                            # the lane's turn ends every DECODE_CHUNK steps
        self.assertIn('if sl is None:', sf); self.assertIn('slots[key] = {"seen": 1}', sf)                       # first occurrence eager, capture on the second
        self.assertIn("dst.copy_(src)", sf); self.assertIn('sl["graph"].replay()', sf)
        self.assertIn("return self.decoder_scores(hVb, hEb, Ib, Sb, maskb, chain_Mb, randnb", sf)                # CPU / graph levers off: eager, the same statements
        self.assertIn("out[rows, :L] = lp", m["forward_scores"])                                                 # a replayed graph's output copied out before the next call

    def test_generators_persist_per_slot_and_are_reseeded_per_backbone(self):
        m, _ = _methods()
        g = m["gen_for"]
        self.assertIn("g.manual_seed(self.seed); g.set_offset(offset)", g); self.assertIn("return make_gen(self.device, self.seed, offset)", g)
        self.assertIn('gens = [self.gen_for(c["lane"], b, self.offsets[proteins[b]["name"]]) for b in range(K)]', m["prep"])   # the lane's own generator bank

    def test_encode_writes_each_backbone_into_the_batch_outputs_in_place(self):
        m, _ = _methods()
        enc = m["encode"]
        self.assertNotIn("torch.cat(hE_l", enc); self.assertNotIn("hE_l.append", enc)
        self.assertIn("h_V_all[rows, :L] = hV; h_E_all[rows, :L] = hE; E_idx_all[rows, :L] = Ib", enc)


    def test_fused_draw_is_probed_and_all_or_refuse(self):
        m, src = _methods()
        self.assertIn("def _fused_draw_kernel(probs_ptr, out_ptr, seed_ptr, off_ptr, ndraw_ptr, active_ptr, B: tl.constexpr, BP: tl.constexpr, A: tl.constexpr, AP: tl.constexpr, INC: tl.constexpr, F64: tl.constexpr):", src)
        for piece in ("_tl_libdevice.fast_logf(u32)", "_tl_libdevice.log(u)", "zz = c0.to(tl.uint64) ^ (c1.to(tl.uint64) << 21)", "idx = tl.argmax(w, axis=1)", "for _ in range(10):"):
            self.assertIn(piece, src)                                                           # curand's uniform (float | double), aten's transform, Philox4x32-10, leftmost argmax
        probe = m["decide_fused_draw"]
        self.assertIn("torch.multinomial(probs[b*B:(b+1)*B], 1, generator=gens[b])", probe); self.assertIn("bad = int((ref != out).any(dim=1).sum().item())", probe)
        self.assertIn("self.fused_draw = ok", probe)
        main = src[src.index("def main():"):]
        self.assertIn('T["fused_draw_probe"] = w.decide_fused_draw(args.batch_size, requested=bool(args.fused_draw), dtype=torch.promote_types(torch.float32, w.constant_bias.dtype))', main)
        self.assertIn('T["refused"] = "fused_draw: " + ("PROBE FAIL" if HAVE_TRITON else "triton unavailable")', main)   # the line is all of its levers: refused by name before any output
        self.assertLess(main.index('T["refused"] = "fused_draw: "'), main.index("t_run0 = time.time()"))
        # started beside the parse step: the input-independent start-up (weights, the draw probe) runs first, the inputs are read once the driver's flag exists
        order = [main.index(x) for x in ("model, ck = load_model(", 'T["fused_draw_probe"] = w.decide_fused_draw(', "while not os.path.exists(args.inputs_ready):", "ds = StructureDatasetPDB(", 'T["hybrid_gemm_probe"] = w.decide_hybrid(', "t_run0 = time.time()")]
        self.assertEqual(order, sorted(order))
        v2 = m["sample_v2"]
        self.assertIn("_fused_draw_kernel[(K,)](probs, S_t, rng_seed, rng_off, ndraw, active_vec(pattern), B=B, BP=_pow2(B), A=21, AP=32, INC=self.draw_inc, F64=(probs.dtype == torch.float64))", v2)
        self.assertIn("S_t[b*B:(b+1)*B] = torch.multinomial(probs[b*B:(b+1)*B], 1, generator=gens[b])", v2)   # torch's own draw stays the path without the lever (CPU)
        self.assertIn("rows = tl.arange(0, BP)[:, None]", src); self.assertIn("valid = (cols < A) & (rows < B)", src)   # any --batch_size: a power-of-two tile, rows beyond B masked
        self.assertIn("ndraw.copy_(snap_nd)", v2); self.assertIn("if not fused:", v2)             # warm-up restores the draw counter; no generator registration for the fused graphs
        self.assertIn("g.set_offset(off0[b] + n_draws * step_inc)", v2)                          # each generator ends where stock's stream stands

    def test_native_record_follows_stocks_per_round_condition(self):
        """finalize writes the native ('>name, score=…') record where protein_mpnn_run.py's round loop writes it (`if b_ix == 0 and j==0 and
        temp==temperatures[0]`): the worker's own rule, executed here on round lists — once for distinct temperatures, once per batch-0 round of a
        repeated first temperature."""
        m, src = _methods()
        tree = ast.parse(src)
        fn, = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "native_header_due"]
        ns = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "native_header_due", "exec"), ns)
        due = ns["native_header_due"]
        def headers(temps, num_batches):
            rounds = [(t, j) for t in temps for j in range(num_batches)]                          # main(): stock's `for temp in temperatures: for j in range(NUM_BATCHES)`
            return sum(due(t, j, rounds[0][0]) for t, j in rounds)
        self.assertEqual([headers([0.1], 1), headers([0.1], 3), headers([0.1, 0.2], 2), headers([0.1, 0.1], 1), headers([0.2, 0.1, 0.2], 2)], [1, 1, 1, 2, 2])
        fin = m["finalize"]
        self.assertIn("if native_header_due(T, j, self.rounds[0][0]):", fin); self.assertLess(fin.index("native_header_due(T, j"), fin.index("for b_ix in range(B):"))
        main = src[src.index("def main():"):]
        for opt in ("fixed_positions_jsonl", "omit_AA_jsonl", "bias_by_res_jsonl", "pssm_jsonl", "bias_AA_jsonl", "chain_id_jsonl"):   # one loading rule for every dictionary: protein_mpnn_run.py's
            self.assertRegex(main, r"load_jsonl_dict\(args\.%s\b" % opt)
        self.assertNotIn("splitlines()[0]", main)

    def test_every_tensor_the_step_graphs_touch_is_the_slots(self):
        """A decode-step graph captured for one batch is replayed for every later batch of its shape: nothing it reads or writes may be a
        per-call tensor (freed when the call returns, its storage reused — the row index `ar` once was)."""
        m, src = _methods()
        v2 = m["sample_v2"]
        self.assertIn('"ar": torch.arange(N, device=device)}', v2); self.assertIn('ar = slot["ar"]', v2)
        body = v2[v2.index("def step_core(t):"):v2.index("def build(pattern):")]
        for name in ("E_idx", "h_E", "S_true", "chain_mask", "mask", "bias_by_res", "mask_bw", "mask_fw", "h_V_stack", "all_probs", "h_S", "S", "zeros_EX", "ar"):
            self.assertRegex(v2, r'(?m)^\s*(?:\w+, )*%s(?:, \w+)* = .*slot\[' % name, name)      # bound to the slot's tensor before the step body
        setup = v2[:v2.index("def step_core(t):")]
        per_call = [l.strip() for l in setup.splitlines() if "device=device)" in l and "slot" not in l and '"' not in l]
        self.assertEqual(per_call, ["row_of_bb = torch.arange(N, device=device) // B"], per_call)   # the one per-call device tensor left is read eagerly (keep_vec), never inside a graph
        graph_body = v2[v2.index("with torch.no_grad(), torch.cuda.graph(g, pool="):v2.index("torch.cuda.synchronize()", v2.index("with torch.no_grad(), torch.cuda.graph(g, pool="))]
        self.assertNotIn("row_of_bb", graph_body); self.assertNotIn("torch.arange(", graph_body); self.assertNotIn("torch.tensor(", graph_body)   # the captured statements allocate no index tensors of their own
        kv = v2[v2.index("def keep_vec(pattern):"):v2.index("def commit(")] if v2.index("def keep_vec(pattern):") < v2.index("def commit(") else v2[v2.index("def keep_vec(pattern):"):]
        self.assertIn("keep_cache[pattern] = (~sel[row_of_bb])", kv)                          # read once, eagerly, into the slot's cache

if __name__ == "__main__":
    unittest.main()
