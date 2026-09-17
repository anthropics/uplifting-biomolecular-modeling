# OpenFold3 → AlphaFold3 Weight Porting: Code Changes and Implications

This document records every change made to the AlphaFold3 (JAX/Haiku) codebase
to run OpenFold3 (PyTorch) weights, and what each change implies for anyone trying to
reproduce or re-implement AlphaFold3 from the released source.

**Repositories**
- AF3 source: [sokrypton/alphafold3](https://github.com/sokrypton/alphafold3) (fork of google-deepmind/alphafold3)
- OF3 source: [aqlaboratory/openfold3](https://github.com/aqlaboratory/openfold)
- OF3 weights: `s3://openfold/staging/of3-p2-155k.pt` (public, no sign required)

---

## Background

Google DeepMind released the AlphaFold3 source code but not the model weights
(commercial use restricted). AlQuraishi Lab released OpenFold3 with public weights
for the same architecture. Since both implement the same algorithm, the weights are
portable — but not without changes.

All OF3-specific branches in AF3 are gated on `global_config.of3_weights = True`,
set automatically via `--of3_weights` at the command line. Standard AF3 behavior
is unchanged when the flag is absent.

---

## Model Architecture Changes

These changes live in `src/alphafold3/model/`. They cannot be handled in the weight
converter alone because they reflect genuine differences in how OF3 and AF3 implement
or apply the same algorithm.

| File | Change | Why |
|---|---|---|
| `model_config.py` | Added `GlobalConfig.of3_weights: bool = False` | Master switch for all OF3 branches |
| `model.py` | Element one-hot index shifted by −1 | OF3 featurizes elements as `GetAtomicNumber() - 1` (0-indexed); AF3 uses `GetAtomicNum()` (1-indexed) |
| `network/atom_cross_attention.py` | `queries_single_cond *= queries_mask` | Masks padded query atoms from single conditioning, consistent with OF3's `atom_pair_mask` behavior |
| `network/atom_cross_attention.py` | `keys_ref_space_uid` sourced from `queries_ref_space_uid` | Matches OF3's uid lookup convention |
| `network/atom_cross_attention.py` | `offsets_valid &= keys_mask` | OF3 multiplies uid-match by `atom_pair_mask`; AF3 did not mask padded key atoms — `ref_space_uid=0` of padded atoms collided with token 0, distorting N-terminal backbone (CA–C was ~0.85 Å instead of 1.52 Å) |
| `network/evoformer.py` | Bond matrix symmetrized in `_embed_bonds` | OF3's `create_token_bonds` sets both `[i,j]` and `[j,i]` per bond; AF3 featurizes only one direction — ring bonds broken without this (ATP ribose C–C ~2.0 Å instead of ~1.5 Å) |
| `network/modules.py` | Pair bias axes swapped for column attention (`transpose=True`) | OF3 computes column attention by transposing the sequence input before `linear_z`, producing `Linear(z[k,q])` as the bias; AF3 computes `Linear(z[q,k])`. The weights were trained with OF3's convention so the bias must be swapped |
| `network/diffusion_transformer.py` | Per-block pair LayerNorm + linear projection branch | OF3's `AttentionPairBias` contains its own `layer_norm_z` + `linear_z` per block; AF3 originally used a single shared LN before the entire block stack |
| `network/diffusion_head.py` | Fourier `w`/`b` loaded as Haiku params instead of AF3's hardcoded constants | OF3 stores these as `register_buffer` (saved to `state_dict`); we convert them directly from the checkpoint so AF3's hardcoded JAX constants are never used |
| `network/noise_level_embeddings.py` | Optional `weight`/`bias` args added to `noise_embeddings()` | Allows passing the converted Fourier values from `diffusion_head` instead of falling back to AF3's hardcoded values |
| `network/diffusion_head.py` | Two zero columns re-inserted into `features_1d` | OF3's restype/profile blocks carry 32 classes to AF3's 31. Everywhere else the extra class is simply dropped from the weights — a zero input column contributes nothing to a bare `Linear`. Here it cannot be: `single_cond_initial_norm` is a LayerNorm, which maps a zero input to −mean/std, so OF3 always adds a trained contribution through those columns *and* normalises over 833 channels rather than 831. Dropping them cost ~1.6e-3 relative error in the diffusion single conditioning (verified: rel 2.2e-3 → 3.4e-7 against the OF3 module) |

---

## Weight Converter

`convert_of3_weights.py` + `src/alphafold3/model/of3_weight_converter.py`

Handles the systematic PyTorch → JAX/Haiku translation:

- **Parameter renames**: `weight` → `weights`, `LayerNorm.weight` → `scale`, `LayerNorm.bias` → `offset`
- **Linear transpositions**: PyTorch stores `(out, in)`; AF3 stores `(in, out)` for most projections
- **Attention head reshapes**: Q/K stored as `(H, D, in)` (transpose_weights=True); V as `(in, H, D)`
- **SwiGLU concatenation**: OF3 has separate `linear_a` (gate) and `linear_b`; AF3 concatenates them as `[gate | linear]` for a single fused projection
- **Layer stack aggregation**: OF3 has per-block parameter dicts (`blocks.0.*, blocks.1.*`); AF3's `hk.experimental.layer_stack` expects a leading stacked axis
- **Residue alphabet permutation**: the two codebases order the residue classes differently, so every weight matrix whose input rows are indexed by residue type must be permuted (`_AF3_TO_OF3_AATYPE` / `_AF3_TO_OF3_MSA`):

  | | ordering |
  |---|---|
  | AF3 (31 classes, `POLYMER_TYPES_WITH_UNKNOWN_AND_GAP`) | `0-19` protein, `20` UNK, `21` GAP, `22-25` A/G/C/U, `26-29` DA/DG/DC/DT, `30` N |
  | OF3 (32 classes, `STANDARD_RESIDUES_WITH_GAP_3`) | `0-19` protein, `20` UNK, `21-25` A/G/C/U/N, `26-30` DA/DG/DC/DT/DN, `31` GAP |

  Protein and DNA indices coincide; **RNA is offset by one and GAP/N are in completely different slots**. The permuted matrices are `input_embedder.linear_s/linear_z_i/linear_z_j`, `msa_module_embedder.linear_s_input/linear_m`, `aux_heads.pairformer_embedding.linear_i/linear_j`, `template_pair_embedder.aatype_linear_1/2`, and `diffusion_conditioning.layer_norm_s/linear_s` (the `features_1d` block).

**Critical bugs fixed during conversion**:
- TriangularMultiplication incoming `a`/`b` projection weights were swapped.
- `msa_module_embedder.linear_m` was transposed but **not** permuted through the residue alphabet. Its input is `[restype one-hot(32), has_deletion, deletion_value]` in both codebases, so the shapes matched (34 rows) and the error was silent. Consequences: MSA gaps (AF3 class 21) were embedded as OF3 RNA adenine, and RNA A/G/C/U were read one slot high (A→G, G→C, C→U, U→N). Protein residues and DNA bases were unaffected — but since gaps dominate any real MSA, this degraded protein predictions too.
- **Row (i) / column (j) pair projections were swapped in the confidence head and the template embedder.** See below — this one is a trap laid by AF3's own naming.

### The i/j trap: AF3's `left`/`right` naming is inconsistent

AF3 builds a pair representation from a single representation in three places, and the meaning of `left`/`right` is **not the same in all of them**:

```python
# evoformer._seq_pair_embedding
left_single[:, None] + right_single[None]        # out[i,j] = left[i] + right[j]   → left is row i

# confidence_head._embed_features
out  = Linear(name='left_target_feat_project')(target_feat)          # [N, c]
out += Linear(name='right_target_feat_project')(target_feat)[:, None] # [N, 1, c]
                                                 # out[i,j] = left[j] + right[i]   → left is column j

# template_modules: to_concat order
to_concat.append((aatype[None, :, :], 1))        # index 2 → column j
to_concat.append((aatype[:, None, :], 1))        # index 3 → row i
```

OF3 consistently uses the evoformer convention — `linear_i` is always the row-i projection:

```python
z   = s_input_emb_i[..., None, :] + s_input_emb_j[..., None, :, :]   # input_embedder
zij = zij + linear_i(si_input.unsqueeze(-2)) + linear_j(si_input.unsqueeze(-3))  # pairformer_embedding
a   = ... + aatype_linear_1(template_restype_ti) + aatype_linear_2(template_restype_tj)
```

So mapping `linear_i → left` is **correct in the trunk and wrong in the confidence head**, and `aatype_linear_1 → template_pair_embedding_2` is wrong for the same reason. The converter had both by-name mappings; they are now crossed where required.

**Symptom**: a swap here transposes the pair embedding that the confidence pairformer starts from. `PAE` is where it shows: it is the only asymmetric confidence output — PDE is explicitly symmetrized (`logits + logitsᵀ`), and pLDDT / experimentally-resolved come from the single representation. pTM and ipTM are derived from `pae_probs`, so they were affected as well.

**Checked and confirmed consistent** (no fix needed): the PAE/PDE bin decoding — AF3's `linspace(0, 31.0, 63)` + half-step + catch-all and OF3's `get_bin_centers(0, 32, 64)` both yield centers `0.25, 0.75, …, 31.75`; and the representative-atom choice feeding `linear_distance` — AF3's pseudo-beta (CB, CA for Gly, C4 for purines, C2 for pyrimidines, atom 0 for ligands) matches OF3's `get_token_representative_atoms` exactly.

---

## Verification

The conversion was audited three ways, using the scripts in
`scripts/of3_verification/` (see the README there). They need the OF3 checkpoint
plus, for the module comparisons, an importable `openfold3`. The fast unit tests
are in `src/alphafold3/model/of3_weight_converter_test.py`.

**Weight coverage.** Instrumenting `_get`/`_has` over a full conversion shows
4172 of 4936 checkpoint tensors are read. The remaining 764 are all under
`sample_diffusion.*`, which is a bit-identical duplicate of `diffusion_module.*`
(763 pairs compared, max|Δ| = 0), plus `version_tensor`. No tensor is silently
dropped and no key is probed without being used.

**Structural comparison against real AF3 params.** Every converted array matches
the native `af3.bin.zst` parameter tree in name and shape, except:
* the diffusion transformer, which legitimately differs under `of3_weights`
  (`__layer_stack_no_per_layer` because the OF3 branch omits
  `with_per_layer_inputs=True`; per-block `pair_input_layer_norm` /
  `pair_logits_projection` inside the stack; the two Fourier params), and
* `single_cond_initial_norm` / `single_cond_initial_projection`, which are 833
  rather than 831 rows for the LayerNorm reason described above.

Zero shape mismatches elsewhere also rules out every head-dimension swap, since
`num_head != head_dim` in all attention configs.

**Numerical module comparison.** Real OF3 PyTorch modules were run against the
real AF3 Haiku modules loaded with converted weights, on identical inputs. This
is what catches the error classes no shape check can see: square-matrix
transposes, SwiGLU gate-vs-value order, and block ordering within a layer stack.

| module | relative error |
|---|---|
| trunk PairFormer block (0 and 47 of 48) | 2e-5 |
| MSA module block (0 and 3 of 4) | 3e-6 |
| diffusion transformer (all 24 blocks) | 3e-5 |
| template pair-stack block (0 and 1) | 2e-6 |
| confidence pairformer block (0 and 3 of 4) | 1e-5 |
| diffusion conditioning (Algorithm 21) | 4e-7 |
| trunk/MSA input embeddings, confidence pair embed | 1e-5 |
| pae / pde / plddt / experimentally-resolved / distogram heads | exactly 0 |

Testing both ends of each stack confirms block order is not reversed. Residuals
are float32 accumulation noise; each harness was validated to *fail* when a
single weight is deliberately transposed or scaled. This covers 99% of the 368M
converted parameters (364.7M). The remaining 1% is the atom cross-attention
transformers (2.81M), the atom feature embedders (0.60M) and evoformer
miscellany (0.32M — prev embeddings, relpos, bond embedding, template output);
AF3's `CrossAttTransformer` needs a constructed `queries_to_keys` gather layout,
and a hand-built stand-in risks a false result more than it buys confidence.

### Benign asymmetries confirmed during the audit

* **`_stack_blocks` zero-fills** any key a later block lacks. This fires for
  real: OF3's last MSA block drops `msa_att_row`/`msa_transition`
  (`last_block=True`). Zero weights make AF3's corresponding sub-modules output
  exactly zero, so adding them is a true skip — the MSA output difference for
  that block is 0. Correct, but note it would also silently hide a genuine
  omission.
* **pLDDT / experimentally-resolved heads** are sized for 23 atoms per token in
  OF3 versus AF3's 24, and are zero-padded. The padded slot is unreachable: the
  largest standard residue is RNA G with 23 heavy atoms, and non-standard
  residues are atomized to one atom per token.
* **The non-`_1` atom pair-conditioning params are dead.** AF3 calls
  `token_atoms_single_cond, _ = _per_atom_conditioning(...)`, discarding that
  site's pair output, so only the `_1` copies (the real queries×keys
  conditioning) matter. Writing OF3's single weight set into both is harmless.
* **Relative-position encoding layouts agree**: both are
  `[rel_pos(66) | rel_token(66) | same_entity(1) | rel_chain(6)]` with identical
  clipping and out-of-range sentinels, so `linear_relpos` needs no reordering.
  Verified numerically via the diffusion conditioning, which feeds relpos and the
  trunk pair rep through one linear.

---

## Implications for AlphaFold3 Reproduction

### 1. Atom cross-attention padding (N-terminal backbone distortion)

The `offsets_valid` bug in `atom_cross_attention.py` affects anyone using AF3 with padded atom sequences. AF3 pads atom arrays to a fixed size; padded slots are zero-filled, giving them `ref_space_uid=0`. Since the first real residue also has `ref_space_uid=0`, every padded key atom was treated as a valid neighbor of residue 0. This produced severe backbone geometry errors at the N-terminus in all test cases (CA–C ~0.85 Å, C–O ~0.95 Å vs. ideals of 1.52 Å and 1.23 Å).

**This is a bug in the released AF3 source**, not specific to OF3 weights. Fix:
```python
offsets_valid = (
    queries_ref_space_uid[:, :, None] == keys_ref_space_uid[:, None, :]
) & keys_mask[:, None, :].astype(jnp.bool_)
```

### 2. Bond matrix asymmetry

AF3's featurizer provides bonds in one direction only (lower-index → higher-index from the CCD bond table). OF3 trained with a symmetric bond matrix. Whether this matters for AF3's own weights is unknown — but ring-topology ligands (saccharides, ATP ribose) are significantly affected: ATP ribose C–C bonds were ~2.0 Å (broken) vs. ~1.5 Å (correct) without symmetrization.

Anyone training AF3 from scratch should verify which convention is used in their featurizer and ensure consistency with the bond embedding weights.

### 3. Column attention pair bias (OF3 deviation from paper)

OF3's `TriangleAttention` (`starting=False`) transposes the pair representation **before** applying `linear_z`:

```python
# OF3 — column attention
x = x.transpose(-2, -3)           # z → z.T
triangle_bias[h, q, k] = Linear(z.T[q, k]) = Linear(z[k, q])
```

AF3's Algorithm 15 specifies the pair bias between query q and key k as `Linear(z[q, k])`. OF3 computes `Linear(z[k, q])` — the **transposed** pair bias. This is a deviation from the paper. The model trained with this convention and learned to compensate, so predictions are good in practice. Our AF3 fix reproduces the OF3 convention:

```python
if self.transpose and self.global_config.of3_weights:
    nonbatched_bias = jnp.swapaxes(nonbatched_bias, -1, -2)
```

Anyone reimplementing triangle attention ending node should use `Linear(z[q, k])` per the paper if training from scratch.

### 4. Diffusion transformer pair conditioning (per-block vs. shared)

OF3 applies a separate `LayerNorm + Linear` to the pair representation inside **each** diffusion transformer block. AF3's original code applied a single shared LayerNorm before all blocks. This is a genuine architectural difference that produces incompatible parameter layouts. If training AF3 from scratch, the choice must match whatever convention the weights were trained with.

### 5. Fourier noise embeddings

Both AF3 and OF3 initialize the random Fourier embedding weights with seed 42, but PyTorch's `torch.Generator().manual_seed(42)` + `normal_` + `uniform_` produces different values than JAX's equivalent. AF3 hardcodes its values as `_WEIGHT` and `_BIAS` constants; OF3 saves them as `register_buffer` in the checkpoint. When using OF3 weights, the checkpoint values must be used — not AF3's hardcoded constants.

---

## Summary: What the Released AF3 Code Gets Wrong

| Issue | Severity | Affects AF3 weights too? |
|---|---|---|
| `offsets_valid` missing `keys_mask` gate (N-terminal distortion) | High — ~0.85 Å CA–C at residue 0 | Likely yes |
| Bond matrix one-directional (ring ligand geometry) | Medium — ring bonds broken | Depends on training convention |
| Column attention pair bias transposed (OF3 deviation from paper) | Low — model compensates during training | No — OF3 weights only |
| Diffusion transformer pair conditioning per-block vs. shared | Architectural — incompatible param layouts | Only relevant if training from scratch |
| Fourier embedding constants differ between JAX and PyTorch RNG | Medium — wrong noise conditioning without fix | No — OF3 weights only |
