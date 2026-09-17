"""Layout flag for the converted OpenFold3 weights in the AF3 haiku layout (sokrypton/alphafold3 `global_config.of3_weights`).
Set ONCE before building the model:  from xfold import of3; of3.OF3 = True
Mirrors the 7 code-path differences of the JAX fork (model.py, modules.py, evoformer.py, atom_cross_attention.py, diffusion_head.py x2, diffusion_transformer.py)."""
OF3 = False
