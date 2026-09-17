"""Modules the tree loads INTO the model process (the fork's interpreter) by file path from its launchers — stdlib at import; jax, tokamax and
alphafold3 are imported inside install() only, where the model process already has them. Nothing here is imported by the wrapper's commands."""
