"""stand-in jax.tree_util (tests only): tree_map over dicts / lists / tuples of leaves."""


def tree_map(f, *trees):
    t = trees[0]
    if isinstance(t, dict):
        return {k: tree_map(f, *[x[k] for x in trees]) for k in t}
    if isinstance(t, (list, tuple)):
        return type(t)(tree_map(f, *[x[i] for x in trees]) for i in range(len(t)))
    return f(*trees)
