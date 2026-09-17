"""enformer_deepmind_opt._graph — rewriting the Enformer SavedModel's prediction graph.

``hub.load(...).model.predict_on_batch`` is a restored ``tf.function`` with one concrete function: a single flat graph (no nested function
calls, no control flow) whose inputs are the sequence tensor and the model's variables as captured resource handles. The kit edits that
graph's ``GraphDef`` — a lever replaces a group of nodes with nodes computing the same values — and builds a new concrete function from the
edited ``GraphDef`` bound to the SAME captured variables (no weight is copied; TensorFlow's own optimizer passes then run on the new function
exactly as they run on the restored one). Everything outside the replaced groups is the stock graph node for node.
"""
from __future__ import annotations

from typing import Callable, Dict, List


# ----------------------------------------------------------------------------------------------------------------- GraphDef helpers
class Graph:
    """A mutable view of a GraphDef: nodes by name, consumers, node construction, output rewiring."""

    def __init__(self, graph_def, cf=None):
        self.gd = graph_def
        self.cf = cf                                                   # the stock concrete function (for evaluate(): its inputs and captured variables)
        self.nodes = {n.name: n for n in graph_def.node}
        self.removed = set()
        self.alias = {}                                                # rewired producers: old node name -> the node now feeding its consumers
        self.input_crop = None                                         # (lo, hi): the positions of each window the graph reads, once a lever moves the input crop to the host

    def evaluate(self, names: List[str]) -> list:
        """Run the CURRENT graph once for the tensors ``names`` (input-independent ones: the sequence input is fed zeros of batch 1) with the
        stock function's captured variables, on the device the model runs on; returns numpy arrays."""
        import numpy as np
        import tensorflow as tf
        if self.cf is None:
            raise RuntimeError("Graph.evaluate needs the concrete function")
        fn = rebuild_outputs(self.cf, self.sub_graph(names), [n + ":0" for n in names])   # import only what feeds `names`
        spec = self.cf.structured_input_signature[0][0]
        dims = self.node(self.producer(self.cf.inputs[0].name)).attr["shape"].shape.dim   # the input placeholder's shape as the graph now declares it
        x = tf.zeros([1] + [int(d.size) for d in dims][1:], spec.dtype)
        out = fn._call_flat([x], fn.captured_inputs)                    # noqa: SLF001
        out = out if isinstance(out, (list, tuple)) else [out]
        return [np.asarray(t.numpy()) for t in out]

    def sub_graph(self, names: List[str]):
        """The current GraphDef cut down to the nodes feeding ``names`` plus every Placeholder (the inputs ``rebuild_outputs`` binds)."""
        from tensorflow.core.framework import graph_pb2
        gd = self.finish()
        nodes = {n.name: n for n in gd.node}
        keep, stack = set(), list(names) + [n.name for n in gd.node if n.op == "Placeholder"]
        while stack:
            nm = stack.pop()
            if nm not in keep:
                keep.add(nm)
                stack.extend(self.producer(i) for i in nodes[nm].input)      # KeyError: a name not in the graph
        out = graph_pb2.GraphDef()
        out.CopyFrom(gd)
        del out.node[:]
        out.node.extend(n for n in gd.node if n.name in keep)
        return out

    def resolve(self, name: str) -> str:
        """The node now standing in for ``name`` after rewires (``name`` itself when it was never rewired)."""
        seen = set()
        while name in self.alias and name not in seen:
            seen.add(name); name = self.alias[name]
        return name

    def const_f32(self, name: str, value):
        """A float32 Const node holding ``value`` (numpy array)."""
        import numpy as np
        from tensorflow.core.framework import attr_value_pb2, node_def_pb2, types_pb2
        from tensorflow.python.framework import tensor_util
        n = node_def_pb2.NodeDef(name=name, op="Const")
        n.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
        n.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(np.ascontiguousarray(value, dtype=np.float32))))
        self.gd.node.append(n)
        self.nodes[name] = self.gd.node[-1]
        return name

    def node(self, name: str):
        n = self.nodes.get(name)
        if n is None or name in self.removed:
            raise KeyError(f"node {name!r} not in the graph")
        return n

    def has(self, name: str, op: str | None = None) -> bool:
        n = self.nodes.get(name)
        return n is not None and name not in self.removed and (op is None or n.op == op)

    @staticmethod
    def producer(input_name: str) -> str:
        return input_name.lstrip("^").split(":")[0]

    def consumers(self, name: str) -> List:
        return [n for n in self.gd.node if n.name not in self.removed and any(self.producer(i) == name for i in n.input)]

    def add(self, name: str, op: str, inputs: List[str], **attrs):
        """Append a node; attrs: T=True / dtype=True set that attr to float32, ``<attr>_i32=True`` sets ``<attr>`` to int32 (T_i32, Index_i32,
        Tshape_i32, out_type_i32), ints / floats / strings / lists of ints / bools by value."""
        from tensorflow.core.framework import attr_value_pb2, node_def_pb2, types_pb2
        if name in self.nodes and name not in self.removed:
            raise ValueError(f"node {name!r} already exists")
        n = node_def_pb2.NodeDef(name=name, op=op)
        n.input.extend(inputs)
        for k, v in attrs.items():
            if v is True and k in ("T", "dtype"):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
            elif v is True and k.endswith("_i32"):
                n.attr[k[:-4]].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_INT32))
            elif isinstance(v, bool):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(b=v))
            elif isinstance(v, int):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(i=v))
            elif isinstance(v, float):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(f=v))
            elif isinstance(v, str):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(s=v.encode()))
            elif isinstance(v, (list, tuple)):
                n.attr[k].CopyFrom(attr_value_pb2.AttrValue(list=attr_value_pb2.AttrValue.ListValue(i=list(v))))
            else:
                raise TypeError(f"attr {k}={v!r}")
        self.gd.node.append(n)
        self.nodes[name] = self.gd.node[-1]
        return name

    def const_i32(self, name: str, value):
        """An int32 Const node (scalar or list)."""
        import numpy as np
        from tensorflow.core.framework import attr_value_pb2, node_def_pb2, types_pb2
        from tensorflow.python.framework import tensor_util
        n = node_def_pb2.NodeDef(name=name, op="Const")
        n.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_INT32))
        n.attr["value"].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(np.asarray(value, dtype=np.int32))))
        self.gd.node.append(n)
        self.nodes[name] = self.gd.node[-1]
        return name

    def rewire(self, old: str, new: str) -> int:
        """Point every consumer of tensor ``old`` (``node`` or ``node:k``) at ``new``; returns the number of inputs rewired."""
        old_full = old if ":" in old else old + ":0"
        count = 0
        for n in self.gd.node:
            if n.name in self.removed:
                continue
            for i, inp in enumerate(n.input):
                full = inp if (":" in inp or inp.startswith("^")) else inp + ":0"
                if full == old_full:
                    n.input[i] = new
                    count += 1
        self.alias[old] = new
        return count

    def remove(self, names: List[str]) -> None:
        self.removed.update(names)

    def finish(self):
        """The edited GraphDef: removed nodes dropped (a removed node still consumed is an error, by name)."""
        from tensorflow.core.framework import graph_pb2
        for n in self.gd.node:
            if n.name in self.removed:
                continue
            for inp in n.input:
                if self.producer(inp) in self.removed:
                    raise RuntimeError(f"graph rewrite: {n.name} ({n.op}) still consumes removed node {self.producer(inp)}")
        out = graph_pb2.GraphDef()
        out.CopyFrom(self.gd)
        del out.node[:]
        out.node.extend(n for n in self.gd.node if n.name not in self.removed)
        return out


# ----------------------------------------------------------------------------------------------------------------- rebuild
def concrete_function(model):
    """The SavedModel's prediction function: exactly one concrete function taking one float32 (None, L, 4) tensor."""
    fns = list(model.predict_on_batch.concrete_functions)
    if len(fns) != 1:
        raise RuntimeError(f"predict_on_batch has {len(fns)} concrete functions (expected 1)")
    return fns[0]


def rebuild_outputs(cf, graph_def, outputs):
    """A ConcreteFunction over ``graph_def`` with ``cf``'s inputs and captured variables, returning the tensors named by ``outputs`` (a list,
    or any structure of tensor names: the function returns that structure)."""
    from tensorflow.python.eager import wrap_function
    captures = {internal.name.split(":")[0]: external for external, internal in cf.graph.captures}
    return wrap_function.function_from_graph_def(graph_def, [t.name for t in cf.inputs], outputs, captures)


def rebuild(cf, graph_def):
    """A new ConcreteFunction over ``graph_def`` with ``cf``'s inputs, outputs, output structure and captured variables."""
    from tensorflow.python.util import nest
    return rebuild_outputs(cf, graph_def, nest.pack_sequence_as(cf.graph.structured_outputs, [t.name for t in cf.outputs]))


def rewrite(cf, levers: Dict[str, Callable[[Graph], dict]]):
    """Apply each lever's rewrite to ``cf``'s GraphDef; returns (predict callable taking the stock call's argument and returning the stock
    output structure, {lever: stats}). ``predict`` hands the argument to the rebuilt function's flat call — one positional tensor plus the
    captured variables, structured outputs back — without the per-call signature binding of ``ConcreteFunction.__call__``."""
    g = Graph(cf.graph.as_graph_def(), cf)
    stats = {}
    for name, fn in levers.items():
        stats[name] = fn(g)
    new = rebuild(cf, g.finish())
    call, captured = new._call_flat, new.captured_inputs               # noqa: SLF001
    crop = g.input_crop
    device = next((t.device for t in captured if t.dtype.name == "resource"), None)   # where the model's variables — and so the graph's kernels — are

    def predict(x):
        return call([as_tensor(x) if crop is None else as_cropped_tensor(x, crop[0], crop[1], device)], captured)
    predict.concrete_function = new                                     # takes the cropped window when input_crop is set
    predict.input_crop = crop
    return predict, stats


def as_tensor(x):
    """The call's argument as a float32 tensor. ``tf.constant`` — what calling the stock function with an array amounts to — first copies the
    array into a fresh host buffer (a pageable host copy the call waits on) and uploads that. In an eager, synchronous
    call, a C-contiguous, writeable, native float32 numpy array is instead handed to TensorFlow through DLPack: the tensor IS the array's memory,
    and the function's own host-to-device copy — the tensor's only consumer, so no host kernel ever reads the (16-byte-aligned) buffer — uploads it
    and completes before the first kernel runs, hence before the call returns. The graph reads the same values either way. Anything else (another
    dtype or layout, a read-only array, a tensor, a call being traced into a graph or dispatched asynchronously — where TensorFlow may read the
    argument after the call returns, and stock's copy is what the caller relies on) converts as stock converts it."""
    import numpy as np
    import tensorflow as tf
    if isinstance(x, np.ndarray) and x.dtype == np.dtype(np.float32) and x.flags.c_contiguous and x.flags.writeable and hasattr(x, "__dlpack__") \
            and tf.executing_eagerly() and tf.config.experimental.get_synchronous_execution():
        return tf.experimental.dlpack.from_dlpack(x.__dlpack__())
    return tf.convert_to_tensor(x, tf.float32)


def as_cropped_tensor(x, lo: int, hi: int, device):
    """Positions ``[lo, hi)`` of every window of the batch ``x`` (B, L, 4) as the call's argument — the crop the stock graph applies on the
    device, applied before the upload instead, so only the positions the network reads cross the bus. One window: the cropped view of a
    C-contiguous array is itself contiguous and is handed over exactly as ``as_tensor`` hands over a whole array (no host copy). Several windows:
    each window's cropped view is handed over on its own and the B uploads (together half the window bytes) are stacked on ``device``. A tensor,
    or anything else TensorFlow converts, is sliced by TensorFlow itself, as the stock graph slices it."""
    import numpy as np
    import tensorflow as tf
    if not isinstance(x, (np.ndarray, tf.Tensor)):
        x = tf.convert_to_tensor(x, tf.float32)
    if not isinstance(x, np.ndarray) or x.shape[0] <= 1:
        return as_tensor(x[:, lo:hi, :])
    with tf.device(device):
        return tf.stack([as_tensor(x[i, lo:hi, :]) for i in range(x.shape[0])])
