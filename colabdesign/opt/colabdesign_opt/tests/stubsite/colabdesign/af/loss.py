"""stand-in colabdesign.af.loss: the names BindCraft's colabdesign_utils imports (used inside loss closures only, never called here)."""


def get_ptm(inputs, outputs, interface=False):
    raise NotImplementedError("stand-in")


def mask_loss(x, mask=None, mask_grad=False):
    raise NotImplementedError("stand-in")


def get_dgram_bins(outputs):
    raise NotImplementedError("stand-in")


def _get_con_loss(dgram, dgram_bins, cutoff=None, binary=True):
    raise NotImplementedError("stand-in")
