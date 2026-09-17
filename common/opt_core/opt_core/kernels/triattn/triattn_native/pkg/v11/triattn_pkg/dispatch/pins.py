"""Kernel directories `best` routes to, pinned at the commit that sealed them: route name -> (directory relative to the tree that holds
dispatch/, commit).  "HEAD" = the directory as checked out beside this file.  test_dispatch.py pins checks that the checked-out trees still
equal these commits; a directory that changes is re-pinned explicitly, never followed silently."""
PINS = {
    "tri":     ("triton", "HEAD"),                  # k10 / k11 / k12 (+ launch, masking, errors); `tri` = triton/candidate.py
    "k13":     ("dispatch/kernels", "HEAD"),        # the small-S Gluon kernel beside this router
    "cuda_b":  ("cuda_b", "6fb95c4b646"),           # triattn_m1, default instantiation (flags 0) + its SAFE partner, with the dead-row early exit
    "cuda_c":  ("cuda_c", "d424d883727"),           # triattn_mw, geometry 3x4 (the default; MW_GEOM is refused by best), with the dead-row early exit
    "cuda":    ("cuda", "010f83b752a"),           # triattn_cuda: the binding instantiates only its kernel of record (tree unchanged since it was imported)
    "cuda_80": ("cuda_80", "20bbc5dc491"),           # triattn_sm80: the sm_80 member (cc 8.0) as frozen for generation 11
}
