"""The fused-sampler words' account (ditfast_ptx1.describe() shape) for the kit tests' synthetic runs: every word engaged and serving
(the exact arm reads atom_attn_exact only, the fast arms the four tolerance words; report.kit_evidence ignores words the arm does not list)."""


def _w(word, served, **facts):
    return {"word": word, "requested": True, "state": "on", "reason": None, "served": served, "gated": {}, "fallback": {}, "card": None, "aside": None, "facts": dict(facts)}


DF_OK = {"card": "sm_90", "words": {
    "cond_dedupe": _w("cond_dedupe", 3, guard="stride0"),
    "dit_fused": _w("dit_fused", 3, blocks=24, lowp="fp16"),
    "dit_lowp": _w("dit_lowp", 3, precision="fp16"),
    "atom_fused": _w("atom_fused", 6, enc_blocks=3, dec_blocks=3),
    "atom_attn_exact": _w("atom_attn_exact", 1200, row="atom_exact", modules=6),
}}
