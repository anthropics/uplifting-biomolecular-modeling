"""Carried module ``ef2_trimul_v5`` (byte-identical to its kit) + ``ef2_w4_fpf_trimul_v4_cells.json`` (the kit's (cc|triton) launch-cell table for
this line) + ``route`` (the core's plain-torch weight relayout and forward entry: row esm_v5_fwd).  Imports torch and triton only -- no model
package of the ESM-family image is on the import path, so the row loads on any torch stack whose triton has tensor descriptors."""
