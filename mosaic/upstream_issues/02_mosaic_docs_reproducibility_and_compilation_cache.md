# mosaic doc reproducibility

See README item 2. Evidence: two fresh stock processes at the same seed do not reproduce each other's trajectory (each XLA compile re-autotunes); with the persistent compilation cache + autotune file (the `exact` mode's P1) they do, bitwise.
