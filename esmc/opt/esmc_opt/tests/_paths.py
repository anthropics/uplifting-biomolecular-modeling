"""Shared paths for the tests: the tree (esmc/), the kits package dir, the stock dir."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                     # esmc_opt/
OPT = os.path.dirname(PKG)                      # opt/
TREE = os.path.dirname(OPT)                     # esmc/
KITS = os.path.join(PKG, "kits")                # esmc_opt/kits — the levers' package dir
STOCK = os.path.join(TREE, "stock")
