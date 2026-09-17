"""Modes exact | fast: the kit arm — `python -s -m colabdesign_opt.kit_launch --pins ... --mode M [...] -- <stock_design.py args>` (launch.py)."""
import sys

from .launch import main

if __name__ == "__main__":
    sys.exit(main("kit"))
