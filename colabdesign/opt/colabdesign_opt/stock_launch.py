"""Mode off: the stock arm — `python -s -m colabdesign_opt.stock_launch --pins ... -- <stock_design.py args>` (launch.py)."""
import sys

from .launch import main

if __name__ == "__main__":
    sys.exit(main("stock"))
