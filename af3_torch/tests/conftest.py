"""The tree tests import the package without an install: opt/ on the path (the package tests' conftest does the same for their dir)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opt"))
