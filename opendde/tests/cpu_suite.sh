#!/bin/sh
# The CPU conformance suite's route (opt/opendde_opt/tests/test_registry_kits.py checks this file's shape).
# Run from opendde/ in a fresh virtual environment of Python 3.11 or 3.12.
set -e
pip install 'torch==2.7.1' --index-url https://download.pytorch.org/whl/cpu
pip install -e ../common/opt_core -e 'opt[test]'
python -m pytest opt/opendde_opt/tests "$@"
