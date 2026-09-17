"""Build hooks the way pip's in-process hook runner calls them: a script, the kit directory (backend-path) first on sys.path,
the hooks called from the kit directory. Usage: python _build_driver.py <kit dir> <wheel dir>."""
import os
import sys

kit, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
os.chdir(kit)
sys.path.insert(0, kit)
import _build_backend as backend  # noqa: E402

print("WHEEL:" + backend.build_wheel(out))
print("WHEEL:" + backend.build_editable(out))
