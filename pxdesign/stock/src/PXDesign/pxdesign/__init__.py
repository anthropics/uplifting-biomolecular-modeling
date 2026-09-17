import os
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent

if os.environ.get("PROTENIX_DATA_ROOT_DIR") is None:
    os.environ["PROTENIX_DATA_ROOT_DIR"] = str(
        parent_dir / "release_data" / "ccd_cache"
    )

if os.environ.get("TOOL_WEIGHTS_ROOT") is None:
    os.environ["TOOL_WEIGHTS_ROOT"] = str(parent_dir / "tool_weights")
