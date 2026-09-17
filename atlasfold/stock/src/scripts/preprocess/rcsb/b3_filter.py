import argparse
import pathlib

import msgpack


def parse_args():
    parser = argparse.ArgumentParser(description="Construct synthetic data set.")
    parser.add_argument(
        "--data_dir",
        type=pathlib.Path,
        required=True,
        help="Path to the preprocessed data directory.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="manifest_confidence",
        help="Path to output manifest file (msgpack).",
    )
    parser.add_argument(
        "--min_resolution",
        type=float,
        default=0.1,
        help="Minimum resolution threshold for filtering structures.",
    )
    parser.add_argument(
        "--max_resolution",
        type=float,
        default=3.0,
        help="Maximum resolution threshold for filtering structures.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / "rcsb/"

    print("Loading metadata...")
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        manifest = msgpack.load(f)
    print(f"Loaded {len(manifest)} entries from {manifest_path}")

    # Filter entries based on resolution
    def filter_fn(entry):
        resolution = entry["exp"].get("resolution")
        return (
            resolution is not None
            and args.min_resolution <= resolution <= args.max_resolution
        )

    manifest_filtered = [entry for entry in manifest if filter_fn(entry)]
    print(
        f"Filtered {len(manifest_filtered)} entries with resolution between "
        f"{args.min_resolution} and {args.max_resolution}"
    )

    # Save the filtered manifest
    out_path = data_dir / f"{args.out_prefix}.msgpack"
    print(f"Saved manifest (msgpack) to {out_path}")
    with open(out_path, "wb") as f:
        msgpack.dump(manifest_filtered, f)


if __name__ == "__main__":
    main()
