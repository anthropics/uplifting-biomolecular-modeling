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
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="manifest_plddt70",
        help="Path to output manifest file (msgpack).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=70.0,
        help="pLDDT threshold for filtering structures.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data_dir: pathlib.Path = args.data_dir / args.name

    print("Loading metadata...")
    manifest_path: pathlib.Path = data_dir / "manifest.msgpack"
    with open(manifest_path, "rb") as f:
        manifest = msgpack.load(f)
    print(f"Loaded {len(manifest)} entries from {manifest_path}")

    # Filter entries based on pLDDT threshold
    manifest_filtered = [
        entry for entry in manifest if entry["pred"]["plddt"] >= args.threshold
    ]
    for entry in manifest_filtered:
        entry["pred"]["plddt"] = round(
            entry["pred"]["plddt"], 2
        )  # Round pLDDT to 2 decimal places
    print(f"Filtered {len(manifest_filtered)} entries with pLDDT >= {args.threshold}")

    # Save the filtered manifest
    out_path = data_dir / f"{args.out_prefix}.msgpack"
    print(f"Saved manifest (msgpack) to {out_path}")
    with open(out_path, "wb") as f:
        msgpack.dump(manifest_filtered, f)


if __name__ == "__main__":
    main()
