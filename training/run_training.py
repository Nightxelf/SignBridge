import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_command(command):
    print("Running:", " ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Prepare WLASL data and train the SignBridge model.")
    parser.add_argument("--json", required=True, help="WLASL metadata JSON file path")
    parser.add_argument("--raw-root", default="training/dataset/raw", help="Directory for raw videos")
    parser.add_argument("--out-root", default="training/dataset/processed", help="Directory for processed landmarks")
    parser.add_argument("--subset", default="all", help="Dataset split to prepare: train/val/test/all")
    parser.add_argument("--max-items", type=int, default=512, help="Maximum number of samples to prepare")
    parser.add_argument("--download", action="store_true", help="Download missing raw videos from URL metadata")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo samples when raw videos are missing")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=512, help="Maximum number of processed samples to train on")
    parser.add_argument("--save-path", default="backend/checkpoints/latest.pt", help="Checkpoint file to save")
    parser.add_argument("--save-label-map", default="backend/label_map.json", help="Label map JSON path for inference")
    parser.add_argument("--num-classes", type=int, default=0, help="Number of target classes (0 to infer)")
    args = parser.parse_args()

    wlasl_prepare = ROOT / "wlasl_prepare.py"
    train_script = ROOT / "train.py"

    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = ROOT.parent / args.json

    if not json_path.exists():
        metadata_dir = ROOT.parent / "training" / "dataset" / "metadata"
        sample_files = []
        if metadata_dir.exists():
            sample_files = [str(p.name) for p in metadata_dir.iterdir() if p.suffix.lower() == ".json"]
        raise FileNotFoundError(
            f"WLASL metadata JSON not found: {json_path}\n"
            f"Place your JSON file in {metadata_dir} or pass the correct --json path."
            + (f"\nAvailable metadata files: {sample_files}" if sample_files else "")
        )

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    if not raw_root.is_absolute():
        raw_root = ROOT.parent / args.raw_root
    if not out_root.is_absolute():
        out_root = ROOT.parent / args.out_root

    prepare_cmd = [
        sys.executable,
        str(wlasl_prepare),
        "--json",
        str(json_path),
        "--raw-root",
        str(raw_root),
        "--out-root",
        str(out_root),
        "--subset",
        args.subset,
        "--max-items",
        str(args.max_items),
    ]
    if args.download:
        prepare_cmd.append("--download")
    if args.demo:
        prepare_cmd.append("--demo")

    run_command(prepare_cmd)

    train_cmd = [
        sys.executable,
        str(train_script),
        "--data-dir",
        args.out_root,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-samples",
        str(args.max_samples),
        "--save-path",
        args.save_path,
        "--save-label-map",
        args.save_label_map,
    ]
    if args.num_classes:
        train_cmd.extend(["--num-classes", str(args.num_classes)])

    run_command(train_cmd)
    print("Data preparation and training completed.")


if __name__ == "__main__":
    main()
