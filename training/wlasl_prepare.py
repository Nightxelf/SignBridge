"""WLASL preparation utilities.

This module prepares WLASL samples for model training by converting raw video segments
into landmark sequences using MediaPipe. It supports local raw video directories and
optional video download from WLASL URLs.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from tqdm import tqdm

from backend.mediapipe_handler import MediaPipeHandler


def download_video(youtube_url, out_path, timeout=120):
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise ImportError("yt-dlp is required for video download. Install with `pip install yt-dlp`." ) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path

    opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": str(out_path),
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "socket_timeout": timeout,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([youtube_url])
    return out_path


def load_wlasl_json(json_path):
    with open(json_path, "r", encoding="utf-8") as fp:
        dataset = json.load(fp)
    if isinstance(dataset, dict):
        if "data" in dataset:
            dataset = dataset["data"]
        elif "samples" in dataset:
            dataset = dataset["samples"]
    return dataset


def extract_landmarks_from_segment(video_path, frame_start, frame_end, fps=25, bbox=None):
    mp = MediaPipeHandler()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    start_frame = max(int(frame_start) - 1, 0)
    end_frame = int(frame_end) if frame_end is not None else None

    frames = []
    current = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current < start_frame:
            current += 1
            continue
        if end_frame is not None and current >= end_frame:
            break
        if bbox and len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            xmin = max(0, int(xmin))
            ymin = max(0, int(ymin))
            xmax = min(frame.shape[1], int(xmax))
            ymax = min(frame.shape[0], int(ymax))
            if xmax > xmin and ymax > ymin:
                frame = frame[ymin:ymax, xmin:xmax]
        coords = mp.extract_from_bgr(frame)
        if coords is None:
            coords = np.zeros(63, dtype=np.float32).tolist()
        frames.append(coords)
        current += 1
    cap.release()
    return np.array(frames, dtype=np.float32)


def generate_synthetic_landmarks(item, num_frames=16):
    seed_value = abs(hash(item.get("video_id") or item.get("id") or item.get("sample_id") or item.get("gloss") or "demo")) % (2**32)
    rng = np.random.RandomState(seed_value)
    return rng.normal(loc=0.0, scale=0.5, size=(num_frames, 63)).astype(np.float32)


def prepare_wlasl_dataset(
    json_path,
    raw_root="training/dataset/raw",
    out_root="training/dataset/processed",
    subset="all",
    max_items=None,
    download=False,
    synthetic=False,
):
    json_path = Path(json_path)
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset = load_wlasl_json(json_path)
    processed_files = []
    label_map = {}
    label_counter = 0
    mapping = {}

    for item in tqdm(dataset, desc="Preparing WLASL samples", unit="sample"):
        split = item.get("split")
        if subset != "all" and split != subset:
            continue

        gloss = item.get("gloss") or item.get("label") or "unknown"
        if gloss not in label_map:
            label_map[gloss] = label_counter
            label_counter += 1

        video_id = item.get("video_id") or item.get("id") or item.get("youtube_id")
        if not video_id:
            continue

        url = item.get("url") or item.get("youtube_url")
        if not url:
            continue

        frame_start = item.get("frame_start") or item.get("start_frame")
        frame_end = item.get("frame_end") or item.get("end_frame")
        bbox = item.get("bbox")

        raw_video_path = raw_root / f"{video_id}.mp4"
        if not raw_video_path.exists():
            if download:
                try:
                    download_video(url, raw_video_path)
                except Exception as exc:
                    print(f"Download failed for {video_id}: {exc}")
            if not raw_video_path.exists():
                if synthetic:
                    print(f"Generating synthetic sample for {video_id} ({gloss})")
                    landmarks = generate_synthetic_landmarks(item)
                    filename = f"{video_id}_{gloss}_{item.get('instance_id', 0)}.npz"
                    filename = filename.replace(" ", "_").replace("/", "_")
                    sample_out = out_root / filename
                    np.savez_compressed(sample_out, landmarks=landmarks, gloss=gloss, split=split, source="synthetic")
                    mapping[sample_out.name] = label_map[gloss]
                    processed_files.append(sample_out.name)
                    if max_items and len(processed_files) >= max_items:
                        break
                    continue
                print(f"Raw video missing for {video_id}. Skipping.")
                continue

        filename = f"{video_id}_{gloss}_{item.get('instance_id', 0)}.npz"
        filename = filename.replace(" ", "_").replace("/", "_")
        sample_out = out_root / filename

        try:
            landmarks = extract_landmarks_from_segment(raw_video_path, frame_start, frame_end, bbox=bbox)
        except Exception as exc:
            print(f"Failed to extract segment for {video_id}: {exc}")
            continue

        np.savez_compressed(sample_out, landmarks=landmarks, gloss=gloss, split=split, source=url)
        mapping[sample_out.name] = label_map[gloss]
        processed_files.append(sample_out.name)

        if max_items and len(processed_files) >= max_items:
            break

    with open(out_root / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    inverse_map = {str(v): k for k, v in label_map.items()}
    with open(out_root / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(inverse_map, f, indent=2, ensure_ascii=False)

    with open(out_root / "mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    print(f"Prepared {len(processed_files)} files in {out_root}")
    return processed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare WLASL samples into landmark sequences.")
    parser.add_argument("--json", required=True, help="WLASL JSON annotation file path")
    parser.add_argument("--raw-root", default="training/dataset/raw", help="Directory for raw videos")
    parser.add_argument("--out-root", default="training/dataset/processed", help="Directory for processed landmarks")
    parser.add_argument("--subset", default="all", help="Split to process: train/val/test/all")
    parser.add_argument("--max-items", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--download", action="store_true", help="Download raw videos if missing")
    parser.add_argument("--demo", action="store_true", help="Generate synthetic demo samples when raw videos are missing")
    args = parser.parse_args()
    prepare_wlasl_dataset(
        args.json,
        raw_root=args.raw_root,
        out_root=args.out_root,
        subset=args.subset,
        max_items=args.max_items,
        download=args.download,
        synthetic=args.demo,
    )
