"""Training script for LSTM model.

This script trains the sign language model on preprocessed landmark data.
It supports small-sample training via `--max-samples` and prints progress for each epoch.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from backend.models.lstm_model import SignLSTM
from training.dataset import LandmarkDataset


def load_label_map(data_dir, label_map_path=None):
    data_dir = Path(data_dir)
    if label_map_path:
        path = Path(label_map_path)
    else:
        path = data_dir / "label_map.json"

    def parse_map(loaded):
        if not isinstance(loaded, dict):
            return {}
        if all(str(k).isdigit() for k in loaded.keys()):
            return {int(k): str(v) for k, v in loaded.items()}
        if all(isinstance(v, int) or (isinstance(v, str) and v.isdigit()) for v in loaded.values()):
            inverted = {}
            for k, v in loaded.items():
                try:
                    label_id = int(v)
                except ValueError:
                    continue
                inverted[label_id] = str(k)
            return inverted
        return {}

    if path.exists():
        with open(path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        parsed = parse_map(loaded)
        if parsed:
            return parsed

    names_path = data_dir / "label_names.json"
    if names_path.exists():
        with open(names_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        parsed = parse_map(loaded)
        if parsed:
            return parsed

    mapping_path = data_dir / "mapping.json"
    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as fp:
            mapping = json.load(fp)
        label_ids = sorted({int(v) for v in mapping.values() if str(v).isdigit()})
        return {label_id: str(label_id) for label_id in label_ids}

    return {}


def infer_num_classes(label_map, num_classes, fallback_ids=None):
    if num_classes and num_classes > 0:
        return num_classes
    label_ids = []
    if label_map:
        label_ids.extend(label_map.keys())
    if fallback_ids:
        label_ids.extend(fallback_ids)
    if label_ids:
        return max(label_ids) + 1
    return 4


def pad_sequences(seqs, max_len=None, pad_value=0.0):
    if not seqs:
        return np.zeros((0, 0, 0), dtype=np.float32)

    seq_lengths = [seq.shape[0] for seq in seqs]
    feature_dims = [seq.shape[1] for seq in seqs]
    max_len = max_len or max(seq_lengths)
    feature_dim = max(feature_dims)

    padded = np.full((len(seqs), max_len, feature_dim), pad_value, dtype=np.float32)
    for i, seq in enumerate(seqs):
        length = min(seq.shape[0], max_len)
        padded[i, :length, : seq.shape[1]] = seq[:length]
    return padded


def collate_batch(batch):
    seqs, labels = zip(*batch)
    seqs = pad_sequences(seqs)
    labels = np.array(labels, dtype=np.int64)
    return seqs, labels


def train(
    data_dir=None,
    epochs=3,
    batch_size=8,
    num_classes=0,
    save_path='backend/checkpoints/latest.pt',
    max_samples=128,
    use_synthetic=False,
    mapping_path=None,
    label_map_path=None,
    save_label_map='backend/label_map.json',
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map = {}
    if data_dir and os.path.isdir(data_dir) and not use_synthetic:
        label_map = load_label_map(data_dir, label_map_path=label_map_path)
        ds = LandmarkDataset(data_dir, label_map_path=mapping_path, max_samples=max_samples)
        print(f"Loaded dataset from {data_dir} with {len(ds)} samples")

        if len(ds) == 0:
            print("No .npz files found in the dataset directory. Falling back to synthetic data.")
            use_synthetic = True
        else:
            actual_labels = [int(l) for l in ds.labels if str(l).isdigit() or isinstance(l, (int, np.integer))]
            if actual_labels:
                num_classes = infer_num_classes(label_map, num_classes, fallback_ids=actual_labels)
            else:
                num_classes = infer_num_classes(label_map, num_classes)
            print(f"Inferred num_classes={num_classes} from dataset labels and label map")
    else:
        use_synthetic = True

    if use_synthetic:
        if max_samples is None:
            max_samples = 64
        num_classes = max(4, num_classes or 4)

        class SynthDataset:
            def __init__(self, length, num_classes):
                self.length = length
                self.num_classes = num_classes

            def __len__(self):
                return self.length

            def __getitem__(self, idx):
                seq = np.random.randn(16, 63).astype(np.float32)
                label = np.random.randint(0, self.num_classes)
                return seq, int(label)

        ds = SynthDataset(max_samples, num_classes)
        print(f"Using synthetic dataset with {len(ds)} samples and num_classes={num_classes}")

    dl = DataLoader(ds, batch_size=batch_size, collate_fn=collate_batch)
    model = SignLSTM(input_size=63, hidden_size=128, num_layers=2, num_classes=num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Starting training: epochs={epochs}, batch_size={batch_size}, num_classes={num_classes}, max_samples={max_samples}")
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for batch_idx, (seqs, labels) in enumerate(dl, start=1):
            seqs_t = torch.from_numpy(seqs).to(device)
            labels_t = torch.from_numpy(labels).to(device)
            logits = model(seqs_t)
            loss = loss_fn(logits, labels_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1
            if batch_idx % 5 == 0 or batch_idx == len(dl):
                print(f"  Epoch {ep+1}/{epochs} batch {batch_idx}/{len(dl)} loss={loss.item():.4f}")

        avg = total_loss / max(1, steps)
        print(f"Epoch {ep+1}/{epochs} completed. avg_loss={avg:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Saved checkpoint to {save_path}")

    if label_map and save_label_map:
        save_path_obj = Path(save_label_map)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path_obj, "w", encoding="utf-8") as fp:
            json.dump({str(k): v for k, v in label_map.items()}, fp, indent=2, ensure_ascii=False)
        print(f"Saved label map to {save_path_obj}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the SignBridge LSTM model.")
    parser.add_argument("--data-dir", default="training/dataset/processed", help="Path to processed landmark dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num-classes", type=int, default=0, help="Number of target classes (infer from label map if unset)")
    parser.add_argument("--save-path", default="backend/checkpoints/latest.pt", help="Where to save the checkpoint")
    parser.add_argument("--max-samples", type=int, default=128, help="Maximum number of processed samples to train on")
    parser.add_argument("--use-synthetic", action="store_true", help="Force training on synthetic data")
    parser.add_argument("--mapping-path", default=None, help="Path to mapping.json for training labels")
    parser.add_argument("--label-map-path", default=None, help="Path to label_map.json for class label names")
    parser.add_argument("--save-label-map", default="backend/label_map.json", help="Where to save label map for inference")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        save_path=args.save_path,
        max_samples=args.max_samples,
        use_synthetic=args.use_synthetic,
        mapping_path=args.mapping_path,
        label_map_path=args.label_map_path,
        save_label_map=args.save_label_map,
    )


