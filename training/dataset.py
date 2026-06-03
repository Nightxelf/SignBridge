import json
import os
import numpy as np
from torch.utils.data import Dataset


class LandmarkDataset(Dataset):
    """Loads pre-extracted .npz files containing `landmarks` arrays.

    Expects each file in `data_dir` to be an `.npz` containing `landmarks` shape (T,63).
    If a `mapping.json` file exists in the data directory, it is used to provide labels.
    Otherwise, a provided label_map dict or JSON file path may be used.
    """
    def __init__(self, data_dir, label_map=None, label_map_path=None, max_samples=None, transform=None):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.files.sort()
        if max_samples is not None:
            self.files = self.files[:max_samples]
        self.transform = transform
        self.label_map = {}

        mapping_path = os.path.join(data_dir, "mapping.json")
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as fp:
                self.label_map = json.load(fp)
        elif os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as fp:
                self.label_map = json.load(fp)
        elif isinstance(label_map, dict):
            self.label_map = label_map

        self.labels = [self.label_map.get(os.path.basename(f), 0) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path)
        landmarks = data["landmarks"].astype("float32")
        if self.transform:
            landmarks = self.transform(landmarks)
        label = int(self.labels[idx])
        return landmarks, label
