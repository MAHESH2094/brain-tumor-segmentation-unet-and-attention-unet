import math
import threading

import h5py
import numpy as np
import tensorflow as tf


class HDF5Generator(tf.keras.utils.Sequence):
    def __init__(
        self,
        hdf5_path,
        split,
        batch_size,
        shuffle=True,
        target_mode="segmentation",
        classification_num_classes=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hdf5_path = hdf5_path
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.target_mode = str(target_mode).strip().lower()
        self.classification_num_classes = int(classification_num_classes)
        if self.target_mode == "classification" and self.classification_num_classes not in {1, 2}:
            raise ValueError("classification_num_classes must be 1 or 2 when target_mode='classification'.")
        self.file = None
        self.lock = threading.Lock()
        with h5py.File(hdf5_path, "r") as f:
            self.n = int(f[f"{split}/images"].shape[0])
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return math.ceil(self.n / self.batch_size)

    def _open(self):
        with self.lock:
            if self.file is None:
                self.file = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx):
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Batch index {idx} out of range")
        self._open()
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n)
        batch_indices = np.sort(self.indices[start:end])
        with self.lock:
            x = self.file[f"{self.split}/images"][batch_indices].astype(np.float32)
            y = self.file[f"{self.split}/masks"][batch_indices].astype(np.float32)

        if self.target_mode == "classification":
            has_tumor = np.max(y, axis=(1, 2, 3)) > 0.5
            if self.classification_num_classes == 1:
                y = has_tumor.astype(np.float32).reshape((-1, 1))
            else:
                y = has_tumor.astype(np.int32)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def close(self):
        if self.file is not None:
            try:
                self.file.close()
            except Exception:
                pass
            self.file = None

    def __del__(self):
        self.close()

