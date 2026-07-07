from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from .windowing import (
    SCREENED_CMAPSS_COLUMNS,
    compute_handcrafted_degradation_features,
    interpolate_short_engine_sequence,
)


class CMAPSSDataset(Dataset):
    def __init__(
        self,
        mode='train',
        dataset=None,
        rul_result=None,
        feature_stats=None,
        window_length=30,
        rul_cap=150,
    ):
        self.data = np.loadtxt(fname=Path(dataset), dtype=np.float32)
        self.data = np.delete(self.data, SCREENED_CMAPSS_COLUMNS, axis=1)
        self.window_length = window_length
        self.rul_cap = rul_cap
        self.sample_num = int(self.data[-1][0])
        self.mode = mode

        if self.mode == 'test' and rul_result is not None:
            self.rul_result = np.loadtxt(fname=Path(rul_result), dtype=np.float32)
        if self.mode == 'test' and rul_result is None:
            raise ValueError('rul_result is required when mode="test".')
        if self.mode not in {'train', 'test'}:
            raise ValueError('mode must be "train" or "test".')
        if self.mode == 'train' and rul_result is not None:
            warnings.warn('rul_result is ignored when mode="train".')

        self.windowed_sequences = []
        self.rul_labels = []

        if self.mode == 'train':
            self._build_training_windows()
        else:
            self._build_test_windows()

        self.windowed_sequences = np.array(self.windowed_sequences)
        self.rul_labels = np.array(self.rul_labels) / self.rul_cap
        self.handcrafted_degradation_features = np.array(
            [compute_handcrafted_degradation_features(sample) for sample in self.windowed_sequences]
        )

        if feature_stats is None:
            feature_mean = np.mean(self.handcrafted_degradation_features, axis=0)
            feature_std = np.std(self.handcrafted_degradation_features, axis=0)
        else:
            feature_mean, feature_std = feature_stats
        self.feature_stats = (feature_mean, feature_std)
        self.handcrafted_degradation_features = (
            self.handcrafted_degradation_features - feature_mean
        ) / (feature_std + 1e-10)

    def _build_training_windows(self):
        for engine_id in range(1, self.sample_num + 1):
            engine_indices = np.where(self.data[:, 0] == engine_id)[0]
            engine_data = self.data[engine_indices, :]
            for start_index in range(len(engine_data) - self.window_length + 1):
                self.windowed_sequences.append(engine_data[start_index:start_index + self.window_length, 2:])
                rul = len(engine_data) - self.window_length - start_index
                self.rul_labels.append(min(rul, self.rul_cap))

    def _build_test_windows(self):
        for engine_id in range(1, self.sample_num + 1):
            engine_indices = np.where(self.data[:, 0] == engine_id)[0]
            engine_data = self.data[engine_indices, :]
            if len(engine_data) < self.window_length:
                engine_data = interpolate_short_engine_sequence(engine_data, self.window_length)
            self.windowed_sequences.append(engine_data[-self.window_length:, 2:])
            self.rul_labels.append(min(self.rul_result[engine_id - 1], self.rul_cap))

    def __len__(self):
        return len(self.rul_labels)

    def __getitem__(self, index):
        sequence_tensor = torch.from_numpy(self.windowed_sequences[index]).to(torch.float32)
        feature_tensor = torch.from_numpy(self.handcrafted_degradation_features[index]).to(torch.float32)
        label_tensor = torch.Tensor([self.rul_labels[index]]).to(torch.float32)
        return sequence_tensor, feature_tensor, label_tensor
