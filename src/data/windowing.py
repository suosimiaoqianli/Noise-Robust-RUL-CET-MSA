from pathlib import Path

import numpy as np


SCREENED_CMAPSS_COLUMNS = [5, 9, 10, 14, 20, 22, 23]


def normalize_cmapss_subset(subset, data_dir='data/CMAPSS', processed_dir='data/processed'):
    data_dir = Path(data_dir)
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.loadtxt(data_dir / f'train_{subset}.txt', dtype=np.float32)
    test_data = np.loadtxt(data_dir / f'test_{subset}.txt', dtype=np.float32)

    train_prefix = train_data[:, [0, 1]]
    test_prefix = test_data[:, [0, 1]]
    train_inputs = train_data[:, 2:]
    test_inputs = test_data[:, 2:]

    train_min = np.min(train_inputs, axis=0)
    train_max = np.max(train_inputs, axis=0)
    eps = 1e-12

    normalized_train = (train_inputs - train_min) / (train_max - train_min + eps)
    normalized_test = (test_inputs - train_min) / (train_max - train_min + eps)

    train_output = processed_dir / f'train_{subset}_normed.txt'
    test_output = processed_dir / f'test_{subset}_normed.txt'
    np.savetxt(train_output, np.concatenate((train_prefix, normalized_train), axis=1), fmt='%f')
    np.savetxt(test_output, np.concatenate((test_prefix, normalized_test), axis=1), fmt='%f')
    return train_output, test_output


def interpolate_short_engine_sequence(engine_data, window_length):
    interpolated = np.zeros((window_length, engine_data.shape[1]))
    for column_index in range(engine_data.shape[1]):
        old_x = np.linspace(0, len(engine_data) - 1, len(engine_data), dtype=np.float64)
        slope, intercept = np.polyfit(old_x, engine_data[:, column_index].flatten(), deg=1)
        new_x = np.linspace(0, window_length - 1, window_length, dtype=np.float64)
        interpolated[:, column_index] = new_x * len(engine_data) / window_length * slope + intercept
    return interpolated


def compute_handcrafted_degradation_features(window):
    features = []
    time_index = np.array(range(window.shape[0]))
    for sensor_index in range(window.shape[1]):
        features.append(np.mean(window[:, sensor_index]))
        features.append(np.polyfit(time_index.flatten(), window[:, sensor_index].flatten(), deg=1)[0])
    return features
