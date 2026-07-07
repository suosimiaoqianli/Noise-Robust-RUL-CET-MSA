import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.windowing import normalize_cmapss_subset


def main():
    parser = argparse.ArgumentParser(description='Normalize a C-MAPSS subset using train-set min/max statistics.')
    parser.add_argument('--subset', default='FD001', choices=['FD001', 'FD002', 'FD003', 'FD004'])
    parser.add_argument('--data-dir', default='data/CMAPSS')
    parser.add_argument('--processed-dir', default='data/processed')
    args = parser.parse_args()

    train_output, test_output = normalize_cmapss_subset(
        subset=args.subset,
        data_dir=args.data_dir,
        processed_dir=args.processed_dir,
    )
    print(f'[OK] saved {train_output}')
    print(f'[OK] saved {test_output}')


if __name__ == '__main__':
    main()
