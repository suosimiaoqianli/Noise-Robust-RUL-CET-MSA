import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Expected raw C-MAPSS directory: {data_dir}")
    print(f"Processed data will be saved to: {output_dir}")
    print("Raw NASA C-MAPSS files are not included in this repository.")


if __name__ == "__main__":
    main()