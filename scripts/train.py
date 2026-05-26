import argparse
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("Loaded config:", args.config)
    print("Dataset:", config["dataset"]["name"])
    print("Model:", config["model"]["name"])
    print("Training pipeline placeholder for CET-MSA.")


if __name__ == "__main__":
    main()