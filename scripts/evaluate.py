import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_model
from src.data.cmapss_dataset import CMAPSSDataset
from src.training.trainer import RULTrainer


def main():
    parser = argparse.ArgumentParser(description='Evaluate a CET-MSA checkpoint on a C-MAPSS subset.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    dataset_config = config['dataset']
    subset = dataset_config['name']
    processed_dir = Path(dataset_config['processed_dir'])
    data_dir = Path(dataset_config['data_dir'])

    trainset = CMAPSSDataset(
        mode='train',
        dataset=processed_dir / f'train_{subset}_normed.txt',
        window_length=dataset_config.get('window_length', 30),
        rul_cap=dataset_config.get('rul_cap', 150),
    )
    testset = CMAPSSDataset(
        mode='test',
        dataset=processed_dir / f'test_{subset}_normed.txt',
        rul_result=data_dir / f'RUL_{subset}.txt',
        feature_stats=trainset.feature_stats,
        window_length=dataset_config.get('window_length', 30),
        rul_cap=dataset_config.get('rul_cap', 150),
    )
    test_loader = DataLoader(dataset=testset, batch_size=config['training'].get('eval_batch_size', 64), shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = RULTrainer(
        model=model,
        model_optimizer=dummy_optimizer,
        print_every=999999,
        epochs=1,
        device=device,
        prefix=f'{subset}_eval',
        rul_cap=dataset_config.get('rul_cap', 150),
    )
    trainer.evaluate_rul_metrics(test_loader)


if __name__ == '__main__':
    main()
