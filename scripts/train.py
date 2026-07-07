import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch import optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cmapss_dataset import CMAPSSDataset
from src.models.cet_msa import CETMSARULPredictor
from src.training.trainer import RULTrainer
from src.utils.seed import set_random_seed


def build_model(config):
    model_config = config['model']
    dataset_config = config['dataset']
    return CETMSARULPredictor(
        input_dim=model_config.get('input_dim', 17),
        hidden_dim=model_config.get('hidden_dim', 50),
        window_length=dataset_config.get('window_length', 30),
        heads=model_config.get('heads', 8),
        dropout=model_config.get('dropout', 0.2),
        tlce_groups=model_config.get('tlce_groups', 5),
        pyramidal_ratio=model_config.get('pyramidal_ratio', 2),
    )


def build_loaders(config):
    dataset_config = config['dataset']
    training_config = config['training']
    subset = dataset_config['name']
    processed_dir = Path(dataset_config['processed_dir'])
    data_dir = Path(dataset_config['data_dir'])
    window_length = dataset_config.get('window_length', 30)
    rul_cap = dataset_config.get('rul_cap', 150)

    trainset = CMAPSSDataset(
        mode='train',
        dataset=processed_dir / f'train_{subset}_normed.txt',
        window_length=window_length,
        rul_cap=rul_cap,
    )
    testset = CMAPSSDataset(
        mode='test',
        dataset=processed_dir / f'test_{subset}_normed.txt',
        rul_result=data_dir / f'RUL_{subset}.txt',
        feature_stats=trainset.feature_stats,
        window_length=window_length,
        rul_cap=rul_cap,
    )
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=training_config.get('batch_size', 100),
        shuffle=True,
        num_workers=training_config.get('num_workers', 2),
    )
    test_loader = DataLoader(
        dataset=testset,
        batch_size=training_config.get('eval_batch_size', 64),
        shuffle=False,
        num_workers=training_config.get('num_workers', 2),
    )
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train CET-MSA on a C-MAPSS subset.')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    set_random_seed(config['training'].get('seed', 0))
    train_loader, test_loader = build_loaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_scores = []
    best_rmses = []
    for iteration in range(args.iterations):
        model = build_model(config)
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training'].get('lr', 1e-3),
            weight_decay=config['training'].get('weight_decay', 0.0),
        )
        trainer = RULTrainer(
            model=model,
            model_optimizer=optimizer,
            print_every=config['training'].get('print_every', 50),
            epochs=config['training'].get('epochs', 50),
            device=device,
            prefix=config['dataset']['name'],
            rul_cap=config['dataset'].get('rul_cap', 150),
        )
        result = trainer.train(train_loader, test_loader, iteration)
        best_scores.append(result['best_score'])
        best_rmses.append(result['best_RMSE'])

    print('Best Score values:', best_scores)
    print('Best RMSE values:', best_rmses)


if __name__ == '__main__':
    main()
