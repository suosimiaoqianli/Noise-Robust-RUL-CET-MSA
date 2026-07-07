import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_model
from src.data.cmapss_dataset import CMAPSSDataset
from src.training.losses import TemporalSmoothnessRegularizer
from src.training.trainer import RULTrainer
from src.utils.seed import set_random_seed


class DegradationAwareTransferTrainer(RULTrainer):
    def __init__(self, *args, regularizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.regularizer = regularizer

    def train_one_epoch(self, dataloader):
        running_loss = 0.0
        length = len(dataloader)
        for batch_index, data in enumerate(dataloader, 0):
            inputs, handcrafted_degradation_features, labels = data
            inputs = inputs.to(self.device)
            handcrafted_degradation_features = handcrafted_degradation_features.to(self.device)
            labels = labels.to(self.device)

            self.model_optimizer.zero_grad()
            predictions = self.model(inputs, handcrafted_degradation_features)
            loss = self.criterion(predictions, labels)
            if self.regularizer is not None:
                loss = loss + self.regularizer.penalty(self.model)
            running_loss += loss.item()
            loss.backward()
            self.model_optimizer.step()

            if (batch_index + 1) % self.print_every == 0:
                print(
                    'batch:{}/{}, loss(avg. on {} batches): {}'.format(
                        batch_index + 1,
                        length,
                        self.print_every,
                        running_loss / self.print_every,
                    )
                )
                running_loss = 0.0


def freeze_temporal_local_context_enhancement(model, freeze=True):
    for parameter in model.cet_msa.temporal_local_context_enhancement.parameters():
        parameter.requires_grad_(not freeze)


def build_transfer_loaders(config):
    dataset_config = config['dataset']
    training_config = config['training']
    target_subset = dataset_config['target']
    processed_dir = Path(dataset_config['processed_dir'])
    data_dir = Path(dataset_config['data_dir'])

    trainset = CMAPSSDataset(
        mode='train',
        dataset=processed_dir / f'train_{target_subset}_normed.txt',
        window_length=dataset_config.get('window_length', 30),
        rul_cap=dataset_config.get('rul_cap', 150),
    )
    testset = CMAPSSDataset(
        mode='test',
        dataset=processed_dir / f'test_{target_subset}_normed.txt',
        rul_result=data_dir / f'RUL_{target_subset}.txt',
        feature_stats=trainset.feature_stats,
        window_length=dataset_config.get('window_length', 30),
        rul_cap=dataset_config.get('rul_cap', 150),
    )
    train_loader = DataLoader(dataset=trainset, batch_size=training_config.get('batch_size', 100), shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=testset, batch_size=training_config.get('eval_batch_size', 64), shuffle=False, num_workers=2)
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Degradation-aware transfer learning for CET-MSA.')
    parser.add_argument('--config', default='configs/transfer.yaml')
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    set_random_seed(config['training'].get('seed', 0))
    train_loader, test_loader = build_transfer_loaders(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_scores = []
    best_rmses = []
    source_checkpoint = config['training']['source_checkpoint']
    output_prefix = config['training'].get('output_prefix', 'degradation_aware_transfer')

    for iteration in range(args.iterations):
        model_config = {
            'dataset': dict(config['dataset'], name=config['dataset']['target']),
            'model': config['model'],
        }
        model = build_model(model_config)
        checkpoint = torch.load(source_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)

        freeze_epochs = config['training'].get('freeze_tlce_epochs', 0)
        freeze_temporal_local_context_enhancement(model, freeze=freeze_epochs > 0)
        regularizer = TemporalSmoothnessRegularizer(model, regularization_weight=1e-5)
        optimizer = optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=config['training'].get('lr', 1e-4))
        trainer = DegradationAwareTransferTrainer(
            model=model,
            model_optimizer=optimizer,
            print_every=config['training'].get('print_every', 50),
            epochs=config['training'].get('epochs', 50),
            device=device,
            prefix=output_prefix,
            regularizer=regularizer,
            rul_cap=config['dataset'].get('rul_cap', 150),
        )
        result = trainer.train(train_loader, test_loader, iteration)
        best_scores.append(result['best_score'])
        best_rmses.append(result['best_RMSE'])

    print('Transfer Score values:', best_scores)
    print('Transfer RMSE values:', best_rmses)
    np.savetxt(f'{output_prefix}_result.txt', np.array([best_scores, best_rmses]), fmt='%.4f')


if __name__ == '__main__':
    main()
