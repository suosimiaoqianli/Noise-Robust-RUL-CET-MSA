import numpy as np
import torch


class RULTrainer:
    def __init__(self, model, model_optimizer, print_every, epochs=200, device='cpu', prefix='FD001', rul_cap=150):
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.prefix = prefix
        self.rul_cap = rul_cap

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

    def train(self, train_loader, evaluation_loader, iteration):
        best_score = None
        best_rmse = None
        best_score_path = None
        best_rmse_path = None

        for epoch in range(self.epochs):
            print('Epoch: {}'.format(epoch + 1))
            self.model.train()
            self.train_one_epoch(train_loader)
            current_score, current_rmse = self.evaluate_rul_metrics(evaluation_loader)

            if best_score is None or current_score < best_score:
                best_score = current_score
                best_score_path = self.save_checkpoint(iteration + 1, epoch + 1, 'best_score')
            if best_rmse is None or current_rmse < best_rmse:
                best_rmse = current_rmse
                best_rmse_path = self.save_checkpoint(iteration + 1, epoch + 1, 'best_RMSE')

        return {
            'best_score': float(best_score),
            'best_RMSE': float(best_rmse),
            'best_score_path': best_score_path,
            'best_RMSE_path': best_rmse_path,
        }

    def save_checkpoint(self, iteration, epoch, checkpoint_type):
        import os
        os.makedirs('checkpoints', exist_ok=True)
        state = {
            'iter': iteration,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.model_optimizer.state_dict(),
        }
        path = 'checkpoints/{}_iteration{}_{}.pth.tar'.format(self.prefix, iteration, checkpoint_type)
        torch.save(state, path)
        print('{}_checkpoint saved successfully: {}'.format(checkpoint_type, path))
        return path

    @staticmethod
    def asymmetric_score(y_true, y_pred):
        score = 0.0
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        for i in range(len(y_pred)):
            error = y_pred[i] - y_true[i]
            if error >= 0:
                score = score + np.exp(error / 10.0) - 1
            else:
                score = score + np.exp(-error / 13.0) - 1
        return score

    def evaluate_rul_metrics(self, evaluation_loader):
        score = 0.0
        squared_error = 0.0
        self.model.eval()
        criterion = torch.nn.MSELoss()
        for data in evaluation_loader:
            with torch.no_grad():
                inputs, handcrafted_degradation_features, labels = data
                inputs = inputs.to(self.device)
                handcrafted_degradation_features = handcrafted_degradation_features.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(inputs, handcrafted_degradation_features)
                score += self.asymmetric_score(labels * self.rul_cap, predictions * self.rul_cap)
                squared_error += criterion(labels * self.rul_cap, predictions * self.rul_cap) * len(labels)
        rmse = (squared_error / len(evaluation_loader.dataset)) ** 0.5
        print('evaluation result: Score: {}, RMSE: {}'.format(score.item(), rmse))
        return score.item(), rmse
