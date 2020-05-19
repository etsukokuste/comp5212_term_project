import torch
import numpy as np
import os
from tqdm import tqdm

INF = 1.0e8


class Trainer():
    '''
    The trainer class for training models.
    '''
    def __init__(self,
                 model,
                 epochs,
                 dataloaders,
                 criterion,
                 optimizer,
                 device,
                 print_iter,
                 patience,
                 log_file,
                 save=False):

        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.print_iter = print_iter
        self.patience = patience
        self.patience_cnt = 0
        self.log_file = log_file
        self.save = save

        # Evaluation results
        self.best_train_epoch = 0
        self.best_train_loss = 0
        self.best_train_acc = INF
        self.best_val_epoch = 0
        self.best_val_loss = 0
        self.best_val_acc = INF
        self.best_model = model
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            print('Training...')
            self.train_one_epoch(epoch)
            print('Validating...')
            self.evaluate(epoch)
            print('=' * 20)

        self.save_logging()
        print(f'Best train results: epoch {self.best_train_epoch}, loss {self.best_train_loss}')
        print(f'Best valid results: epoch {self.best_val_epoch}, loss {self.best_val_loss}')
        print('=' * 20)
        self.save_model(self.save)
        return self.best_model

    def train_one_epoch(self, epoch):
        self.model.train()
        dataloader = self.dataloaders['train']
        loss, acc = 0, 0
        iters_per_epoch = 0
        for iteration, (h_re, h_im, snr, pos) in enumerate(dataloader):
            iters_per_epoch += 1

            h_re = h_re.to(device=self.device).float()
            h_im = h_im.to(device=self.device).float()
            snr = snr.to(device=self.device).float()
            pos = pos.to(device=self.device).float()

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                pred = self.model(h_re, h_im, snr)
                _loss = self.criterion(pred, pos)
                loss += _loss.item()
                _acc = torch.dist(pred, pos, 2) / pred.shape[0]
                acc += _acc.item()

                # Backward
                _loss.backward()
                self.optimizer.step()
                if iteration % self.print_iter == 0:
                    print(f'Iteration {iteration}: loss = {_loss:.4f} acc = {_acc:.4f}')

        loss /= iters_per_epoch
        acc /= iters_per_epoch
        if self.best_train_acc > acc:
            self.best_train_loss = loss
            self.best_train_epoch = epoch
            self.best_train_acc = acc

        self.printing(loss, acc)
        self.train_losses.append(loss)

    def evaluate(self, epoch):
        self.model.eval()
        dataloader = self.dataloaders['valid']
        loss, acc = 0, 0
        iters_per_epoch = 0
        for iteration, (h_re, h_im, snr, pos) in enumerate(dataloader):
            iters_per_epoch += 1

            h_re = h_re.to(device=self.device).float()
            h_im = h_im.to(device=self.device).float()
            snr = snr.to(device=self.device).float()
            pos = pos.to(device=self.device).float()

            with torch.set_grad_enabled(False):
                pred = self.model(h_re, h_im, snr)
                _loss = self.criterion(pred, pos)
                loss += _loss.item()
                _acc = torch.dist(pred, pos, 2) / pred.shape[0]
                acc += _acc.item()

        loss /= iters_per_epoch
        acc /= iters_per_epoch
        self.patience_cnt += 1
        if self.best_val_acc > acc:
            self.best_val_acc = acc
            self.best_val_loss = loss
            self.best_val_epoch = epoch
            self.best_model = self.model
            self.patience_cnt = 0

        self.printing(loss, acc)

        self.val_losses.append(loss)

        if self.patience_cnt >= self.patience:
            print(f'Epoch {epoch} out of patience!')
            print(f'Best train results: epoch {self.best_train_epoch}, loss {self.best_train_loss}, acc {self.best_train_acc}')
            print(f'Best valid results: epoch {self.best_val_epoch}, loss {self.best_val_loss}. acc {self.best_val_acc}')
            self.test(self.best_model, save_pred=True)
            self.save_model(True)
            self.save_logging()
            exit(1)

    def printing(self, loss, acc):
        print(f'Loss={loss:.4f} Acc={acc:.4f}')

    def save_logging(self):
        filename = 'save/log/' + self.log_file + '.txt'
        logging = np.array([self.train_losses, self.train_accs, self.val_losses, self.val_accs], dtype=object)
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        np.savetxt(filename, logging, fmt='%s')

    def test(self, model, save_pred=False):
        model.eval()
        dataloader = self.dataloaders['test']
        loss, acc = 0, 0
        iters_per_epoch = 0
        preds = np.zeros((0, 3))
        for iteration, (h_re, h_im, snr, pos) in enumerate(dataloader):
            iters_per_epoch += 1

            h_re = h_re.to(device=self.device).float()
            h_im = h_im.to(device=self.device).float()
            snr = snr.to(device=self.device).float()
            pos = pos.to(device=self.device).float()

            with torch.set_grad_enabled(False):
                pred = self.model(h_re, h_im, snr)
                preds = np.concatenate((preds, pred.cpu().detach().numpy()))
                _loss = self.criterion(pred, pos)
                loss += _loss.item()
                _acc = torch.dist(pred, pos, 2) / pred.shape[0]
                acc += _acc.item()

        acc /= iters_per_epoch
        loss /= iters_per_epoch
        print(f'Test on test set: loss {loss} acc {acc}')
        print('=' * 20)
        if save_pred:
            preds = np.array(preds)
            filename = 'save/pred/' + self.log_file + '_pred.npy'
            np.save(filename, preds)

    def save_model(self, save=False):
        if save:
            filename = 'save/model/' + self.log_file + '.pt'
            dirname = os.path.dirname(filename)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            torch.save(self.best_model.state_dict(), filename)
        else:
            pass

    def load_model_and_test(self, model_path):
        """function to load trained model from given file path and test on test set"""
        self.model.load_state_dict(torch.load(model_path))
        self.test(model=self.model, save_pred=True)

    def self_train(self, model_path):
        # self.model.load_state_dict(torch.load(model_path))
        # # label unlabelled data
        # self.model.eval()
        # dataloader = self.unlabelled_dataloader
        # preds = np.zeros((0, 3))
        # for iteration, (h_re, h_im, snr) in enumerate(dataloader):

        #     h_re = h_re.to(device=self.device).float()
        #     h_im = h_im.to(device=self.device).float()
        #     snr = snr.to(device=self.device).float()

        #     with torch.set_grad_enabled(False):
        #         pred = self.model(h_re, h_im, snr)
        #         preds = np.concatenate((preds, pred.cpu().detach().numpy()))
        
        # preds = np.array(preds)
        # filename = 'save/pred/unlabelled_pred.npy'
        # np.save(filename, preds)
        self.model.load_state_dict(torch.load(model_path))
        self.log_file = self.log_file + '_selftrain_'
        self.train()