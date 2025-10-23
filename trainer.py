from typing import Tuple

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from data_loader import Loader


class Trainer():

    def __init__(
        self,
        model: nn.Module,
        optimizer: T.optim.Optimizer,
        data_loader: Loader,
        device: T.device,
        permute_interval: int = 5000,
    ):

        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.permute_interval = permute_interval
        self.num_steps = 1
        self.num_epochs = 1


    def train_epoch(
        self,
    ):

        # Toggle model to train mode
        self.model.train()

        # Load batch using DataLoader
        progress_bar = tqdm(
            self.data_loader.train_loader,
            desc=f"Epoch {self.num_epochs}"
        )
        for data, target in progress_bar:

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)  # Negative Loss Likelihood
            loss.backward()  # Calculate gradients
            self.optimizer.step()  # update weights

            # TODO: to tensorboard
            progress_bar.set_postfix({
                'Loss': f"{loss.detach().cpu().item():.4f}",
                'Num Steps': self.num_steps,
            })
            self.num_steps += 1

            # Permute every self.permute_interval step
            if self.num_steps % self.permute_interval == 0:
                self.data_loader.permute()

        self.num_epochs += 1

    def test(
        self
    ) -> Tuple[float, float]:

        # Toggle model to evaluation mode
        self.model.eval()
        test_loss = 0
        correct = 0
        progress_bar = tqdm(
            self.data_loader.test_loader,
            desc=f"Test"
        )

        len_test = len(self.data_loader.test_loader.dataset)

        with T.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                # Count the number of correct prediction
                correct += pred.eq(target.view_as(pred)).sum().item()

                accuracy = 100. * correct / len_test
                progress_bar.set_postfix({
                    'Average Loss': f"{float(test_loss):.4f}",
                    'Accuracy': f"{correct}/{len_test} ({accuracy:.2f}%)"
                })

        test_loss /= len_test

        return test_loss, accuracy
