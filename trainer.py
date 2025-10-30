from typing import Tuple
import pickle
from os import makedirs

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from data_loader import Loader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: T.optim.Optimizer,
        data_loader: Loader,
        device: T.device,
    ):

        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.device = device
        self.num_steps = 1
        self.num_epochs = 1

        self.criterion = nn.CrossEntropyLoss()

    def train_steps(
        self,
        steps: int,
        seed: int,
        writer: SummaryWriter,
        writer_tag: str,
        dataset_name: str,
    ):

        # Toggle model to train mode
        self.model.train()

        track = {"average online accuracy": []}
        correct = 0
        losses = []
        while steps > 0:

            # Load batch using DataLoader
            progress_bar = tqdm(
                self.data_loader.train_loader, desc=f"Epoch {self.num_epochs}"
            )
            for data, target in progress_bar:

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)  # Negative Loss Likelihood
                loss.backward()  # Calculate gradients
                self.optimizer.step()  # update weights

                losses.append(loss.detach().cpu().item())

                self.num_steps += 1

                steps -= 1  # count step
                if steps <= 0:
                    break

                # =======================
                # Average Online Accuracy
                # =======================

                pred = output.argmax(dim=1, keepdim=True)
                # Count the number of correct prediction
                correct += pred.eq(target.view_as(pred)).sum().item()
                average_online_accuracy = correct / self.num_steps
                track["average online accuracy"].append(average_online_accuracy)

                progress_bar.set_postfix(
                    {
                        "Avg Loss": f"{sum(losses)/len(losses):.2f}",
                        "Online Avg Acc": f"{average_online_accuracy:.2f}",
                        "Num Steps": self.num_steps,
                    }
                )

                # Tensorboard
                writer.add_scalar(
                    f"Online Average Accuracy/{writer_tag}",
                    average_online_accuracy,
                    self.num_steps,
                )

                # =======================

            self.num_epochs += 1

        makedirs(f"./results/{dataset_name}", exist_ok=True)
        with open(f"./results/{dataset_name}/{writer_tag}_{seed}.pkl", "wb") as file:
            pickle.dump(track, file)

    def train_epoch(
        self,
    ):

        # Toggle model to train mode
        self.model.train()

        # Load batch using DataLoader
        progress_bar = tqdm(
            self.data_loader.train_loader, desc=f"Epoch {self.num_epochs}"
        )

        for data, target in progress_bar:

            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)  # Negative Loss Likelihood
            loss.backward()  # Calculate gradients
            self.optimizer.step()  # update weights

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.detach().cpu().item():.4f}",
                    "Num Steps": self.num_steps,
                }
            )
            self.num_steps += 1

        self.num_epochs += 1

    def test(self) -> Tuple[float, float]:

        # Toggle model to evaluation mode
        self.model.eval()
        test_loss = 0
        correct = 0
        progress_bar = tqdm(self.data_loader.test_loader, desc=f"Test")

        len_test = len(self.data_loader.test_loader.dataset)

        with T.no_grad():
            for data, target in progress_bar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                # Count the number of correct prediction
                correct += pred.eq(target.view_as(pred)).sum().item()

                accuracy = 100.0 * correct / len_test
                progress_bar.set_postfix(
                    {
                        "Average Loss": f"{float(test_loss):.4f}",
                        "Accuracy": f"{correct}/{len_test} ({accuracy:.2f}%)",
                    }
                )

        test_loss /= len_test

        return test_loss, accuracy
