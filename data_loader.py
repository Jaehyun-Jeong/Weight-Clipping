from typing import Tuple, Dict
import pickle

import numpy as np
from PIL import Image
import torch as T
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset


class Loader:
    def __init__(
        self,
    ):
        raise NotImplementedError

    def permute(self):
        raise NotImplementedError


def loader_selector(
    name: str,
    no_accel: bool,
    permute_interval: int,
    train_batch_size: int,
    test_batch_size: int,
) -> Tuple[Loader, int, int]:

    if name == "IP-MNIST":
        loader = IP_MNIST_Loader(
            no_accel=no_accel,
            permute_interval=permute_interval,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
    elif name == "LP-EMNIST":
        loader = LP_EMNIST_Loader(
            no_accel=no_accel,
            permute_interval=permute_interval,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
    elif name == "LP-MINI-IMAGENET":
        loader = LP_MINI_IMAGENET_Loader(
            no_accel=no_accel,
            permute_interval=permute_interval,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
        )
    else:
        raise ValueError(f"No dataset named {name}")

    n_inputs = loader.n_inputs
    n_outputs = loader.n_outputs

    return loader, n_inputs, n_outputs


class Permutation:
    def __init__(
        self,
        permute_size: int,
        permute_interval: int,
    ):

        self.permute_size = permute_size
        self.__reset_permutation()

        self.num_step = 0
        self.permute_interval = permute_interval

    def __call__(
        self,
        x: T.tensor,  # 1d tensor
    ):

        self.num_step += 1
        permuted_x = x[self.permutation_idx]

        if self.num_step % self.permute_interval == 0:
            self.__reset_permutation()

        return permuted_x

    def __reset_permutation(
        self,
    ):
        self.permutation_idx = T.randperm(self.permute_size)


class IP_MNIST_Loader(Loader):
    def __init__(
        self,
        no_accel: bool,
        permute_interval: int,
        train_batch_size: int,
        test_batch_size: int,
    ):

        self.n_inputs = 28 * 28
        self.n_ouputs = 10

        self.permutation = Permutation(
            permute_size=self.n_inputs,
            permute_interval=permute_interval,
        )

        self.transform_order = [
            transforms.ToTensor(),  # It does also normalizing by deviding by 255
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.flatten()),
        ]

        use_accel = not no_accel and T.accelerator.is_available()

        self.train_kwargs = {"batch_size": train_batch_size}
        self.test_kwargs = {"batch_size": test_batch_size}
        if use_accel:
            accel_kwargs = {
                "num_workers": 1,
                "persistent_workers": True,
                "pin_memory": True,
                "shuffle": True,
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        transform = transforms.Compose(
            [
                *self.transform_order,
                self.permutation,
            ]
        )

        self.train_dataset = datasets.MNIST(
            "./dataset",
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.MNIST(
            "./dataset",
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            self.train_dataset, **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            self.test_dataset, **self.test_kwargs
        )


class LP_EMNIST_Loader(Loader):
    def __init__(
        self,
        no_accel: bool,
        permute_interval: int,
        train_batch_size: int,
        test_batch_size: int,
    ):

        self.n_inputs = 28 * 28
        self.n_outputs = 47
        self.permutation = Permutation(
            permute_size=self.n_outputs,
            permute_interval=permute_interval,
        )

        self.transform_order = [
            transforms.ToTensor(),  # It does also normalizing by deviding by 255
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.flatten()),
        ]

        use_accel = not no_accel and T.accelerator.is_available()

        self.train_kwargs = {"batch_size": train_batch_size}
        self.test_kwargs = {"batch_size": test_batch_size}
        if use_accel:
            accel_kwargs = {
                "num_workers": 1,
                "persistent_workers": True,
                "pin_memory": True,
                "shuffle": True,
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        transform = transforms.Compose(
            self.transform_order,
        )

        def int_to_dist(idx: int) -> np.ndarray:
            tmp_arr = np.zeros(self.n_outputs)
            tmp_arr[idx] = 1
            return tmp_arr

        def dist_to_int(dist: np.ndarray) -> int:
            return int(np.where(dist == 1)[0])

        target_transform = transforms.Compose(
            [
                transforms.Lambda(int_to_dist),
                self.permutation,
                transforms.Lambda(dist_to_int),
            ]
        )

        self.train_dataset = datasets.EMNIST(
            "./dataset",
            train=True,
            download=True,
            split="balanced",
            transform=transform,
            target_transform=target_transform,
        )
        self.test_dataset = datasets.EMNIST(
            "./dataset",
            train=False,
            download=True,
            split="balanced",
            transform=transform,
            target_transform=target_transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            self.train_dataset, **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            self.test_dataset, **self.test_kwargs
        )


class LP_MINI_IMAGENET_Loader(Loader):
    def __init__(
        self,
        no_accel: bool,
        permute_interval: int,
        train_batch_size: int,
        test_batch_size: int,
    ):

        file_name = "./dataset/processed_mini_imagenet.pkl"

        self.n_inputs = 2048
        self.n_outputs = 100
        self.permutation = Permutation(
            permute_size=self.n_outputs,
            permute_interval=permute_interval,
        )

        use_accel = not no_accel and T.accelerator.is_available()

        self.train_kwargs = {"batch_size": train_batch_size}
        self.test_kwargs = {"batch_size": test_batch_size}
        if use_accel:
            accel_kwargs = {
                "num_workers": 1,
                "persistent_workers": True,
                "pin_memory": True,
                "shuffle": True,
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        def int_to_dist(idx: int) -> np.ndarray:
            tmp_arr = np.zeros(self.n_outputs)
            tmp_arr[idx] = 1
            return tmp_arr

        def dist_to_int(dist: np.ndarray) -> int:
            return int(np.where(dist == 1)[0])

        target_transform = transforms.Compose(
            [
                transforms.Lambda(int_to_dist),
                self.permutation,
                transforms.Lambda(dist_to_int),
            ]
        )

        # Download it by yourself
        data = self.get_dataset()
        self.mini_imagenet_dataset = DatasetFromTensor(
            data["data"],
            data["target"],
            target_transform=target_transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            self.mini_imagenet_dataset, shuffle=True, **self.train_kwargs
        )

        self.test_loader = None

    def get_dataset(self) -> Dict[T.tensor, T.tensor]:

        file_name = "./dataset/processed_mini_imagenet.pkl"

        try:
            with open(file_name, "rb") as f:
                data = pickle.load(f)
            return data
        except:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((82, 82)),
                    transforms.Normalize((0.5,), (0.5,), (0.5,)),
                ]
            )

            self.mini_imagenet_dataset = datasets.ImageFolder(
                root="./dataset/mini-IMAGENET",
                transform=transform,
            )

            batch_size = 10000
            self.train_kwargs["batch_size"] = batch_size
            self.train_loader = T.utils.data.DataLoader(
                self.mini_imagenet_dataset, **self.train_kwargs
            )

            resnet = models.resnet50(pretrained=True)

            for param in resnet.parameters():
                param.requires_grad_(False)
            resnet.eval()

            processed_data = T.zeros(
                (len(self.mini_imagenet_dataset.imgs), resnet.fc.in_features)
            )
            target_data = T.zeros((len(self.mini_imagenet_dataset.imgs),))

            for i, (data, target) in enumerate(self.train_loader):
                print(f"processing: {i*batch_size} to {(i+1)*batch_size}")
                processed_data[i * batch_size : (i + 1) * batch_size] = (
                    self.get_bottle_neck(
                        resnet,
                        data,
                    )
                )
                target_data[i * batch_size : (i + 1) * batch_size] = target

            final_data = {
                "data": processed_data,
                "target": target_data.int().tolist(),
            }

            with open(file_name, "wb") as f:
                pickle.dump(final_data, f)

            return final_data

    @staticmethod
    def get_bottle_neck(model, x):

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)

        return T.flatten(x, 1)


class DatasetFromTensor(Dataset):

    def __init__(
        self, data: T.tensor, target: T.tensor, transform=None, target_transform=None
    ):
        self.data = data
        self.target = target
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]

        if self.transform:
            x = self.transform(x)

        y = self.target[index]
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


if __name__ == "__main__":

    data_loader, _, _ = loader_selector(
        name="LP-MINI-IMAGENET",
        no_accel=True,
        permute_interval=2500,
        train_batch_size=1,
        test_batch_size=1,
    )

    file_name = "./dataset/processed_mini_imagenet.pkl"

    with open(file_name, "rb") as f:
        data = pickle.load(f)

    labels = data["target"]
    print(labels[:10])
    print(type(labels))
