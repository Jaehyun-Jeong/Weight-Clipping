import torch as T
from torchvision import datasets, transforms


class Loader():

    def __init__(
        self,
    ):
        raise NotImplementedError


    def permute(
        self
    ):
        raise NotImplementedError


class CNN_MNIST_Loader(Loader):

    def __init__(
        self,
        no_accel: bool,
        train_batch_size: int,
        test_batch_size: int,
    ):

        self.transform_order = [
            transforms.ToTensor(),  # It does also normalizing by deviding by 255
            transforms.Normalize((0.137,), (0.3081))
        ]

        use_accel = not no_accel and T.accelerator.is_available()

        self.train_kwargs = {'batch_size': train_batch_size}
        self.test_kwargs = {'batch_size': test_batch_size}
        if use_accel:
            accel_kwargs = {
                'num_workers': 1,
                'persistent_workers': True,
                'pin_memory': True,
                'shuffle': True
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        transform = transforms.Compose(self.transform_order)

        train_dataset = datasets.MNIST(
            './dataset',
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            './dataset',
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            train_dataset,
            **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            test_dataset,
            **self.test_kwargs
        )

    def permute(
        self
    ):

        permutation = T.randperm(28*28)

        transform = transforms.Compose([
            *self.transform_order,
            transforms.Lambda(lambda x: x.flatten()[permutation].reshape(x.shape)),  # permutation
        ])

        train_dataset = datasets.MNIST(
            './dataset',
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            './dataset',
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            train_dataset,
            **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            test_dataset,
            **self.test_kwargs
        )


class FC_MNIST_Loader(Loader):

    def __init__(
        self,
        no_accel: bool,
        train_batch_size: int,
        test_batch_size: int,
    ):

        self.transform_order = [
            transforms.ToTensor(),  # It does also normalizing by deviding by 255
            transforms.Normalize((0.5,), (0.5,)),
        ]

        use_accel = not no_accel and T.accelerator.is_available()

        self.train_kwargs = {'batch_size': train_batch_size}
        self.test_kwargs = {'batch_size': test_batch_size}
        if use_accel:
            accel_kwargs = {
                'num_workers': 1,
                'persistent_workers': True,
                'pin_memory': True,
                'shuffle': True
            }
            self.train_kwargs.update(accel_kwargs)
            self.test_kwargs.update(accel_kwargs)

        permutation = T.randperm(28*28)
        transform = transforms.Compose([
            *self.transform_order,
            transforms.Lambda(lambda x: x.flatten()[permutation]),  # permutation
        ])

        train_dataset = datasets.MNIST(
            './dataset',
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            './dataset',
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            train_dataset,
            **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            test_dataset,
            **self.test_kwargs
        )

    def permute(
        self
    ):

        permutation = T.randperm(28*28)
        transform = transforms.Compose([
            *self.transform_order,
            transforms.Lambda(lambda x: x.flatten()[permutation]),  # permutation
        ])

        train_dataset = datasets.MNIST(
            './dataset',
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            './dataset',
            train=False,
            download=True,
            transform=transform,
        )

        self.train_loader = T.utils.data.DataLoader(
            train_dataset,
            **self.train_kwargs
        )

        self.test_loader = T.utils.data.DataLoader(
            test_dataset,
            **self.test_kwargs
        )
