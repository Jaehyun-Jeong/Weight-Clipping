from typing import Iterable

from torch.optim import Adam, SGD


class AdamWC(Adam):

    def __init__(
        self,
        k: float,  # Clipping parameter
        **kwargs,
    ):

        super().__init__(**kwargs)

        print("=========================================")
        for param in kwargs['params']:
            print(param.shape)
        print("=========================================")

        self.params = kwargs['params']
        self.k = k


    def step(
        self,
    ):

        super().step()
        print("=========================================")
        for param in self.params:
            print(param.shape)
        print("=========================================")

        raise ValueError("test")


    def __weight_clipping(
        self,
    ):
        pass
