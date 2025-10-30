from typing import Iterable
from argparse import Namespace
import math as ma

import torch as T
from torch.optim import Optimizer, Adam, SGD


def optimizer_selector(
    model: T.nn.Module,
    args: Namespace,
) -> Optimizer:

    if args.optimizer == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
        )
    elif args.optimizer == "AdamL2Init":
        optimizer = Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "AdamWC":
        optimizer = AdamWC(
            k=args.clipping,
            params=model.parameters(),
            lr=args.lr,
        )
    elif args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGDL2Init":
        optimizer = SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "SGDWC":
        optimizer = SGDWC(k=args.clipping, params=model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"No Optimizer named {name}")

    return optimizer


class AdamWC(Adam):
    def __init__(
        self,
        k: float,  # Clipping parameter
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        if not 0.0 <= k:
            raise ValueError(f"Invalid clipping parameter: {k}")

        self.params = kwargs["params"]
        self.k = k

    def step(
        self,
    ) -> None:

        super().step()

        """ Debuging code
        print("===============================")
        for param in self.param_groups[0]['params']:
            if param.ndim == 2:
                fan_in = param.shape[1]
            upper_bound = self.k/ma.sqrt(fan_in)
            lower_bound = -self.k/ma.sqrt(fan_in)
            max_value = T.max(param)
            min_value = T.min(param)

            if max_value > upper_bound:
                print(f"upper_bound: {upper_bound}")
                print(f"max value: {T.max(param)}")

            if min_value < lower_bound:
                print(f"lower bound: {lower_bound}")
                print(f"min value: {T.min(param)}")
        """

        self.__weight_clipping()

        """ Debuging code
        for param in self.param_groups[0]['params']:
            if param.ndim == 2:
                fan_in = param.shape[1]
            upper_bound = self.k/ma.sqrt(fan_in)
            lower_bound = -self.k/ma.sqrt(fan_in)
            max_value = T.max(param)
            min_value = T.min(param)

            if max_value >= upper_bound:
                print(f"upper_bound: {upper_bound}")
                print(f"max value: {T.max(param)}")

            if min_value <= lower_bound:
                print(f"lower bound: {lower_bound}")
                print(f"min value: {T.min(param)}")

        print("===============================")
        """

    @T.no_grad()
    def __weight_clipping(
        self,
    ) -> None:

        for param in self.param_groups[0]["params"]:
            if param.ndim == 2:
                fan_in = param.shape[1]
            param.copy_(
                T.clip(
                    param,
                    -self.k / ma.sqrt(fan_in),
                    self.k / ma.sqrt(fan_in),
                )
            )


class SGDWC(SGD):
    def __init__(
        self,
        k: float,  # Clipping parameter
        **kwargs,
    ) -> None:

        super().__init__(**kwargs)

        if not 0.0 <= k:
            raise ValueError(f"Invalid clipping parameter: {k}")

        self.params = kwargs["params"]
        self.k = k

    def step(
        self,
    ) -> None:

        super().step()

        """ Debuging code
        print("===============================")
        for param in self.param_groups[0]['params']:
            if param.ndim == 2:
                fan_in = param.shape[1]
            upper_bound = self.k/ma.sqrt(fan_in)
            lower_bound = -self.k/ma.sqrt(fan_in)
            max_value = T.max(param)
            min_value = T.min(param)

            if max_value > upper_bound:
                print(f"upper_bound: {upper_bound}")
                print(f"max value: {T.max(param)}")

            if min_value < lower_bound:
                print(f"lower bound: {lower_bound}")
                print(f"min value: {T.min(param)}")
        """

        self.__weight_clipping()

        """ Debuging code
        for param in self.param_groups[0]['params']:
            if param.ndim == 2:
                fan_in = param.shape[1]
            upper_bound = self.k/ma.sqrt(fan_in)
            lower_bound = -self.k/ma.sqrt(fan_in)
            max_value = T.max(param)
            min_value = T.min(param)

            if max_value >= upper_bound:
                print(f"upper_bound: {upper_bound}")
                print(f"max value: {T.max(param)}")

            if min_value <= lower_bound:
                print(f"lower bound: {lower_bound}")
                print(f"min value: {T.min(param)}")

        print("===============================")
        """

    @T.no_grad()
    def __weight_clipping(
        self,
    ) -> None:

        for param in self.param_groups[0]["params"]:
            if param.ndim == 2:
                fan_in = param.shape[1]
            param.copy_(
                T.clip(
                    param,
                    -self.k / ma.sqrt(fan_in),
                    self.k / ma.sqrt(fan_in),
                )
            )
