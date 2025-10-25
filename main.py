import argparse
from argparse import Namespace
import torch as T
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Net, FCNLeakyReLU
from data_loader import FC_MNIST_Loader
from trainer import Trainer
from optimizer import optimizer_selector


def get_args() -> Namespace:

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # Streaming learning setting
    parser.add_argument('--batch_size', type=int, default=1, metavar='N')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N')
    parser.add_argument('--epochs', type=int, default=14, metavar='N')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M')
    # 'store_true' means executed with --no-accel, then it is true
    # vice versa for 'store_false'
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seed', type=int, default=1, metavar='S')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N')
    parser.add_argument('--save-model', action='store_true')
    # 'Adam', 'SGD'
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--permute-interval', type=int, default=2500)
    parser.add_argument('--clipping', type=float, default=2)
    parser.add_argument('--weight-decay', type=float, default=0.)

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    writer = SummaryWriter()

    # setting seed
    T.manual_seed(args.seed)

    # setting device
    use_accel = not args.no_accel and T.accelerator.is_available()
    if use_accel:
        device = T.accelerator.current_accelerator()
    else:
        device = T.device("cpu")

    data_loader = FC_MNIST_Loader(
        no_accel=args.no_accel,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    model = FCNLeakyReLU(
        n_inputs=28*28,
        n_outputs=10,
    ).to(device)

    optimizer = optimizer_selector(
        model,
        args,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        device=device,
        permute_interval=args.permute_interval,
    )

    trainer.train_steps(
        steps=200*args.permute_interval,
        writer=writer,
        writer_tag=f"/OnlineAvgAcc/{args.optimizer}/step",
    )


if __name__ == '__main__':
    main()
