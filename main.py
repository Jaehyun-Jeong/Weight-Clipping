import argparse
import torch as T
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import Net
from data_loader import MNISTLoader
from trainer import Trainer


def main():

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

    args = parser.parse_args()

    writer = SummaryWriter()

    # setting seed
    T.manual_seed(args.seed)

    # setting device
    use_accel = not args.no_accel and T.accelerator.is_available()
    if use_accel:
        device = T.accelerator.current_accelerator()
    else:
        device = T.device("cpu")

    data_loader = MNISTLoader(
        no_accel=args.no_accel,
        train_batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
    )

    model = Net().to(device)

    if args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        device=device,
        permute_interval=5000,
    )

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch()
        _, average_accuracy = trainer.test()
        self.writer.add_scalar(
            f"{args.optimizer}/AvgAcc/Test",
            average_accuracy,
            epoch,
        )

    if args.save_model:
        T.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
