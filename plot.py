from os import path, listdir
import argparse
from argparse import Namespace
import pickle

import numpy as np
import matplotlib.pyplot as plt


def get_args() -> Namespace:

    parser = argparse.ArgumentParser()
    # "IP-MNIST", "LP-EMNIST", "LP-mini-ImageNet"
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = get_args()
    track = {}
    file_names = listdir(f"./results/{args.dataset}")
    for file_name in file_names:
        name_optimizer = file_name.split(".")[0].split("_")[0]

        if not name_optimizer in track.keys():
            track[name_optimizer] = []

        with open(path.join(f"./results/{args.dataset}", file_name), "rb") as file:
            track[name_optimizer].append(pickle.load(file)["average online accuracy"])

    colors = ["b", "g", "r", "c", "m", "y", "k"]
    means = {}
    stds = {}
    for idx, (name_optimizer, results) in enumerate(track.items()):
        track[name_optimizer] = np.array(track[name_optimizer])
        means[name_optimizer] = np.mean(track[name_optimizer], axis=0)
        stds[name_optimizer] = np.std(track[name_optimizer], axis=0)

        x = np.arange(len(means[name_optimizer]))
        plt.plot(x, means[name_optimizer], f"{colors[idx]}-", label=name_optimizer)
        plt.fill_between(
            x,
            means[name_optimizer] - stds[name_optimizer],
            means[name_optimizer] + stds[name_optimizer],
            color=colors[idx],
            alpha=0.2,
        )

    plt.ylim([0.05, 0.35])
    plt.legend()
    plt.xlabel("step")
    plt.ylabel("Average Online Accuracy")
    plt.savefig(f"./{args.dataset}_experiments.png")
