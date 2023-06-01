from typing import List
from pathlib import Path
import random
import json

import numpy as np
import scipy
import torch
from torch.utils.data import Dataset


class PinnDataset(Dataset):
    def __init__(self, data: List[List[float]]):
        self.data = data
        self.examples = torch.tensor(
            data, dtype=torch.float32, requires_grad=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        headers = ["t", "x", "y", "p", "u", "v"]
        return {key: self.examples[idx, i] for i, key in enumerate(headers)}


def load_jsonl(path, skip_first_lines: int = 0):
    with open(path, "r") as f:
        for _ in range(skip_first_lines):
            next(f)
        return [json.loads(line) for line in f]


def dump_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def get_dataset(data_path: Path):
    data = load_jsonl(data_path, skip_first_lines=1)
    random.shuffle(data)

    # It's weird that the test data is a subset of train data, but
    # that's what the original paper does.
    split_idx = int(len(data) * 0.9)
    train_data = data
    test_data = data[split_idx:]

    # train_data = train_data[:10000]
    # train_data = train_data[:1000]

    min_x = min([d[1] for d in train_data])
    max_x = max([d[1] for d in train_data])

    train_data = PinnDataset(train_data)
    test_data = PinnDataset(test_data)
    return train_data, test_data, min_x, max_x


def get_orig_dataset():
    # Get U star
    print("\n\n\n\n\n U star : ")
    Uarr = np.zeros((10720, 2, 10))
    n = 0
    while (n < 10):
        fileOpen = 50 + 50*n
        fileOpenStr = str(fileOpen)
        file = open("../airFoil2D/"+fileOpenStr+"/U", "r")
        line = ""

        while(line.find("(") == -1):
            line = file.readline()
        line = file.readline()
        
        first = line.partition(" ")[0]
        first = first[1:]
        second = line.partition(" ")[2]
        second = second.partition(" ")[0]

        i = 0
        while(not line.find(".") == -1):
            first = line.partition(" ")[0]
            first = first[1:]
            second = line.partition(" ")[2]
            second = second.partition(" ")[0]
            line = file.readline()
            Uarr[i][0][n] = float(first)
            Uarr[i][1][n] = float(second)
            i += 1
        n += 1
    print(Uarr)
    

    # GET P star
    print("\n\n\n\n\nP star : ")
    Parr = np.zeros((10720, 10))
    n = 0
    while (n < 10):
        fileOpen = 50 + 50*n
        fileOpenStr = str(fileOpen)
        file = open("../airFoil2D/"+fileOpenStr+"/p", "r")
        line = ""

        while(line.find("(") == -1):
            line = file.readline()
        line = file.readline()

        i = 0
        while(line.find(")") == -1):
            addToArr = line[0:-1]
            if (i < 100):
                print(addToArr)
                print(line)
                print(i)
            Parr[i][n] = float(addToArr)
            line = file.readline()            
            i += 1
        n += 1
    print(Parr)

    path = Path("./cylinder_nektar_wake.mat")
    data = scipy.io.loadmat(path)
    X_star = data["X_star"]  # N x 2
    x = X_star[:, 0:1]
    y = X_star[:, 1:2]

    U_star = data["U_star"]  # N x 2 x T
    # CHANGE THIS TO :
    # U_star = Uarr
    # print("\n\n\nU star :", X_star, "\nShape of u star", X_star.shape, "\n\n\n")
    P_star = data["p_star"]  # N x T
    print(P_star.shape)
    t_star = data["t"]  # T x 1

    X_star = data["X_star"]  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    print("\n\n\n\n")

    # Rearrange Data
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1

    min_x = np.min(x)
    max_x = np.max(x)

    NOISE_SCALE = 0.1
    u += NOISE_SCALE * np.std(u) * np.random.randn(*u.shape)
    v += NOISE_SCALE * np.std(v) * np.random.randn(*v.shape)

    train_data = np.hstack((t, x, y, p, u, v))
    # Randomly sample 1000 points as test data
    idx = np.random.choice(train_data.shape[0], 1000, replace=False)
    test_data = train_data[idx, :]
    train_data = PinnDataset(train_data)
    test_data = PinnDataset(test_data)
    return train_data, test_data, min_x, max_x
