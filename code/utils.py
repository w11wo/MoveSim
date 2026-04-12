import os
import pdb
import time
import json
import shutil
import logging
import hashlib
import numpy as np


def hash_args(*args):
    # json.dumps will keep the dict keys always sorted.
    string = json.dumps(args, sort_keys=True, default=str)  # frozenset
    return hashlib.md5(string.encode()).hexdigest()


def use_gpu(idx):
    # 0->2,3->1,1->3,2->0
    map = {0: 2, 3: 1, 1: 3, 2: 0}
    return map[idx]


def get_acc(target, scores):
    """target and scores are torch cuda Variable"""
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t in p[:10] and t > 0:
            acc[0] += 1
        if t in p[:5] and t > 0:
            acc[1] += 1
        if t == p[0] and t > 0:
            acc[2] += 1
    return acc


def get_gps(gps_file):
    with open(gps_file) as f:
        gpss = f.readlines()
    X = []
    Y = []
    for gps in gpss:
        x, y = float(gps.split()[0]), float(gps.split()[1])
        X.append(x)
        Y.append(y)
    return X, Y


def read_data_from_file(fp):
    """
    Read trajectory data from a txt file (one trajectory per line, space-separated ints).
    Returns a list of lists to support variable-length trajectories.
    """
    dat = []
    with open(fp, "r") as f:
        for line in f:
            tmp = line.split()
            dat.append([int(t) for t in tmp])
    return dat


def write_data_to_file(fp, dat):
    """Write a bunch of trajectory data to txt file.
    Parameters
    ----------
    fp : str
        file path of data
    dat : list
        list of trajs
    """
    with open(fp, "w") as f:
        for i in range(len(dat)):
            line = [str(p) for p in dat[i]]
            line_s = " ".join(line)
            f.write(line_s + "\n")


def read_logs_from_file(fp):
    dat = []
    with open(fp, "r") as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[float(t) for t in tmp]]
    return np.asarray(dat, dtype="float")


def get_workspace_logger(datasets):
    data_path = "preprocessed"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    os.makedirs(data_path + "/%s/logs" % (datasets), exist_ok=True)
    fh = logging.FileHandler(data_path + "/%s/logs/all.log" % (datasets), mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
