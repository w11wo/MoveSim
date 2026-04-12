import argparse
import json
import os
import random

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", type=str, required=True, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument(
        "--dispre_n", type=int, default=10000, help="Number of perturbed trajectories for dispre.data (default: 10000)"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def perturb_trajectory(traj, total_locs, p_replace=0.5):
    """
    Create a fake trajectory by perturbing the real one in two ways
    (matching the paper's discriminator pre-train strategy):
      1. Replace a random location with a randomly chosen distant one.
      2. Shuffle a contiguous segment (simulates disrupting periodicity).
    """
    traj = list(traj)
    n = len(traj)
    if n < 2:
        return traj

    if random.random() < p_replace:
        # Replace one location with a random location
        idx = random.randint(0, n - 1)
        new_loc = random.randint(0, total_locs - 1)
        traj[idx] = new_loc
    else:
        # Shuffle a contiguous segment
        if n >= 4:
            i = random.randint(0, n // 2)
            j = random.randint(i + 2, n)
            seg = traj[i:j]
            random.shuffle(seg)
            traj[i:j] = seg
    return traj


def write_data_file(path, trajs):
    with open(path, "w") as f:
        for traj in trajs:
            f.write(" ".join(str(r) for r in traj) + "\n")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    city = args.city
    data_root = "data/"

    train_traj_csv = os.path.join(data_root, city, "train.csv")
    test_traj_csv = os.path.join(data_root, city, "test.csv")
    val_traj_csv = os.path.join(data_root, city, "val.csv")

    rid_gps_path = os.path.join(data_root, city, "rid_gps.json")

    out_dir = os.path.join("preprocessed", city)
    os.makedirs(out_dir, exist_ok=True)

    with open(rid_gps_path, "r") as f:
        rid_gps = json.load(f)

    # Road IDs are expected to be 0-indexed contiguous integers
    total_locs = max(int(k) for k in rid_gps.keys()) + 1

    gps_path = os.path.join(out_dir, "gps")
    with open(gps_path, "w") as f:
        for rid in range(total_locs):
            lon, lat = rid_gps[str(rid)]
            f.write(f"{lat} {lon}\n")

    def read_traj(traj_csv):
        df = pd.read_csv(traj_csv)
        trajs = []
        for rid_list_str in df["rid_list"]:
            rids = [int(r) for r in str(rid_list_str).split(",")]
            trajs.append(rids)
        return trajs

    train_trajs = read_traj(train_traj_csv)
    test_trajs = read_traj(test_traj_csv)
    val_trajs = read_traj(val_traj_csv)

    write_data_file(os.path.join(out_dir, "real.data"), train_trajs)
    write_data_file(os.path.join(out_dir, "val.data"), val_trajs)
    write_data_file(os.path.join(out_dir, "test.data"), test_trajs)

    dispre = []
    pool = train_trajs * (args.dispre_n // len(train_trajs) + 1)
    for traj in pool[: args.dispre_n]:
        dispre.append(perturb_trajectory(traj, total_locs))
    write_data_file(os.path.join(out_dir, "dispre.data"), dispre)

    start_counts = np.zeros(total_locs, dtype=np.float64)
    for traj in train_trajs:
        start_counts[traj[0]] += 1.0
    start_dist = start_counts / start_counts.sum()
    np.save(os.path.join(out_dir, "start.npy"), start_dist)


if __name__ == "__main__":
    main()
