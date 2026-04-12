import os

import numpy as np
from scipy.spatial.distance import cdist

from utils import get_gps


def gen_matrix(data_path, max_locs):
    """Generate transition (M1) and distance (M2) matrices.

    Both have shape (max_locs + 1, max_locs + 1). The extra row/column at
    index max_locs is all-zeros so that pad-token lookups return a zero vector.
    """
    if os.path.exists(f"{data_path}/M1.npy") and os.path.exists(f"{data_path}/M2.npy"):
        print("Matrix files already exist, skipping generation.")
        return

    gps = get_gps(f"{data_path}/gps")
    n = max_locs + 1

    # --- M1: transition counts via sparse COO -> dense ---
    with open(f"{data_path}/real.data") as fh:
        lines = fh.readlines()

    srcs = []
    dsts = []
    for line in lines:
        traj = np.fromstring(line, dtype=np.int32, sep=" ")
        if len(traj) < 2:
            continue
        s, d = traj[:-1], traj[1:]
        mask = (s < max_locs) & (d < max_locs)
        srcs.append(s[mask])
        dsts.append(d[mask])

    src_all = np.concatenate(srcs)
    dst_all = np.concatenate(dsts)

    from scipy.sparse import coo_matrix

    data = np.ones(len(src_all), dtype=np.float32)
    reg1 = coo_matrix((data, (src_all, dst_all)), shape=(n, n)).toarray()

    # --- M2: pairwise distance in chunks to avoid OOM ---
    coords = np.column_stack(
        [
            np.asarray(gps[0][:max_locs], dtype=np.float32),
            np.asarray(gps[1][:max_locs], dtype=np.float32),
        ]
    )

    reg2 = np.zeros((n, n), dtype=np.float32)
    chunk = 2048
    for i in range(0, max_locs, chunk):
        ie = min(i + chunk, max_locs)
        # cdist returns (ie-i, max_locs) — only one chunk in memory at a time
        reg2[i:ie, :max_locs] = cdist(coords[i:ie], coords, metric="euclidean").astype(np.float32)

    np.fill_diagonal(reg2, 0.0)

    np.save(f"{data_path}/M1.npy", reg1)
    np.save(f"{data_path}/M2.npy", reg2)
    print("Matrix Generation Finished")
