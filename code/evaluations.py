# encoding: utf-8

import numpy as np
import scipy.stats
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from utils import get_gps, read_data_from_file


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000
    distance = round(distance / 1000, 3)
    return distance


class EvalUtils(object):
    """Commonly-used evaluation tools and functions."""

    @staticmethod
    def filter_zero(arr):
        arr = np.array(arr)
        return np.array(list(filter(lambda x: x != 0.0, arr)))

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        distribution, base = np.histogram(arr, np.arange(min, max, float(max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1.0 / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30.0, bins=100):
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0.0, 1.0 / bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        p1 = p1 / (p1.sum() + 1e-14)
        p2 = p2 / (p2.sum() + 1e-14)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js


class IndividualEval(object):

    def __init__(self, data_path, max_locs, max_distance, pad_id=None):
        """
        Parameters
        ----------
        data_path : str
            Directory containing the gps file (e.g. '../data/Xian').
        max_locs : int
            Number of real location IDs (= TOTAL_LOCS, before padding).
        max_distance : float
            Upper bound for distance histogram binning.
        pad_id : int or None
            Pad token ID to strip when computing per-trajectory metrics.
            None means no stripping.
        """
        self.X, self.Y = get_gps(f"{data_path}/gps")
        self.max_locs = max_locs
        self.max_distance = max_distance
        self.pad_id = pad_id

    def _strip_padding(self, traj):
        """Return traj as a list, truncated at the first pad token."""
        if self.pad_id is None:
            return list(traj)
        result = []
        for loc in traj:
            if loc == self.pad_id:
                break
            result.append(loc)
        return result

    def get_topk_visits(self, trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            traj_clean = self._strip_padding(traj)
            n = max(len(traj_clean), 1)
            topk = Counter(traj_clean).most_common(k)
            for i in range(len(topk), k):
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f / n for _, f in topk]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)

    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)

    def get_geodistances(self, trajs):
        distances = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            for i in range(len(traj) - 1):
                lng1 = self.X[traj[i]]
                lat1 = self.Y[traj[i]]
                lng2 = self.X[traj[i + 1]]
                lat2 = self.Y[traj[i + 1]]
                distances.append(geodistance(lng1, lat1, lng2, lat2))
        return np.array(distances, dtype=float)

    def get_distances(self, trajs):
        distances = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            for i in range(len(traj) - 1):
                dx = self.X[traj[i]] - self.X[traj[i + 1]]
                dy = self.Y[traj[i]] - self.Y[traj[i + 1]]
                distances.append(dx**2 + dy**2)
        return np.array(distances, dtype=float)

    def get_durations(self, trajs):
        d = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            n = len(traj)
            if n == 0:
                continue
            num = 1
            for i, lc in enumerate(traj[1:]):
                if lc == traj[i]:
                    num += 1
                else:
                    d.append(num / n)
                    num = 1
        return np.array(d, dtype=float)

    def get_gradius(self, trajs):
        gradius = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            if not traj:
                continue
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = np.mean(dxs**2 + dys**2)
            gradius.append(rad)
        return np.array(gradius, dtype=float)

    def get_periodicity(self, trajs):
        reps = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            n = len(traj)
            if n == 0:
                continue
            reps.append(float(len(set(traj))) / n)
        return np.array(reps, dtype=float)

    def get_geogradius(self, trajs):
        gradius = []
        for traj in trajs:
            traj = self._strip_padding(traj)
            if not traj:
                continue
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):
                distance = geodistance(lng1, lat1, xs[i], ys[i])
                rad.append(distance)
            gradius.append(np.mean(np.array(rad, dtype=float)))
        return np.array(gradius, dtype=float)

    def get_individual_jsds(self, t1, t2):
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)
        d1_dist, _ = EvalUtils.arr_to_distribution(d1, 0, self.max_distance, 10000)
        d2_dist, _ = EvalUtils.arr_to_distribution(d2, 0, self.max_distance, 10000)
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)

        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g1_dist, _ = EvalUtils.arr_to_distribution(g1, 0, self.max_distance**2, 10000)
        g2_dist, _ = EvalUtils.arr_to_distribution(g2, 0, self.max_distance**2, 10000)
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)

        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 0, 1, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 0, 1, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)

        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)

        l1 = CollectiveEval.get_visits(t1, self.max_locs, pad_id=self.pad_id)
        l2 = CollectiveEval.get_visits(t2, self.max_locs, pad_id=self.pad_id)
        l1_dist, _ = CollectiveEval.get_topk_visits(l1, 100)
        l2_dist, _ = CollectiveEval.get_topk_visits(l2, 100)
        l1_dist, _ = EvalUtils.arr_to_distribution(l1_dist, 0, 1, 100)
        l2_dist, _ = EvalUtils.arr_to_distribution(l2_dist, 0, 1, 100)
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        f1 = self.get_overall_topk_visits_freq(t1, 100)
        f2 = self.get_overall_topk_visits_freq(t2, 100)
        f1_dist, _ = EvalUtils.arr_to_distribution(f1, 0, 1, 100)
        f2_dist, _ = EvalUtils.arr_to_distribution(f2, 0, 1, 100)
        f_jsd = EvalUtils.get_js_divergence(f1_dist, f2_dist)

        return d_jsd, g_jsd, du_jsd, p_jsd, l_jsd, f_jsd


class CollectiveEval(object):
    """Collective evaluation metrics."""

    @staticmethod
    def get_visits(trajs, max_locs, pad_id=None):
        visits = np.zeros(shape=(max_locs,), dtype=float)
        for traj in trajs:
            for t in traj:
                if pad_id is not None and t == pad_id:
                    break
                if 0 <= t < max_locs:
                    visits[t] += 1
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_topk_visits(visits, K):
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K
