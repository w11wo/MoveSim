import math
import random

import numpy as np
import torch


class GenDataIter(object):
    """Data iterator for generator pre-training (prepends zero start token)."""

    def __init__(self, data_file, batch_size, seq_len=48, pad_id=0):
        super(GenDataIter, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = list(range(self.data_num))
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def _pad_or_trunc(self, seq):
        seq = list(seq)[: self.seq_len]
        return seq + [self.pad_id] * (self.seq_len - len(seq))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self._pad_or_trunc(self.data_lis[i]) for i in index]
        d = torch.LongTensor(np.asarray(d, dtype="int64"))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.full((self.batch_size, 1), self.pad_id).long()], dim=1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(" ")
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class NewGenIter(object):
    """Data iterator for generator pre-training with fixed starting location."""

    def __init__(self, data_file, batch_size, seq_len=48, pad_id=0):
        super(NewGenIter, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = list(range(self.data_num))
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def _pad_or_trunc(self, seq):
        seq = list(seq)[: self.seq_len]
        return seq + [self.pad_id] * (self.seq_len - len(seq))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self._pad_or_trunc(self.data_lis[i]) for i in index]
        d = np.asarray(d, dtype="int64")  # (batch, seq_len)
        data = torch.LongTensor(d[:, :-1])  # (batch, seq_len - 1)
        target = torch.LongTensor(d[:, 1:])  # (batch, seq_len - 1)
        self.idx += self.batch_size
        return data, target

    def read_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(" ")
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class DisDataIter(object):
    """Data iterator for discriminator training (real vs fake sequences)."""

    def __init__(self, real_data_file, fake_data_file, batch_size, seq_len=48, pad_id=0):
        super(DisDataIter, self).__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pad_id = pad_id

        real_data_lis = [self._pad_or_trunc(t) for t in self.read_file(real_data_file)]
        fake_data_lis = [self._pad_or_trunc(t) for t in self.read_file(fake_data_file)]

        # Subsample real data to match fake count: keeps the dataset balanced and
        # avoids the 20:1 real:fake imbalance that inflates discriminator training time.
        if len(real_data_lis) > len(fake_data_lis):
            real_data_lis = random.sample(real_data_lis, len(fake_data_lis))

        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] + [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = list(range(self.data_num))
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0

    def _pad_or_trunc(self, seq):
        seq = list(seq)[: self.seq_len]
        return seq + [self.pad_id] * (self.seq_len - len(seq))

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype="int64"))
        label = torch.LongTensor(np.asarray(label, dtype="int64"))
        self.idx += self.batch_size
        return data, label

    def read_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(" ")
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class TCGenDataIter(object):
    """Data iterator with time context."""

    def __init__(self, data_file, batch_size):
        super(TCGenDataIter, self).__init__()
        self.batch_size = batch_size
        self.data_lis = self.read_file(data_file)
        self.data_num = len(self.data_lis)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0
        time_list = [23] + [i % 24 for i in range(48)]
        self.time = torch.stack([torch.ones(self.batch_size) * i for i in time_list], dim=1).long()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.data_lis)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        d = [self.data_lis[i] for i in index]
        d = torch.LongTensor(np.asarray(d, dtype="int64"))
        data = torch.cat([torch.zeros(self.batch_size, 1).long(), d], dim=1)
        target = torch.cat([d, torch.zeros(self.batch_size, 1).long()], dim=1)
        self.idx += self.batch_size
        return self.time, data, target

    def read_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(" ")
            l = [int(s) for s in l]
            lis.append(l)
        return lis


class TCDisDataIter(object):
    """Discriminator iterator with time context."""

    def __init__(self, real_data_file, fake_data_file, batch_size):
        super(TCDisDataIter, self).__init__()
        self.batch_size = batch_size
        real_data_lis = self.read_file(real_data_file)
        fake_data_lis = self.read_file(fake_data_file)
        self.data = real_data_lis + fake_data_lis
        self.labels = [1 for _ in range(len(real_data_lis))] + [0 for _ in range(len(fake_data_lis))]
        self.pairs = list(zip(self.data, self.labels))
        self.data_num = len(self.pairs)
        self.indices = range(self.data_num)
        self.num_batches = int(math.ceil(float(self.data_num) / self.batch_size))
        self.idx = 0
        time_list = [i % 24 for i in range(48)]
        self.time = torch.stack([torch.ones(self.batch_size) * i for i in time_list], dim=1).long()

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random.shuffle(self.pairs)

    def next(self):
        if self.idx >= self.data_num - self.batch_size:
            raise StopIteration
        index = self.indices[self.idx : self.idx + self.batch_size]
        pairs = [self.pairs[i] for i in index]
        data = [p[0] for p in pairs]
        label = [p[1] for p in pairs]
        data = torch.LongTensor(np.asarray(data, dtype="int64"))
        label = torch.LongTensor(np.asarray(label, dtype="int64"))
        self.idx += self.batch_size
        return self.time, data, label

    def read_file(self, data_file):
        with open(data_file, "r") as f:
            lines = f.readlines()
        lis = []
        for line in lines:
            l = line.strip().split(" ")
            l = [int(s) for s in l]
            lis.append(l)
        return lis
