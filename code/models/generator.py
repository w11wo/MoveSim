# coding: utf-8
import sys

sys.path.append("../")

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import *


def gen_gaussian_dist(sigma=10):
    """Return a single-sided gaussian distribution weight array and its index."""
    u = 0
    x = np.linspace(0, 1, 100)
    y = np.exp(-((x - u) ** 2) / (2 * sigma**2)) / (math.sqrt(2 * math.pi) * sigma)
    return x, y


class Generator(nn.Module):
    """Basic Generator."""

    def __init__(
        self,
        total_locations=8606,
        embedding_net=None,
        embedding_dim=32,
        hidden_dim=64,
        bidirectional=False,
        cuda=None,
        starting_sample="zero",
        starting_dist=None,
    ):
        super(Generator, self).__init__()
        self.total_locations = total_locations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.use_cuda = cuda
        self.starting_sample = starting_sample
        if self.starting_sample == "real":
            self.starting_dist = torch.tensor(starting_dist).float()

        if embedding_net:
            self.embedding = embedding_net
        else:
            self.embedding = nn.Embedding(num_embeddings=total_locations, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=True)
        self.linear = nn.Linear(self.linear_dim, total_locations)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def init_hidden(self, batch_size):
        from torch.autograd import Variable

        h = Variable(torch.zeros((2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        c = Variable(torch.zeros((2 if self.bidirectional else 1, batch_size, self.hidden_dim)))
        if self.use_cuda is not None:
            h, c = h.cuda(), c.cuda()
        return h, c

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        h0, c0 = self.init_hidden(x.size(0))
        x, (h, c) = self.lstm(x, (h0, c0))
        pred = F.log_softmax(self.linear(x.contiguous().view(-1, self.linear_dim)), dim=-1)
        return pred

    def step(self, x, h, c):
        self.lstm.flatten_parameters()
        x = self.embedding(x)
        x, (h, c) = self.lstm(x, (h, c))
        pred = F.softmax(self.linear(x.view(-1, self.linear_dim)), dim=-1)
        return pred, h, c

    def sample(self, batch_size, seq_len, x=None):
        from torch.autograd import Variable

        res = []
        flag = False
        if x is None:
            flag = True
        s = 0
        if flag:
            if self.starting_sample == "zero":
                x = Variable(torch.zeros((batch_size, 1)).long())
            elif self.starting_sample == "rand":
                x = Variable(torch.randint(high=self.total_locations, size=(batch_size, 1)).long())
            elif self.starting_sample == "real":
                x = Variable(torch.stack([torch.multinomial(self.starting_dist, 1) for i in range(batch_size)], dim=0))
                s = 1
        self.lstm.flatten_parameters()
        if self.use_cuda is not None:
            x = x.cuda()
        h, c = self.init_hidden(batch_size)
        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                x, h, c = self.step(x, h, c)
                x = x.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(x.size(1), dim=1)
            for i in range(given_len):
                x, h, c = self.step(lis[i], h, c)
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                x, h, c = self.step(x, h, c)
                x = x.multinomial(1)
        output = torch.cat(samples, dim=1)
        return output


class ATGenerator(nn.Module):
    """Attention Generator."""

    def __init__(
        self,
        total_locations=8606,
        embedding_net=None,
        loc_embedding_dim=256,
        tim_embedding_dim=16,
        hidden_dim=64,
        bidirectional=False,
        data_path="data/geolife",
        device=None,
        function=False,
        starting_sample="zero",
        starting_dist=None,
    ):
        super(ATGenerator, self).__init__()
        self.total_locations = total_locations
        self.loc_embedding_dim = loc_embedding_dim
        self.tim_embedding_dim = tim_embedding_dim
        self.embedding_dim = loc_embedding_dim + tim_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.linear_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.device = device
        self.starting_sample = starting_sample
        self.function = function

        # M1/M2: (total_locations+1) × (total_locations+1), float32.
        # Stored as non-persistent buffers so they move with .to(device) but are
        # not saved in state_dict (they're derived from data files).
        # NOTE: each matrix is ~1.2 GB for 17 k locations; ensure enough VRAM.
        M1 = torch.from_numpy(np.load(f"{data_path}/M1.npy"))  # already float32
        M2 = torch.from_numpy(np.load(f"{data_path}/M2.npy"))
        self.register_buffer("M1", M1, persistent=False)
        self.register_buffer("M2", M2, persistent=False)

        if self.starting_sample == "real":
            self.starting_dist = torch.tensor(starting_dist).float()

        if embedding_net:
            self.embedding = embedding_net
        else:
            # +1 so the pad token (index total_locations) has a valid embedding slot
            self.loc_embedding = nn.Embedding(
                num_embeddings=self.total_locations + 1,
                embedding_dim=self.loc_embedding_dim,
                padding_idx=self.total_locations,
            )
            self.tim_embedding = nn.Embedding(num_embeddings=24, embedding_dim=self.tim_embedding_dim)

        self.attn = nn.MultiheadAttention(self.hidden_dim, 4)
        self.Q = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.V = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.K = nn.Linear(self.embedding_dim, self.hidden_dim)

        self.attn2 = nn.MultiheadAttention(self.hidden_dim, 1)
        self.Q2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.V2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.K2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.linear = nn.Linear(self.linear_dim, self.total_locations)
        # M1/M2 rows are (total_locations + 1) wide after the pad-row addition
        self.linear_mat1 = nn.Linear(self.total_locations + 1, self.linear_dim)
        self.linear_mat1_2 = nn.Linear(self.linear_dim, self.total_locations)

        self.linear_mat2 = nn.Linear(self.total_locations + 1, self.linear_dim)
        self.linear_mat2_2 = nn.Linear(self.linear_dim, self.total_locations)

        self.final_linear = nn.Linear(self.linear_dim, self.total_locations)

        if function:
            M3 = torch.from_numpy(np.load(f"{data_path}/M3.npy"))
            self.register_buffer("M3", M3, persistent=False)
            self.linear_mat3 = nn.Linear(self.total_locations + 1, self.linear_dim)

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            param.data.uniform_(-0.05, 0.05)

    def _lookup_mat(self, locs):
        """Row-index M1/M2 using a GPU LongTensor — no CPU/numpy roundtrip."""
        mat1 = self.M1[locs]
        mat2 = self.M2[locs]
        return mat1, mat2

    def forward(self, x_l, x_t):
        """
        :param x_l: (batch_size, seq_len) location indices
        :param x_t: (batch_size, seq_len) time indices
        :return: (batch_size * seq_len, total_locations) log-softmax predictions
        """
        locs = x_l.contiguous().view(-1)  # stays on device — no CPU copy
        mat1, mat2 = self._lookup_mat(locs)  # pure GPU indexing

        lemb = self.loc_embedding(x_l)
        temb = self.tim_embedding(x_t)
        x = torch.cat([lemb, temb], dim=-1)

        x = x.transpose(0, 1)
        Query = F.relu(self.Q(x))
        Value = F.relu(self.V(x))
        Key = F.relu(self.K(x))
        x, _ = self.attn(Query, Key, Value)

        Query = F.relu(self.Q2(x))
        Value = F.relu(self.V2(x))
        Key = F.relu(self.K2(x))
        x, _ = self.attn2(Query, Key, Value)

        x = x.transpose(0, 1).reshape(-1, self.linear_dim)
        x = F.relu(self.linear(x))

        mat1 = F.normalize(torch.sigmoid(self.linear_mat1_2(F.relu(self.linear_mat1(mat1)))))
        mat2 = F.normalize(torch.sigmoid(self.linear_mat2_2(F.relu(self.linear_mat2(mat2)))))

        if self.function:
            mat3 = torch.sigmoid(self.linear_mat3(self.M3[locs]))
            pred = self.final_linear(x + torch.mul(x, mat1) + torch.mul(x, mat2) + torch.mul(x, mat3))
        else:
            pred = x + torch.mul(x, mat1) + torch.mul(x, mat2)
        return F.log_softmax(pred, dim=-1)

    def step(self, l, t):
        """
        :param l: (batch_size, 1) current location
        :param t: (batch_size, 1) current time index
        :return: (batch_size, total_locations) softmax probabilities
        """
        locs = l.contiguous().view(-1)  # stays on device
        mat1, mat2 = self._lookup_mat(locs)

        lemb = self.loc_embedding(l)
        temb = self.tim_embedding(t)
        x = torch.cat([lemb, temb], dim=-1)

        x = x.transpose(0, 1)
        Query = F.relu(self.Q(x))
        Value = F.relu(self.V(x))
        Key = F.relu(self.K(x))
        x, _ = self.attn(Query, Key, Value)

        Query = F.relu(self.Q2(x))
        Value = F.relu(self.V2(x))
        Key = F.relu(self.K2(x))
        x, _ = self.attn2(Query, Key, Value)

        x = x.transpose(0, 1).reshape(-1, self.linear_dim)
        x = F.relu(self.linear(x))

        mat1 = F.normalize(torch.sigmoid(self.linear_mat1_2(F.relu(self.linear_mat1(mat1)))))
        mat2 = F.normalize(torch.sigmoid(self.linear_mat2_2(F.relu(self.linear_mat2(mat2)))))

        if self.function:
            mat3 = torch.sigmoid(self.linear_mat3(self.M3[locs]))
            pred = self.final_linear(x + torch.mul(x, mat1) + torch.mul(x, mat2) + torch.mul(x, mat3))
        else:
            pred = x + torch.mul(x, mat1) + torch.mul(x, mat2)
        return F.softmax(pred, dim=-1)

    def sample(self, batch_size, seq_len, x=None):
        """
        :param batch_size: int
        :param seq_len: int, length of the generated sequence
        :param x: (batch_size, k) optional prefix of known locations
        :return: (batch_size, seq_len) generated location IDs
        """
        flag = x is None
        s = 0

        if flag:
            if self.starting_sample == "zero":
                x = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)
            elif self.starting_sample == "rand":
                x = torch.randint(self.total_locations, (batch_size, 1), device=self.device)
            elif self.starting_sample == "real":
                x = torch.stack([torch.multinomial(self.starting_dist, 1) for _ in range(batch_size)], dim=0).to(
                    self.device
                )
                s = 1
        else:
            x = x.to(self.device)

        # Pre-compute all time tokens to avoid per-step tensor creation
        all_t = (torch.arange(seq_len, device=self.device) % 24).unsqueeze(0)  # (1, seq_len)
        all_t = all_t.expand(batch_size, -1)  # (batch, seq_len)

        samples = []
        if flag:
            if s > 0:
                samples.append(x)
            for i in range(s, seq_len):
                t = all_t[:, i : i + 1]  # (batch, 1) — view, no copy
                x = self.step(x, t)
                x = x.multinomial(1)
                samples.append(x)
        else:
            given_len = x.size(1)
            lis = x.chunk(given_len, dim=1)
            for i in range(given_len):
                t = all_t[:, i : i + 1]
                x = self.step(lis[i], t)
                samples.append(lis[i])
            x = x.multinomial(1)
            for i in range(given_len, seq_len):
                samples.append(x)
                t = all_t[:, i : i + 1]
                x = self.step(x, t)
                x = x.multinomial(1)

        return torch.cat(samples, dim=1)
