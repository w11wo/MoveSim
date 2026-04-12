# -*- coding:utf-8 -*-

import copy

import numpy as np
import torch
from tqdm import tqdm


class Rollout(object):
    """Roll-out policy"""

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x, num, discriminator):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # Expand x to (batch_size * num, seq_len) so all rollouts run at once
        x_exp = x.repeat(num, 1)  # (batch_size * num, seq_len)

        rewards = torch.zeros(batch_size, seq_len, device=x.device)

        self.own_model.eval()
        discriminator.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for l in tqdm(range(1, seq_len), desc="Rollout"):
                data = x_exp[:, :l]
                samples = self.own_model.sample(batch_size * num, seq_len, data)
                pred = discriminator(samples)[:, 1]  # (batch_size * num,)
                # Reshape to (num, batch_size) and average over num
                pred = pred.view(num, batch_size).mean(dim=0)
                rewards[:, l - 1] = pred.float()

            # Last token — just evaluate the real sequence
            pred = discriminator(x)[:, 1]
            rewards[:, seq_len - 1] = pred.float()

        self.own_model.train()
        discriminator.train()
        return rewards.cpu().numpy()

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if name.startswith("emb") or name.startswith("Emb"):
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]


class TCRollout(object):
    """Roll-out policy"""

    def __init__(self, model, update_rate):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate

    def get_reward(self, x_t, x_s, num, discriminator):
        """
        Args:
            x : (batch_size, seq_len) input data
            num : roll-out number
            discriminator : discrimanator model
        """
        rewards = []
        batch_size = x_t.size(0)
        seq_len = x_t.size(1)
        for i in range(num):
            for l in range(1, seq_len):
                _x_t = x_t[:, 0:l]
                _x_s = x_s[:, 0:l]
                time, samples = self.own_model.sample(batch_size, seq_len, _x_t, _x_s)
                pred = discriminator(time, samples)
                pred = pred.cpu().data[:, 1].numpy()
                if i == 0:
                    rewards.append(pred)
                else:
                    rewards[l - 1] += pred

            # for the last token
            pred = discriminator(x_t, x_s)
            pred = pred.cpu().data[:, 1].numpy()
            if i == 0:
                rewards.append(pred)
            else:
                rewards[seq_len - 1] += pred
        rewards = np.transpose(np.array(rewards)) / (1.0 * num)  # batch_size * seq_len
        return rewards

    def update_params(self):
        dic = {}
        for name, param in self.ori_model.named_parameters():
            dic[name] = param.data
        for name, param in self.own_model.named_parameters():
            if "emb" in name or "Emb" in name:
                param.data = dic[name]
            else:
                param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
