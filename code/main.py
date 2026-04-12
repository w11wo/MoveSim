# coding=utf-8
import argparse
import random
import os

import numpy as np
import torch
from torch import nn, optim

from data_iter import DisDataIter, GenDataIter, NewGenIter
from gen_data import gen_matrix
from models.discriminator import Discriminator
from models.gan_loss import GANLoss, distance_loss, period_loss
from models.generator import ATGenerator
from rollout import Rollout
from tqdm import tqdm

from train import generate_samples, pretrain_model, train_epoch
from utils import get_workspace_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--cuda", default="0", type=str)
    parser.add_argument("--task", default="attention", type=str)
    parser.add_argument("--ploss", default=3.0, type=float)
    parser.add_argument("--dloss", default=1.5, type=float)
    parser.add_argument("--city", required=True, type=str, choices=["Beijing", "Porto", "San_Francisco"])
    parser.add_argument("--seq_len", default=48, type=int)
    return parser.parse_args()


def generate_test_trajectories(generator, test_file, seq_len, batch_size, device, output_file):
    """Generate one trajectory per test trajectory, conditioned on the same origin.

    Each output trajectory has the same length as the corresponding test trajectory
    (capped at seq_len).  Written one per line, space-separated location IDs.
    """
    with open(test_file) as f:
        test_trajs = [[int(t) for t in line.strip().split()] for line in f if line.strip()]

    generator.eval()
    all_generated = []
    with torch.no_grad():
        for start in tqdm(range(0, len(test_trajs), batch_size), desc="Gen test trajectories"):
            batch = test_trajs[start : start + batch_size]
            actual_bs = len(batch)
            # Fix the starting location to each test trajectory's origin
            origins = torch.LongTensor([[traj[0]] for traj in batch]).to(device)
            # Match the length of each real trajectory, capped at seq_len
            lengths = [min(len(traj), seq_len) for traj in batch]
            max_len = max(lengths)
            generated = generator.sample(actual_bs, max_len, x=origins)
            generated = generated.cpu().numpy().tolist()
            for traj_gen, length in zip(generated, lengths):
                all_generated.append(traj_gen[:length])

    with open(output_file, "w") as f:
        for traj in all_generated:
            f.write(" ".join(str(loc) for loc in traj) + "\n")
    print(f"Generated {len(all_generated)} test trajectories -> {output_file}")
    generator.train()


def main():
    opt = parse_args()
    print(opt)

    SEED = 88
    EPOCHS = 30
    BATCH_SIZE = 32
    SEQ_LEN = opt.seq_len
    GENERATED_NUM = 10000

    DATA_PATH = "preprocessed"
    CITY_PATH = f"{DATA_PATH}/{opt.city}"
    REAL_DATA = f"{CITY_PATH}/real.data"
    TEST_DATA = f"{CITY_PATH}/test.data"
    GENE_DATA = f"{CITY_PATH}/gene.data"
    GEN_TEST_DATA = f"{CITY_PATH}/gen_test.data"
    os.makedirs(f"{CITY_PATH}/pretrain", exist_ok=True)

    assert SEQ_LEN >= 20, f"SEQ_LEN={SEQ_LEN} is too short; discriminator filter sizes go up to 20."

    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.benchmark = True

    # Derive TOTAL_LOCS from the gps file (one line per location)
    with open(f"{CITY_PATH}/gps") as f:
        TOTAL_LOCS = sum(1 for _ in f)
    PAD_ID = TOTAL_LOCS  # pad token is one index beyond the real locations
    print(f"TOTAL_LOCS = {TOTAL_LOCS}, PAD_ID = {PAD_ID}, SEQ_LEN = {SEQ_LEN}")

    device = torch.device("cuda:" + opt.cuda)

    print("Pre-processing Data...")
    gen_matrix(CITY_PATH, TOTAL_LOCS)

    if opt.task == "attention":
        d_pre_epoch = 20
        g_pre_epoch = 110
        ploss_alpha = opt.ploss
        dloss_alpha = opt.dloss
        generator = ATGenerator(
            device=device,
            total_locations=TOTAL_LOCS,
            starting_sample="real",
            starting_dist=np.load(f"{CITY_PATH}/start.npy"),
            data_path=CITY_PATH,
        )
        discriminator = Discriminator(total_locations=TOTAL_LOCS)
        gen_train_fixstart = True

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    generator = torch.compile(generator, mode="reduce-overhead")
    discriminator = torch.compile(discriminator, mode="reduce-overhead")

    logger = get_workspace_logger(opt.city)

    if opt.pretrain:
        # Pre-train discriminator
        logger.info("pretrain discriminator ...")
        dis_data_iter = DisDataIter(
            f"{CITY_PATH}/real.data", f"{CITY_PATH}/dispre.data", BATCH_SIZE, seq_len=SEQ_LEN, pad_id=PAD_ID
        )
        dis_criterion = nn.NLLLoss(reduction="sum")
        dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.000001)
        pretrain_model(
            "D",
            d_pre_epoch,
            discriminator,
            dis_data_iter,
            dis_criterion,
            dis_optimizer,
            BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        # Pre-train generator
        logger.info("pretrain generator ...")
        if gen_train_fixstart:
            gen_data_iter = NewGenIter(REAL_DATA, BATCH_SIZE, seq_len=SEQ_LEN, pad_id=PAD_ID)
        else:
            gen_data_iter = GenDataIter(REAL_DATA, BATCH_SIZE, seq_len=SEQ_LEN, pad_id=PAD_ID)
        gen_criterion = nn.NLLLoss(reduction="sum", ignore_index=PAD_ID)
        gen_optimizer = optim.Adam(generator.parameters(), lr=0.0001)
        pretrain_model(
            "G",
            g_pre_epoch,
            generator,
            gen_data_iter,
            gen_criterion,
            gen_optimizer,
            BATCH_SIZE,
            seq_len=SEQ_LEN,
            device=device,
        )

        torch.save(generator.state_dict(), f"{CITY_PATH}/pretrain/generator.pth")
        torch.save(discriminator.state_dict(), f"{CITY_PATH}/pretrain/discriminator.pth")

    else:
        generator.load_state_dict(torch.load(f"{CITY_PATH}/pretrain/generator.pth"))
        discriminator.load_state_dict(torch.load(f"{CITY_PATH}/pretrain/discriminator.pth"))

    print("advtrain generator and discriminator ...")
    rollout = Rollout(generator, 0.8)

    gen_gan_loss = GANLoss()
    gen_gan_optm = optim.Adam(generator.parameters(), lr=0.0001)

    dis_criterion = nn.NLLLoss(reduction="sum")
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=0.00001)

    gen_gan_loss = gen_gan_loss.to(device)
    dis_criterion = dis_criterion.to(device)

    epoch_pbar = tqdm(range(EPOCHS), desc="Adversarial training")
    for epoch in epoch_pbar:
        # Train the generator for one step
        for it in range(1):
            with torch.no_grad():
                samples = generator.sample(BATCH_SIZE, SEQ_LEN)
            zeros = torch.zeros((BATCH_SIZE, 1)).type(torch.LongTensor).to(device)
            inputs = torch.cat([zeros, samples.data], dim=1)[:, :-1].contiguous()
            tim = torch.LongTensor([i % 24 for i in range(SEQ_LEN)]).to(device)
            tim = tim.repeat(BATCH_SIZE).reshape(BATCH_SIZE, -1)

            targets = samples.contiguous().view((-1,))
            rewards = rollout.get_reward(samples, 16, discriminator)
            rewards = torch.Tensor(rewards)
            rewards = torch.exp(rewards.to(device)).contiguous().view((-1,))

            gen_gan_optm.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                prob = generator.forward(inputs, tim)
                gloss = gen_gan_loss(prob, targets, rewards, device)

                if ploss_alpha != 0.0:
                    p_crit = period_loss(24)
                    p_crit = p_crit.to(device)
                    gloss += ploss_alpha * p_crit(samples.float())
                if dloss_alpha != 0.0:
                    d_crit = distance_loss(data_path=CITY_PATH, device=device)
                    d_crit = d_crit.to(device)
                    gloss += dloss_alpha * d_crit(samples.float())

            gloss.backward()
            gen_gan_optm.step()

        rollout.update_params()
        for _ in range(4):
            generate_samples(generator, BATCH_SIZE, SEQ_LEN, GENERATED_NUM, GENE_DATA)
            dis_data_iter = DisDataIter(REAL_DATA, GENE_DATA, BATCH_SIZE, seq_len=SEQ_LEN, pad_id=PAD_ID)
            for _ in range(2):
                dloss = train_epoch(
                    "D",
                    discriminator,
                    dis_data_iter,
                    dis_criterion,
                    dis_optimizer,
                    BATCH_SIZE,
                    seq_len=SEQ_LEN,
                    device=device,
                )

        epoch_pbar.set_postfix(gloss=f"{gloss.item():.4f}", dloss=f"{dloss:.4f}")
        logger.info("Epoch [%d] Generator Loss: %f, Discriminator Loss: %f" % (epoch, gloss.item(), dloss))
        with open(f"{CITY_PATH}/logs/loss.log", "a") as f:
            f.write(" ".join([str(j) for j in [epoch, float(gloss.item()), dloss]]) + "\n")

    torch.save(generator.state_dict(), f"{CITY_PATH}/generator.pth")
    torch.save(discriminator.state_dict(), f"{CITY_PATH}/discriminator.pth")

    # Generate one trajectory per test trajectory, starting from the same origin.
    # Results are written to gen_test.data for external evaluation.
    generate_test_trajectories(generator, TEST_DATA, SEQ_LEN, BATCH_SIZE, device, GEN_TEST_DATA)


if __name__ == "__main__":
    main()
