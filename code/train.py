import numpy as np
import torch
from tqdm import tqdm


def generate_samples(model, batch_size, seq_len, generated_num, output_file):
    samples = []
    n_batches = int(generated_num / batch_size)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for _ in tqdm(range(n_batches), desc="Sampling"):
            sample = model.sample(batch_size, seq_len).cpu().data.numpy().tolist()
            samples.extend(sample)
    model.train()
    with open(output_file, "w") as fout:
        for sample in samples:
            string = " ".join([str(s) for s in sample])
            fout.write("%s\n" % string)


def generate_samples_to_mem(model, batch_size, seq_len, generated_num):
    samples = []
    n_batches = int(generated_num / batch_size)
    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(n_batches), desc="Sampling"):
            sample = model.sample(batch_size, seq_len).cpu().data.numpy().tolist()
            samples.extend(sample)
    model.train()
    return np.array(samples)


def pretrain_model(name, pre_epochs, model, data_iter, criterion, optimizer, batch_size, seq_len=48, device=None):
    lloss = 0.0
    criterion = criterion.to(device)
    pbar = tqdm(range(pre_epochs), desc=f"Pretrain {name}")
    for epoch in pbar:
        loss = train_epoch(name, model, data_iter, criterion, optimizer, batch_size, seq_len, device)
        pbar.set_postfix(loss=f"{loss:.4f}")
        if loss < 0.01 or 0 < lloss - loss < 0.01:
            print(f"Early stop at epoch {epoch + 1}")
            break
        lloss = loss


def train_epoch(name, model, data_iter, criterion, optimizer, batch_size, seq_len=48, device=None):
    total_loss = 0.0
    use_amp = device is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if name == "G":
        tim = torch.LongTensor([i % 24 for i in range(seq_len - 1)]).to(device)
        tim = tim.repeat(batch_size).reshape(batch_size, -1)
    pbar = tqdm(enumerate(data_iter), total=len(data_iter), desc=f"  {name} batches")
    for i, (data, target) in pbar:
        data = torch.LongTensor(data).to(device)
        target = torch.LongTensor(target).to(device)
        target = target.contiguous().view(-1)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            if name == "G":
                pred = model(data, tim)
            else:
                pred = model(data)
        loss = criterion(pred.float(), target)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / (i + 1):.4f}")
    data_iter.reset()
    return total_loss / (i + 1)
