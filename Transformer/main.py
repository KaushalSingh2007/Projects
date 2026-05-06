#!/usr/bin/env python3
"""
mini_gpt2_plus.py

Mini GPT-2+ training script optimized for local GPU (e.g., RTX 4050).

Features:
- HuggingFace GPT-2 tokenizer
- Decoder-only Transformer (GPT-style) implemented in PyTorch
- Mixed precision training (torch.cuda.amp)
- Checkpoint saving & resume
- Cosine LR schedule
- Dataset building from a single .txt or all .txt files in a directory
- Periodic sampling with top-k + temperature
- Logging to console and to `train_log.txt`

Example:
    python mini_gpt2_plus.py --data_path ./my_corpus.txt --epochs 6 --batch_size 8

Author: Generated for you
"""

import os
import math
import time
import argparse
import glob
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

# --------------------------
# Model: decoder-only GPT style
# --------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.1, resid_dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.resid_dropout = nn.Dropout(resid_dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, T, C)
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, T, head_dim)

        # compute scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, n_heads, T, T)
        if attn_mask is not None:
            # attn_mask expected shape broadcastable to (B, n_heads, T, T)
            att = att.masked_fill(attn_mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, n_heads, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.resid_dropout(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, attn_dropout=dropout, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None):
        # Pre-LN residual blocks
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT2(nn.Module):
    def __init__(self, vocab_size, block_size=512, n_layer=8, n_head=8, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        tok = self.tok_emb(idx)  # (B, T, C)
        pos = self.pos_emb[:, :T, :]  # (1, T, C)
        x = self.drop(tok + pos)

        # create causal mask once per forward: shape (1, 1, T, T) broadcastable to (B, n_heads, T, T)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).unsqueeze(0).unsqueeze(0)  # bool mask allowed
        for b in self.blocks:
            x = b(x, attn_mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        # idx: (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)  # (B, T, V)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                min_v = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# --------------------------
# Dataset utilities
# --------------------------
def read_texts_from_path(path: str) -> List[str]:
    p = Path(path)
    texts = []
    if p.is_file():
        texts.append(p.read_text(encoding='utf-8', errors='ignore'))
    elif p.is_dir():
        for fname in sorted(glob.glob(os.path.join(path, "*.txt"))):
            texts.append(Path(fname).read_text(encoding='utf-8', errors='ignore'))
    else:
        raise ValueError(f"data_path {path} is not file or directory")
    return texts


class CausalLMDataset(Dataset):
    """
    Concatenate all tokens and chunk into block_size windows (like GPT training).
    Returns (x, y) where y is x shifted left by one token.
    """
    def __init__(self, ids: List[int], block_size: int):
        super().__init__()
        self.block_size = block_size
        # total number of full blocks
        self.n = (len(ids) - 1) // block_size
        self.ids = ids

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = torch.tensor(self.ids[start:start + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[start + 1:start + 1 + self.block_size], dtype=torch.long)
        return x, y

# --------------------------
# Training / helpers
# --------------------------
def top_k_sample_logits(logits, top_k=40):
    v, _ = torch.topk(logits, top_k)
    min_v = v[:, -1].unsqueeze(-1)
    return torch.where(logits < min_v, torch.full_like(logits, -1e10), logits)


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device='cuda'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and 'optim_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optim_state'])
    if scheduler is not None and 'sched_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['sched_state'])
    if scaler is not None and 'scaler_state' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state'])
    return checkpoint.get('epoch', 0), checkpoint.get('step', 0), checkpoint

# --------------------------
# Main training script
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a Mini GPT-2 locally")
    parser.add_argument('--data_path', type=str, required=True, help='Path to .txt file or directory of .txt files')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Where to save checkpoints')
    parser.add_argument('--model_name', type=str, default='mini-gpt2-plus', help='Model name prefix for checkpoints')
    parser.add_argument('--block_size', type=int, default=512, help='Context window length')
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_every_steps', type=int, default=500)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample_prompt', type=str, default="In artificial intelligence")
    parser.add_argument('--sample_length', type=int, default=120)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # tokenizer
    print("Loading tokenizer (gpt2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    vocab_size = tokenizer.vocab_size
    print(f"Tokenizer vocab size: {vocab_size}")

    # read texts
    print("Reading texts from:", args.data_path)
    texts = read_texts_from_path(args.data_path)
    if len(texts) == 0:
        raise RuntimeError("No text found at data_path")

    # build ids
    print("Tokenizing and concatenating corpus...")
    all_ids = []
    for t in texts:
        if t is None:
            continue
        # optional minimal cleaning
        s = t.strip()
        if len(s) == 0:
            continue
        ids = tokenizer.encode(s)
        all_ids.extend(ids + [tokenizer.eos_token_id])

    total_tokens = len(all_ids)
    print(f"Total tokens in corpus: {total_tokens:,}")

    if total_tokens < args.block_size:
        raise RuntimeError(f"Corpus is too small ({total_tokens} tokens) for block_size {args.block_size}")

    # dataset and dataloader
    dataset = CausalLMDataset(all_ids, args.block_size)
    print("Num training samples (blocks):", len(dataset))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    # instantiate model
    print("Building model...")
    model = MiniGPT2(vocab_size=vocab_size,
                     block_size=args.block_size,
                     n_layer=args.n_layer,
                     n_head=args.n_head,
                     d_model=args.d_model,
                     d_ff=args.d_ff,
                     dropout=args.dropout).to(device)

    # optimizer, scaler, scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (len(dataloader) // args.gradient_accumulation_steps) * args.epochs
    if total_steps <= 0:
        total_steps = args.epochs
    # Cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    scaler = GradScaler()

    start_epoch = 0
    start_step = 0
    # resume if provided
    if args.resume:
        print("Resuming from checkpoint:", args.resume)
        start_epoch, start_step, ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device=args.device)
        print(f"Resumed at epoch {start_epoch}, step {start_step}")

    # print model size
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {params_m:.2f}M")

    # logging
    log_path = os.path.join(args.save_dir, "train_log.txt")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"Training started: {time.ctime()}\n")
        f.write(json.dumps(vars(args)) + "\n")
    step = start_step
    global_step = start_step
    model.train()

    print("Beginning training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()
        for it, (x_batch, y_batch) in enumerate(dataloader):
            # basic step index accounting
            step += 1
            global_step += 1

            x = x_batch.to(device, non_blocking=True)
            y = y_batch.to(device, non_blocking=True)

            with autocast():
                logits = model(x)  # (B, T, V)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss_value = loss.item()  # float for logging

            # backprop with gradient scaling for FP16
            scaler.scale(loss).backward()

            if (step % args.gradient_accumulation_steps) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += loss_value
            epoch_steps += 1

            # periodic save
            if (global_step % args.save_every_steps) == 0:
                ckpt_path = os.path.join(args.save_dir, f"{args.model_name}_step{global_step}.pt")
                save_checkpoint({
                    'model_state': model.state_dict(),
                    'optim_state': optimizer.state_dict(),
                    'sched_state': scheduler.state_dict(),
                    'scaler_state': scaler.state_dict(),
                    'epoch': epoch,
                    'step': global_step,
                    'args': vars(args)
                }, ckpt_path)
                print(f"[Saved checkpoint] {ckpt_path}")

            # light console logging
            if (global_step % 50) == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{args.epochs} step {global_step} loss {loss_value:.4f} lr {lr:.2e}")

        t1 = time.time()
        avg_loss = epoch_loss / max(1, epoch_steps)
        print(f"Epoch {epoch+1} finished — avg_loss {avg_loss:.4f} — time {(t1-t0):.1f}s")

        # save epoch checkpoint
        ckpt_path = os.path.join(args.save_dir, f"{args.model_name}_epoch{epoch+1}.pt")
        save_checkpoint({
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
            'sched_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'epoch': epoch+1,
            'step': global_step,
            'args': vars(args)
        }, ckpt_path)
        print(f"[Saved checkpoint] {ckpt_path}")

        # sampling after each epoch
        with torch.no_grad():
            prompt = args.sample_prompt
            tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
            out_ids = model.generate(tokens, max_new_tokens=args.sample_length, temperature=args.temperature, top_k=args.top_k)
            sample_text = tokenizer.decode(out_ids[0].tolist(), skip_special_tokens=True)
            print("\n=== Sample ===")
            print(sample_text)
            print("=== End Sample ===\n")

            # append sample to log file
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nEpoch {epoch+1} sample:\n")
                f.write(sample_text + "\n\n")

    print("Training complete!")

if __name__ == "__main__":
    main()
