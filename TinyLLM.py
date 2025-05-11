#-Author: Anup Buchke (anup.estuff@gmail.com)
# ===========================================
# Forward Pass Flow (Step-by-Step with Tensors)
#
# #1: x → embedding[x]                     → shape: [batch_size, seq_len, embed_dim]
# #2: embed + position[seq_idx]           → positional embedding added
# #3: Q = Wq * embed, K = Wk * embed, V = Wv * embed → linear projection
# #4: attention_scores = Q @ K.T / sqrt(d_k)        → [batch, heads, seq, seq]
# #5: weights = softmax(attention_scores)           → normalized attention
# #6: context = weights @ V                         → [batch, heads, seq, head_dim]
# #7: context_merged = concat(heads)                → [batch, seq, embed_dim]
# #8: mlp_output = MLP(context_merged)              → feed-forward transformation
# #9: logits = output_layer(mlp_output[:, -1, :])   → predict token
#
# This flow maps input token indices to vocab logits used in CrossEntropyLoss.
# ===========================================

# ===========================================
# Backpropagation Flow (Step-by-Step with Formulas)
#
# Suppose L is the total loss and y_hat = logits from self.output
#
# #1: dL/d(logits) is computed via CrossEntropyLoss
# #2: dL/d(output) = dL/d(logits) * d(logits)/d(output)
# #3: dL/d(MLP_2)  = dL/d(output) (#2) * d(output)/d(MLP_2)
# #4: dL/d(MLP_1)  = dL/d(MLP_2) (#3) * d(MLP_2)/d(MLP_1)
# #5: dL/d(context)= dL/d(MLP_1) (#4) * d(MLP)/d(context)
#
# -- Now backprop into attention layers --
#
# #6: dL/d(Q) = dL/d(context) (#5) * d(context)/d(Q)  (similarly for K, V)
# #7: dL/d(Wq) = dL/d(Q) (#6) * d(Q)/d(Wq)            (same for Wk, Wv)
# #8: dL/d(embed) = dL/d(Q) (#7) * d(Q)/d(embed) + \
#                  dL/d(K) (#7) * d(K)/d(embed) + \
#                  dL/d(V) (#7) * d(V)/d(embed)
#
# Embedding, position, attention (QKV), MLP and output all receive gradients.
# ===========================================

# ===========================================
# Backpropagation + Logits Explanation
#
# For each sequence in a batch:
# - The final context vector from attention has shape: [batch_size, embed_dim]
# - This is fed into a linear output projection layer (W_out) of shape [embed_dim, vocab_size]
# - This produces 'logits' of shape [batch_size, vocab_size]
#   → which represent the model's confidence scores for **every word in the vocabulary**
# - These logits are compared with y_batch (target token index) using CrossEntropyLoss
#   (which internally applies softmax + negative log likelihood)
#
# NOTE:
# W_out is **not** the same as the original embedding matrix,
# but it projects from the same embedding space **into vocabulary space**
# (i.e., reverse direction: embed_dim → vocab_size)
# ===========================================

# ===========================================
# Backpropagation flow through this model
# Given: Loss L and final output logits (from self.output)
# The gradients are calculated layer by layer during backward() as:
#
# dL/d(output_layer)     = ∂L/∂logits → from CrossEntropy
# dL/d(MLP_2)            = dL/d(output) * d(output)/d(MLP)
# dL/d(MLP_1)            = dL/d(MLP_2) * d(MLP_2)/d(MLP_1)
# dL/d(attn_context)     = dL/d(MLP) * d(MLP)/d(attn_context)
# dL/d(Q, K, V)          = ∂L/∂context * d(context)/d(Q, K, V)
# dL/d(Wq/Wk/Wv weights) = ∂L/∂Q * dQ/dWq (and similarly for K, V)
# dL/d(embedding)        = ∂L/∂Q * dQ/d(embed) + ∂L/∂K * dK/d(embed) + ∂L/∂V * dV/d(embed)
#
# The embedding and all linear layers (including Wq, Wk, Wv, MLP, output)
# receive gradients automatically during .backward() and are updated with .step()
#
# Batch size determines how many such sequences are processed in parallel
# Attention runs for all batch entries concurrently using tensorized operations
# ===========================================

# ===========================================
# This script trains a simple multi-head self-attention model.
# Each batch contains multiple sequences (sentences), where:
# - Each sequence is a list of 20 token indices (words)
# - The first 19 tokens (x_batch) are the input to the attention block
# - The 20th token (y_batch) is the actual label/target to predict
# So for batch_size = 32:
# - x_batch has shape [32, 19]
# - y_batch has shape [32]
# ===========================================

import time
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import os

# === Load and tokenize data ===
file_path = "wiki_sentences.txt"
with open(file_path, "r") as f:
    sentences = [line.strip().lower() for line in f if line.strip()]

vocab = {"<pad>": 0, "<unk>": 1}
word2idx = defaultdict(lambda: 1, vocab)
idx2word = ["<pad>", "<unk>"]
for sentence in sentences:
    for word in sentence.replace(".", "").split():
        if word not in word2idx:
            word2idx[word] = len(idx2word)
            idx2word.append(word)
vocab_size = len(idx2word)

def encode(sentence, max_len=20):
    tokens = sentence.replace(".", "").split()
    ids = [word2idx[w] for w in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids[:-1], ids[-1]

encoded = [encode(s) for s in sentences]
random.shuffle(encoded)
split = int(0.8 * len(encoded))
train_data = encoded[:split]
dev_data = encoded[split:]

class SentenceDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

train_loader = DataLoader(SentenceDataset(train_data), batch_size=32, shuffle=True)
dev_loader = DataLoader(SentenceDataset(dev_data), batch_size=32)

# === Model Definition with limited logs ===
class CustomSelfAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, max_len=20, num_heads=2):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.logged_once = False

        # This embedding matrix has shape [vocab_size, embed_dim]
        # Each row represents a vocabulary word as a dense vector of length embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.position = nn.Embedding(max_len, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.output = nn.Linear(embed_dim, vocab_size)

    def compute_multihead_attention(self, Q, K, V, batch_size, seq_len):
        if not self.logged_once:
            print("Q/K/V pre-split:", Q.shape)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        if not self.logged_once:
            print("Q/K/V post-split:", Q.shape)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        if not self.logged_once:
            print("Attention scores:", scores.shape)

        weights = F.softmax(scores, dim=-1)
        if not self.logged_once:
            print("Attention weights:", weights.shape)

        context = weights @ V
        if not self.logged_once:
            print("Context (per head):", context.shape)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        if not self.logged_once:
            print("Context (merged heads):", context.shape)
        return context

    def forward(self, x):
        batch_size, seq_len = x.size()

        if not self.logged_once:
            print("Input x:", x.shape)

        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        embed = self.embedding(x) + self.position(positions)

        if not self.logged_once:
            print("Embed+Position:", embed.shape)

        Q = self.Wq(embed)
        K = self.Wk(embed)
        V = self.Wv(embed)

        context = self.compute_multihead_attention(Q, K, V, batch_size, seq_len)
        context = self.mlp(context)

        if not self.logged_once:
            print("Post-MLP context:", context.shape)
            self.logged_once = True

        return self.output(context[:, -1, :])

# === Training with validation tracking ===
def train_model(model, train_loader, dev_loader, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    train_losses = []
    dev_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x_batch, y_batch in train_loader:
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_dev_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dev_loader:
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
                total_dev_loss += loss.item()
        avg_dev_loss = total_dev_loss / len(dev_loader)
        dev_losses.append(avg_dev_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Dev Loss = {avg_dev_loss:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(dev_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    plt.show()

# === Inference Function ===
def generate_sequence(model, prompt, max_words=10, max_len=20):
    model.eval()
    tokens = prompt.lower().replace(".", "").split()
    generated = tokens[:]
    for _ in range(max_words):
        ids = [word2idx[w] for w in generated[-(max_len - 1):]]
        if len(ids) < max_len - 1:
            ids = [0] * (max_len - 1 - len(ids)) + ids
        x = torch.tensor([ids])
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            next_word = idx2word[next_token_id]
            generated.append(next_word)
            if next_word == "<pad>":
                break
    return " ".join(generated)

# === Run ===
start_time = time.time()
model = CustomSelfAttention(vocab_size)
train_model(model, train_loader, dev_loader)
prompt = "A new grammar book was"
print(f"Prompt: {prompt}")
print(f"Generated: {generate_sequence(model, prompt)}")
print(f"Elapsed time: {time.time() - start_time:.2f} sec")
