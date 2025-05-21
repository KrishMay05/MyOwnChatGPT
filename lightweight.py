import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ------------------ Configuration ------------------
class Config:
    # Smaller batch for faster iterations
    batch_size = 6
    # Reduced context length for lightweight training
    block_size = 12
    # Fewer total iterations to speed up training
    max_iters = 2000
    # Evaluate more frequently on a smaller sample
    eval_interval = 200
    # Learning rate remains moderate
    learning_rate = 3e-4
    # Fewer batches for quick loss estimation
    eval_iters = 50
    # Smaller embedding dimension for lightweight model
    n_embd = 200
    # Fewer attention heads to match embedding size
    n_head = 4
    # Fewer transformer layers
    n_layer = 2
    # Moderate regularization
    dropout = 0.1
    # Device config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoding = "o200k_base"

# ------------------ Data Preparation ------------------
enc = tiktoken.get_encoding(Config.encoding)
with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
text_tokens = torch.tensor(enc.encode(raw_text), dtype=torch.long)
train_split = int(0.9 * len(text_tokens))
train_data = text_tokens[:train_split]
val_data = text_tokens[train_split:]
vocab_size = enc.n_vocab

# ------------------ Batch Generator ------------------
def get_batch(split: str):
    """Returns x, y batches of shape (batch_size, block_size)"""
    data = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i + Config.block_size] for i in idx])
    y = torch.stack([data[i + 1:i + Config.block_size + 1] for i in idx])
    return x.to(Config.device), y.to(Config.device)

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = []
        for _ in range(Config.eval_iters):
            X, Y = get_batch(split)
            _, l = model.forward(X, Y)
            losses.append(l.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# ------------------ Model Components ------------------
class Head(nn.Module):
    # A single head of self-attention
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v

class MultiHeadAttention(nn.Module):
    # Multi-head attention is a mechanism that allows the model to attend to different parts of the input sequence, by combining the attention heads
    def __init__(self):
        super().__init__()
        hs = Config.n_embd // Config.n_head
        self.heads = nn.ModuleList([Head(hs) for _ in range(Config.n_head)])
        self.proj = nn.Linear(Config.n_head * hs, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)
    def forward(self, x):
        out = torch.cat([h.forward(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    # FeedForward provides further processing of the output of the multi-head attention by putting it through a linear layer and a non-linear layer
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.n_embd, 4 * Config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * Config.n_embd, Config.n_embd),
            nn.Dropout(Config.dropout),
        )
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    # A single transformer block that contains a multi-head attention and a feedforward network
    # The residual connection is a technique used to help the model learn better by adding the original input back to the output
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(Config.n_embd)
        # Layer normalization normalizes the activations across the feature dimension for each token independently.
        self.ln2 = nn.LayerNorm(Config.n_embd)
    def forward(self, x):
        # x is the input to the layer
        x = x + self.sa.forward(self.ln1(x))
        return x + self.ff.forward(self.ln2(x))

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.pos_emb = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.head = nn.Linear(Config.n_embd, vocab_size)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=Config.device))
        #  turns each token ID into a 128-dim vector 
        x = self.blocks.forward(x)
        # passes the input through the 2 transformer blocks
        x = self.ln_f(x)
        # Layer normalization normalizes the activations across the feature dimension for each token independently.
        logits = self.head(x)
        # logits is the output of the model
        loss = None
        # if targets is not None, then we are in training mode
        if targets is not None:
            logits, targets = logits.view(-1, logits.size(-1)), targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    def generate(self, idx, max_new):
        for _ in range(max_new):
            cond = idx[:, -Config.block_size:]
            # cond is the last 12 tokens of the input
            logits, _ = self.forward(cond)
            # logits is the output of the model
            probs = F.softmax(logits[:, -1, :], dim=-1)
            # probs is the probability of the next token
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            # idx is the input to the next transformer block
        return idx

# ------------------ Training Loop ------------------
def main():
    model = GPTLanguageModel().to(Config.device)
    opt = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    for step in range(Config.max_iters):
        if step % Config.eval_interval == 0 or step == Config.max_iters-1:
            losses = estimate_loss(model)
            print(f"step {step}: train {losses['train']:.4f}, val {losses['val']:.4f}")
        xb, yb = get_batch('train')
        _, loss = model.forward(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    ctx = torch.zeros((1,1), dtype=torch.long, device=Config.device)
    print(enc.decode(model.generate(ctx, max_new=600)[0].tolist()))

if __name__ == '__main__': main()
