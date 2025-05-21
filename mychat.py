import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

# ------------------ Configuration ------------------
class Config:
    # Controls how many sequences the model processes simultaneously during each training step
    batch_size = 8
    # Maximum context length: model looks at up to this many previous tokens to predict the next one
    block_size = 16
    # Total number of weight update iterations
    max_iters = 5000
    # How often (in iterations) to evaluate model performance on train/val splits
    eval_interval = 500
    # Learning rate for the optimizer: step size for weight updates
    learning_rate = 3e-4
    # Number of batches to estimate loss during evaluation
    eval_iters = 200
    # Dimension of token and positional embeddings
    n_embd = 384
    # Number of attention heads: splits embedding dimension into these many subspaces
    n_head = 6
    # Number of transformer blocks (layers) in the model
    n_layer = 6
    # Dropout rate: fraction of activations to zero during training for regularization
    dropout = 0.2
    # Device configuration: GPU if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Name of the subword encoding scheme
    encoding = "o200k_base"

# ------------------ Data Preparation ------------------
# Initialize tokenizer based on Config.encoding
enc = tiktoken.get_encoding(Config.encoding)
# Read raw text from file
with open('input.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
# Convert text to a sequence of token IDs
text_tokens = torch.tensor(enc.encode(raw_text), dtype=torch.long)
# Split 90% train, 10% validation
train_size = int(0.9 * len(text_tokens))
train_data = text_tokens[:train_size]
val_data = text_tokens[train_size:]
# Vocabulary size equals tokenizer's vocabulary count
vocab_size = enc.n_vocab

# ------------------ Batch Generator ------------------
def get_batch(split: str):
    """
    Fetches a batch of inputs (x) and targets (y) for training or validation.
    x has shape (batch_size, block_size)
    y has the same shape but shifted by one token ahead.
    """
    data = train_data if split == 'train' else val_data
    # Random starting indices for each sequence in the batch
    ix = torch.randint(0, len(data) - Config.block_size, (Config.batch_size,))
    x = torch.stack([data[i:i + Config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + Config.block_size + 1] for i in ix])
    return x.to(Config.device), y.to(Config.device)

# ------------------ Loss Estimation ------------------
@torch.no_grad()
def estimate_loss(model):
    """
    Evaluates and returns average training and validation loss over multiple batches,
    without computing gradients.
    """
    model.eval()
    losses = {}
    for split in ('train', 'val'):
        split_losses = []
        for _ in range(Config.eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses

# ------------------ Model Components ------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # Linear projections for key, query, and value
        self.key = nn.Linear(Config.n_embd, head_size, bias=False)
        self.query = nn.Linear(Config.n_embd, head_size, bias=False)
        self.value = nn.Linear(Config.n_embd, head_size, bias=False)
        # Lower-triangular mask to prevent attention to future tokens
        self.register_buffer('mask', torch.tril(torch.ones(Config.block_size, Config.block_size)))
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, embedding dim
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        # Compute raw attention scores, scale by sqrt of head_size
        scores = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        # Mask out future positions
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        # Convert scores to probabilities
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        v = self.value(x) # (B, T, head_size)
        # Weighted sum of values
        return attn @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = Config.n_embd // Config.n_head
        # Create multiple Heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(Config.n_head)])
        # Final linear layer to combine head outputs back to embedding size
        self.proj = nn.Linear(Config.n_head * head_size, Config.n_embd)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project back and apply dropout
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        # Two-layer MLP with ReLU activation in between
        self.net = nn.Sequential(
            nn.Linear(Config.n_embd, 4 * Config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * Config.n_embd, Config.n_embd),
            nn.Dropout(Config.dropout),
        )

    def forward(self, x):
        # Applies feed-forward network token-wise
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ff = FeedForward()
        self.ln1 = nn.LayerNorm(Config.n_embd)
        self.ln2 = nn.LayerNorm(Config.n_embd)

    def forward(self, x):
        # Rusted skip connections: add attention output to input
        x = x + self.sa(self.ln1(x))
        # Add feed-forward output to current representation
        x = x + self.ff(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, Config.n_embd)
        self.pos_emb = nn.Embedding(Config.block_size, Config.n_embd)
        # Stacked transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)    # Final layer norm
        self.head = nn.Linear(Config.n_embd, vocab_size)  # Project to vocab size
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            # Normal initialization for stability
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass: returns logits and cross-entropy loss (if targets provided).
        idx: (B, T) indices of tokens
        targets: (B, T) next-token indices for loss computation
        """
        B, T = idx.shape
        # Sum token and position embeddings
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=Config.device))
        # Pass through transformer blocks
        x = self.blocks(x)
        # Final normalization
        x = self.ln_f(x)
        logits = self.head(x)  # Raw scores for each token
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive text generation:
        - Repeatedly predict next token, append to sequence
        - Returns final sequence indices
        """
        for _ in range(max_new_tokens):
            # Crop to last block_size tokens
            idx_cond = idx[:, -Config.block_size:]
            logits, _ = self(idx_cond)
            # Focus on last time step
            next_logits = logits[:, -1, :]
            probs = F.softmax(next_logits, dim=-1)
            # Sample next token id
            next_idx = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, next_idx), dim=1)
        return idx

# ------------------ Training Loop ------------------
def main():
    # Instantiate model and optimizer
    model = GPTLanguageModel().to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)

    # Print parameter count for reference
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params / 1e6:.2f}M parameters")

    # Training iterations
    for step in range(Config.max_iters):
        # Periodic evaluation
        if step % Config.eval_interval == 0 or step == Config.max_iters - 1:
            losses = estimate_loss(model)
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Fetch batch, compute loss, backpropagate, and update weights
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Example: generate new text from an empty context
    context = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
    generated = model.generate(context, max_new_tokens=500)
    print(enc.decode(generated[0].tolist()))

if __name__ == '__main__':
    main()