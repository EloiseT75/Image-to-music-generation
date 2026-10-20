import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # (B, T, head_dim)
        q = self.query(x)    # (B, T, head_dim)

        att = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)   # (B, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        v = self.value(x)
        out = att @ v        # (B, T, head_dim)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        head_dim = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_dim, block_size, dropout)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ABCTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device)
        tok_emb = self.token_embedding(idx)              # (B, T, C)
        pos_emb = self.position_embedding(pos)           # (T, C)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                         # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

        return idx