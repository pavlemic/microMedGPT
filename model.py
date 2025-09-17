"""
MicroMedGPT model architecture and components
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *

class AttentionHead(nn.Module):
    """Single attention head implementation"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """Multi-head attention implementation"""
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    """Feed forward network"""
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed forward"""
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    Advanced sampling function with temperature, top-k, and top-p sampling
    
    Args:
        logits: tensor of shape (B, vocab_size) for the current time step
        temperature: float, >0, higher values = more random
        top_k: int, keep only top-k tokens
        top_p: float in (0,1], keep smallest set of tokens with cumulative probability >= top_p
    """
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        top_probs, top_idx = probs.topk(top_k)
        mask = torch.zeros_like(probs).scatter_(1, top_idx, top_probs)
        probs = mask / mask.sum(dim=-1, keepdim=True)

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[:, 0] = False  # always keep the top token
        sorted_probs[mask] = 0
        probs = torch.zeros_like(probs).scatter_(1, sorted_idx, sorted_probs)
        probs = probs / probs.sum(dim=-1, keepdim=True)

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

class MicroMedGPTModel(nn.Module):
    """Main MicroMedGPT model"""
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, decode_fn, sep_token="<SEP>", temperature=1.0, top_k=None, top_p=None):
        """
        Generate text from the model
        
        Args:
            idx: (B, T) tensor of input tokens (notes + <SEP>)
            max_new_tokens: maximum number of tokens to generate
            decode_fn: function to decode token ids to text
            sep_token: separator token to split on
            temperature: sampling temperature
            top_k: top-k sampling parameter
            top_p: top-p sampling parameter
            
        Returns:
            Generated report text only
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            idx_next = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            idx = torch.cat((idx, idx_next), dim=1)

        decoded = decode_fn(idx[0].tolist())
        if sep_token in decoded:
            return decoded.split(sep_token)[-1].strip()
        else:
           return decoded

    def count_parameters(self):
        """Count the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save(self.state_dict(), path)
        
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.load_state_dict(torch.load(path, map_location=device))