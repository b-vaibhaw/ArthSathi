"""
arthasathi_model.py
====================
ArthaSathi LLM — Complete Decoder-Only Transformer from Scratch
Architecture: GPT-2 family + LLaMA improvements (RoPE, SwiGLU, RMSNorm, weight tying)
Parameters: 117M / 345M / 774M — configurable
Languages: Hindi, English, Marathi, Tamil, Kannada, Bhojpuri, Assamese, Bengali, Telugu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json
from dataclasses import dataclass, asdict
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 1. MODEL CONFIGURATIONS
# ─────────────────────────────────────────────────────────────

@dataclass
class ArthaSathiConfig:
    """
    Model hyperparameters.

    Small  (117M): d_model=768,  n_heads=12, n_layers=12, d_ff=3072  – Colab T4 free
    Medium (345M): d_model=1024, n_heads=16, n_layers=24, d_ff=4096  – RECOMMENDED
    Large  (774M): d_model=1280, n_heads=20, n_layers=36, d_ff=5120  – 24GB VRAM
    """
    vocab_size:     int   = 60000    # custom multilingual BPE
    context_length: int   = 1024
    d_model:        int   = 1024
    n_heads:        int   = 16
    n_layers:       int   = 24
    d_ff:           int   = 4096
    dropout:        float = 0.10
    attn_dropout:   float = 0.10
    rope_base:      float = 10000.0
    pad_token_id:   int   = 0
    bos_token_id:   int   = 2
    eos_token_id:   int   = 3

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ArthaSathiConfig":
        with open(path) as f:
            return cls(**json.load(f))


def get_small_config() -> ArthaSathiConfig:
    return ArthaSathiConfig(vocab_size=60000, context_length=512,
                            d_model=768, n_heads=12, n_layers=12, d_ff=3072)

def get_medium_config() -> ArthaSathiConfig:
    return ArthaSathiConfig(vocab_size=60000, context_length=1024,
                            d_model=1024, n_heads=16, n_layers=24, d_ff=4096)

def get_large_config() -> ArthaSathiConfig:
    return ArthaSathiConfig(vocab_size=60000, context_length=2048,
                            d_model=1280, n_heads=20, n_layers=36, d_ff=5120)


# ─────────────────────────────────────────────────────────────
# 2. FUNDAMENTAL BUILDING BLOCKS
# ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler and faster than LayerNorm; used by LLaMA, Mistral, Gemma.
    Formula: x / RMS(x) * weight
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then back to original dtype
        return self._norm(x.float()).type_as(x) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    Encodes position by rotating Q and K vectors.
    Advantages: relative position awareness, generalises to unseen lengths.
    """
    def __init__(self, dim: int, base: float = 10000.0, max_seq: int = 4096):
        super().__init__()
        self.dim  = dim
        self.base = base
        inv_freq  = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t    = torch.arange(seq_len, device=self.inv_freq.device).float()
        freq = torch.einsum("i,j->ij", t, self.inv_freq)
        emb  = torch.cat([freq, freq], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None])  # (1,1,T,dim)
        self.register_buffer("sin_cache", emb.sin()[None, None])

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cache.shape[2]:
            self._build_cache(seq_len)
        return (self.cos_cache[:, :, :seq_len].to(device),
                self.sin_cache[:, :, :seq_len].to(device))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor,
               cos: torch.Tensor, sin: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward block.
    FFN(x) = W_out(SiLU(W_gate * x) ⊙ (W_up * x))
    Used by LLaMA, PaLM, Gemma. Outperforms GELU/ReLU FFNs.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_out  = nn.Linear(d_ff, d_model, bias=False)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_out(self.drop(F.silu(self.w_gate(x)) * self.w_up(x)))


# ─────────────────────────────────────────────────────────────
# 3. MULTI-HEAD SELF-ATTENTION
# ─────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with:
    - RoPE positional embeddings
    - Causal (autoregressive) masking
    - Flash Attention 2 (PyTorch 2.0+ scaled_dot_product_attention)
    - KV Cache for fast inference
    """
    def __init__(self, config: ArthaSathiConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads  = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.d_model  = config.d_model

        self.Wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wk = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wv = nn.Linear(config.d_model, config.d_model, bias=False)
        self.Wo = nn.Linear(config.d_model, config.d_model, bias=False)

        self.rope       = RotaryEmbedding(self.head_dim, config.rope_base, config.context_length)
        self.attn_drop  = nn.Dropout(config.attn_dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        # KV Cache — populated during inference, None during training
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                cache_pos: int = 0) -> torch.Tensor:
        B, T, C = x.shape

        q = self.Wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(cache_pos + T, x.device)
        cos, sin = cos[:, :, cache_pos:cache_pos + T], sin[:, :, cache_pos:cache_pos + T]
        q, k     = apply_rope(q, k, cos, sin)

        # KV cache for autoregressive inference
        if use_cache:
            self.cache_k = k if self.cache_k is None else torch.cat([self.cache_k, k], dim=2)
            self.cache_v = v if self.cache_v is None else torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v

        # Attention computation — Flash Attention if available
        try:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=(mask is None and not use_cache),
            )
        except Exception:
            # Manual fallback
            scale  = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is None:
                causal = torch.triu(
                    torch.full((q.shape[2], k.shape[2]), float('-inf'), device=x.device), 1
                )
                scores = scores + causal
            weights = F.softmax(scores, dim=-1)
            out     = torch.matmul(self.attn_drop(weights), v)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.Wo(out))

    def clear_cache(self):
        self.cache_k = None
        self.cache_v = None


# ─────────────────────────────────────────────────────────────
# 4. TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Single pre-norm transformer decoder block.
    Pre-norm: normalize BEFORE attention/FFN → more stable training.
    Two residual connections: x += attn(norm1(x)); x += ffn(norm2(x))
    """
    def __init__(self, config: ArthaSathiConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = MultiHeadAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = SwiGLU(config.d_model, config.d_ff, config.dropout)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                cache_pos: int = 0) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask, use_cache, cache_pos)
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────
# 5. FULL LANGUAGE MODEL
# ─────────────────────────────────────────────────────────────

class ArthaSathiLLM(nn.Module):
    """
    ArthaSathi — Full decoder-only language model.

    Key design choices:
    1. Weight tying: lm_head.weight = embedding.weight
       → 60K * 1024 = 61.4M fewer parameters, better quality
    2. Pre-norm architecture: stabler training than post-norm
    3. RoPE: no learned positional embeddings needed
    4. SwiGLU: better than ReLU/GELU in every benchmark
    5. RMSNorm: simpler and faster than LayerNorm
    """
    def __init__(self, config: ArthaSathiConfig):
        super().__init__()
        self.config = config

        self.embedding  = nn.Embedding(config.vocab_size, config.d_model,
                                       padding_idx=config.pad_token_id)
        self.embed_drop = nn.Dropout(config.dropout)
        self.blocks     = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm       = RMSNorm(config.d_model)
        self.lm_head    = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # ─── Weight tying (THE most important trick) ───
        self.lm_head.weight = self.embedding.weight

        # Standard init
        self.apply(self._init_weights)

        # Residual path scaled init (GPT-2 paper technique)
        for name, p in self.named_parameters():
            if name.endswith(("Wo.weight", "w_out.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

        total = sum(p.numel() for p in self.parameters())
        print(f"[ArthaSathiLLM] Initialized: {total / 1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids: torch.Tensor,
                targets: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                use_cache: bool = False,
                cache_pos: int = 0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids : (B, T) — integer token ids
            targets   : (B, T) — next-token labels; if provided, returns loss
            mask      : optional attention mask override
            use_cache : enable KV cache for inference
            cache_pos : offset for KV cache

        Returns:
            logits : (B, T, V) if targets given else (B, 1, V)
            loss   : scalar cross-entropy loss or None
        """
        B, T = input_ids.shape
        assert T <= self.config.context_length, (
            f"Input length {T} > context_length {self.config.context_length}"
        )

        x = self.embed_drop(self.embedding(input_ids))           # (B, T, d_model)

        for block in self.blocks:
            x = block(x, mask, use_cache, cache_pos)

        x = self.norm(x)                                          # (B, T, d_model)

        if targets is not None:
            logits = self.lm_head(x)                              # (B, T, V)
            loss   = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])                  # (B, 1, V)
            loss   = None

        return logits, loss

    def clear_cache(self):
        for block in self.blocks:
            block.attn.clear_cache()

    # ─── Text Generation ─────────────────────────────────────
    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 repetition_penalty: float = 1.1,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation with:
          temperature  : controls randomness (0 = greedy, 1 = full stochastic)
          top_k        : keep only top-k logits
          top_p        : nucleus sampling — keep tokens summing to top_p probability
          rep_penalty  : penalise already-generated tokens
        """
        self.eval()
        self.clear_cache()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        generated = input_ids.clone()
        B         = input_ids.shape[0]
        done      = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        for step in range(max_new_tokens):
            if step == 0:
                logits, _ = self.forward(generated, use_cache=True, cache_pos=0)
            else:
                logits, _ = self.forward(generated[:, -1:], use_cache=True,
                                         cache_pos=generated.shape[1] - 1)
            logits = logits[:, -1, :].clone()                     # (B, V)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(B):
                    for tid in set(generated[b].tolist()):
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

            # Temperature
            if temperature > 0:
                logits = logits / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)   # (B, 1)
            done     |= (next_tok.squeeze(-1) == eos)
            generated = torch.cat([generated, next_tok], dim=1)
            if done.all():
                break

        self.clear_cache()
        return generated

    # ─── Persistence ─────────────────────────────────────────
    def save_checkpoint(self, path: str, step: int = 0,
                        optimizer=None, scheduler=None):
        payload = {
            "model_state_dict": self.state_dict(),
            "config":           asdict(self.config),
            "step":             step,
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state_dict"] = scheduler.state_dict()
        torch.save(payload, path)
        print(f"[Checkpoint] Saved step {step} → {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ArthaSathiLLM":
        ckpt   = torch.load(path, map_location=device)
        config = ArthaSathiConfig(**ckpt["config"])
        model  = cls(config)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        print(f"[Checkpoint] Loaded step {ckpt.get('step', '?')} from {path}")
        return model


# ─────────────────────────────────────────────────────────────
# 6. SMOKE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n{'='*50}")

    for label, fn in [("Small-117M", get_small_config), ("Medium-345M", get_medium_config)]:
        cfg   = fn()
        model = ArthaSathiLLM(cfg).to(device)
        ids   = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        tgt   = torch.randint(0, cfg.vocab_size, (2, 32), device=device)
        lg, loss = model(ids, tgt)
        print(f"{label}: logits={lg.shape}  loss={loss.item():.4f}")
        gen = model.generate(ids[:1, :8], max_new_tokens=10)
        print(f"  generate: {gen.shape[1]} total tokens")

    print("\nAll architecture tests PASSED ✓")
