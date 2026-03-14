import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# (Assumes MultiHeadAttention, PositionwiseFFN, Encoder from earlier chapters)

# ─────────────────────────────────────────────────────────────────
# FULL DECODER LAYER (with cross-attention)
# ─────────────────────────────────────────────────────────────────
class FullDecoderLayer(nn.Module):
    """
    Decoder layer with THREE sub-layers:
      1. Masked self-attention   (Q=K=V=tgt, causal mask)
      2. Cross-attention         (Q=tgt, K=V=encoder_memory)
      3. Position-wise FFN

    Each sub-layer: residual connection + LayerNorm.
    """
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # Sub-layer 1: Masked self-attention on target
        self.self_attn   = MultiHeadAttention(d_model, h, dropout)
        # Sub-layer 2: Cross-attention — attends to encoder output
        self.cross_attn  = MultiHeadAttention(d_model, h, dropout)
        # Sub-layer 3: FFN
        self.ffn         = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.norm3   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(seq_len, device):
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

    def forward(self, tgt, memory, src_mask=None):
        """
        Args:
            tgt:      target representations    (B, T_tgt, d_model)
            memory:   encoder output (fixed)    (B, T_src, d_model)
            src_mask: padding mask on source    (B, 1, 1, T_src)
        """
        T, device = tgt.size(1), tgt.device
        tgt_mask = self._causal_mask(T, device)

        # Step 1: Masked Self-Attention on target sequence
        #         Q = K = V = tgt, masked so token i only sees ≤ i
        _self = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt   = self.norm1(tgt + self.dropout(_self))

        # Step 2: Cross-Attention
        #         Q from decoder, K and V from encoder memory
        _cross = self.cross_attn(tgt, memory, memory, src_mask)
        tgt    = self.norm2(tgt + self.dropout(_cross))

        # Step 3: Feed-Forward
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt


# ─────────────────────────────────────────────────────────────────
# FULL ENCODER-DECODER TRANSFORMER
# ─────────────────────────────────────────────────────────────────
class EncoderDecoderTransformer(nn.Module):
    """
    Full Transformer [1].

    Encoder: reads source → encoder memory
    Decoder: attends to memory → generates target tokens

    Input:  src token ids (B, src_len),  tgt token ids (B, tgt_len)
    Output: logits (B, tgt_len, tgt_vocab_size)
    """
    def __init__(self,
                 src_vocab_size, tgt_vocab_size,
                 d_model=512, N=6, h=8, d_ff=2048,
                 max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # ── Encoder ─────────────────────────────────────────────
        self.encoder  = Encoder(src_vocab_size, d_model, N, h,
                                d_ff, max_len, dropout)

        # ── Decoder ─────────────────────────────────────────────
        self.tgt_embed  = nn.Embedding(tgt_vocab_size, d_model)
        self.dec_layers = nn.ModuleList(
            [FullDecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.dec_norm   = nn.LayerNorm(d_model)
        self.dec_dropout = nn.Dropout(dropout)

        # ── Output projection ────────────────────────────────────
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Sinusoidal positional encoding (shared by encoder and decoder)
        self.register_buffer('pe', Encoder._build_pe(max_len, d_model))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """Run the encoder. Returns memory (B, src_len, d_model)."""
        return self.encoder(src, src_mask)

    def decode(self, tgt, memory, src_mask=None):
        """
        Run the decoder.
        Args:
            tgt:    target token ids  (B, tgt_len)
            memory: encoder output    (B, src_len, d_model)
        Returns:
            decoder output            (B, tgt_len, d_model)
        """
        # Embed target tokens and add positional encoding
        tgt_len = tgt.size(1)
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pe[:, :tgt_len, :]
        x = self.dec_dropout(x)

        # Pass through all N decoder layers
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask)

        return self.dec_norm(x)   # (B, tgt_len, d_model)

    def forward(self, src, tgt, src_mask=None):
        """
        Full forward pass.

        During training, tgt is the target sequence shifted right:
        e.g., if target = [BOS, w1, w2, w3, EOS]
        then tgt_in  = [BOS, w1, w2, w3]  (input to decoder)
        and  tgt_out = [w1, w2, w3, EOS]  (labels for loss)
        """
        memory = self.encode(src, src_mask)       # (B, src_len, d_model)
        dec_out = self.decode(tgt, memory, src_mask)  # (B, tgt_len, d_model)
        logits  = self.output_proj(dec_out)       # (B, tgt_len, tgt_vocab)
        return logits


# ─────────────────────────────────────────────────────────────────
# TRAINING WITH LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────────
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing (epsilon = 0.1).

    Instead of a one-hot target, the true class gets probability (1-ε)
    and all other classes share ε / (V-1). This prevents overconfidence
    and acts as regularisation.

    Loss = -(1-ε)·log P(y_true) - ε/V · Σ log P(y_i)
    """
    def __init__(self, vocab_size, pad_idx=0, epsilon=0.1):
        super().__init__()
        self.epsilon   = epsilon
        self.pad_idx   = pad_idx
        self.vocab_size = vocab_size

    def forward(self, logits, targets):
        # logits:  (B*T, vocab_size)
        # targets: (B*T,)

        B = logits.size(0)
        log_probs = F.log_softmax(logits, dim=-1)  # (B*T, V)

        # Smooth target distribution
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.epsilon / (self.vocab_size - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
            # Zero out padding tokens
            smooth[targets == self.pad_idx] = 0

        loss = -(smooth * log_probs).sum() / (targets != self.pad_idx).sum().float()
        return loss


# ─────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────
def train_step(model, optimizer, src, tgt, criterion, pad_idx=0):
    """
    One training step for the encoder-decoder Transformer.

    Args:
        src:       source token ids     (B, src_len)
        tgt:       target token ids     (B, tgt_len+1)  includes BOS and EOS
        criterion: LabelSmoothingLoss or nn.CrossEntropyLoss
    """
    model.train()

    # Shift target right for teacher forcing
    tgt_in  = tgt[:, :-1]   # BOS + words (decoder input)
    tgt_out = tgt[:, 1:]    # words + EOS  (target labels)

    # Build source padding mask
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,src_len)

    # Forward pass
    logits = model(src, tgt_in, src_mask)              # (B, tgt_len, vocab)

    # Compute loss (flatten batch and sequence dimensions)
    loss = criterion(
        logits.view(-1, logits.size(-1)),              # (B*T, vocab)
        tgt_out.reshape(-1)                            # (B*T,)
    )

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping — prevents exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    return loss.item()


# ─────────────────────────────────────────────────────────────────
# AUTOREGRESSIVE INFERENCE (GREEDY)
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def translate_greedy(model, src, bos_idx, eos_idx, max_len=100, pad_idx=0):
    """
    Greedy decoding for sequence-to-sequence generation.
    Generates one token at a time until EOS or max_len.
    """
    model.eval()
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    # Encode source once
    memory = model.encode(src, src_mask)   # (B, src_len, d_model)

    # Start decoding with BOS token
    B = src.size(0)
    tgt = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)

    for step in range(max_len):
        # Decode current target sequence
        dec_out = model.decode(tgt, memory, src_mask)   # (B, T, d_model)
        logits  = model.output_proj(dec_out[:, -1, :])  # (B, vocab)

        # Greedy: pick highest-probability token
        next_token = logits.argmax(dim=-1, keepdim=True)  # (B, 1)
        tgt = torch.cat([tgt, next_token], dim=1)

        # Stop if all sequences have generated EOS
        if (next_token.squeeze(-1) == eos_idx).all():
            break

    return tgt  # (B, generated_length)
