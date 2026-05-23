import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
 
#──────────────────────────────────────────────────────────────── 
# DECODER LAYER (GPT style — no cross-attention) 
#──────────────────────────────────────────────────────────────── 
class DecoderLayer(nn.Module): 
    """ 
    Single decoder layer with two sub-layers: 
      1. Masked Multi-Head Self-Attention + Add & Norm 
      2. Position-wise FFN                + Add & Norm 
    """ 
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1): 
        super().__init__() 
        self.masked_attn = MultiHeadAttention(d_model, h, dropout) 
        self.ffn         = PositionwiseFFN(d_model, d_ff, dropout) 
        self.norm1       = nn.LayerNorm(d_model) 
        self.norm2       = nn.LayerNorm(d_model) 
        self.dropout     = nn.Dropout(dropout) 
 
    @staticmethod 
    def _causal_mask(seq_len, device): 
        """ 
        Lower-triangular matrix: position i can only attend to positions 
<= i. 
        Shape: (1, 1, seq_len, seq_len) — broadcasts over (B, h, seq, 
seq). 
        """ 
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)) 
        return mask.unsqueeze(0).unsqueeze(0) 
 
    def forward(self, x): 
        T, device = x.size(1), x.device 
 
        # Build causal mask on-the-fly for this sequence length 
        causal_mask = self._causal_mask(T, device) 
 
        # Sub-layer 1: Masked self-attention + residual + LayerNorm 
        _attn = self.masked_attn(x, x, x, causal_mask) 
        x = self.norm1(x + self.dropout(_attn)) 
 
        # Sub-layer 2: FFN + residual + LayerNorm 
        x = self.norm2(x + self.dropout(self.ffn(x))) 
        return x 
 
 
 
#──────────────────────────────────────────────────────────────── 
# FULL GPT-STYLE DECODER MODEL 
#──────────────────────────────────────────────────────────────── 
class GPTDecoder(nn.Module): 
    """ 
    Full GPT-style decoder-only Transformer. 
 
    Uses learned positional embeddings (not sinusoidal). 
    Causal masking is applied in every decoder layer. 
 
    Input:  token indices (B, seq_len) 
    Output: logits over vocabulary (B, seq_len, vocab_size) 
    """ 
    def __init__(self, vocab_size, d_model=512, N=6, h=8, 
                 d_ff=2048, max_len=1024, dropout=0.1): 
        super().__init__() 
        self.d_model    = d_model 
 
        # Token and position embeddings (both learned) 
        self.token_embed = nn.Embedding(vocab_size, d_model) 
        self.pos_embed   = nn.Embedding(max_len, d_model) 
 
        # Stack of decoder layers 
        self.layers   = nn.ModuleList( 
            [DecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)] 
        ) 
        self.norm     = nn.LayerNorm(d_model) 
        self.dropout  = nn.Dropout(dropout) 
 
        # Language model head: projects d_model → vocab_size 
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False) 
 
        # Weight tying: share embedding and lm_head weights 
        # (common optimisation that reduces parameters and often improves 
performance) 
        self.lm_head.weight = self.token_embed.weight 
 
        self._init_weights() 
 
    def _init_weights(self): 
        """Initialise weights with small Gaussian values.""" 
        for module in self.modules(): 
            if isinstance(module, nn.Linear): 
                nn.init.normal_(module.weight, mean=0.0, std=0.02) 
                if module.bias is not None: 
                    nn.init.zeros_(module.bias) 
            elif isinstance(module, nn.Embedding): 
                nn.init.normal_(module.weight, mean=0.0, std=0.02) 
 
    def forward(self, tokens): 
        B, T = tokens.shape 
 
        # Token embeddings scaled by √d_model 
        tok_emb = self.token_embed(tokens) * math.sqrt(self.d_model) 
 
        # Positional embeddings for positions 0, 1, ..., T-1 
        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # 
(1, T) 
        pos_emb   = self.pos_embed(positions)                             
# (1, T, d_model) 
 
        # Combine and apply dropout 
        x = self.dropout(tok_emb + pos_emb)   # (B, T, d_model) 
 
        # Pass through all decoder layers 
        for layer in self.layers: 
            x = layer(x) 
 
        x = self.norm(x)                       # Final layer norm 
        return self.lm_head(x)                 # (B, T, vocab_size) — 
logits 
 
    def compute_loss(self, tokens): 
        """ 
        Compute cross-entropy language modelling loss. 
        Input:  tokens (B, T+1) 
        Predicts: tokens[1:] from tokens[:-1] (next-token prediction) 
        """ 
        src    = tokens[:, :-1]    # input:  first T tokens 
        target = tokens[:, 1:]     # target: last  T tokens (shifted by 
1) 
 
        logits = self.forward(src)                         # (B, T, 
vocab_size) 
        loss   = F.cross_entropy( 
            logits.view(-1, logits.size(-1)),  # (B*T, vocab_size) 
            target.reshape(-1),                # (B*T,) 
        ) 
        return loss 
 
 
#──────────────────────────────────────────────────────────────── 
# TEXT GENERATION WITH SAMPLING STRATEGIES 
#──────────────────────────────────────────────────────────────── 
@torch.no_grad() 
def generate(model, prompt_ids, max_new_tokens=100, 
             temperature=1.0, top_k=None, top_p=None): 
    """ 
    Auto-regressive text generation with multiple sampling strategies. 
 
    Args: 
        model:          trained GPTDecoder 
        prompt_ids:     token indices of the prompt (1, T) 
        max_new_tokens: number of tokens to generate 
        temperature:    softmax temperature (lower = more deterministic) 
        top_k:          if set, sample from top-k tokens 
        top_p:          if set, sample from top-p nucleus 
    """ 
    model.eval() 
    ids = prompt_ids.clone()  # (1, T) 
 
    for _ in range(max_new_tokens): 
        # Forward pass 
        logits = model(ids)[:, -1, :]  # (1, vocab_size) — last position 
only 
 
        # Apply temperature scaling 
        logits = logits / temperature 
 
        # Top-k filtering: keep only the k highest logits 
        if top_k is not None: 
            topk_vals, _ = torch.topk(logits, top_k) 
            threshold = topk_vals[:, -1].unsqueeze(-1)   # k-th highest 
value 
            logits = logits.masked_fill(logits < threshold, -1e9) 
 
        # Top-p (nucleus) filtering 
        if top_p is not None: 
            sorted_logits, sorted_idx = torch.sort(logits, 
descending=True) 
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), 
dim=-1) 
            # Remove tokens with cumulative prob above p 
            remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p 
            sorted_logits[remove] = -1e9 
            # Scatter back to original ordering 
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits) 
 
        # Sample from the filtered distribution 
        probs   = F.softmax(logits, dim=-1) 
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1) 
 
        # Append new token and continue 
        ids = torch.cat([ids, next_id], dim=1) 
 
    return ids  # (1, T + max_new_tokens) 
 
# Example usage: 
# model = GPTDecoder(vocab_size=50257, d_model=768, N=12, h=12) 
# prompt = torch.tensor([[50256, 1211, 318, 257]])  # 'This is a' 
# output_ids = generate(model, prompt, max_new_tokens=50, top_p=0.9) 
