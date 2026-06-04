from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FourierPositionEncoding2D(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError("FourierPositionEncoding2D dim must be a positive even integer")
        if scale <= 0.0:
            raise ValueError("FourierPositionEncoding2D scale must be positive")
        generator = torch.Generator()
        generator.manual_seed(42)
        self.register_buffer(
            "gaussian_matrix", scale * torch.randn((2, dim // 2), generator=generator)
        )

    def forward(self, coords: Tensor) -> Tensor:
        projected = (2.0 * coords.float() - 1.0) @ self.gaussian_matrix
        projected = 2.0 * math.pi * projected
        return torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)


class _MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _AttentionLayer(nn.Module):
    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        hidden_dim: int,
        *,
        num_heads: int,
        dropout: float,
        use_bias: bool = False,
        use_v_proj: bool = True,
        use_out_proj: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not use_v_proj and kv_dim != hidden_dim:
            raise ValueError("kv_dim must equal hidden_dim when value projection is disabled")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(q_dim, hidden_dim, bias=use_bias)
        self.k_proj = nn.Linear(kv_dim, hidden_dim, bias=use_bias)
        self.v_proj = nn.Linear(kv_dim, hidden_dim, bias=use_bias) if use_v_proj else nn.Identity()
        self.out_proj = (
            nn.Linear(hidden_dim, hidden_dim, bias=use_bias) if use_out_proj else nn.Identity()
        )
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        nn.init.kaiming_normal_(self.q_proj.weight, mode="fan_in", nonlinearity="linear")
        nn.init.kaiming_normal_(self.k_proj.weight, mode="fan_in", nonlinearity="linear")
        if isinstance(self.v_proj, nn.Linear):
            nn.init.kaiming_normal_(self.v_proj.weight, mode="fan_in", nonlinearity="linear")
        if isinstance(self.out_proj, nn.Linear):
            nn.init.kaiming_normal_(self.out_proj.weight, mode="fan_in", nonlinearity="linear")

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, q_len, _ = q.shape
        _, kv_len, _ = k.shape
        query = (
            self.q_proj(q).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        )
        key = self.k_proj(k).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = (
            self.v_proj(v).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        )

        query = self.q_norm(query)
        key = self.k_norm(key)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1))
        out = torch.matmul(attn_weights, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)
        return self.out_proj(out), attn_weights


class _CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, *, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.cross_attn = _AttentionLayer(dim, dim, dim, num_heads=num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = _MLPBlock(dim, 2 * dim, dropout)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        x, _ = self.cross_attn(self.query_norm(query), context, context)
        x = query + self.dropout(x)
        return self.out_norm(x + self.mlp(self.mlp_norm(x)))


class TRENRegionEncoder(nn.Module):
    """T-REN-style point-prompted region token encoder.

    The module follows the public T-REN architecture: learned k-per-prompt
    query tokens, Fourier 2D prompt/patch position encodings, alternating
    cross-attention and per-prompt self-attention, then final single-head
    attention that pools unprojected patch tokens into region tokens.
    """

    def __init__(
        self,
        vision_dim: int,
        text_dim: int,
        *,
        hidden_dim: int | None = None,
        num_region_tokens: int = 3,
        num_decoder_layers: int = 2,
        num_attention_heads: int = 8,
        prompt_grid_size: int = 7,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_region_tokens <= 0:
            raise ValueError("num_region_tokens must be positive")
        if num_decoder_layers <= 0:
            raise ValueError("num_decoder_layers must be positive")
        if prompt_grid_size <= 0:
            raise ValueError("prompt_grid_size must be positive")
        hidden_dim = int(hidden_dim or vision_dim)
        if hidden_dim != vision_dim:
            raise ValueError("TRENRegionEncoder currently requires hidden_dim == vision_dim")
        if hidden_dim % 2 != 0:
            raise ValueError("TRENRegionEncoder hidden_dim must be even for Fourier features")
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                "TRENRegionEncoder hidden_dim must be divisible by num_attention_heads"
            )

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_region_tokens = num_region_tokens
        self.prompt_grid_size = prompt_grid_size
        self.position_encoder = FourierPositionEncoding2D(hidden_dim)
        self.region_token_embeddings = nn.Embedding(num_region_tokens, hidden_dim)
        nn.init.normal_(self.region_token_embeddings.weight, std=0.02)
        self.region_attention_layers = nn.ModuleList(
            [
                _CrossAttentionBlock(hidden_dim, num_heads=num_attention_heads, dropout=dropout)
                for _ in range(num_decoder_layers)
            ]
        )
        self.region_attention_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_decoder_layers)]
        )
        self.prompt_attention_layers = nn.ModuleList(
            [
                _AttentionLayer(
                    hidden_dim,
                    hidden_dim,
                    hidden_dim,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )
        self.prompt_attention_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_decoder_layers)]
        )
        self.token_prediction_head = _AttentionLayer(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            num_heads=1,
            dropout=0.0,
            use_v_proj=False,
            use_out_proj=False,
        )
        self.text_alignment_block = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, text_dim),
        )

    def forward(self, image_tokens: Tensor) -> dict[str, Tensor]:
        patch_tokens, patch_grid = _patch_tokens_and_grid(image_tokens)
        batch_size, patch_count, _ = patch_tokens.shape
        patch_coords = _grid_coords(patch_grid, patch_grid, patch_tokens.device)
        prompt_coords = _grid_coords(
            self.prompt_grid_size, self.prompt_grid_size, patch_tokens.device
        )
        prompt_count = prompt_coords.size(0)

        feature_pos = self.position_encoder(patch_coords).to(dtype=patch_tokens.dtype)
        prompt_pos = self.position_encoder(prompt_coords).to(dtype=patch_tokens.dtype)
        kv = patch_tokens + feature_pos.unsqueeze(0)
        prompt_pos = prompt_pos.view(1, prompt_count, 1, self.hidden_dim)

        q = self.region_token_embeddings.weight.to(dtype=patch_tokens.dtype)
        q = q.view(1, 1, self.num_region_tokens, self.hidden_dim).expand(
            batch_size,
            prompt_count,
            self.num_region_tokens,
            self.hidden_dim,
        )
        for region_layer, region_norm, prompt_layer, prompt_norm in zip(
            self.region_attention_layers,
            self.region_attention_norms,
            self.prompt_attention_layers,
            self.prompt_attention_norms,
            strict=True,
        ):
            q = q + prompt_pos
            q = q.reshape(batch_size, prompt_count * self.num_region_tokens, self.hidden_dim)
            q = region_layer(q, kv)
            q = q.reshape(batch_size, prompt_count, self.num_region_tokens, self.hidden_dim)
            q = region_norm(q)
            q = q.reshape(batch_size * prompt_count, self.num_region_tokens, self.hidden_dim)
            q, _ = prompt_layer(q, q, q)
            q = prompt_norm(q)
            q = q.reshape(batch_size, prompt_count, self.num_region_tokens, self.hidden_dim)

        flat_q = q.reshape(batch_size, prompt_count * self.num_region_tokens, self.hidden_dim)
        visual_tokens, attn_weights = self.token_prediction_head(flat_q, kv, patch_tokens)
        visual_tokens = visual_tokens.reshape(
            batch_size, prompt_count, self.num_region_tokens, self.hidden_dim
        )
        attn_weights = attn_weights.squeeze(1).reshape(
            batch_size, prompt_count, self.num_region_tokens, patch_count
        )
        region_masks = attn_weights / attn_weights.amax(dim=-1, keepdim=True).clamp_min(
            torch.finfo(attn_weights.dtype).eps
        )
        region_masks = region_masks.reshape(
            batch_size, prompt_count, self.num_region_tokens, patch_grid, patch_grid
        )
        text_aligned_tokens = self.text_alignment_block(visual_tokens)
        return {
            "visual_tokens": visual_tokens,
            "text_aligned_tokens": text_aligned_tokens,
            "region_masks": region_masks,
            "prompt_coords": prompt_coords,
        }


def _patch_tokens_and_grid(tokens: Tensor) -> tuple[Tensor, int]:
    if tokens.ndim != 3:
        raise ValueError("TRENRegionEncoder expects image tokens with shape [batch, tokens, dim]")
    token_count = tokens.size(1)
    grid = int(math.isqrt(token_count))
    if grid * grid == token_count:
        return tokens, grid
    grid = int(math.isqrt(token_count - 1))
    if grid * grid == token_count - 1:
        return tokens[:, 1:, :], grid
    raise ValueError(f"Cannot infer a square patch grid from {token_count} image tokens")


def _grid_coords(height: int, width: int, device: torch.device) -> Tensor:
    y = torch.linspace(0.5 / height, 1.0 - 0.5 / height, height, device=device)
    x = torch.linspace(0.5 / width, 1.0 - 0.5 / width, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)
