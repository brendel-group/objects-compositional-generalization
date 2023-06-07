import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional


class SlotAttention(nn.Module):
    def __init__(
        self, num_iterations=1, qkv_size=None, mlp_size=None, epsilon=1e-8, num_heads=1
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.qkv_size = qkv_size
        self.mlp_size = mlp_size
        self.epsilon = epsilon
        self.num_heads = num_heads

    def forward(self, slots, inputs, padding_mask=None, train=False):
        qkv_size = self.qkv_size or slots.shape[-1]
        head_dim = qkv_size // self.num_heads

        dense_q = nn.Linear(qkv_size, self.num_heads * head_dim, bias=False)
        layernorm_q = nn.LayerNorm(qkv_size)
        inverted_attention = InvertedDotProductAttention(
            norm_type="mean", multi_head=self.num_heads > 1
        )
        gru = nn.GRUCell(qkv_size, qkv_size)

        if self.mlp_size is not None:
            mlp = nn.Sequential(
                nn.LayerNorm(qkv_size),
                nn.Linear(qkv_size, self.mlp_size),
                nn.ReLU(),
                nn.Linear(self.mlp_size, qkv_size),
            )

        inputs = nn.LayerNorm(inputs.size(-1))(inputs)
        k = nn.Linear(inputs.size(-1), qkv_size)(inputs)
        v = nn.Linear(inputs.size(-1), qkv_size)(inputs)

        for _ in range(self.num_iterations):
            slots_n = layernorm_q(slots)
            q = dense_q(slots_n).view(*slots.shape[:-1], self.num_heads, head_dim)
            q = q.transpose(-2, -3).flatten(-2, -3)
            updates = inverted_attention(query=q, key=k, value=v)
            slots = gru(slots, updates)

            if self.mlp_size is not None:
                slots = mlp(slots)

        return slots


class InvertedDotProductAttention(nn.Module):
    """Inverted version of dot-product attention (softmax over query axis)."""

    def __init__(
        self,
        norm_type: Optional[str] = "mean",
        multi_head: bool = False,
        epsilon: float = 1e-8,
        dtype: torch.float32 = torch.float32,
    ):
        super().__init__()

        self.norm_type = norm_type
        self.multi_head = multi_head
        self.epsilon = epsilon
        self.dtype = dtype

        self.attn = GeneralizedDotProductAttention(
            inverted_attn=True,
            renormalize_keys=True if self.norm_type == "mean" else False,
            epsilon=self.epsilon,
            dtype=self.dtype,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        train: bool = False,
    ) -> torch.Tensor:
        """Computes inverted dot-product attention.

        Args:
            query: Queries with shape of `[batch..., q_num, qk_features]`.
            key: Keys with shape of `[batch..., kv_num, qk_features]`.
            value: Values with shape of `[batch..., kv_num, v_features]`.
            train: Indicating whether we're training or evaluating.

        Returns:
            Output of shape `[batch_size..., n_queries, v_features]`
        """
        # Apply attention mechanism.
        output = self.attn(query=query, key=key, value=value, train=train)

        if self.multi_head:
            # Multi-head aggregation. Equivalent to concat + dense layer.
            output = nn.Linear(output.size(-1), output.size(-1))(
                output.view(*output.shape[:-2], -1)
            )
        else:
            # Remove head dimension.
            output = torch.squeeze(output, dim=-2)

        if self.norm_type == "layernorm":
            output = nn.LayerNorm(output.size(-1), eps=self.epsilon)(output)

        return output


class GeneralizedDotProductAttention(nn.Module):
    def __init__(
        self,
        dtype=torch.float32,
        precision=None,
        epsilon=1e-8,
        inverted_attn=False,
        renormalize_keys=False,
        attn_weights_only=False,
    ):
        super().__init__()
        self.dtype = dtype
        self.precision = precision
        self.epsilon = epsilon
        self.inverted_attn = inverted_attn
        self.renormalize_keys = renormalize_keys
        self.attn_weights_only = attn_weights_only

    def forward(self, query, key, value, train=False, **kwargs):
        assert (
            query.ndim == key.ndim == value.ndim
        ), "Queries, keys, and values must have the same rank."
        assert (
            query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
        ), "Query, key, and value batch dimensions must match."
        assert (
            query.shape[-2] == key.shape[-2] == value.shape[-2]
        ), "Query, key, and value num_heads dimensions must match."
        assert (
            key.shape[-3] == value.shape[-3]
        ), "Key and value cardinality dimensions must match."
        assert (
            query.shape[-1] == key.shape[-1]
        ), "Query and key feature dimensions must match."

        if kwargs.get("bias") is not None:
            raise NotImplementedError(
                "Support for masked attention is not yet implemented."
            )

        if "dropout_rate" in kwargs:
            if kwargs["dropout_rate"] > 0.0:
                raise NotImplementedError("Support for dropout is not yet implemented.")

        qk_features = query.size(-1)
        query = query / torch.sqrt(torch.tensor(qk_features, dtype=self.dtype))

        attn = torch.einsum("...qhd,...khd->...hqk", query, key)

        if self.inverted_attn:
            attention_axis = -2  # Query axis.
        else:
            attention_axis = -1  # Key axis.

        attn = F.softmax(attn, dim=attention_axis).to(self.dtype)

        if not train:
            self.sow("intermediates", "attn", attn)

        if self.renormalize_keys:
            normalizer = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn = attn / normalizer

        if self.attn_weights_only:
            return attn

        return torch.einsum("...hqk,...khd->...qhd", attn, value)
