import torch
import torch.nn as nn
import torch.nn.functional as func
from math import sqrt, prod

from backbones.base import ExtendedModule, numeric_tuple
from backbones.counting import CountedAdd, CountedLinear, CountedMatmul
from backbones.modules import (
    SimpleSTGTGate,
    TokenBuffer,
    TokenDeltaGate,
    TokenGate,
    MatmulDeltaAccumulator,
    MatmulBuffer,
)
from backbones.utils import (
    DropPath,
    RelativePositionEmbedding,
    expand_row_index,
)
from utils.image import pad_to_size
from backbones.utils import PositionEncoding

LN_EPS = 1e-6


class Block(ExtendedModule):
    """
    Defines a base (non-eventful) Transformer block. Includes a couple
    of extra features: a simple implementation of Adaptive Token
    Sampling (ATS - Fayyaz et al. 2022) and self-attention pooling.
    These features are controlled via the ats_fraction and pool_size
    parameters.
    """

    def __init__(
        self,
        dim,
        heads,
        input_size,
        mlp_ratio,
        ats_fraction=None,
        drop_path_rate=0.0,
        relative_embedding_size=None,
        matmul_2_cast=None,
        pool_size=None,
        window_size=None,
        input_block=False,
        output_block=False
    ):
        """
        :param dim: The number of dimensions in a token
        :param heads: The number of attention heads (None for no
        multi-headed attention)
        :param input_size: The expected size of the inputs in tokens
        :param mlp_ratio: The ratio of the MLP dimensionality to the
        token dimensionality
        :param ats_fraction: The fraction of tokens to retain if
        using Adaptive Token Sampling (ATS)
        :param drop_path_rate: Drop path ratio (for use when training)
        :param relative_embedding_size: The size (in tokens) assumed for
        relative position embeddings
        :param matmul_2_cast: Typecast for the attention-value product
        (None, "float16", or "bfloat16"). Helps save some memory when
        using an A-gate, without a noticeable impact on accuracy.
        :param pool_size: Pooling ratio to use with self-attention
        pooling.
        :param window_size: Self-attention window size (None to use
        global, non-windowed attention).
        """
        super().__init__()
        self.heads = heads
        self.input_size = tuple(input_size)
        if ats_fraction is not None:
            assert pool_size is None
            assert window_size is None
            assert not (ats_fraction < 0.0 or ats_fraction > 1.0)
        assert not (drop_path_rate < 0.0 or drop_path_rate > 1.0)
        assert matmul_2_cast in [None, "float16", "bfloat16"]
        self.ats_fraction = ats_fraction
        self.last_ats_indices = None
        self.matmul_2_cast = matmul_2_cast
        if pool_size is None:
            self.pool_size = None
        else:
            self.pool_size = numeric_tuple(pool_size, length=2)
        if window_size is None:
            self.window_size = None
            attention_size = input_size
        else:
            self.window_size = numeric_tuple(window_size, length=2)
            attention_size = self.window_size
            if relative_embedding_size is not None:
                relative_embedding_size = self.window_size
        self.scale = sqrt(dim // heads)

        # Set up submodules.
        self.input_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.qkv = CountedLinear(in_features=dim, out_features=dim * 3)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        if relative_embedding_size is not None:
            self.relative_position = RelativePositionEmbedding(
                attention_size,
                relative_embedding_size,
                dim // heads,
                pool_size=self.pool_size,
            )
        else:
            self.relative_position = None
        self.matmul = CountedMatmul()
        self.projection = CountedLinear(in_features=dim, out_features=dim)
        self.add = CountedAdd()
        self.mlp_layer_norm = nn.LayerNorm(dim, eps=LN_EPS)
        self.mlp_1 = CountedLinear(in_features=dim, out_features=dim * mlp_ratio)
        self.gelu = nn.GELU()
        self.mlp_2 = CountedLinear(in_features=dim * mlp_ratio, out_features=dim)
        self.input_block = input_block
        self.output_block = output_block
        self.b = None

    def forward(self, x, index=None):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Linearly project x into qkv space.
        x = self.qkv(x)

        if index is not None:
            idx = index.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = self.b.scatter(1, idx, x)
        self.b = x

        # Compute attention on the qkv representation.
        x = self._forward_attention(x)

        if index is not None:
            idx = index.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = torch.gather(x, 1, idx)

        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        return x

    def reset_self(self):
        self.last_ats_indices = None

    def _cast_matmul_2(self, x, v):
        old_dtype = x.dtype
        if self.matmul_2_cast is not None:
            dtype = getattr(torch, self.matmul_2_cast)
            x = x.to(dtype)
            v = v.to(dtype)
        return x, v, old_dtype

    def _compute_window_padding(self):
        pad_h = -self.input_size[0] % self.window_size[0]
        pad_w = -self.input_size[1] % self.window_size[1]
        return pad_h, pad_w

    def _forward_attention(self, x):
        # (batch, token, dim)

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)
        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        return x

    def _forward_mlp(self, x):
        x = self.mlp_1(x)
        x = self.gelu(x)
        x = self.mlp_2(x)
        return x

    def _partition_heads(self, x):
        # (batch, token, dim)

        x = x.view(x.shape[:-1] + (3, self.heads, x.shape[-1] // (3 * self.heads)))
        q, k, v = x.permute(2, 0, 3, 1, 4)
        # (batch, heads, token, dim / heads)

        return q, k, v

    def _partition_windows(self, x, in_qkv_domain):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        # (batch, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(x.shape[:1] + self.input_size + x.shape[2:])
        # (batch, height, width, dim)

        if any(p):
            s = x.shape
            pad_tensor = torch.zeros(
                (1,) * (x.ndim - 1) + s[-1:], dtype=x.dtype, device=x.device
            )

            # The attention computation expects padded tokens to equal
            # _forward_qkv(zero). If x has already been mapped to the
            # QKV domain, then we need to transform the padded zero
            # values to the QKV domain. Only the bias portion of the
            # linear transform has an effect on the zero padding vector.
            if in_qkv_domain:
                pad_tensor = self.qkv.forward_bias(pad_tensor)

            # Pad to a multiple of the window size.
            # func.pad seems broken (see the comments in pad_to_size).
            # In the meantime we'll use pad_to_size.
            # x = func.pad(x, (0, 0, 0, p[1], 0, p[0]))
            x = pad_to_size(x, (s[-3] + p[0], s[-2] + p[1], s[-1]), pad_tensor)
            # (batch, height, width, dim)

        # Partition into windows.
        s = x.shape
        x = x.view(-1, s[-3] // d[0], d[0], s[-2] // d[1], d[1], s[-1])
        x = x.transpose(-3, -4)
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Re-flatten the spatial dimensions. Can't use x.view here
        # because of the transpose.
        x = x.reshape(-1, prod(d), s[-1])
        # (batch * window, token, dim)

        return x

    def _pool_tokens(self, x):
        # (batch, heads, token, dim)

        if self.pool_size is None:
            return x
        w = self.input_size if (self.window_size is None) else self.window_size
        s = x.shape

        # Can't use x.view here because of the permutation in
        # _partition_heads.
        x = x.reshape((-1,) + w + x.shape[-1:])
        # (batch * heads, token_y, token_x, dim)

        x = x.permute(0, 3, 1, 2)
        x = func.avg_pool2d(x, self.pool_size)
        # (batch * heads, dim, token_y, token_x)

        x = x.permute(0, 2, 3, 1)
        # (batch * heads, token_y, token_x, dim)

        x = x.view(s[:-2] + (-1,) + s[-1:])
        # (batch, heads, token, dim)

        return x

    @staticmethod
    def _recombine_heads(x):
        # (batch, heads, token, dim / heads)

        # Can't use x.view here because of the permutation.
        x = x.permute(0, 2, 1, 3)
        x_reshaped = x.reshape(x.shape[:-2] + (-1,))
        # (batch, token, dim)

        # We assume that x.reshape actually copies the data. We can run
        # into problems if this is not the case, i.e., we may end up
        # with a gate being passed a raw reference to an accumulator
        # state. For an example, see EventfulMatmul1Block.
        assert x.data_ptr() != x_reshaped.data_ptr()
        x = x_reshaped

        return x

    def _recombine_windows(self, x):
        if self.window_size is None:
            return x

        p = self._compute_window_padding()
        d = self.window_size
        s = self.input_size
        total_h = p[0] + s[0]
        total_w = p[1] + s[1]
        # (batch * window, token, dim)

        # Unflatten the spatial dimensions.
        x = x.view(-1, total_h // d[0], total_w // d[1], d[0], d[1], x.shape[-1])
        # (batch, window_y, window_x, token_y, token_x, dim)

        # Recombine the window partitions. Can't use x.view here because
        # of the transpose.
        x = x.transpose(-3, -4)
        x = x.reshape(-1, total_h, total_w, x.shape[-1])
        # (batch, height, width, dim)

        # Remove padding.
        if any(p):
            x = x[:, : s[0], : s[1]]
            # (batch, height, width, dim)

        # Re-flatten the spatial dimensions.
        x = x.flatten(start_dim=1, end_dim=2)
        # (batch, token, dim)

        return x

    def _uncast_matmul_2(self, x, old_dtype):
        if self.matmul_2_cast is not None:
            x = x.to(old_dtype)
        return x
    

class MaskedBlock(Block):
    """
    An EventfulMatmul1Block that also adds eventfulness to the
    attention-value product.
    """
    def __init__(self, **super_kwargs):
        """
        :param super_kwargs: Kwargs for the super class (
        EventfulTokenwiseBlock)
        """
        super().__init__(**super_kwargs)
        # self.v_gate = TokenDeltaGate()
        # self.matmul_gate = TokenDeltaGate(structure="col")
        # self.matmul_accumulator_2 = MatmulDeltaAccumulator()
        self.x_window = None

    def _forward_attention(self, x, index=None, window_index=None, window_keep_ratio=0.5):
        # (batch, token, dim)
        # skip_1 = x
        # x = self.input_layer_norm(x)

        # Partition the windows and attention heads. _window_partition
        # is a noop if self.window_size is None. Windows are arranged
        # along the batch dimension.
        x = self._partition_windows(x, in_qkv_domain=True)

        if window_index is not None:
            # n_win = x.shape[0]
            # x_window = x
            # #  TODO: change how to select windows
            # window_id = torch.topk(x.sum((1,2)), int(window_keep_ratio * n_win)).indices 
            win_idx = window_index.unsqueeze(1).unsqueeze(2).expand(-1, x.shape[1], x.shape[2]).to(x.device)
            x = torch.gather(x, 0, win_idx)
        
        # x = self.qkv(x) # moving this after windows partition
            
        q, k, v = self._partition_heads(x)
        # (batch, heads, token, dim / heads)

        # Token pooling is a noop if self.pool_size is None.
        k = self._pool_tokens(k)
        v = self._pool_tokens(v)

        # Perform the actual attention computation.
        # The output of this first matmul is huge - hence it's much
        # faster to scale one of the inputs than it is to scale the
        # output.
        x = self.matmul(q / self.scale, k.transpose(-2, -1))
        if self.relative_position is not None:
            x = self.relative_position(x, q)
        x = x.softmax(dim=-1)

        x, v, old_dtype = self._cast_matmul_2(x, v)
        x = self.matmul(x, v)
        # (batch, heads, token, dim / heads)

        x = self._recombine_heads(x)

        # Apply the post-attention linear transform and add the skip.
        # x = self.projection(x)

        if window_index is not None:
            win_idx = window_index.unsqueeze(1).unsqueeze(2).expand(-1, x.shape[1], x.shape[2]).to(x.device)
            x = self.x_window.scatter(0, win_idx, x)
        self.x_window = x

        x = self._recombine_windows(x)
        x = self._uncast_matmul_2(x, old_dtype)
        # (batch, token, dim)

        # x = self.add(self.drop_path(x), skip_1)

        return x

    def forward(self, x, index=None, window_index=None, windowed=False):
        skip_1 = x
        x = self.input_layer_norm(x)

        # Linearly project x into qkv space.
        x = self.qkv(x)

        if index is not None:
            idx = index.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = self.b.scatter(1, idx, x)
        self.b = x

        # Compute attention on the qkv representation.
        x = self._forward_attention(x, index, window_index)

        if index is not None:
            idx = index.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = torch.gather(x, 1, idx)

        # Apply the post-attention linear transform and add the skip.
        x = self.projection(x)
        x = self.add(self.drop_path(x), skip_1)

        # Apply the token-wise MLP.
        skip_2 = x
        x = self.mlp_layer_norm(x)
        x = self._forward_mlp(x)
        x = self.add(self.drop_path(x), skip_2)
        return x




class ViTBackbone(ExtendedModule):
    """
    Common backbone for vision Transformers.
    """

    def __init__(
        self,
        block_config,
        depth,
        position_encoding_size,
        input_size,
        block_class="Block",
        has_class_token=False,
        window_indices=(),
        windowed_class=None,
        windowed_overrides=None,
        **kwargs,
    ):
        """
        :param block_config: A dict containing kwargs for the
        block_class constructor
        :param depth: The number of blocks to use
        :param position_encoding_size: The size (in tokens) assumed for
        position encodings
        :param input_size: The expected size of the inputs in tokens
        :param block_class: The specific Block class to use (see
        blocks.py for options)
        :param has_class_token: Whether to add an extra class token
        :param window_indices: Block indices that should use windowed
        attention
        :param windowed_class: The specific Block class to use with
        windowed attention (if None, fall back to block_class)
        :param windowed_overrides: A dict containing kwargs overrides
        for windowed_class
        """
        super().__init__()
        self.position_encoding = PositionEncoding(
            block_config["dim"], position_encoding_size, input_size, has_class_token
        )
        self.window_indices = window_indices
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_class_i = block_class
            block_config_i = block_config.copy()
            if i in window_indices:
                if windowed_class is not None:
                    block_class_i = windowed_class
                if windowed_overrides is not None:
                    block_config_i |= windowed_overrides
            else:
                block_config_i["window_size"] = None
            self.blocks.append(
                Block(input_size=input_size, **block_config_i)
            )

    def forward(self, x, mask_id=None, window_id=None):
        x = self.position_encoding(x)
        if (mask_id is not None):
            index = mask_id.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = torch.gather(x, 1, index)
        for i, block in enumerate(self.blocks):
            # if i in self.window_indices:
            #     x = block(x, index=mask_id, window_index=window_id, windowed=True)
            # else:
            x = block(x, index=mask_id)
        if (mask_id is not None):
            index = mask_id.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.device)
            x = self.ref_out.scatter(1, index, x)
        self.ref_out = x.detach().clone()

        return x


