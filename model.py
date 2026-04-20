import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from typing import Optional, NamedTuple
from MultiHeadAttention import TransformerBlock, GELU, batch_vmap


class ModelOutput(NamedTuple):
    logits: Float[jnp.ndarray, "batch_size sequence_length vocab_size"]
    cache: Optional[tuple]
    attention_weights: Optional[list]


class Model(eqx.Module):
    embeddings: eqx.nn.Embedding
    transformer_block: list[TransformerBlock]
    linear: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    layer_norm_out: eqx.nn.LayerNorm
    layers: int
    return_attention_weights: bool

    def __init__(self,d_model: int,num_layers: int,num_heads: int, vocab_size: int, key, dropout_rate: float = 0.3, use_bias = False, return_attention_weights = False):

        key_emb, key_blocks, key_linear, key_linear_out = jax.random.split(key, 4)

        self.embeddings = eqx.nn.Embedding(vocab_size, d_model, key = key_emb)
        
        # Stack of Transformer Blocks
        self.transformer_block = [
            TransformerBlock(d_model, num_heads, key = k_block, return_attention_weights = return_attention_weights,
                use_bias = use_bias, dropout_rate = 0.1)
            for k_block in jax.random.split(key_blocks, num_layers)]

        self.linear = eqx.nn.Linear(d_model, d_model, key = key_linear, use_bias = use_bias)
        self.linear_out = eqx.nn.Linear(d_model, vocab_size, key = key_linear_out, use_bias = use_bias)

        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.layer_norm_out = eqx.nn.LayerNorm(d_model)

        self.layers = num_layers
        self.return_attention_weights = return_attention_weights

    def num_params(self) -> int:
      # Count trainable parameters
      params = eqx.filter(self, eqx.is_array)
      return sum(x.size for x in jax.tree_util.tree_leaves(params))

    @staticmethod
    def positional_encoding(seq_len, d_model, start_pos=0):
        pos = (jnp.arange(seq_len) + start_pos)[:, jnp.newaxis]
        divide_term = 10000 ** (jnp.arange(0, d_model, 2) / d_model)

        pe = jnp.zeros((seq_len, d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(pos / divide_term))
        pe = pe.at[:, 1::2].set(jnp.cos(pos / divide_term))
        return pe

    def __call__(self, x, key=None, inference=False, use_cache=False, cache=None) -> Float[jnp.ndarray, "batch_size sequence_length vocab_size"]:
        if (not inference) and (key is None):
            raise ValueError("Provide key when inference = False (training)")

        # Token embeddings (batch, seq_len) -> (batch, seq_len, d_model)
        x = batch_vmap(self.embeddings)(x)

        cache_len = 0
        if use_cache and cache is not None:
            cache_len = cache[0][0].shape[1]
        # Shift positional encoding if using KV cache
        x = x + self.positional_encoding(x.shape[1], x.shape[2], start_pos=cache_len if use_cache else 0)

        if use_cache:
            if cache is None:
                cache = [None] * self.layers

        if not inference:
            split_keys = jax.random.split(key, self.layers + 1)
            block_keys = split_keys[:-1]
            drop_key = split_keys[-1]
        else:
            block_keys = [None] * self.layers
            drop_key = None

        next_cache = [] if use_cache else None
        layer_activations = [] if self.return_attention_weights else None

        # Forward through Transformer blocks
        for i, block in enumerate(self.transformer_block):
            layer_cache = cache[i] if use_cache else None
            x, layer_cache, attn_weights = block(x,use_cache = True,cache = layer_cache,inference = inference,key = block_keys[i])

            if use_cache:
                next_cache.append(layer_cache)

            if self.return_attention_weights:
                layer_activations.append(attn_weights)

        # Final projection head
        z = batch_vmap(self.layer_norm_out)(x)
        h = batch_vmap(self.linear)(z)
        h = GELU(h)
        x = self.dropout(h, key=drop_key, inference=inference)

        logits = batch_vmap(self.linear_out)(x)

        return ModelOutput(logits, next_cache, layer_activations)
