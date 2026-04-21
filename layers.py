import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float
from typing import Optional, Tuple, NamedTuple


def GELU(x, key=None):
    scale_factor = 0.044715
    return 0.5 * x * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (x + scale_factor * x**3)))


def update_cache(K, V, cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]]):
    if cache is None:
        return K, V
    else:
        # Concatenate past and current keys/values along sequence dimension
        K_new = jnp.concatenate([cache[0], K], axis=1)
        V_new = jnp.concatenate([cache[1], V], axis=1)
        return K_new, V_new


# Apply function over batch and sequence dimensions
def batch_vmap(fn):
    return jax.vmap(jax.vmap(fn))


class TransformerOut(NamedTuple):
    output: jnp.ndarray
    next_cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]]
    attn_weights: Optional[jnp.ndarray]


# Define the MLP used in Transformer Block
class Feedforward(eqx.Module):
    ffn: eqx.nn.Sequential

    def __init__(self, d_model: int, key, use_bias: bool = False):
        k1, k2 = jax.random.split(key, 2)
        self.ffn = eqx.nn.Sequential([
            eqx.nn.Linear(d_model, d_model * 4, key=k1, use_bias=use_bias),
            GELU,
            eqx.nn.Linear(d_model * 4, d_model, key=k2, use_bias=use_bias)])

    def __call__(self, x):
        return self.ffn(x)


class Multiheadattention(eqx.Module):
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    d_model: int
    num_heads: int
    return_attention_weights: bool

    def __init__(self, d_model: int, num_heads: int, key,return_attention_weights: bool = False, use_bias: bool = False, dropout_rate: float = 0.1):
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.return_attention_weights = return_attention_weights

        # Define the parameters for the linear projections
        kq, kw, kv, k_out = jax.random.split(key, 4)

        # Projections for Q, K, V
        self.wq = eqx.nn.Linear(d_model, d_model, use_bias=use_bias, key=kq)
        self.wk = eqx.nn.Linear(d_model, d_model, use_bias=use_bias, key=kw)
        self.wv = eqx.nn.Linear(d_model, d_model, use_bias=use_bias, key=kv)
        self.proj_out = eqx.nn.Linear(d_model, d_model, use_bias=use_bias, key=k_out)

    def __call__(self, x, use_cache = True, cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None, inference: bool = False, key = None) -> Float[jnp.ndarray, "batch_size sequence_length d_model"]:
        if (not inference) and (key is None):
            raise ValueError("Provide `key` when inference=False (training)")

        batch, seq_length = x.shape[0], x.shape[1]

        # Linear projections for Q, K, V (batch_size, seq_length, d_model)
        Q = batch_vmap(self.wq)(x)
        K = batch_vmap(self.wk)(x)
        V = batch_vmap(self.wv)(x)

        # Tensor shape: (batch_size, seq_length, d_model) -> (batch_size, seq_length, num_heads, head_dim)
        Q = Q.reshape(batch, seq_length, self.num_heads, self.d_model // self.num_heads)
        K = K.reshape(batch, seq_length, self.num_heads, self.d_model // self.num_heads)
        V = V.reshape(batch, seq_length, self.num_heads, self.d_model // self.num_heads)

        # Update KV cache with new key/value tensors
        if use_cache:
            K, V = update_cache(K, V, cache)
            next_cache = (K, V)

        # Tensors shape: (batch_size, seq_length, num_heads, head_dim) -> (batch_size, num_heads, seq_length, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Compute attention score (batch_size, num_heads, q_len, k_len)
        attention_score = Q @ K.transpose(0, 1, 3, 2)
        # Causal mask (lower triangular with offset for cached tokens)
        q_len = Q.shape[2]
        k_len = K.shape[2]
        past_len = 0 if cache is None else cache[0].shape[1]
        mask = jnp.tril(jnp.ones((q_len, k_len), dtype=bool), k=past_len)

        # Shape broadcasted to (1, 1, q_len, k_len)
        mask = mask[None, None, :, :]
        attention_score = jnp.where(mask, attention_score, -jnp.inf)

        # compute attention weights
        attn_weights = jax.nn.softmax(attention_score / K.shape[-1]**0.5, axis=-1)
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vector = (attn_weights @ V).transpose(0, 2, 1, 3)

        # Concatenate heads: (batch_size, seq_length, num_heads * head_dim) -> (batch_size, seq_length, d_model)
        context_vector = context_vector.reshape(batch, seq_length, self.d_model)

        out = batch_vmap(self.proj_out)(context_vector)

        return TransformerOut(out, next_cache if use_cache else None, attn_weights if self.return_attention_weights else None)


class TransformerBlock(eqx.Module):
    attention: Multiheadattention
    LayerNorm1: eqx.nn.LayerNorm
    LayerNorm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    ffn: Feedforward

    def __init__(self, d_model: int, h: int, key, dropout_rate: float = 0.1, return_attention_weights: bool = False, use_bias: bool = False):
        # Initialize the parameters for the multi-head attention
        mha_key, ffn_key = jax.random.split(key, 2)
        self.attention = Multiheadattention(d_model, h,mha_key, return_attention_weights = return_attention_weights,use_bias = use_bias,dropout_rate = dropout_rate)

        self.LayerNorm1 = eqx.nn.LayerNorm(d_model)
        self.LayerNorm2 = eqx.nn.LayerNorm(d_model)
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.ffn = Feedforward(d_model, key = ffn_key, use_bias=use_bias)

    def __call__(self, x, use_cache = True, cache = None, inference = False, key = None) -> Float[jnp.ndarray, "batch_size sequence_length d_model"]:
        if (not inference) and (key is None):
            raise ValueError("Provide `key` when inference = False (training)")

        if not inference:
            k_attn, k1, k2 = jax.random.split(key, 3)
        else:
            k_attn = k1 = k2 = None

        z = batch_vmap(self.LayerNorm1)(x)

        # Attention + residual connections
        attn_out, next_cache, attn_weights = self.attention(z,use_cache = use_cache,cache = cache,inference = inference,key = k_attn)

        x = x + self.dropout(attn_out, key=k1, inference=inference)

        # Feedforward network with residual connections
        z = batch_vmap(self.LayerNorm2)(x)
        z_out = batch_vmap(self.ffn)(z)
        out = x + self.dropout(z_out, key=k2, inference=inference)

        return TransformerOut(out, next_cache if use_cache else None, attn_weights if self.attention.return_attention_weights else None)
