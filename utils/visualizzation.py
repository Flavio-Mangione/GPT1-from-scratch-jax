import matplotlib.pyplot as plt
import jax.numpy as jnp

def plot_attention_bar(attention_weights, tokens, num_heads, num_layers):
    fig, ax = plt.subplots(nrows=num_layers, ncols=num_heads, figsize=(4*num_heads, 3*num_layers))

    # Align tokens to key sequence length
    k_len = attention_weights[0].shape[-1]

    tokens_for_plot = tokens[:k_len] 

    for layers_id in range(num_layers):
        for head in range(num_heads):

            # Shape: (batch, heads, q_len, k_len)
            attn_2d = jnp.array(attention_weights[layers_id][0, head])
            attentions = attn_2d.flatten()

            valid_x = jnp.arange(len(tokens_for_plot))
            valid_att = jnp.array(attentions)

            cmap = plt.cm.viridis
            norm = lambda x: (x - valid_att.min()) / (valid_att.max() - valid_att.min())
            colors = cmap(norm(valid_att))

            ax[layers_id, head].bar(valid_x, valid_att, color=colors, alpha=0.8, edgecolor='black')
            ax[layers_id, head].set_title(f"Layer {layers_id}, Head {head+1}")
            ax[layers_id, head].set_xticks(valid_x)
            ax[layers_id, head].set_xticklabels(tokens_for_plot, rotation=60, ha="right", fontsize=8)
            ax[layers_id, head].set_xlabel("Token (key)")
            ax[layers_id, head].set_ylabel("Attention weight")

    plt.tight_layout()
    plt.show()