import jax
import jax.numpy as jnp
import equinox as eqx

# Cross-entropy for autoregressive language modeling
def cross_entropy_loss(logits, y, pad_id: int = 0):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_indices = y[:, :, None]
    target_log_probs = jnp.take_along_axis(log_probs, target_indices, axis=-1).squeeze(-1)

    valid_mask = (y != pad_id)
    target_log_probs = jnp.where(valid_mask, target_log_probs, 0.0)

    loss = -jnp.sum(target_log_probs) / jnp.maximum(jnp.sum(valid_mask), 1)
    return loss

# Define Traning steps
@eqx.filter_jit
def train_step(my_model, state, x_ids, y_ids, key):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(my_model, x_ids, y_ids, key)
    updates, state = optimizer.update(grads, state, my_model)
    my_model = eqx.apply_updates(my_model, updates)
    return my_model, state, loss


# Define validation steps
@eqx.filter_jit
def val_step(model, xb, yb):
    out = model(xb, inference=True)
    loss = cross_entropy_loss(out.logits, yb)
    return loss

def compute_val_loss(model, x, y, batch_size=128):
    total_loss = 0.0
    n = len(x)
    n_batches = n // batch_size

    for i in range(n_batches):
        xb = x[i * batch_size:(i + 1) * batch_size]
        yb = y[i * batch_size:(i + 1) * batch_size]

        loss = val_step(model, xb, yb)
        total_loss += loss

    return float(total_loss / n_batches)

@eqx.filter_jit
def _top_k_batch(model, xb, yb, k):
    out = model(xb, inference=True)
    top_k = jnp.argsort(out.logits, axis=-1)[:, :, -k:]
    correct = jnp.any(top_k == yb[:, :, None], axis=-1)
    mask = (yb != PAD_ID)
    correct = jnp.where(mask, correct, 0)
    return jnp.sum(correct), jnp.sum(mask)

def top_k_accuracy(model, x, y, k = 5, batch_size = 128):
    total_correct, total_tokens = 0, 0
    n_batches = len(x) // batch_size

    for i in range(n_batches):
        xb = x[i * batch_size:(i + 1) * batch_size]
        yb = y[i * batch_size:(i + 1) * batch_size]

        correct, tokens = _top_k_batch(model, xb, yb, k)
        total_correct += correct
        total_tokens += tokens

    return total_correct / jnp.maximum(total_tokens, 1)