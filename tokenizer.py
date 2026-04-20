import re
import html
import contractions
import jax.numpy as jnp


def tokenize(text):
    # Fix malformed HTML entities (e.g. #39;)
    text = re.sub(r'(?<!&)#(\d+);', r'&#\1;', text)
    text = re.sub(r'(?<!&)#([a-zA-Z]+);', r'&\#\1;', text)
    # Decode HTML entities
    text = html.unescape(text)
    text = text.lower()
    # Remove residual HTML tags (<b>, </b>, <br>)
    text = re.sub(r'<[^>]+>', ' ', text)
    # Expand contractions
    text = contractions.fix(text)
    # Remove punctuation and underscores
    text = re.sub(r'[^\w\s]|_', ' ', text)
    # Remove single-char tokens except "a and i" and white space
    text = re.sub(r'\b(?![ai]\b)[a-zA-Z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove news agency tags
    text = re.sub(r'\b(reuters|ap|afp|quot)\b', '', text)
    return text.split()


# Build token frequency dictionary from the training texts
def build_vocab(examples):
    vocab = dict()
    for text in examples["text"]:
        tokens = tokenize(text)
        for t in tokens:
            vocab[t] = vocab.get(t, 0) + 1
    return vocab


def tokenize_and_encode(example):
    ids = [token_to_id.get(tok, UNK_ID) for tok in tokenize(example["text"])]
    ids = [BOS_ID] + ids + [EOS_ID]
    ids = ids[:seq_len + 1]

    if len(ids) < seq_len + 1:
        ids += [PAD_ID] * (seq_len + 1 - len(ids))

    return {"x": ids[:-1], "y": ids[1:]}


def mask_token(logits):
    for token_id in [UNK_ID, PAD_ID, BOS_ID]:
        logits = logits.at[:, :, token_id].set(-jnp.inf)
    return logits


def decode_ids(ids):
    tokens = (id_to_token[int(i)] for i in ids)
    out = []
    for t in tokens:
        if t == "<EOS>":
          break
        if t not in {"<PAD>", "<BOS>"}:
            out.append(t)
    return " ".join(out)


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):

    mask = jnp.zeros_like(logits)
    mask = mask.at[jnp.array(generated_ids)].set(1.0)

    penalty_condition = jnp.where(logits > 0, logits / penalty, logits * penalty)

    return jnp.where(mask, penalty_condition, logits)
