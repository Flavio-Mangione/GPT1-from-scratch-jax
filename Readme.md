<div align="center">
  <img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png">
  <h1>
    <bold> GPT-1 from Scratch using Jax </bold>
  </h1>
  <p>
    <a href = "https://www.python.org/downloads/release/python-3110/">
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/>
    </a>
    <a href = "https://www.uniroma1.it/en/pagina-strutturale/home">
    <img src="https://img.shields.io/badge/Sapienza-Università_di_Roma-822433" alt="Sapienza"/>
    </a>
  </p>
  </p>
</div>


This repository contains a custom implementation of [**GPT-1**](https://huggingface.co/openai-community/openai-gpt#model-details) using the [JAX](https://github.com/jax-ml/jax) library 
for the course of *Neural Networks for Data Science* at Sapienza University of Rome. In this implementation, a Key-Value (KV) cache is used to store key and value tensors, avoiding recomputation during attention and improving text generation efficiency.

> **Note**  
> This project is for educational purposes and does not aim to fully reproduce the original GPT-1 training setup.

The pipeline for text generation using cache is defined as follows:

<div align="center">
<img src="https://jax-ml.github.io/scaling-book/assets/img/cached-inference.png" width = 800 height = 350>
</div>

## Repository Structure 

- `Layers`: My implementantion of multi-head-attention for Transformer blocks layers, using a KV cache mechanism. 
- `Model`: My custom version of GPT-1. It is composed of sinusoidal positional encodings, a stack of Transformer blocks and a final linear projection layer for next token prediction.
  The model returns up to three outputs:

  * The model's logits of shape (batch_size, seq_length, model_size).
  * Key-Value Cache when `use_cache = True`.
  * Attention weights when `return_weights = True`, to investigate the specific model's activations across tokens, revealing which ones influence each other during text generation.

- `utils.metrics`: Metrics used for training and evaluating the model.
- `utils.visualizzation`: Utilities to visualize model behavior and attention patterns.
- [`Demo Notebook.ipynb`](https://github.com/Flavio-Mangione/GPT1-from-scratch-jax/blob/master/Notebook/Demo%20Notebook.ipynb): example notebook used to test the model.

## How to use

### Install dependencies 

```bash
pip install jax equinox optax
```

### Import and initialize model

```bash
!git clone https://github.com/Flavio-Mangione/GPT1-from-scratch-jax.git

# Example of Model initialization with GPT-1 configuration

model = GPT1(
    d_model = 768,
    num_heads = 12,
    num_layers = 12,
    vocab_size = 50257,
    key = model_key,
    dropout_rate = 0.1,
    return_attention_weights = True)

```

## Training 

The model was evaluated during training using Cross-Entropy loss and Perplexity, defined as::

<div align="center">

$$
\text{CE}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

$$
\text{PPL}(y) = \exp(\text{CE}(y, \hat{y}))
$$

</div>

## Cache Compression

The KV cache is compressed using L2-norm-based pruning: entries with higher norm are discarded according to a threshold, reducing memory usage based on a compression ratio (between 0 and 1).

```bash

generate_text_compress(model, prompt, max_new_tokens = 50, temperature = 0.50, penalty = 1.4, ratio = 0.70, cache_cap = 25)

```

## Citation

If you use this repository in academic work, consider citing the following paper:

> 1. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

> 2. [Attention Is all You Need](http://arxiv.org/abs/1706.03762)

> 3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)

> 4. [A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://arxiv.org/abs/2406.11430)






