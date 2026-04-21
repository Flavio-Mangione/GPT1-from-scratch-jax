<div align="center">
  <img src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png"
  </div>

  <h1>
    <bold> GPT1 from Scratch using Jax </bold>
  </h1>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python"/>
    <img src="https://img.shields.io/badge/Sapienza-Università_di_Roma-822433" alt="Sapienza"/>
  </p>
  </p>
</div>


This repository contains my custom implmentation of [**GPT1**](https://huggingface.co/openai-community/openai-gpt#model-details) using [Jax](https://github.com/jax-ml/jax) library 
for the course of *Neural Networks for Data Science* at Sapienza University of Rome. 

## Repository Structure 

- `Layers`: My implementantion of Multi-head-attention for Transformer blocks layers, applying a Kv Cache to store Key and Value tensors for memory usage during the generation phase. 
- `Model`: My custom version of GPT1. It is composed of sinusoidal positional encodings, a stack of Transformer blocks and a final linear projection layer for next token prediction.
  It returns up to three outputs:

  * The model's logits of shape (batch_size, seq_length, model_size).
  * Key-Value Cache when `use_cache = True`.
  * Attention weights when `return_weights = True`, to investigate the specific model's activations across tokens, revealing which ones influence each other during text generation.

- `Demo Notebook.ipynb`: example of notebook that I used to test my model

## How to usage

### Install dependecies 

```bash
pip install jax, equinox, optax
```

### Import and initialize model

```bash
!git clone https://github.com/Flavio-Mangione/GPT1-with-JAX.git

# Example of Model initialization with GPT1 configuration

model = Model(
    d_model=768,
    num_heads=12,
    num_layers=12,
    vocab_size=50000,
    key=model_key,
    dropout_rate=0.1,
    return_attention_weights=True)

```

## Traning 

This custom implementation was evaluated over the epoches, using Cross-entropy loss and Perplexity. Define in these way:

<div align="center">

$$
\text{CE}(y, \hat{y}) = - \frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

$$
\text{PPL}(y) = \exp(\text{CE}(y, \hat{y}))
$$

</div>

## Cache Compression

Kv cache was compressed to reduce memory usage, applying **l2 normalizzation** with a pruning treshold to reduce cache size according to a ratio parameter with a value between 0 and 1. 

```bash

generate_text_compress(model, prompt, max_new_tokens = 50, temperature=0.50, penalty = 1.4 , ratio = 0.70)

```

## Citation

If you use this repository in academic work, consider citing the following paper:

> 1. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

> 2. [Attention Is all You Need](http://arxiv.org/abs/1706.03762)

> 3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)

> 4. [A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://arxiv.org/abs/2406.11430)






