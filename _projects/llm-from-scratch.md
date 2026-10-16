
---
layout: page
title: "LLM from Scratch: Architectures and Optimizers"
description: "An educational implementation and comparison of language-model components, attention mechanisms, positional methods, and optimizers."
importance: 4
category: llm-architectures
---

## Overview

This project is a structured exploration of the components behind modern language models. It aims to make architectural choices and optimization methods easier to understand through implementations, comparisons, and executable notebooks.

## Topics covered

### Tokenization and representations

- Character-level and byte-level BPE tokenization.
- WordPiece and Unigram tokenization.
- Token embeddings and positional representations.

### Positional information

- Sinusoidal and learned positional embeddings.
- Rotary Position Embeddings (RoPE).
- RoPE scaling approaches, including NTK-aware scaling and YaRN.
- ALiBi and NoPE.

### Attention and transformer components

- Multi-Head Attention (MHA).
- Multi-Query Attention (MQA) and Grouped-Query Attention (GQA).
- Multi-head Latent Attention (MLA).
- Sliding-window attention and QK normalization.
- KV caching and normalization methods such as LayerNorm and RMSNorm.

### Model architecture and optimization

- Feed-forward network variants.
- Mixture-of-Experts (MoE), top-k routing, Switch-style routing, and related techniques.
- SGD, Adam, AdamW, Lion, and Muon.
- Learning-rate schedules, exponential moving averages, and checkpoint resumption.

## Approach

The repository combines modular implementations, explanatory notebooks, and automated tests to support learning through experimentation. It is intended as an educational resource rather than a claim to reproduce every production implementation exactly.

## Technologies

Python, PyTorch, tokenization, Transformers, optimization, automated testing.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/lm-from-scratch-architectures-and-optimizers)
