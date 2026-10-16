---
layout: page
title: "Fine-Tuning Open Language Models"
description: "Experiments with instruction tuning and parameter-efficient fine-tuning of open language models."
importance: 8
category: llm-post-training
---

## Overview

This work explores adapting open language models to instruction-following tasks using supervised fine-tuning and parameter-efficient methods.

The experiments include models from the Qwen, Gemma, and SmolLM families, with an emphasis on understanding the practical trade-offs between model size, training resources, and adaptation techniques.

## Topics explored

- Instruction data preparation and formatting.
- Supervised fine-tuning.
- Low-Rank Adaptation (LoRA) and Quantized LoRA (QLoRA).
- Quantization and memory-aware training.
- Model evaluation and experiment tracking.
- Using the Hugging Face ecosystem to share models and reproducible work.

## Research motivation

Parameter-efficient fine-tuning makes it possible to investigate model adaptation under limited computational resources. It also provides a practical foundation for later work on preference optimization and reinforcement learning for language models.

## Technologies

Python, PyTorch, Hugging Face Transformers, PEFT, LoRA, QLoRA.

## Model portfolio

[Explore my models on Hugging Face 🤗](https://huggingface.co/Mouhamedamar)
