
---
layout: page
title: "LLM Supervised Fine-Tuning Pipeline"
description: "An instruction fine-tuning pipeline exploring training mechanics and experiment tracking."
importance: 3
category: llm-post-training
---

## Overview

This project focuses on supervised fine-tuning (SFT) for large language models. It explores how a pretrained model can be adapted to follow instructions through carefully constructed training data and a reproducible training procedure.

## Key components

- Instruction-format data preparation.
- Prompt and response masking for training.
- Gradient accumulation to manage memory requirements.
- Learning-rate warmup and decay.
- Training monitoring and experiment tracking with Weights & Biases.
- Fine-tuning experiments with an OLMo-2 model.

## Learning objectives

The project examines the practical details behind instruction tuning, including which tokens contribute to the training loss, how optimization is configured, and how experiments can be monitored and reproduced.

## Technologies

Python, PyTorch, Transformers, supervised fine-tuning, Weights & Biases.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/llm-sft-pipeline)
