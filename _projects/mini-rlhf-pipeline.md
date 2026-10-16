---
layout: page
title: "Mini RLHF Pipeline: SFT, Reward Modeling and PPO"
description: "A compact implementation exploring the main stages of Reinforcement Learning from Human Feedback."
importance: 2
category: llm-post-training
---

## Overview

This project explores the main stages of Reinforcement Learning from Human Feedback (RLHF) in a compact, educational pipeline.

Rather than treating RLHF as a black box, the implementation studies how supervised fine-tuning, preference modeling, and reinforcement learning can be combined to adapt a language model to a reward signal.

## Pipeline

1. **Supervised Fine-Tuning (SFT):** train a small language model on instruction-style examples.
2. **Reward Modeling:** learn a reward model from pairwise preference data using a Bradley–Terry-style objective.
3. **PPO Optimization:** optimize the language model using a reward signal and a KL-divergence penalty to constrain deviation from a reference policy.
4. **Analysis:** inspect reward evolution, generated responses, and possible reward-hacking behavior.

## Experiments

The experiments include monitoring the SFT training loss, evaluating preference prediction, and tracking changes in the reward signal during PPO optimization. The results are specific to this small experimental setup and should not be interpreted as a benchmark against production-scale RLHF systems.

## Research motivation

This project provides a practical foundation for studying policy optimization, reward misspecification, KL regularization, and alignment methods for language models.

## Technologies

Python, PyTorch, language modeling, reward modeling, PPO.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/mini-RLHF-pipeline)
