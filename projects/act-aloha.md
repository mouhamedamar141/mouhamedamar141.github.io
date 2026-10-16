
---
layout: page
title: "Action Chunking Transformer on ALOHA"
description: "Exploring imitation learning and chunked action prediction for robotic manipulation."
importance: 5
category: robot-learning
---

## Overview

This project explores the Action Chunking Transformer (ACT), an imitation-learning approach for learning robot behavior from demonstrations.

Instead of predicting only the next action, ACT predicts a sequence, or chunk, of future actions. This provides a useful framework for studying temporally coherent behavior in robotic manipulation.

## Main concepts

- Learning robot policies from demonstrations.
- Predicting chunks of actions.
- Temporal ensembling to combine predictions across time.
- Understanding the relationship between sequence modeling and robot control.
- Working with the ALOHA robotic manipulation setup through the LeRobot ecosystem.

## Research motivation

Imitation learning is an important approach to robot learning because it allows policies to learn from demonstrations without requiring a manually designed reward for every task.

This project is part of my exploration of learning-based control, manipulation, and embodied AI.

## Technologies

Python, PyTorch, ACT, LeRobot, imitation learning.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/action-chunking-transformer-Aloha)
