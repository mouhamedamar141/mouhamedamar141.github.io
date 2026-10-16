
---
layout: page
title: "Behavior Cloning from Scratch: PushT"
description: "A practical implementation of behavior cloning using the PushT manipulation task."
importance: 6
category: robot-learning
---

## Overview

This project studies behavior cloning, a form of imitation learning in which a policy learns to reproduce actions from expert demonstrations.

The PushT task provides a manipulation setting for exploring how a model maps observations to actions and how the quality of demonstrations affects learned behavior.

## Main components

- Working with demonstration datasets and LeRobot data tools.
- Preparing observations and action targets.
- Implementing a behavior-cloning training pipeline in PyTorch.
- Training and evaluating a policy in a constrained computing environment.
- Exploring the strengths and limitations of supervised imitation learning.

## Key takeaway

Behavior cloning offers a simple starting point for learning robot policies, but performance depends on demonstration quality and the distribution of states encountered during training and evaluation.

## Technologies

Python, PyTorch, LeRobot, imitation learning, robot manipulation.

## Repository

[View the source code on GitHub](https://github.com/mouhamedamar141/BC-pusht)
