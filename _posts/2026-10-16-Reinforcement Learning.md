---
layout: post
title: "Reinforcement Learning: From Foundations to Modern Policy Optimization for Large Language Models"
date: 2026-10-16
description: "A comprehensive mathematical guide to reinforcement learning, from MDPs and policy gradients to PPO, RLHF, DPO, GRPO, and RLVR for LLM post-training."
tags: [reinforcement-learning, policy-gradients, PPO, RLHF, DPO, GRPO, RLVR, LLMs]
categories: [Reinforcement Learning]
featured: true
---

# Reinforcement Learning: From Foundations to Modern Policy Optimization for Large Language Models

**Author:** Mouhamed Amar

**Subtitle:** A comprehensive guide to reinforcement learning theory, from Markov decision processes and Bellman equations to policy gradients, PPO, RLHF, DPO, GRPO, and verifiable-reward training for large language models.

## Abstract

Reinforcement learning has evolved from a niche technique for sequential decision-making into a central framework for aligning and fine-tuning large language models. This article provides a complete and self-contained treatment of the mathematical foundations that connect classical reinforcement learning to modern language model post-training.

We begin with the Markov decision process, derive the Bellman equations, and explain value-based and policy-based methods. We then present the policy gradient theorem from first principles, showing how the score-function estimator enables gradient-based optimization without knowledge of environment dynamics. We develop actor-critic architectures, generalized advantage estimation, and the clipped surrogate objective of PPO.

Turning to language models, we formalize autoregressive generation as a token-level Markov decision process and derive the RLHF objective, including the KL-regularized optimal policy that underlies both PPO-based RLHF and Direct Preference Optimization. We present GRPO as a critic-free alternative that estimates advantages from group-relative rewards, and we situate RLVR as a reward paradigm rather than an algorithm.

Throughout, we distinguish mathematical guarantees from empirical regularities, derive every major objective from its assumptions, and identify the failure modes such as reward hacking, distribution shift, and verifier exploitation that remain active research questions.

## Table of Contents

1. Introduction
2. Mathematical Foundations of Reinforcement Learning
3. Policy Gradients: The Central Mathematical Connection
4. From Policy Gradients to PPO
5. Connecting Classical RL to Large Language Models
6. RLHF: Reinforcement Learning from Human Feedback
7. DPO: Direct Preference Optimization
8. RLVR: Reinforcement Learning with Verifiable Rewards
9. GRPO and Group-Based Policy Optimization
10. Additional Modern Methods and Connections
11. A Unified Mathematical View
12. Practical and Scientific Limitations
13. Reproducible Experiments and Implementation Intuition
14. Open Research Questions
15. Conclusion
16. Glossary of Symbols and Concepts
17. References

## 1. Introduction

Reinforcement learning (RL) is a branch of machine learning in which an agent learns to make sequential decisions by interacting with an environment. The goal of the agent is to maximize cumulative reward over time by discovering which actions produce the best outcomes in different states. Unlike supervised learning, where a model is trained on labeled data, RL is built on exploration and exploitation: the agent must try various actions to discover high-reward strategies while exploiting what it has already learned to achieve long-term success.

The striking successes of RL are well known. In 2016, AlphaGo defeated the world champion Lee Sedol at the game of Go, a feat widely considered a milestone in artificial intelligence because Go has a search space far larger than chess and was long thought to be beyond the reach of machine learning [14]. In 2017, AlphaZero demonstrated that a single algorithm could master Go, chess, and shogi from scratch through self-play, without any human knowledge beyond the rules of the game [15]. In robotics, RL has enabled robots to learn dexterous manipulation, locomotion, and grasping policies directly from sensor data [10]. In autonomous driving, RL is used to train decision-making policies that handle complex, dynamic environments. And in large language models, RL now sits at the center of how models are aligned, fine-tuned, and taught to reason [3], [16].

Deep RL extends traditional RL by leveraging deep neural networks to handle complex environments with high-dimensional state spaces. This allows agents to learn directly from raw, unstructured data such as pixels in video games or sensor readings in robotic control. Deep RL algorithms like Deep Q-Networks (DQN) and policy gradient methods such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) have achieved breakthroughs in domains including video games, robotics, and autonomous systems [6], [9], [17].

This article argues that policy optimization provides a unifying mathematical framework connecting these seemingly disparate domains. The same score-function estimator that lets a robot learn to grasp an object also lets a language model learn to solve mathematical problems. The same advantage function that reduces variance in continuous control also reduces variance in token-level credit assignment. The same KL-regularized objective that constrains a policy from drifting too far from a reference appears, with minor modifications, in both RLHF and Direct Preference Optimization.

But unification should not be confused with identity. RLHF is not just PPO applied to text. DPO is not just RLHF with the reward model removed. GRPO is not simply PPO without a critic. Each method makes distinct assumptions, optimizes distinct objectives, and fails in distinct ways. The goal of this article is to derive these methods from their foundations, make the assumptions explicit, and clarify what is guaranteed versus what is merely observed.

## 2. Mathematical Foundations of Reinforcement Learning

### 2.1 What is Reinforcement Learning?

Reinforcement learning is a type of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning, where a model learns from a fixed dataset of labeled examples, RL focuses on learning from the consequences of actions rather than from predefined correct behavior. The interaction between the agent and the environment is guided by the concepts of states, actions, rewards, and policies, which form the foundation of RL. The agent seeks to maximize cumulative rewards by exploring different actions and learning which ones yield the best outcomes over time.

Deep RL extends this framework by incorporating neural networks to handle high-dimensional, complex problems that traditional RL methods struggle with. By using deep learning techniques, Deep RL can tackle challenges like visual input or other high-dimensional data, allowing it to solve problems that are intractable for classical RL approaches. This combination of RL and neural networks enables agents to perform well in more complex environments with minimal manual intervention.

### 2.2 Key Components of Reinforcement Learning

At the core of RL is the interaction between an agent and an environment. In this interaction, the agent takes actions in the environment and receives feedback in the form of states and rewards. The goal is for the agent to learn a strategy, or policy, that maximizes the cumulative reward over time.

The critical components of RL are:

**Agent/Learner:** The agent is the learner or decision-maker. It is responsible for selecting actions based on the current state of the environment.

**Environment:** Everything the agent interacts with. The environment defines the rules of the game, transitioning from one state to another based on the agent's actions.

**State ($s$):** A representation of the environment at a particular point in time. States encapsulate all the information that the agent needs to know to make a decision. For example, in a video game, a state might be the current configuration of the game board.

**Action ($a$):** A decision taken by the agent in response to the current state. In each state, the agent must choose an action from a set of possible actions, which will affect the future state of the environment.

**Reward ($r$):** A scalar value that the agent receives from the environment after taking an action. The reward provides feedback on how good or bad an action was in that particular state. The agent's objective is to maximize the cumulative reward over time, often referred to as the return.

**Policy ($\pi$):** A policy is the strategy the agent uses to determine the actions to take based on the current state. It is implemented as a mapping from states to action probabilities. It can be tabular, i.e., a simple lookup table mapping states to actions, or it can be more complex, such as a neural network in the case of deep RL. The policy can be deterministic (always taking the same action for a given state) or stochastic (taking different actions with some probability).

**Value Function:** The value function estimates how good (i.e., how much total expected reward) it is to be in a particular state (in which case, it is called the state-value function) or to take a specific action in that state (in which case, it is called the action-value function). It does so by accounting for both the immediate reward and the expected future rewards from subsequent states, helping the agent understand long-term reward potential rather than focusing only on immediate rewards. While value functions are of two types, the state-value function is also commonly called the value function. The relationship between the two is given by:

$$
V^\pi(s) = \sum_a \pi(a \mid s) Q^\pi(s, a),
$$

which means the value of a state under policy $\pi$ is the expected action-value, averaged over all possible actions the policy might take in that state.

**State-Value Function (V-function):** Denoted as $V^\pi(s)$, the state-value function measures the expected return (total discounted reward) starting from state $s$ and then following policy $\pi$ thereafter. It is formally defined as:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s \right],
$$

where $\pi$ is the policy, $\gamma \in [0, 1)$ is the discount factor weighting future rewards, $r_t$ is the reward received at time step $t$, and the expectation $\mathbb{E}_\pi[\cdot]$ is taken over trajectories generated by following policy $\pi$.

**Action-Value Function (Q-function):** Denoted as $Q(s, a)$ (where $Q$ stands for "quality"), the action-value function measures the expected return for taking action $a$ in state $s$ and then following the policy $\pi$ thereafter. It plays a central role in algorithms like Q-learning and Deep Q-Networks (DQN). It is formally defined as:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid s_0 = s, a_0 = a \right],
$$

where the terms have the same meanings as above, except that the expectation begins from both a given state $s$ and an initial action $a$.

**Advantage Function ($A$):** The advantage function quantifies how much better taking a specific action $a$ in state $s$ is compared to the average action according to the policy. Put simply, the advantage function measures how much better the actual return was from a particular state than what was expected from that state before acting. It is commonly used in policy gradient methods such as Actor-Critic and Proximal Policy Optimization (PPO) to reduce variance in gradient estimates. It is defined as:

$$
A(s, a) = Q(s, a) - V(s),
$$

where $Q(s, a)$ is the action-value function and $V(s)$ is the state-value function.

**Return ($G$):** The total accumulated reward from a given time step onward, often discounted to prioritize near-term rewards:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1},
$$

where $G$ stands for "gain" and $\gamma$ is the discount factor that determines how much future rewards are valued relative to immediate rewards.

**Discount Factor ($\gamma$):** A scalar between 0 and 1 that controls the importance of future rewards. Smaller values make the agent myopic (focusing on immediate rewards), while larger values encourage long-term planning. Typically set closer to 1 to focus on long-term rewards rather than immediate ones.

**Exploration vs. Exploitation:** The trade-off between exploring new actions to discover potentially better rewards and exploiting known actions that already yield high rewards. Balancing these two is crucial for effective learning.

**Trajectory/Episode/Rollout:** A sequence of states, actions, rewards, and next states from the beginning of an episode to its termination, representing one complete interaction of the agent with the environment.

**Temporal-Difference (TD) Error:** The difference between the predicted value of a state and the observed reward plus the estimated value of the next state. It is used to update value estimates dynamically in methods like TD-learning, where the TD error is given by:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),
$$

with $r_t$ as the immediate reward, $\gamma$ the discount factor, and $V(s_t)$ and $V(s_{t+1})$ being the predicted values of the current and next states respectively.

**Replay Buffer (Experience Replay):** In Deep RL, a replay buffer stores past transitions (state, action, reward, next state) for sampling during training. This helps break correlation between consecutive samples, since experiences are drawn randomly rather than sequentially, allowing the agent to learn from a more diverse and independent set of experiences, which improves data efficiency and stabilizes training.

**Actor-Critic Architecture:** A hybrid approach combining a policy-based (actor) component that selects actions and a value-based (critic) component that evaluates them. The critic's feedback stabilizes the actor's learning.

### 2.3 The Bellman Equation

The Bellman Equation is a fundamental concept in RL, used to describe the relationship between the value of a state and the value of its successor states. It breaks down the value function into immediate rewards and the expected value of future states.

For a given policy $\pi$, the state-value function $V^\pi(s)$ can be written as:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ r_t + \gamma V^\pi(s_{t+1}) \mid s_t = s \right]
$$

where:

- $V^\pi(s)$ is the value of state $s$ under policy $\pi$,
- $r_t$ is the reward received after taking an action at time $t$,
- $\gamma$ is the discount factor $(0 \leq \gamma \leq 1)$ that determines the importance of future rewards,
- $s_{t+1}$ is the next state after taking an action from state $s$.

This equation expresses that the value of a state $s$ is the immediate reward $r_t$ plus the discounted value of the next state $V^\pi(s_{t+1})$. The Bellman equation is central to many RL algorithms, as it provides the basis for recursively solving the optimal value function.

### 2.4 The RL Process: Trial and Error Learning

The agent interacts with the environment in a loop:

1. At each time step, the agent observes the current state of the environment.
2. Based on this state, it selects an action according to its policy.
3. The environment transitions to a new state, and the agent receives a reward.
4. The agent uses this feedback to update its policy, gradually improving its decision-making over time.

This process of learning from trial and error allows the agent to explore different actions and outcomes, eventually finding the optimal policy that maximizes the long-term reward.

### 2.5 Mathematical Formulation: Markov Decision Process (MDP)

RL problems are typically framed as Markov Decision Processes (MDP), which provide a mathematical framework for modeling decision-making where outcomes are partly random and partly under the control of the agent. An MDP is defined by:

- **States ($S$):** The set of all possible states in the environment.
- **Actions ($A$):** The set of all possible actions the agent can take.
- **Transition function ($P$):** The probability distribution $P(s_{t+1} \mid s_t, a_t)$ describing transitions from one state to another given an action.
- **Reward function ($R$):** The immediate reward $R(s_t, a_t, s_{t+1})$ received after transitioning between states.
- **Discount factor ($\gamma$):** A scalar $\gamma \in [0, 1]$ controlling the importance of future rewards; values close to 0 emphasize immediate rewards, while values close to 1 emphasize long-term rewards.

The agent's goal is to learn a policy $\pi(s)$ that maximizes the expected cumulative reward (i.e., expected return), often expressed as:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}
$$

where:

- $G_t$ is the total return starting from time step $t$,
- $\gamma$ is the discount factor,
- $r_{t+k+1}$ is the reward received at time $t + k + 1$.

### 2.6 Value-Based Methods

Value-based methods form the cornerstone of RL. Their core principle is to learn value functions that estimate how good it is for an agent to be in a given state or to perform a specific action in that state. These methods do not learn policies directly; instead, they infer the optimal policy from the learned values by choosing actions that maximize expected future rewards.

Two central value functions define this class of methods:

**State-Value Function:**

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid S_0 = s \right]
$$

This represents the expected cumulative reward when starting from state $s$ and following policy $\pi$ thereafter.

**Action-Value Function:**

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid S_0 = s, A_0 = a \right]
$$

This quantifies the expected return when taking action $a$ in state $s$ and then following policy $\pi$.

The optimal policy $\pi^*$ can then be derived as:

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

where $Q^*(s, a)$ is the optimal action-value function.

**Dynamic Programming (DP)** represents the earliest and most theoretically grounded approach to solving RL problems. It assumes that a complete model of the environment is known, specifically the transition probabilities $P(s' \mid s, a)$ and reward function $R(s, a)$. Introduced by Bellman (1957), DP methods are built upon the Bellman Optimality Equation, which recursively expresses the relationship between the value of a state and the values of its successor states:

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$

Two major DP algorithms are:

- **Value Iteration:** Alternates between evaluating and improving the value function until convergence to $V^*(s)$.
- **Policy Iteration:** Alternates between policy evaluation (estimating $V^\pi$) and policy improvement (updating $\pi$) until the policy stabilizes.

DP is exact and guaranteed to converge for finite MDPs, but it is computationally infeasible in large state spaces due to the curse of dimensionality.

**Monte Carlo (MC) methods** learn value functions from experience, without requiring a model of the environment. They estimate expected returns by averaging the actual returns observed after complete episodes of experience. For a state $s$, the Monte Carlo estimate of the value is:

$$
V(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_i
$$

where $G_i$ is the total return following the $i$-th visit to $s$, and $N(s)$ is the number of visits to $s$.

Advantages of Monte Carlo methods include being model-free (no need for transition probabilities) and providing simple and unbiased estimates after enough samples. Limitations include requiring episodes to terminate (not suitable for continuing tasks) and slow convergence due to reliance on complete trajectories.

**Temporal Difference (TD) learning** blends the key ideas of Monte Carlo and Dynamic Programming: learning directly from raw experience without requiring a model, and updating value estimates based on bootstrapping from other estimates. The core update rule for TD is:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[ r_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]
$$

Here, the agent updates its estimate of $V(S_t)$ using the observed reward plus the discounted value of the next state, rather than waiting for the episode to finish. TD learning provides the foundation for most modern value-based algorithms, including SARSA and Q-Learning.

Advantages of TD learning include online, incremental updates; works for both episodic and continuing tasks; and converges faster than Monte Carlo in many settings.

### 2.7 Policy-Based Methods

While value-based methods focus on estimating the long-term value of states or state-action pairs, policy-based methods take a more direct approach: they learn a parameterized policy that maps states to actions and optimize it to maximize expected return. These methods are particularly useful in environments with continuous or stochastic action spaces, where value-based techniques like Q-learning are difficult to apply effectively.

In policy-based RL, the agent's behavior is represented by a stochastic policy $\pi_\theta(a \mid s)$, parameterized by $\theta$. The goal is to find parameters that maximize the expected return:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

Unlike value-based methods, which derive a policy indirectly from learned value estimates, policy-based approaches directly optimize this objective by computing its gradient with respect to the parameters $\theta$.

The key insight enabling policy optimization is the **Policy Gradient Theorem** [2]. It provides a way to estimate the gradient of the expected return without differentiating through the environment's dynamics:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]
$$

This formulation allows gradient ascent on $J(\theta)$ using trajectories sampled from the current policy. Intuitively, the update increases the probability of actions that yield higher returns and decreases it for less rewarding ones.

The **REINFORCE algorithm** [7] is the simplest and most influential policy gradient method. It estimates the gradient using complete episodes of experience, updating the policy parameters as follows:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t
$$

where:

- $\alpha$ is the learning rate,
- $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ is the return following time $t$.

The algorithm works by reinforcing (increasing the probability of) actions that lead to higher observed returns.

Because the variance of gradient estimates can be large, REINFORCE often includes a baseline $b(s_t)$, typically the state value $V^\pi(s_t)$, to reduce variance without introducing bias:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left[ G_t - b(s_t) \right]
$$

This concept laid the foundation for later actor-critic methods, where the critic effectively serves as a learned baseline.

Standard gradient ascent can be inefficient in policy space due to curvature distortions caused by parameterization. The **Natural Policy Gradient (NPG)** method, introduced by Kakade (2001) [12], addresses this by using the Fisher information matrix $F(\theta)$ to compute updates invariant to the parameter scaling:

$$
\theta \leftarrow \theta + \alpha F(\theta)^{-1} \nabla_\theta J(\theta)
$$

This ensures that updates are taken in directions that respect the geometry of the policy distribution, leading to faster and more stable convergence.

Advantages of policy-based methods include naturally handling continuous and stochastic action spaces, enabling stochastic exploration without explicit noise, and offering smooth policy improvement without discontinuities. Limitations include high variance in gradient estimates, often requiring large numbers of trajectories for accurate estimation, and sensitivity to hyperparameters like learning rate and baseline design.

### 2.8 Actor-Critic Methods

Actor-Critic methods bridge the conceptual gap between value-based and policy-based RL. While policy-based methods optimize the policy directly and value-based methods estimate the expected return, actor-critic frameworks do both simultaneously. They maintain two distinct components:

- The **Actor**, which updates the policy parameters in the direction suggested by the critic's evaluation.
- The **Critic**, which estimates value functions and provides a baseline to stabilize and guide policy updates.

This architecture allows actor-critic methods to combine the low variance of value-based updates with the expressive flexibility of policy-based optimization.

The actor-critic approach builds upon the policy gradient theorem and temporal difference (TD) learning. At time $t$, the policy $\pi_\theta(a \mid s)$ selects an action, and the critic evaluates it using a value function $V_w(s)$ or $Q_w(s, a)$, parameterized by weights $w$. The actor updates its policy parameters according to:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) \delta_t
$$

where $\delta_t$ is the TD error, defined as:

$$
\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)
$$

This TD error acts as a critic signal, indicating whether the action taken was better or worse than expected.

To improve stability and efficiency, actor-critic methods often use the advantage function, which measures how much better an action $a$ is compared to the average action in a given state:

$$
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)
$$

Using the advantage function instead of raw returns reduces variance in policy gradient estimates, leading to smoother learning. The resulting update rule becomes:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) A^\pi(s_t, a_t) \right]
$$

This formulation unifies the critic's evaluative feedback with the actor's improvement mechanism.

The actor-critic paradigm originated with the Adaptive Heuristic Critic (AHC) architecture proposed by Barto, Sutton & Anderson (1983) [8]. It introduced the two-network idea: one learning to evaluate (critic) and another learning to control (actor). Subsequent developments expanded this framework into more specialized variants, including Incremental Natural Actor-Critic (INAC) by Peters & Schaal (2008), Continuous Actor-Critic Learning Automaton (CACLA) by Van Hasselt & Wiering (2007), and Asynchronous Advantage Actor-Critic (A3C), whose theoretical roots lie in classical actor-critic formulations.

Actor-Critic algorithms can be seen as implementing a generalized policy iteration (GPI) process, alternating between policy evaluation (the critic estimates $V^\pi(s)$ or $Q^\pi(s, a)$ using TD learning or Monte Carlo rollouts) and policy improvement (the actor updates $\pi_\theta(a \mid s)$ using gradient ascent based on the critic's feedback). This dynamic mirrors classical policy iteration by Howard (1960), but operates incrementally and stochastically, enabling online learning in complex environments.

Advantages of actor-critic methods include combining the strengths of policy and value methods (low bias, low variance), suitability for continuous action spaces, support for online and incremental learning, and natural extension to partially observable and stochastic domains. Limitations include sensitivity to critic accuracy (unstable when critic is poorly estimated), requiring careful tuning of learning rates for actor and critic, and potential oscillatory dynamics if updates are not synchronized.

### 2.9 Model-Based Reinforcement Learning

Model-Based Reinforcement Learning (MBRL) refers to a family of techniques that explicitly learn or exploit a model of the environment's dynamics to predict future states and rewards, enabling planning and sample-efficient policy optimization. Unlike model-free methods that learn purely from experience, model-based approaches simulate potential futures to guide decision-making. This distinction makes model-based methods conceptually closer to optimal control theory and planning algorithms used in operations research and robotics.

The central concept in model-based RL is the MDP model, represented by:

**Transition Function:**

$$
P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

**Reward Function:**

$$
R(s, a) = \mathbb{E}[r_{t+1} \mid s, a]
$$

With access to these functions, one can compute expected returns, plan trajectories, and compute optimal policies using classical algorithms such as Value Iteration and Policy Iteration. The model can either be given (known dynamics) or learned (unknown dynamics), where the agent estimates $P(s' \mid s, a)$ and $R(s, a)$ from collected experience.

Given a known model, the agent can perform planning: evaluating and improving policies without interacting with the real environment. This is accomplished by recursively solving the Bellman Optimality Equation:

$$
V^*(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$

and deriving the corresponding optimal policy:

$$
\pi^*(s) = \arg\max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^*(s') \right]
$$

In more realistic settings, the transition and reward models are not known a priori. In such cases, the agent must learn an approximate model from experience:

$$
\hat{P}(s' \mid s, a) \approx P(s' \mid s, a), \quad \hat{R}(s, a) \approx R(s, a)
$$

Learning these models transforms the RL problem into a supervised learning task, where the goal is to predict next states and rewards from observed transitions $(s, a, s', r)$. Model-learning can use tabular frequency estimates in small discrete environments, regression or Gaussian processes in continuous spaces, or function approximators.

A seminal hybrid framework combining learning, planning, and acting was proposed in Dyna by Sutton (1990). Dyna integrates:

- **Model learning:** Build an internal model from experience.
- **Planning:** Generate synthetic experiences from the model to update the value function.
- **Real experience:** Continue updating from actual environment interactions.

This allows the agent to perform imaginary rollouts using its learned model, accelerating learning while maintaining adaptability. Formally, Dyna's process alternates between direct RL updates from real experiences:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

and simulated updates using the learned model $\hat{P}, \hat{R}$:

$$
\tilde{Q}(s, a) \leftarrow \tilde{Q}(s, a) + \alpha \left[ \hat{R}(s, a) + \gamma \max_{a'} \tilde{Q}(\hat{s}', a') - \tilde{Q}(s, a) \right]
$$

Advantages of model-based methods include sample efficiency (learns faster due to simulated experience), planning capability (can evaluate long-term effects before acting), and flexibility (unifies learning and control). Challenges include model bias (imperfect models can lead to suboptimal or unstable policies), complexity (model estimation adds computational and representational burden), and scalability (accurate models are difficult in large or stochastic environments).

### 2.10 Model-Free Reinforcement Learning

Model-Free Reinforcement Learning (MFRL) refers to a broad class of algorithms that learn optimal behavior without explicitly modeling the environment's dynamics. Instead of estimating transition probabilities $P(s' \mid s, a)$ or reward functions $R(s, a)$, model-free agents learn value functions or policies directly from raw experience tuples $(s, a, r, s')$. This makes MFRL algorithms simpler and more general, at the expense of sample efficiency. They form the practical foundation for most online RL systems and are closely tied to the concept of trial-and-error learning.

In a model-free setting, the agent's objective remains to learn an optimal policy $\pi^*(a \mid s)$ that maximizes the expected return:

$$
J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

However, since the agent does not possess an explicit model of the environment, it must approximate this expectation using empirical experience collected through exploration. Learning proceeds incrementally by adjusting estimates of value functions or policies based on observed temporal-difference (TD) errors.

A key distinction in model-free RL is how experiences are gathered and used:

- **On-Policy Methods:** Learn from actions taken by the current policy (e.g., SARSA). The agent learns to evaluate and improve the same policy it uses for exploration.
- **Off-Policy Methods:** Learn from actions generated by a different policy (e.g., Q-Learning). This allows leveraging historical or exploratory data for more efficient learning.

This dichotomy was formalized by Precup, Sutton & Singh (2000), who introduced importance sampling corrections to enable off-policy evaluation.

**SARSA (State-Action-Reward-State-Action)**, proposed by Rummery & Niranjan (1994) [4], is an on-policy temporal-difference control algorithm. It updates the action-value function $Q(s, a)$ based on the transition sequence $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

This update reflects the return expected from continuing to act according to the current policy, which makes it safer and more stable for non-stationary environments, though sometimes slower to converge. Key properties include evaluating the current behavior policy directly, naturally balancing exploration and exploitation, and being more robust under stochasticity.

**Q-Learning**, introduced by Watkins & Dayan (1992) [5], is the archetypal off-policy model-free algorithm. It estimates the optimal action-value function $Q^*(s, a)$ by updating toward the maximum value achievable from the next state, regardless of the current behavior policy:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

This formulation separates policy evaluation (learning from exploratory behavior) from policy improvement (acting greedily with respect to $Q$), enabling learning from arbitrary data sources or replay buffers. Key properties include convergence to $Q^*$ under standard assumptions (finite state-action space, decaying learning rate) and high flexibility, as it can learn from off-policy or logged data. It is the foundation for most modern off-policy algorithms, including Deep Q-Networks (DQN).

Model-free RL requires effective exploration to ensure sufficient coverage of the state-action space. Common strategies include:

**$\epsilon$-Greedy Exploration:** With probability $1 - \epsilon$, choose the greedy action; with probability $\epsilon$, pick a random one. Balances exploitation of known high-value actions with exploration of new ones.

**Softmax / Boltzmann Exploration:** Selects actions probabilistically according to their estimated Q-values:

$$
P(a \mid s) = \frac{e^{Q(s, a)/\tau}}{\sum_b e^{Q(s, b)/\tau}}
$$

where $\tau$ controls exploration temperature.

**Upper Confidence Bounds (UCB):** Encourages exploration of actions with higher uncertainty in their value estimates.

These techniques are crucial for preventing premature convergence to suboptimal policies, especially in stochastic or large environments.

Advantages of model-free methods include being simpler and easier to implement than model-based methods, requiring no explicit environment model, and being robust across varied environments and tasks. Limitations include poor sample efficiency due to reliance on real experience, limited ability to plan or simulate long-term outcomes, and exploration-exploitation trade-offs that can be difficult to tune.

### 2.11 On-Policy vs. Off-Policy Reinforcement Learning

In RL, a critical design choice is how experience is collected and used to update the agent's knowledge. This gives rise to two fundamental paradigms, on-policy and off-policy learning, which differ in the relationship between the policy being improved and the policy being used to generate data. These paradigms span across value-based, policy-based, and actor-critic methods, and understanding their trade-offs is essential for algorithm design and stability.

Let $\pi$ denote the target policy (the policy being optimized) and $\mu$ denote the behavior policy (the policy used to generate experience data). Then:

- **On-Policy Learning:** $\pi = \mu$. The agent learns from data generated by its current policy.
- **Off-Policy Learning:** $\pi \neq \mu$. The agent learns from data collected under a different policy (e.g., past versions of itself, exploratory policies, or logged data).

In on-policy methods, the agent continuously improves the same policy it uses to interact with the environment. This ensures consistency between learning and behavior, but requires ongoing exploration and data collection. Mathematically, for a policy $\pi$, the value function satisfies:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^{\infty} \gamma^t r_t \mid S_0 = s \right]
$$

A classical example is SARSA, which updates its Q-values based on the actual next action taken by the same policy:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

This results in a learning process that closely tracks the policy's real performance, leading to greater stability, though potentially slower convergence.

In off-policy methods, the agent can learn from experience generated by another policy, allowing it to leverage past data, demonstrations, or exploration strategies. For example, Q-Learning uses the behavior policy to collect data, but learns about the optimal (greedy) target policy:

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

Here, the agent's learning policy (greedy) differs from its behavior policy (exploratory), enabling data reuse, offline learning, and greater flexibility.

Off-policy learning introduces distribution mismatch between the target policy $\pi$ and behavior policy $\mu$. To correct for this bias, importance sampling re-weights returns by the probability ratio of target and behavior policies:

$$
\rho_t = \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)}
$$

The corrected value estimate becomes:

$$
V^\pi(s_t) = \mathbb{E}_\mu \left[ \rho_t G_t \right]
$$

where $G_t = \sum_{k=t}^{\infty} \gamma^{k-t} r_k$ is the observed return. This technique allows off-policy algorithms to learn about arbitrary target policies from diverse datasets, a foundation for offline RL and batch learning.

The two paradigms exhibit complementary characteristics:

| Property | On-Policy | Off-Policy |
|----------|-----------|------------|
| Bias | Low (samples match learning policy) | Potentially high (distribution mismatch) |
| Variance | Moderate | High (due to importance weights) |
| Sample Efficiency | Low (requires fresh data) | High (reuses past experiences) |
| Stability | High | Can be unstable without correction |
| Applicability | Online / continual learning | Offline / batch learning |

In practice, hybrid approaches such as actor-critic or experience replay systems combine both paradigms to balance stability and efficiency.

## 3. Deep Reinforcement Learning

Deep Reinforcement Learning (Deep RL) refers to the integration of deep neural networks with RL, enabling agents to operate in high-dimensional, raw-input spaces (such as images or sensor feeds) and learn complex policies or value functions with minimal manual feature engineering. Classical RL methods provided the foundational theory; Deep RL extends these by using neural networks as function approximators for value functions, policies, or models.

In Deep RL, one often writes:

$$
\pi_\theta(a \mid s), \quad V_w(s), \quad Q_w(s, a)
$$

where $\theta, w$ are deep network parameters. The networks can approximate large or continuous state and action spaces, enabling Deep RL to surpass classical tabular or linear-function-approximation RL in many applications.

### 3.1 Deep Value-Based Methods

Deep Value-Based methods extend classical value-based RL, such as Q-learning, by using deep neural networks to approximate the value function $Q(s, a)$. This innovation enables agents to operate in high-dimensional observation spaces (like raw images), overcoming the limitations of tabular and linear methods that dominated early RL research.

In classical Q-learning, the optimal action-value function satisfies the Bellman Optimality Equation:

$$
Q^*(s, a) = \mathbb{E}_{s'} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
$$

However, maintaining a tabular representation of $Q(s, a)$ becomes infeasible in large or continuous state spaces. Deep Value-Based methods overcome this by parameterizing $Q(s, a)$ as a deep neural network $Q_\theta(s, a)$, trained to minimize the Temporal Difference (TD) error:

$$
L(\theta) = \mathbb{E}_{(s, a, r, s')} \left[ \left( r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a) \right)^2 \right]
$$

where $\theta^-$ represents the parameters of a target network, updated periodically to stabilize training.

The **Deep Q-Network (DQN)** introduced by Mnih et al. (2015) [6] marked a watershed moment for RL. By integrating convolutional neural networks with Q-learning, DQN achieved human-level control on Atari 2600 games from raw pixel inputs. DQN introduced two key innovations to stabilize learning:

- **Experience Replay:** Transitions $(s, a, r, s')$ are stored in a replay buffer and sampled uniformly to break correlation between sequential updates.
- **Target Network:** A separate network $Q_{\theta^-}$ is used for target computation, updated less frequently to prevent divergence.

One major limitation of the original DQN was overestimation bias in value updates due to the use of $\max_{a'} Q(s', a')$ both for action selection and evaluation. To address this, **Double DQN** by van Hasselt et al. (2016) decouples these steps:

$$
L(\theta) = \left( r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a')) - Q_\theta(s, a) \right)^2
$$

This reduces overestimation and yields more accurate Q-value estimates, improving both stability and performance.

The **Dueling Network Architecture** by Wang et al. (2016) decomposes the Q-function into two separate estimators: a state-value function $V(s)$ and an advantage function $A(s, a)$. The combined Q-function is then reconstructed as:

$$
Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a'; \theta, \alpha) \right)
$$

This structure improves learning efficiency by allowing the network to learn which states are valuable, independent of the specific actions.

**Prioritized Experience Replay** by Schaul et al. (2016) instead samples transitions with probability proportional to their TD error magnitude:

$$
P(i) = \frac{|\delta_i|^\alpha}{\sum_k |\delta_k|^\alpha}
$$

This focuses updates on transitions where the model is most surprised, improving data efficiency and convergence rates. To correct for the bias introduced by non-uniform sampling, importance sampling weights are applied:

$$
w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
$$

Several extensions of DQN further improved stability and performance, including NoisyNet DQN (Fortunato et al., 2018), which adds parameterized noise for exploration; Rainbow DQN (Hessel et al., 2018), which integrates multiple DQN enhancements; and Distributional DQN (Bellemare et al., 2017), which learns a distribution over returns rather than a scalar expected value.

### 3.2 Deep Policy-Based Methods

While value-based methods estimate $Q(s, a)$ or $V(s)$ and act greedily with respect to those values, policy-based RL directly optimizes a parameterized policy $\pi_\theta(a \mid s)$ to maximize expected return. This direct optimization allows the handling of continuous or stochastic action spaces and yields smoother learning dynamics.

Deep Policy-Based Methods extend classical policy-gradient ideas by representing $\pi_\theta(a \mid s)$ as a deep neural network, enabling end-to-end learning from high-dimensional inputs such as images or sensor data.

The goal is to maximize the expected return:

$$
J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

The Policy Gradient Theorem gives the gradient of $J(\theta)$:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) Q^{\pi_\theta}(s_t, a_t) \right]
$$

This elegant result allows gradient-based optimization of policies without differentiating through the environment dynamics.

The REINFORCE algorithm is the foundational Monte-Carlo policy-gradient method. It estimates the gradient using complete episode returns:

$$
\nabla_\theta J(\theta) \approx \sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) (G_t - b)
$$

where $G_t$ is the empirical return and $b$ is a baseline (often the mean return) that reduces variance without biasing the gradient. Despite high variance, REINFORCE provides an unbiased estimator and demonstrates the feasibility of learning stochastic deep policies.

To make policy-gradient learning practical, variance-reduction techniques are crucial:

**State-Value Baselines:** Replace raw return $G_t$ with an estimate of the advantage $\hat{A}_t = Q_t - V_t$, where $V_t$ is a learned value baseline.

**Generalized Advantage Estimation (GAE):** Introduced by Schulman et al. (2016) [11], GAE computes a bias-variance-controlled estimator of advantage by exponentially weighting multi-step TD errors:

$$
\hat{A}_t^{(\lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \text{where} \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This innovation enabled the training stability of modern deep policy-gradient algorithms.

One challenge in policy-gradient methods is catastrophic policy collapse due to overly large updates. **Trust Region Policy Optimization (TRPO)** [13] constrains the policy step within a trust region to ensure monotonic improvement:

$$
\max_\theta \mathbb{E}_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ \frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \hat{A}^{\pi_{\theta_{\text{old}}}}(s, a) \right] \quad \text{subject to} \quad \mathbb{E}\left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \| \pi_\theta(\cdot \mid s)) \right] \leq \delta
$$

This optimization ensures conservative updates, improving stability across large neural-network policies.

**Proximal Policy Optimization (PPO)** [9] is a policy-gradient algorithm designed to balance learning stability and efficiency. It improves on TRPO by enforcing a soft trust region through a clipped surrogate objective, which discourages updates that move the policy too far from its previous version. The clipped objective is defined as:

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where:

- $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ is the probability ratio between new and old policies.
- $\hat{A}_t$ is the advantage estimate, typically $r_t - V_\psi(s_t)$, showing how much better the taken action was than expected.
- $\epsilon$ is a small constant (often 0.1–0.3) controlling the trust-region width.

The min operator ensures that the policy update never increases the objective beyond what the clipped surrogate allows, preventing overly large steps that can harm performance. Intuitively, PPO encourages policy improvement proportional to the advantage while clipping the update if the new policy deviates too far from the old one. This yields a stable and robust optimization process widely used in RLHF for aligning large language models.

To encourage exploration and avoid premature convergence to deterministic policies, **entropy regularization** augments the objective:

$$
J'(\theta) = J(\theta) + \beta \mathbb{E}_{\pi_\theta} \left[ H(\pi_\theta(\cdot \mid s)) \right]
$$

where $H(\pi) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$. This technique, introduced in Soft Actor-Critic and earlier A3C methods, keeps the policy sufficiently stochastic to explore effectively.

### 3.3 Deep Actor-Critic Methods

Actor-Critic methods combine the advantages of value-based and policy-based RL by maintaining two distinct components:

- **Actor:** A policy network $\pi_\theta(a \mid s)$ that selects actions.
- **Critic:** A value or Q-network $V_w(s)$ or $Q_w(s, a)$ that estimates expected returns and provides feedback to the actor.

The actor updates its parameters to maximize the critic's estimated value, while the critic updates to better predict the returns observed from the actor's behavior. Deep Actor-Critic methods extend this paradigm using deep neural networks for both components, enabling scalability to complex, continuous, or high-dimensional environments.

The policy gradient for an actor-critic setup is given by:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s_t, a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}(s_t, a_t) \right]
$$

where $\hat{A}(s_t, a_t)$ is the advantage estimate that quantifies how much better action $a_t$ is compared to the average performance at state $s_t$. The critic learns this advantage by minimizing a regression loss, typically using Temporal Difference (TD) learning:

$$
L(w) = \mathbb{E} \left[ \left( r_t + \gamma V_w(s_{t+1}) - V_w(s_t) \right)^2 \right]
$$

Thus, the actor improves its policy using gradients from the critic's evaluation, creating a feedback loop that balances bias (from bootstrapping) and variance (from sampling).

The **Asynchronous Advantage Actor-Critic (A3C)** algorithm, introduced by Mnih et al. (2016), demonstrated that multiple agents (workers) can interact with independent environment instances in parallel, asynchronously updating a shared global model. Each worker learns both an actor and a critic, using an advantage-based update:

$$
\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) (R_t - V_w(s_t))
$$

$$
w \leftarrow w - \beta \nabla_w (R_t - V_w(s_t))^2
$$

This asynchronous setup increases data throughput and decorrelates experience, enabling training without replay buffers. A3C achieved state-of-the-art performance on a variety of Atari and continuous control benchmarks.

The **Advantage Actor-Critic (A2C)** algorithm is a synchronous variant of A3C that aggregates gradients from multiple parallel environments before performing a single update. Although less parallelized, A2C offers improved training stability and is widely used in implementations such as OpenAI Baselines. The advantage function is often estimated using Generalized Advantage Estimation (GAE), which balances bias and variance for stable learning.

For continuous control tasks (e.g., robotic movement), discrete action selection is infeasible. **Deep Deterministic Policy Gradient (DDPG)**, introduced by Lillicrap et al. (2015) [10], extends the actor-critic framework to deterministic policies:

$$
a = \mu_\theta(s)
$$

The actor is updated using the gradient of the critic's Q-value:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim D} \left[ \nabla_a Q_w(s, a) \mid_{a = \mu_\theta(s)} \nabla_\theta \mu_\theta(s) \right]
$$

DDPG employs a replay buffer for decorrelated training data, target networks for stable updates, and Ornstein-Uhlenbeck noise for exploration in continuous spaces. This made DDPG a foundational algorithm for robotic and control applications.

While DDPG is powerful, it suffers from overestimation bias similar to Q-learning. **Twin Delayed DDPG (TD3)**, by Fujimoto et al. (2018), mitigates this through three improvements: clipped double Q-learning (two critics are trained, and the smaller Q-value is used for the target), target policy smoothing (adds noise to target actions for robustness), and delayed policy updates (updates the actor less frequently than the critic for stability). Target computation in TD3 becomes:

$$
y = r + \gamma \min_{i=1, 2} Q_{w'_i}(s', \mu_{\theta'}(s') + \epsilon)
$$

where $\epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma), -c, c)$.

The **Soft Actor-Critic (SAC)** algorithm, proposed by Haarnoja et al. (2018), extends actor-critic learning to the maximum-entropy RL framework, optimizing not only expected returns but also policy entropy:

$$
J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \pi} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot \mid s_t)) \right]
$$

This encourages exploration by maximizing randomness in action selection while maintaining performance. SAC combines off-policy replay buffers with entropy regularization and is one of the most sample-efficient continuous control algorithms available.

### 3.4 Deep Model-Based Methods

Deep Model-Based Reinforcement Learning (MBRL) integrates the predictive structure of classical model-based RL with the representational power of deep neural networks. Rather than learning purely through trial and error, the agent first learns an internal world model, a neural approximation of the environment's dynamics and rewards, and then plans or trains policies within this learned model. This approach promises greater sample efficiency, safety, and generalization, since much of the learning occurs through simulated rollouts rather than direct environment interaction.

An MBRL system typically learns three components:

$$
\hat{P}_\phi(s' \mid s, a), \quad \hat{R}_\phi(s, a), \quad \pi_\theta(a \mid s)
$$

where $\hat{P}_\phi$ is a learned transition model, $\hat{R}_\phi$ is a reward predictor, and $\pi_\theta$ is the policy. The model can be explicit (predicting next states) or latent (predicting compact internal representations). Training alternates between collecting real experience using the current policy, updating the learned model $(\hat{P}_\phi, \hat{R}_\phi)$, and improving $\pi_\theta$ via rollouts simulated inside the model. This inner simulation loop enables learning with fewer real interactions, a major advantage over model-free Deep RL.

**World Models**, introduced by Ha & Schmidhuber (2018), pioneered neural latent-world modeling for RL. Their framework decomposed the agent into a VAE that encodes high-dimensional observations into a latent space, an MDN-RNN that predicts latent transitions over time, and a Controller, a small policy trained entirely in the latent world. This demonstrated that an agent could learn a compact generative model of the environment and achieve competitive control using simulated experience alone.

**Model-Based Policy Optimization (MBPO)**, by Janner et al. (2019), refined model-based learning by coupling short model rollouts with off-policy policy optimization. Instead of long, error-prone simulated trajectories, MBPO performs brief rollouts from real states sampled from the replay buffer. Formally, it optimizes:

$$
J(\theta) = \mathbb{E}_{(s, a, r, s') \sim D_{\text{real}} \cup D_{\text{model}}} \left[ r + \gamma V^{\pi_\theta}(s') \right]
$$

where $D_{\text{model}}$ contains transitions generated by the learned dynamics $\hat{P}_\phi$. This hybrid dataset balances realism and data efficiency, producing state-of-the-art sample efficiency among model-based algorithms.

The **Dreamer** family of algorithms, beginning with Hafner et al. (2019), introduced latent imagination-based planning. Dreamer learns a Recurrent State-Space Model (RSSM) to represent dynamics in a compact latent space, enabling policy updates entirely through "dreamed" trajectories without interacting with the environment. Subsequent versions (Dreamer V2 and V3) improved scalability to visual and continuous-control tasks, achieving human-level or super-human performance on benchmarks such as Atari and DMControl.

**MuZero**, introduced by Schrittwieser et al. (2020), combined deep model-based learning with Monte Carlo Tree Search (MCTS) while discarding explicit environment modeling. Instead of predicting next observations, MuZero learns latent dynamics sufficient for accurate planning in the representation space. Its core components are a representation network $h_\theta$ mapping observations to latent states, a dynamics network $g_\theta$ predicting next latent states and rewards, and a prediction network $f_\theta$ estimating policy and value from latent states. These networks are trained jointly to minimize:

$$
L = \sum_t \left[ l^r_t + l^v_t + l^p_t + c \|\theta\|^2 \right]
$$

where $l^r_t, l^v_t, l^p_t$ denote reward, value, and policy losses, respectively. MuZero achieved state-of-the-art results on Atari, Go, chess, and shogi, matching or surpassing AlphaZero's performance without direct access to the environment's rules.

Advantages of deep model-based methods include high sample efficiency by leveraging learned models for synthetic experience, enhanced planning ability and interpretability via internal simulation, and feasibility in real-world robotics and resource-constrained settings. Challenges include model bias, where compounding errors in long rollouts can degrade policy quality; training instability due to non-stationary data and shared model-policy optimization; and high computational cost for large-scale latent dynamics models.

## 4. Reinforcement Learning for Large Language Models

When fine-tuning Large Language Models (LLMs) to align them with human preferences, instructions, or specialized tasks, one common paradigm is Reinforcement Learning from Human Feedback (RLHF). In that paradigm, an LLM is treated as a policy $\pi_\theta(y \mid x)$ (generating response $y$ given prompt $x$), and the optimization objective becomes:

$$
\max_\theta \mathbb{E}_{x \sim D_{\text{prompt}}, y \sim \pi_\theta(\cdot \mid x)} \left[ r(x, y) \right]
$$

where $r(x, y)$ is a learned or crafted reward that measures how good the response $y$ is for the prompt $x$.

### 4.1 Model Roles

To implement the RLHF pipeline effectively, several models are employed in distinct but interdependent roles. Each contributes to a part of the reward-driven learning loop, from generating responses to evaluating and optimizing them:

**Policy model:** The main LLM we wish to optimize (parameterized by $\theta$). It functions as the environment's actor, generating responses, and is fine-tuned via policy optimization techniques (e.g., PPO).

**Reference model:** A frozen or slowly-updated baseline version of the policy (or a supervised fine-tuned model) used to compute KL or likelihood penalties to ensure the optimized policy does not diverge too far from acceptable behaviors.

**Value model:** A model that estimates the expected return (value) of a given prompt-response pair or sequence, often used to compute advantage estimates in actor-critic style updates.

**Reward model:** A separate model trained (often via human preference data or comparisons) to map a prompt-response pair $(x, y)$ to a scalar reward $r(x, y)$. It encapsulates human or designer preferences and drives the optimization of the policy model.

In typical LLM fine-tuning pipelines, the flow is:

1. The policy model generates responses.
2. The reward model scores them.
3. The value model estimates future return or baseline.
4. A reference model imposes a divergence penalty or acts as a safe anchor.
5. Using a policy-optimization algorithm (e.g., Proximal Policy Optimization), the policy model is updated to increase rewards while constraining divergence from the reference.

### 4.2 Notation Mapping Between Classical RL and Language Modeling

To relate standard RL notation to language-modeling-based policy optimization, it is helpful to make the correspondence between states, actions, trajectories, rewards, returns, and related terms explicit. Making these mappings explicit clarifies how classical RL concepts translate directly into the autoregressive language-model setting used in modern RLHF systems.

Beyond this section, the $x, y$ notation, where $x$ denotes the prompt and $y$ denotes the completion, rollout, or trajectory, is used for consistency with common LLM policy optimization implementations.

In classical RL, rewards are defined over state-action pairs:

$$
r(s_t, a_t)
$$

In the language-modeling setting, these concepts map naturally:

A state corresponds to a prompt plus generated sequence prefix (i.e., partial completion):

$$
s_t \leftrightarrow (x, y_{1:t})
$$

An action corresponds to generating the next token:

$$
a_t \leftrightarrow y_{t+1}
$$

A full trajectory or episode in RL corresponds exactly to a completion or rollout (also called a response or sample) in language modeling. Concretely, a completion:

$$
y = y_{1:T}
$$

corresponds to a full RL trajectory:

$$
(s_1, a_1, s_2, a_2, \ldots, s_T, a_T)
$$

The reward assigned to an entire completion is written in prompt-completion form as:

$$
r(x, y)
$$

where $x$ is the prompt and $y = y_{1:T}$ is the full generated response. Concretely, the reward for a completion or trajectory can be expressed as the sum of step-wise rewards:

$$
r(x, y_{1:T}) \equiv \sum_{t=1}^{T} r(s_t, a_t)
$$

This single expression subsumes both reward formulations: for outcome-based (i.e., terminal-only) rewards, the sum reduces to a single terminal term; for process-based (i.e., step-wise) rewards, the sum includes intermediate step-wise contributions.

In classical RL, the return refers to the cumulative reward along a trajectory. When conditioned on the current state, it is more precisely the expected future reward:

$$
R(s_t) \equiv \mathbb{E} \left[ \sum_{k=t+1}^{T} r(s_k, a_k) \mid s_t \right]
$$

In the realm of policy optimization for LLMs, this corresponds to the expected cumulative reward of the remaining completion given the current prefix. The value model (critic) estimates the expected future reward over all possible continuations under the policy:

$$
V_\psi(s_t) = \mathbb{E}[R \mid s_t]
$$

Additional terminology mappings commonly used across the two domains include:

- Policy $\pi(a_t \mid s_t)$ $\leftrightarrow$ Language model $\pi(y_{t+1} \mid x, y_{1:t})$
- Action distribution $\leftrightarrow$ Next-token distribution over vocabulary
- Environment transition $s_{t+1} \sim P(\cdot \mid s_t, a_t)$ $\leftrightarrow$ Autoregressive update $s_{t+1} = (x, y_{1:t+1})$

In classical RL, policy optimization uses the advantage:

$$
\hat{A}_t = R(s_t) - V(s_t)
$$

In RLHF, this maps to comparing the realized reward of a completion against the value model's baseline for the prefix:

$$
\hat{A}_t = r(x, y_{1:T}) - V_\psi(x, y_{1:t})
$$

for outcome-based rewards, with analogous extensions for process-based rewards.

### 4.3 Policy Model

The policy model in an RLHF-style setup is the LLM that we treat as a policy $\pi_\theta(y \mid x)$, parameterized by $\theta$, which given an input prompt $x$ produces a response $y$. The policy model is the central actor in the RLHF pipeline: it generates responses to prompts and is updated to align with human preferences. It carries the full representational capacity of a large LLM architecture, is trained in multiple phases (pretraining, SFT, RLHF), and must be large enough to enable high-quality responses while still being trainable. Its design must support computing log-probabilities, KL divergences, and synergy with reward/value models.

The policy model is the agent that interacts with the "environment" by generating outputs (responses $y$) to prompts $x$. Its objective is to maximize a reward signal $r(x, y)$, subject to constraints or regularization (for example via KL-divergence to a reference policy). Formally, the objective can be written as:

$$
\max_\theta \mathbb{E}_{x \sim D_{\text{prompt}}, y \sim \pi_\theta(\cdot \mid x)} \left[ r(x, y) - \beta \operatorname{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \right]
$$

where:

- $r(x, y)$: reward signal from human preference or a learned reward model
- $\pi_{\text{ref}}$: reference (often supervised-finetuned) policy
- $\beta$: KL regularization coefficient balancing reward maximization and divergence from the reference policy

During training, the policy model generates responses, receives reward model scores or value-model feedback, and is updated (often via algorithms like Proximal Policy Optimization). The policy model thus evolves from a "supervised fine-tuned" base model into a behavior-aligned model. The policy model must balance helpfulness, accuracy, safety, and alignment (for example to human preferences) [3].

Architecturally, the policy model is typically a causal (autoregressive) transformer with large scale: dozens of layers, high hidden dimensionality, multi-head self-attention, positional embeddings, etc. It is initially pretrained on massive corpora of text, then often fine-tuned via Supervised Fine-Tuning (SFT) on instruction-response pairs. For RLHF, a further head or mechanism may be added or used for value/advantage estimation, but the core remains the transformer. Recent work sometimes uses parameter-efficient tuning (e.g., LoRA, adapters) to limit full-model updates during RL optimization. The architecture must support sampling from $\pi_\theta$, computing log-probabilities $\log \pi_\theta(y \mid x)$, and computing KL divergence between $\pi_\theta$ and $\pi_{\text{ref}}$.

Training data for the policy model includes:

- **Pretraining:** Large unlabeled text corpora (e.g., hundreds of billions to trillions of tokens).
- **Supervised Fine-Tuning (SFT):** Instruction-response pairs collected from humans or human-augmented data; e.g., prompts with "good" responses. Many alignment pipelines begin with this stage to provide a reasonable starting policy.
- **RL Finetuning:** The model generates responses to prompts; responses are scored (via reward model or human ranking). This prompt-response-reward dataset is used in the reinforcement signal. Because the distribution of responses changes as $\pi_\theta$ updates, continuing to sample from updated policy is important.
- **Replay / Off-Policy Data:** Some pipelines incorporate past responses and reward scores into replay buffers or datasets for stability and reuse.

The policy model used in RLHF pipelines tends to be large (tens of billions of parameters or more) to provide strong language understanding and generation capabilities. For example, many state-of-the-art systems use models in the 7B–70B parameter range or larger (100B+). During SFT and then RLHF, often only the base model (e.g., 20B–70B) is used, to manage compute cost and stability. For example, the InstructGPT series used the GPT-3 175B model for SFT, then RLHF [3].

### 4.4 Reference Model

The reference model (also sometimes called the anchor model) is a fixed or slowly updated copy of the policy model used as a baseline or constraint in RLHF and related policy optimization setups for LLMs. Its primary purpose is to ensure that the updated policy model remains linguistically coherent, safe, and semantically aligned with the pre-RL distribution, while still learning to maximize the new reward signal. Put simply, the reference model plays a crucial safety and stability role in RLHF. It anchors the optimization process by maintaining linguistic and factual consistency, ensuring that policy optimization leads to meaningful alignment rather than degenerate exploration.

The reference model $\pi_{\text{ref}}(y \mid x)$ acts as a stability regulator during the RL phase. It appears in the KL-divergence regularization term in the RL objective:

$$
J(\theta) = \mathbb{E}_{x, y \sim \pi_\theta} \left[ r(x, y) - \beta \operatorname{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \right]
$$

where $\pi_\theta$ is the policy model being optimized, and $\beta$ is a scaling factor. The KL term penalizes deviations from the reference model distribution, preventing mode collapse, reward hacking, or drift into incoherent or unfaithful responses.

Conceptually, the reference model anchors the optimization so that the policy model can explore higher-reward regions of response space but does not diverge too far from its pretrained linguistic and factual priors. In practice, the reference model helps maintain fluency, truthfulness, and diversity of outputs throughout training.

Architecturally, the reference model is identical to the policy model. It is often just a frozen copy of the supervised fine-tuned (SFT) model. The typical pipeline is:

1. Begin with a pretrained transformer (e.g., GPT-3, Llama, or PaLM).
2. Fine-tune it with instruction data to get the SFT model.
3. Clone the SFT model to get the reference model (frozen).
4. Train another copy as the policy model (trainable) with PPO or another RL optimizer, using the frozen reference for KL regularization.

Since it shares weights and architecture with the policy model, the reference model uses a causal decoder-only transformer, typically with the same number of layers, hidden dimensions, and parameters. The architectural identity ensures that token-wise probability distributions are directly comparable, allowing exact computation of:

$$
\operatorname{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) = \sum_y \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

Some implementations (e.g., Stiennon et al., 2020) experimented with slowly updating the reference model, but most production pipelines freeze it entirely.

The reference model is not trained during the RL stage. Instead, it is a snapshot of the model before RLHF fine-tuning. It is trained in the supervised fine-tuning (SFT) phase using instruction-following data such as prompt-response pairs written or rated by humans, and curated high-quality datasets covering Q&A, summarization, code generation, reasoning, and dialog. The SFT dataset is usually smaller and more human-curated than pretraining data, ranging from a few thousand to a few hundred thousand high-quality examples. By preserving this SFT policy, the reference model embodies the linguistic priors and alignment baseline learned from human demonstrations before introducing reinforcement signals.

The reference model must match the policy model in architecture and vocabulary to make KL computation meaningful. Therefore, it has the same parameter count as the policy model, commonly in the range of 7B–70B parameters for research-grade or open-source systems (e.g., Llama-2, Falcon, Mistral RLHF variants), and 175B–500B+ parameters for frontier models (e.g., GPT-3 or GPT-4 scale). Because the reference model is frozen, its storage and compute requirements are primarily for forward passes during KL evaluation rather than gradient updates. In distributed training pipelines [3], both the policy and reference models are sharded across GPUs but only the policy model receives gradient updates.

### 4.5 Reward Model

The reward model (RM) is one of the most crucial components in the RLHF pipeline. It provides the scalar feedback signal $r(x, y)$ that quantifies the quality of a model's response $y$ to a prompt $x$, translating human preferences into a form usable by RL algorithms. In modern LLM alignment, the reward model serves as the surrogate objective for human satisfaction, steering the policy model toward behaviors that humans find helpful, truthful, and safe. The reward model provides the human-aligned feedback mechanism that guides RL updates. It bridges subjective human judgment and quantitative optimization, serving as the anchor for policy alignment and safety in LLM fine-tuning.

The reward model approximates a latent human preference function. Given a prompt $x$ and a response $y$, the model outputs a scalar value $r(x, y)$ representing how much a human would prefer that response. Its primary role is to act as a critic that scores generated text, so that the policy model can be optimized to produce higher-reward responses. Formally, the goal is to learn a function $r_\phi(x, y) \approx \text{Expected human preference score}(x, y)$, parameterized by $\phi$.

The reward model is trained using human preference data collected as pairwise or ranked comparisons. The reward modeling methodology, from ranking-based supervision and cross-entropy loss to normalization, originates from Learning to Summarize from Human Feedback by Stiennon et al. (2020) and was directly integrated into Training Language Models to Follow Instructions with Human Feedback (InstructGPT) by Ouyang et al. (2022) [3], forming the standard foundation of modern RLHF systems.

In Stiennon et al. (2020), the RM was trained on a dataset of comparisons between two model outputs on the same input. They used a cross-entropy loss, with the comparisons as labels: the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler. This approach was later adopted and extended in Ouyang et al. (2022) [3], which used this same reward model formulation as the basis of their RLHF training pipeline, with the following changes to sample creation and batching strategy:

- In order to speed up comparison collection (i.e., reduce labeling effort per prompt by ranking multiple responses at once), each labeler is presented with anywhere between $K = 4$ and $K = 9$ responses to rank. Larger $K$ values mean more comparisons per labeling task and richer training signal. This ranking procedure automatically generates up to $\binom{K}{2}$ pairwise comparisons per prompt (since every possible pair of completions produces one "win/loss" label).
- However, because these $\binom{K}{2}$ comparisons are highly correlated within the same labeling task, simply shuffling and treating each pair as an independent datapoint led to severe overfitting: the reward model quickly memorized specific completions and degraded generalization.
- Specifically, if each of the $\binom{K}{2}$ comparisons is treated separately, then each completion may appear in $K - 1$ distinct pairs, resulting in multiple redundant gradient updates per sample. The authors observed that under this setup, the model overfit within a single epoch, and even reusing the data within an epoch caused further overfitting.
- To mitigate this, they changed the training approach to treat all $\binom{K}{2}$ comparisons for a given prompt as a single batch element. This ensures that every completion is processed only once per batch while still contributing to all necessary pairwise comparisons. This modification was both computationally efficient (requiring only one forward pass of the reward model per completion instead of $\binom{K}{2}$) and empirically superior, yielding much improved validation accuracy and log loss. This batching strategy adopted by Ouyang et al. (2022) [3] became a standard approach, forming a key component of modern RLHF reward model training.

Architecturally, the reward model is typically a model derived from the same family as the policy model, sharing the same backbone architecture but differing in its output head and training objective. While some early or alternative setups explored encoder-based reward models, the canonical and most widely adopted approach, used in InstructGPT [3] and all modern large-scale RLHF pipelines, is decoder-only, consistent with the architecture of the policy (causal language) model. Architecturally, it is identical to the policy model but with a scalar regression head added on top of the final hidden state.

In Ouyang et al. (2022) [3], the reward model is based on the GPT-3 architecture (specifically the 6B parameter variant), a decoder-only transformer trained autoregressively. Per the paper, the reward model is initialized from the supervised fine-tuned (SFT) model, with the language modeling head removed (which includes the final unembedding layer, i.e., the linear projection from hidden states to vocabulary logits, along with the subsequent softmax used for next-token prediction) and replaced by a new linear head that outputs a single real-valued reward for a given prompt-completion pair. This added linear layer is a regression head that maps the final hidden (i.e., output) representation to a scalar value representing the model's predicted human preference score. Put simply, the reward model is architecturally identical to the base language model, except for the replacement of the language modeling head (i.e., next-token prediction head) with a linear regression head applied to the final hidden state that outputs a scalar value.

Mathematically speaking, the final token's hidden representation $h_T$ (or an average over all tokens) is passed through a linear projection to output $r_\phi(x, y)$, a single scalar reward representing the model's predicted human preference for the given prompt-completion pair:

$$
r_\phi(x, y) = w^\top h_T + b
$$

where $w, b$ are learned parameters. The model therefore learns to encode text sequences and output a single continuous reward value, capturing human preference judgments.

In practice:

- The scalar head is lightweight (a single dense linear layer).
- The underlying transformer backbone (6B parameters in InstructGPT) is initialized from the SFT model, preserving its linguistic and contextual knowledge.
- During RM training, only the new scalar head and a subset of backbone parameters may be fine-tuned for efficiency and stability.

Several architectural variants are used for reward modeling, including LM Classifiers (language models fine-tuned as binary classifiers to score which response better aligns with human preferences), Value Networks (regression models that predict scalar ratings representing relative human preference), and Critique Generators (language models trained to generate evaluative critiques explaining which response is better and why, used in conjunction with instruction tuning).

The reward model is trained using human-ranked comparison data and assigns a scalar score to each complete model-generated response conditioned on a prompt. Human preferences are modeled implicitly with a preference-based loss, using a pairwise Bradley-Terry (logistic) choice model [18]. Given a prompt $p$ and two candidate responses $r_i$ and $r_j$, the probability that a human rater prefers $r_i$ over $r_j$ is defined as:

$$
P(r_i > r_j \mid p) = \sigma(r_\phi(p, r_i) - r_\phi(p, r_j)) = \frac{\exp(r_\phi(p, r_i))}{\exp(r_\phi(p, r_i)) + \exp(r_\phi(p, r_j))}
$$

Intuitively:

- If $r_\phi(x, y_1) \gg r_\phi(x, y_2)$, then $P(y_1 \succ y_2 \mid x) \to 1$.
- If the scores are equal, $r_\phi(x, y_1) = r_\phi(x, y_2)$, then $P(y_1 \succ y_2 \mid x) = 0.5$.
- Only the difference in rewards matters: $P(y_1 \succ y_2 \mid x) = \sigma(r_\phi(x, y_1) - r_\phi(x, y_2))$, so any additive shift $r_\phi(x, y) \mapsto r_\phi(x, y) + c$ leaves the preference probabilities unchanged.

The reward model is trained by maximizing the likelihood of observed human preferences using the corresponding pairwise loss:

$$
L(\phi) = -\log \sigma(r_\phi(p, r_i) - r_\phi(p, r_j))
$$

where $\sigma$ is the sigmoid function, $r_\phi$ is the reward model, $p$ is the prompt, $r_i$ is the human-preferred response and $r_j$ the human-dispreferred one. This formulation ensures that the reward model learns to assign higher scores to responses more preferred by humans. In other words, the reward model learns a latent scalar scoring function over prompt-response pairs such that differences in scores model human preference probabilities.

Ouyang et al. (2022) [3] offers a practical implementation of the Bradley-Terry formulation. The paper proposes training the reward model with a pairwise cross-entropy ranking loss: the sigmoid of the reward difference defines a Bernoulli probability that one completion (i.e., the winner $y_w$) is preferred over another (i.e., the loser $y_l$), and optimizing it with the cross-entropy (negative log-likelihood) of the observed one-hot preference label under this Bernoulli model:

$$
L_{\text{RM}}(\theta) = -\frac{1}{\binom{K}{2}} \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

where:

- $y_w$ is the preferred (or winning) completion,
- $y_l$ is the dispreferred (or losing) completion,
- $\sigma$ is the sigmoid function mapping the score difference to a probability of preference,
- $r_\phi(x, y)$ is the reward model's scalar output.

The difference $r_\phi(x, y_w) - r_\phi(x, y_l)$ represents the log odds that a human would prefer $y_w$ over $y_l$.

Normalization and practical notes:

- In practice, labelers rank $K$ responses for the same prompt. All induced pairwise comparisons are used jointly, but the reward model computes each response score only once per prompt, improving stability and preventing overfitting.
- Since only reward differences matter, the reward model's loss is invariant to additive shifts in the reward function. Accordingly, the reward model is normalized by adding a bias term so that labeler demonstrations have mean score 0 before PPO training begins. This normalization fixes the otherwise arbitrary offset of the reward function, ensuring that the scale and baseline of rewards remain consistent across updates and thereby stabilizing policy learning.
- A key implementation detail during RLHF with PPO is that the partial responses produce a single scalar for the entire prompt-response pair. Partial responses receive no terminal reward (i.e., practically, a reward of 0); a non-zero scalar reward is only applied once a complete response is generated. This encourages the policy to produce coherent, complete outputs rather than optimizing for high-reward prefixes.

The training data for reward models comes from human preference labeling:

1. A set of prompts $x$ is sampled (often from SFT datasets or model-generated prompts).
2. Multiple responses are generated by one or more models.
3. Human annotators rank or choose preferred responses based on helpfulness, accuracy, harmlessness, or style criteria.
4. The collected comparisons yield tuples $(x, y_w, y_l)$, forming the basis for pairwise training.

Datasets of this form can range from 50,000 to several million comparisons, depending on the scale of the deployment. For example, the InstructGPT reward model used approximately 30,000–40,000 labeled comparisons [3], while larger RLHF systems (e.g., Anthropic's Constitutional AI) use 100K–1M+ pairs [19]. Synthetic preference data (generated using smaller models or heuristics) is also increasingly used to supplement limited human data, as in Self-Instruct by Wang et al. (2022) [20].

The reward model is usually smaller than the policy model (especially since it doesn't have to incorporate the unembedding layers), since it only provides scalar evaluations and doesn't need to generate text. Common sizes range from 1B to 13B parameters for large-scale pipelines. For example, InstructGPT used reward models of 6B parameters, while the policy model was 175B [3]. Open-source Llama 2-Chat models used reward models of 7B–13B parameters. Compact reward models are often used to reduce the cost of reward evaluation during RLHF training (since thousands of responses must be scored per update). Some recent methods, such as Direct Preference Optimization (DPO) by Rafailov et al. (2023) [16], avoid training a separate reward model altogether, instead implicitizing it through log-probability ratios between the policy and reference models.

To prevent the fine-tuned model from overfitting or drifting too far from its pretrained distribution, KL divergence penalties are applied during RL. KL divergence measures the difference between the output distributions of the current policy and the original (pretrained) model. This constraint regularizes learning and ensures that the fine-tuned model does not deviate excessively, preserving safety and coherence. This KL penalty is crucial for maintaining a balance between alignment and generalization.

Reward models are evaluated on held-out preference sets using accuracy metrics: how often the model correctly predicts the human-preferred response. Typical accuracy benchmarks range between 65–80%, depending on domain and data quality. Regular retraining and drift monitoring are essential, since the distribution of policy outputs changes as the policy improves.

### 4.6 Value Model

While the reward model provides scalar feedback signals evaluating model behavior, either as step-wise rewards during generation or as a terminal reward after a response is completed, the value model (also known as the critic) estimates the expected cumulative future reward from a given state or sequence prefix. This enables advantage estimation, variance reduction, and stabilized policy updates, which are foundational to modern policy-gradient methods such as PPO.

In RLHF implementations such as InstructGPT [3], the reward model is first initialized from the supervised fine-tuned (SFT) policy by removing the language modeling head (which includes the final unembedding layer, i.e., the linear projection from hidden states to vocabulary logits, along with the softmax used for next-token prediction) and replacing it with a new linear head that outputs a single real-valued scalar reward for each prompt-completion pair. This reward model is then trained on human preference comparisons.

After the reward model has been trained, the value model (critic) is initialized from this trained reward model and used during PPO fine-tuning to compute advantages:

$$
\hat{A}_t = r_t - V(s_t)
$$

where $r_t$ is the realized scalar reward produced by the trained reward model, and $V(s_t)$ is the value model's prediction of the expected cumulative reward from state $s_t$. This follows the standard actor-critic formulation used in PPO, with the critic providing a learned baseline that reduces variance and stabilizes policy updates.

In the context of LLM alignment, the value model $V_\psi(x)$ or $V_\psi(x, y)$ predicts the expected return (i.e., cumulative reward) for a given prompt $x$ or prompt-response pair $(x, y)$. More precisely, it learns the expected cumulative reward under the policy's full continuation distribution rather than the reward of any single sampled completion. In other words, it approximates the probability-weighted average return over all trajectories that the current policy can generate from a given prompt. It serves the same theoretical role as the critic in an actor-critic framework, providing a learned baseline that allows the policy (actor) to improve using lower-variance gradient estimates.

The fundamental definition is:

$$
V_\psi(s) \approx \mathbb{E}_{a \sim \pi_\theta} \left[ R(s, a) \right]
$$

where $R(s, a)$ is the return or scalar reward obtained when the policy $\pi_\theta$ takes action $a$ in state $s$. In the language modeling context, the "state" corresponds to the prompt or prefix $x$, and the "action" corresponds to the generated token sequence $y$.

The value model enables several key operations:

**Advantage estimation:** Used to compute a baseline-corrected signal for PPO or similar algorithms:

$$
\hat{A}(x, y) = r(x, y) - V_\psi(x)
$$

or, at a token-wise level, via Temporal-Difference (TD) methods:

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \hat{A}_{t+1}
$$

with:

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

In single-step environments (i.e., bandit-style) settings like RLHF (prompt, response, single reward), this simplifies to $\hat{A}_t = r_t - V(s_t)$, as the episode length is one. The short-horizon "bandit" structure still benefits from the critic, since it reduces gradient variance and hence stabilizes training.

**Variance reduction:** By learning a baseline for expected reward, $V_\psi(s)$ removes state-dependent bias and allows the policy gradient to focus on relative action quality.

**Critic-driven generalization:** The critic can generalize expected reward patterns across prompts, enabling continual improvement even when human preference labels are unavailable.

These functions mirror classical actor-critic frameworks, such as those in Konda and Tsitsiklis (2000) [21], but are adapted to the autoregressive, token-level structure of language models.

Architecturally, the value model is typically a decoder-only transformer, sharing its structure with both the policy and reward models, but differing in its output head and training objective. In the implementation described in Ouyang et al. (2022) [3], all components, including the supervised fine-tuned (SFT) policy, the reward model, and the value model used for PPO fine-tuning, are based on the GPT-3 architecture (specifically the 6B parameter variant), a unidirectional transformer decoder trained on autoregressive language modeling. The value function is initialized from the reward model, which itself is trained starting from the SFT model with the final unembedding layer removed and replaced with a new linear head that outputs a scalar regression value for each prompt-completion pair. This added linear layer serves as a learned projection that maps the final hidden state representation of the sequence to a single scalar reward estimate. The same architectural modification is used for the value model, which learns to predict expected returns during RLHF.

Mathematically speaking, the final token's hidden representation $h_T$ (or an average over all tokens) is passed through a linear projection to output $r_\phi(x, y)$, a single scalar reward representing the model's predicted human preference for the given prompt-completion pair:

$$
r_\phi(x, y) = w^\top h_T + b
$$

where $w, b$ are learned parameters. Put simply, the value model outputs a single scalar estimate $V_\psi(x)$ (or per-token estimates $V_\psi(x_t)$), rather than next-token probabilities or pairwise reward scores.

Implementation details:

- The hidden representation of the final token (or the mean of hidden states) is passed through a linear projection layer to produce the scalar value prediction.
- Architecturally, it may share parameters with the policy model up to the final projection layer, forming a multi-head actor-critic structure.
- However, many implementations, including InstructGPT, initialize the value model separately from the reward model rather than sharing parameters with the policy, ensuring that the two objectives (policy improvement and value estimation) do not interfere.
- Some designs explicitly decouple the policy and value networks to prevent gradient interference between actor and critic signals, which helps maintain training stability.

The reason for a dedicated value model or value head is that PPO (and more generally, actor-critic methods) rely on $V(s)$ to compute advantages:

$$
\hat{A}(s, a) \approx r + \gamma V(s') - V(s)
$$

In the short-horizon (bandit-style) case common in RLHF, where each episode consists of a single scalar reward, the episode terminates immediately after the action, so there is no next state $s'$. By convention, the value of a terminal state is zero, i.e., $V(s') = 0$, yielding the advantage expression $\hat{A} = r - V(s)$.

Without a critic, the gradient estimator would have much higher variance, leading to instability. Additionally, LLMs define states as (prompt + partial generation) and actions as next tokens. Having a separate value head or model allows for stable and interpretable gradient flow through long token sequences.

The value model is trained via a Mean Squared Error (MSE) regression objective to predict the expected cumulative reward (i.e., expected return) associated with a prompt:

$$
L_V(\phi) = \mathbb{E}_{(x, y) \sim D} \left[ \left( V_\psi(x) - \hat{R}(x, y) \right)^2 \right]
$$

where:

- $V_\psi(x)$ is the value model's prediction for the prompt (or state) $x$, intended to approximate $\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}[R(\tau)]$, the expected cumulative reward over all possible trajectories (completions) induced by the policy $\pi_\theta$, weighted by their probabilities.
- $\hat{R}(x, y)$ is the observed (i.e., realized) return associated with a single sampled completion $y$. This return is typically produced by the reward model, which approximates human feedback in RLHF, and may be augmented with auxiliary terms such as per-token KL penalties that discourage excessive deviation from the supervised fine-tuned checkpoint used as the reference policy $\pi_{\text{ref}}$.

Although each individual training target $\hat{R}(x, y)$ corresponds to only one sampled trajectory, minimizing the MSE objective over many repeated samples causes $V_\psi(x)$ to converge, in expectation, to the expected return under the policy's full trajectory distribution. Concretely, across training, the same or similar prompts $x$ are encountered many times, and each time the policy stochastically generates a potentially different completion $y$ with its own realized return $\hat{R}(x, y)$. Averaged implicitly through gradient updates, these repeated Monte Carlo samples drive the value model toward the probability-weighted average of cumulative rewards across all trajectories the policy can generate from $x$.

In RLHF pipelines such as InstructGPT [3], the value model is initialized from the reward model and fine-tuned concurrently with the actor (policy) during PPO. This concurrent training ensures that the critic continuously adapts to the policy's evolving expected reward landscape, via repeated sampling from the policy-induced distribution over trajectories, rather than memorizing the rewards of individual rollouts.

The training data underlying this objective consist of tuples $(x, y, r(x, y))$:

- $x$: a prompt sampled from curated datasets or real user queries,
- $y$: a response generated by the current policy $\pi_\theta$, representing one sampled trajectory from the policy-induced distribution,
- $r(x, y)$: a scalar reward produced by the reward model (and, in practice, possibly adjusted by a KL penalty term relative to the SFT checkpoint as the reference policy $\pi_{\text{ref}}$).

Collectively, these tuples define the empirical distribution $D$ over which the value regression loss is computed. Through repeated sampling of prompts and stochastic completions over the course of training, this empirical process provides the coverage needed for the value model to approximate the expected cumulative reward over all policy-generated trajectories.

In RLHF for language models, the expected cumulative reward under a policy cannot be computed in closed form because the space of possible continuations (trajectories) is combinatorially large and defined implicitly by the model's autoregressive token distribution. Enumerating or analytically integrating over all trajectories is intractable. As a result, the expectation $\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}[R(\tau)]$ is approximated using Monte Carlo sampling: trajectories are sampled from the policy, their realized returns are observed, and learning proceeds by averaging information from these samples over time. This approach is standard in policy-gradient and actor-critic methods, where unbiased but noisy samples are preferred over infeasible exact expectations.

Thus, the regression target is not the expectation itself, but a single Monte Carlo sample of the return obtained from one trajectory generated by the policy. For a rollout starting from prompt $x$ (or state $s_t$), the realized return is defined as:

$$
R_t \equiv \sum_{k=t+1}^{T} r(s_k, a_k)
$$

where actions $a_k$ (tokens) are sampled autoregressively from the policy $\pi_\theta$, inducing a trajectory $\tau = (a_{t+1}, \ldots, a_T) \sim \pi_\theta(\cdot \mid s_t)$, and rewards $r(\cdot)$ are provided by the reward model (optionally including KL-based shaping terms). Because the trajectory itself is sampled, $R_t$ is a random variable drawn from the conditional distribution $R \mid x$. Different rollouts from the same prompt can yield different $R_t$ values depending on which tokens are sampled.

In the bandit-style RLHF setting used in InstructGPT [3], the episode terminates immediately after a single completion. As a result, the sum above typically collapses to a single terminal reward assigned to the full prompt-completion pair. The notation highlights that $R_t$ is a single Monte Carlo sample from the policy-induced trajectory distribution $\pi_\theta(\tau \mid s_t)$, not the expectation over that distribution.

Although the value model is trained on individual sampled returns, minimizing the MSE objective drives it toward the conditional expectation of the return under the policy. This follows from a standard result in regression theory:

$$
\arg\min_f \mathbb{E} \left[ (f(x) - R)^2 \mid x \right] = \mathbb{E}[R \mid x]
$$

Crucially, this convergence relies on repeated Monte Carlo sampling over time, not on averaging multiple samples within a single update:

- Although each PPO update uses only one sampled return per prompt, the value model ultimately learns from many such samples across training iterations.
- The same prompt (or closely related prompts) may be encountered repeatedly during training.
- Each time, the policy stochastically generates a different completion according to its token-level probability distribution.
- Each completion yields a different realized return $R_t$, corresponding to a different sampled trajectory.

Across training, this process produces a growing collection of Monte Carlo samples:

$$
R(\tau) : \tau \sim \pi_\theta(\cdot \mid x)
$$

which empirically approximates the full return distribution induced by the policy for that prompt. As gradient updates accumulate, these sampled returns are implicitly averaged through optimization. If $V_\psi(x)$ overestimates the true expected return, subsequent samples will, on average, push it downward; if it underestimates, they will push it upward. Only the conditional mean $\mathbb{E}[R \mid x]$ remains stable under these stochastic updates.

As a result, even though each update uses only one sampled completion, repeated regression over many rollouts causes the value model to converge to $V_\psi(x) \approx \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[R(x, y)]$. Put simply, even though each regression target reflects the reward of a single trajectory, minimizing MSE over many repeated samples causes the value model to converge to the probability-weighted average cumulative reward over all trajectories the policy can generate from $x$, rather than overfitting to any individual rollout.

This mechanism is identical in principle to Monte Carlo value estimation in classical actor-critic methods, as formalized in work such as Actor-Critic Algorithms by Konda and Tsitsiklis (2000) [21], and is what allows the critic to represent an expectation despite being trained on noisy, single-sample returns.

In practice, RLHF systems do not enumerate all continuations under the policy. Instead, the expectation defining the value function is approximated empirically through repeated sampling from the policy-induced distribution over trajectories. The flow is as follows:

1. Each PPO update samples a prompt $x$ from the training distribution.
2. The policy $\pi_\theta$ generates one stochastic completion $y \sim \pi_\theta(\cdot \mid x)$.
3. The reward model assigns a scalar reward $r(x, y)$ (optionally shaped with per-token KL penalties), producing a single Monte Carlo target $R_t$.

In the InstructGPT [3] PPO setup, training is framed as a bandit environment: at each PPO step, the environment presents a randomly sampled customer prompt, the policy generates a single completion, a scalar reward is produced by the reward model (minus the per-token KL penalties), and the episode immediately terminates. As a result, one rollout per prompt per PPO update is used.

Rather than averaging multiple rollouts for the same prompt within a single batch update, the expectation $\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x)}[R(\tau)]$ over the policy's continuation distribution is approximated implicitly through repeated sampling across many PPO iterations via repeated stochastic rollouts. Over the course of training, as prompts are revisited, the policy samples different trajectories, and each trajectory contributes one noisy but unbiased Monte Carlo return sample to the learned expectation.

The empirical return distribution observed by the value model is therefore shaped by the sampling procedure used during rollout generation:

- Decoding strategy (greedy, top-$k$, top-$p$) determines which regions of the policy distribution are explored.
- Temperature scaling controls the entropy of token sampling, trading off exploration and variance.

These choices affect which trajectories are sampled frequently and thus which parts of the policy's trajectory distribution dominate the empirical expectation learned by the value model. Consequently, the value model can be understood as approximating the expected cumulative reward under the empirical policy distribution induced by the rollout procedure.

The value model trained under this objective is used directly in PPO-style advantage estimation:

$$
\hat{A}_t = r_t - V_\psi(s_t)
$$

which is the bandit-style simplification used in RLHF, since prompt, completion, reward forms a single-step episode. This follows the actor-critic formulation introduced in Proximal Policy Optimization Algorithms by Schulman et al. (2017) [9]. By learning $V_\psi$ as an approximation to the expected return, the critic provides a low-variance baseline, stabilizing policy gradients and enabling efficient learning even with sparse, high-variance reward signals from human or AI feedback.

Several extensions are commonly layered on top of the basic MSE objective to improve stability, generalization, and alignment fidelity:

**Per-token value heads** (also called token-level PPO), which estimate $V_\psi(s_t)$ at intermediate prefixes rather than only at the full completion. This supports finer-grained credit assignment across long generations and better aligns the critic with the autoregressive structure of language models.

**Target value networks** $V_{\phi^-}$, updated slowly or periodically, to stabilize bootstrapped targets in TD variants of value learning, following standard deep RL practice.

**Token-wise KL-regularized rewards**, where the effective return includes a per-token KL penalty that constrains the policy from drifting too far from a reference distribution, typically the SFT model. Concretely, at each token step the reward is adjusted by a term proportional to $-\beta \operatorname{KL}(\pi_\theta(\cdot \mid s_t) \| \pi_{\text{SFT}}(\cdot \mid s_t))$ and these penalties are accumulated across the entire completion. This token-level KL control is critical in RLHF, as unconstrained optimization against a learned reward model can otherwise lead to distributional collapse, degraded fluency, or reward hacking. This design follows the PPO-based RLHF formulation used in InstructGPT [3] and earlier work such as Learning to Summarize from Human Feedback by Stiennon et al. (2020).

**PPO-ptx (pretraining-mixed PPO)**, which explicitly mixes PPO reinforcement learning updates with gradients from the original unsupervised language modeling (pretraining) objective. As described in InstructGPT [3], this hybrid objective augments the PPO reward maximization term with an additional log-likelihood term over the pretraining distribution, thereby interleaving reinforcement signals with pretraining signals during optimization. This design ensures that while the model learns from reward feedback, it retains broad linguistic competence and factual grounding acquired during pretraining. By regularizing the optimization trajectory, PPO-ptx mitigates regression on standard NLP benchmarks ("alignment tax"), reduces catastrophic forgetting of general capabilities, and empirically improves both instruction-following behavior and robustness across unseen prompts, striking a balance between alignment and general language proficiency.

Together, these stabilizers ensure that the value model and policy do not simply overfit noisy reward signals from individual rollouts, but instead converge toward a smooth, generalizable estimate of expected cumulative reward under a KL-regularized policy, consistent with modern actor-critic theory and scalable RLHF practice.

The primary source of training data for the value model is on-policy samples collected during RLHF fine-tuning, typically generated from curated instruction datasets. Reward signals are derived from the reward model or human-preference comparisons. The scale is typically hundreds of thousands to millions of prompt-response pairs per training loop. Since LLM reward modeling is usually bandit-like (single scalar reward per completion), the value model relies on Monte Carlo estimates or Generalized Advantage Estimation (GAE) to stabilize learning despite sparse supervision.

The value model's size is often comparable to the reward model and smaller than the policy model. For instance, in InstructGPT, the critic had a similar scale (~6B parameters) to the reward model and served as a critic for a 175B parameter policy [3]. In open-source frameworks such as TRLX or DeepSpeed-Chat, value heads are attached to 7B–13B base LLMs or trained as independent critics. When computational resources are limited, a shared-head architecture is used, where the value head is attached directly to the policy model's hidden states, enabling efficient joint training of actor and critic.

The value model and the reward model are tightly coupled but conceptually distinct components of the RLHF pipeline, each answering a different question about model behavior. They coincide only in degenerate or limiting cases, because they are defined over fundamentally different quantities.

The reward model defines what is being optimized: it maps a completed response (or, in some setups, partial responses) to a scalar reward reflecting human preference. Formally, the reward model defines a reward function:

$$
r_\phi(x, y) \in \mathbb{R}
$$

where $x$ is the prompt and $y$ is the generated completion. More precisely, the reward model produces an observed (i.e., realized) scalar reward for a specific trajectory $r_\phi(x, y_{1:T})$, which is only known once the full response $y_{1:T}$ has been generated and evaluated. This reward can be outcome-based, where $r_\phi(x, y_{1:T})$ is produced only after the full completion, or process-based, where rewards are emitted at intermediate generation steps and summed over time to form $r_\phi(x, y_{1:T})$.

The value model (critic) does not define preferences; instead, it models expectations. Given a state $s_t = (x, y_{1:t})$, corresponding to a prompt plus a partial completion (prefix), the value model predicts the expected cumulative future reward under the current policy $\pi_\theta$:

$$
V_\psi(s_t) = \mathbb{E}_{y_{t+1:T} \sim \pi_\theta(\cdot \mid s_t)} \left[ R(s_t) \right]
$$

Unlike the reward model, this quantity is defined before the future tokens are generated and therefore averages over all possible stochastic continuations of the policy. Because one quantity is a realized outcome and the other is an expectation over possible futures, they are generally not equal:

$$
V_\psi(s_t) \neq r_\phi(x, y_{1:T})
$$

What the return $R(s_t)$ represents is entirely determined by how the reward model is defined:

For outcome-based (i.e., terminal-only) rewards, the return is a single terminal reward:

$$
R(s_t) = r_\phi(x, y_{1:T})
$$

At intermediate prefixes $t < T$, the value model averages this terminal reward over all possible completions:

$$
V_\psi(s_t) = \mathbb{E}_{y_{t+1:T} \sim \pi_\theta} \left[ r_\phi(x, y_{1:T}) \right]
$$

Only at the terminal state $s_T$, when the completion is fixed and no future randomness remains, does the expectation collapse:

$$
V_\psi(s_T) = r_\phi(x, y_{1:T})
$$

For process-based (i.e., step-wise) rewards, the return is the sum of future step-wise rewards:

$$
R(s_t) = \sum_{k=t+1}^{T} r_\phi(s_k, a_k)
$$

and the value model estimates the expected remaining cumulative reward:

$$
V_\psi(s_t) = \mathbb{E}_{\pi_\theta} \left[ \sum_{k=t+1}^{T} r_\phi(s_k, a_k) \right]
$$

Equality with realized rewards again holds only at the final step, when no future rewards remain and the expectation becomes trivial.

Crucially, outcome-based vs. process-based is a property of the reward definition, not of the value model. The distinction concerns when and how the reward model assigns supervision; regardless of how rewards are produced, the value model estimates the expected cumulative future rewards induced by that reward definition:

$$
V_\psi(s) \approx \mathbb{E}[R \mid s]
$$

Although the value model is often initialized from the reward model (as in InstructGPT), the two diverge during training because they optimize different objectives over different data. The reward model is trained offline on human preference data using a pairwise ranking loss, for example:

$$
L_{\text{RM}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) \right]
$$

The value model is trained online using a MSE regression objective, minimizing the squared difference between its predicted value $V_\psi(s_t)$ and a target return $\hat{R}_t$ obtained from policy rollouts:

$$
L_V(\phi) = \mathbb{E}_{(s_t, \hat{R}_t)} \left[ \left( V_\psi(s_t) - \hat{R}_t \right)^2 \right]
$$

In outcome-based (terminal-only) RLHF, the full return is available once the episode ends, so bootstrapping is not strictly required and $\hat{R}_t$ can be set to the realized terminal reward. In process-based or token-level RLHF, rewards may be distributed across multiple steps or the episode may be truncated (e.g., due to fixed context windows, maximum generation length, or early stopping criteria), so the remaining future return is unknown at intermediate states. In these cases, the target $\hat{R}_t$ is formed by bootstrapping from the value model itself, for example $\hat{R}_t = r_t + \gamma V_\psi(s_{t+1})$, or via multi-step or GAE-style targets to balance bias and variance.

As a result, the reward model outputs an observed (i.e., realized) reward $r_\phi(x, y)$ for a specific completion, while the value model outputs a baseline prediction used for variance reduction in advantage estimation, thereby improving training stability. This relationship is formalized through the advantage function, where the value model's baseline is subtracted from the realized return to obtain the advantage signal used for training, which measures how much better or worse the outcome was than expected:

$$
\hat{A}_t = r_t - V_\psi(s_t)
$$

In general, the reward model outputs a realized reward for a specific completion, while the value model outputs an expected return over possible future continuations. They coincide only in degenerate cases where this expectation collapses to a single realized outcome, which has direct implications for policy optimization in LLMs. Specifically, this refers not to their architectures or training objectives, but to the numerical quantity they output for a given state or trajectory.

Concretely, the reward model and value model coincide only in the following degenerate cases:

**Terminal states $s_T$**, where the full completion $y_{1:T}$ is already fixed and no future actions remain. In this case, there is no stochasticity left in the continuation, so the expected return equals the realized reward $V^\pi(s_T) = r_\phi(x, y_{1:T})$. For LLM policy optimization, this means that at the end of generation the critic provides no additional predictive information beyond the reward itself. The advantage used by PPO, $\hat{A}_T = r_\phi(x, y_{1:T}) - V^\pi(s_T)$, becomes zero in expectation, reflecting that no further learning signal can be extracted once the outcome is fully known.

**Single-step (bandit) environments**, where the episode consists of a single action followed immediately by a reward. This closely matches the standard RLHF setup for LLMs with outcome-based rewards, where the policy generates a full completion in one rollout and then receives a single scalar reward. In this case, there is no multi-step future to reason about, and the value function trivially equals the expected reward of that completion:

$$
V^\pi(s_0) = \mathbb{E}[R \mid s_0] = \mathbb{E}_{y \sim \pi(\cdot \mid x)} \left[ r_\phi(x, y) \right]
$$

For PPO-style optimization, the critic's role reduces to estimating the average reward for a prompt, providing a baseline so that advantages $\hat{A} = r_\phi(x, y) - V^\pi(s_0)$ measure whether a sampled completion is better or worse than what the policy typically produces. Even though the environment is bandit-like, this baseline is crucial for reducing gradient variance and stabilizing updates.

Outside these cases, the distinction is essential for LLM policy optimization. At intermediate prefixes $s_t$ with $t < T$, the value model must average over many possible continuations, while the reward model evaluates only the realized completion:

$$
V^\pi(s_t) = \mathbb{E}_{y_{t+1:T} \sim \pi} \left[ r_\phi(x, y_{1:T}) \right] \neq r_\phi(x, y_{1:T})
$$

This separation allows PPO to assign credit across tokens by comparing realized rewards to expected returns conditioned on the current prefix.

In summary, the reward model defines the reward function $r_\phi(\cdot)$, which answers what counts as good for a completed behavior, while the value model approximates the value function $V^\pi(s) = \mathbb{E}[R \mid s]$, which answers how good the future is expected to be, given the current prefix and the policy's stochastic continuation behavior. In LLM policy optimization, which consists of bandit-style regimes typical of RLHF, this distinction underpins advantage estimation, variance reduction, and stable gradient-based updates.

### 4.7 Optimizing the Policy

The policy refers to a strategy or a set of rules that an agent uses to make decisions in an environment. Put simply, the policy defines how the agent selects actions based on its current observations or state. The policy optimization process involves RL techniques that iteratively refine the policy based on reward feedback. The reward model provides feedback based on human preferences, and the policy is optimized iteratively to maximize reward while maintaining a stable learning trajectory. The stability aspect is enforced by maintaining a certain level of similarity to its previous version, to prevent drastic changes that could lead to instability.

Popular policy optimization methods, specifically applied to LLMs, include:

**Proximal Policy Optimization (PPO):** A widely-used RL algorithm that balances exploration and exploitation while maintaining training stability [9].

**Direct Preference Optimization (DPO):** An alternative approach where the policy directly optimizes the relative log probability of preferred responses using a binary cross-entropy loss, balancing human feedback alignment with KL divergence constraints [16].

**Group Relative Policy Optimization (GRPO):** A PPO variant that removes the critic model and estimates the baseline from group scores, improving memory efficiency and performance in complex tasks like mathematical reasoning [17].

Through RLHF, models like InstructGPT and ChatGPT have achieved enhanced alignment with human expectations, producing more beneficial and contextually appropriate responses.

### 4.8 Integration of Policy, Reference, Reward, and Value Models in RLHF

The full RLHF pipeline integrates four central components (the policy, reference, reward, and value models) into a cohesive optimization framework. Together, these models implement a scalable variant of policy-gradient RL (commonly using PPO) for large-scale language model alignment. This section provides a complete description of how these models interact, the mathematical formulation governing their updates, and the system-level architecture of a modern RLHF pipeline.

RLHF transforms large pretrained language models into alignment-optimized conversational agents through a three-phase process:

1. **Supervised Fine-Tuning (SFT):** The base pretrained LLM is fine-tuned on instruction-response data curated by humans. Output: SFT model (used as both the initial policy and the frozen reference model).

2. **Reward Modeling:** Human annotators rank or compare pairs of model responses. A separate reward model is trained on these comparisons to learn a scalar preference function $r_\phi(x, y)$.

3. **Reinforcement Learning (RL) Optimization:** The policy model is optimized to generate responses that maximize the learned reward signal, while staying close to the reference model through KL regularization. The value model acts as a critic, stabilizing the gradient updates.

This procedure was first described comprehensively in Training Language Models to Follow Instructions with Human Feedback by Ouyang et al. (2022) [3], forming the backbone of systems such as InstructGPT and ChatGPT.

The RLHF optimization problem can be expressed as:

$$
\max_\theta \mathbb{E}_{x \sim D_{\text{prompt}}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta \operatorname{KL}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) \right]
$$

where:

- $\pi_\theta$ = policy model (trainable)
- $\pi_{\text{ref}}$ = reference model (frozen)
- $r_\phi$ = reward model (provides scalar reward)
- $\beta$ = KL penalty coefficient controlling exploration-alignment trade-off

The KL term prevents the policy from diverging too far from its linguistic prior, while the reward encourages behaviors that better match human preferences.

To train this objective, Proximal Policy Optimization (PPO) [9] is typically used, which optimizes a clipped surrogate loss:

$$
L^{\text{PPO}}(\theta) = \mathbb{E}_{(x, y) \sim \pi_\theta} \left[ \min\left( r_t(\theta) \hat{A}_t, \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where:

- $r_t(\theta) = \frac{\pi_\theta(y_t \mid x_t)}{\pi_{\theta_{\text{old}}}(y_t \mid x_t)}$ is the likelihood ratio
- $\hat{A}_t = r_\phi(x_t, y_t) - V_\psi(x_t)$ is the advantage estimate
- $V_\psi$ = value model
- $\epsilon$ is a clipping hyperparameter (usually 0.1–0.2)

The advantage term ensures that updates are proportional to how much better a response is than expected, while the clipping stabilizes the step size.

The role of each model in the loop:

**Policy Model $\pi_\theta$:** Generates responses $y$ to prompts $x$. Updated via PPO to maximize the clipped surrogate objective. Receives both reward signals and value-based baselines during training.

**Reference Model $\pi_{\text{ref}}$:** Provides a baseline distribution for KL regularization to prevent over-optimization. Frozen during training; used to compute token-wise divergence:

$$
D_{\text{KL}}(\pi_\theta(\cdot \mid x) \| \pi_{\text{ref}}(\cdot \mid x)) = \sum_y \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

Ensures linguistic stability and mitigates reward hacking by anchoring the policy to its supervised fine-tuned prior.

**Reward Model $r_\phi$:** Maps each generated response $y$ (conditioned on prompt $x$) to a scalar reward: $r_\phi : (x, y) \mapsto \mathbb{R}$. Trained on human preference data (pairwise or ranked comparisons), then frozen during policy optimization. Supplies an approximation of human judgment, encouraging the policy to produce more aligned, preferred responses.

**Value Model $V_\psi$:** Estimates the expected return for a given prompt (or state) $x$, reducing variance in policy-gradient updates. Trained in parallel with the policy to predict the observed or bootstrapped return:

$$
\hat{R}(x, y) = r_\phi(x, y)
$$

and provides advantage estimates:

$$
\hat{A}(x, y) = r_\phi(x, y) - V_\psi(x)
$$

Serves as a critic in the actor-critic framework, enabling stable and efficient optimization.

The full training loop proceeds as follows:

**Step 1: Sampling Responses:** Draw a batch of prompts $x_i$ from the dataset. Generate responses $y_i$ from the current policy $\pi_\theta$.

**Step 2: Reward Evaluation:** Compute scalar rewards $r_\phi(x_i, y_i)$ using the reward model. Compute KL penalties from the reference model.

**Step 3: Advantage Computation:** Use the value model to estimate baselines $V_\psi(x_i)$. Compute advantages $\hat{A}_i = r_\phi(x_i, y_i) - V_\psi(x_i)$.

**Step 4: Policy Update (PPO):** Optimize $L^{\text{PPO}}(\theta)$ with respect to the policy parameters. Clip ratios and advantages to maintain stable updates.

**Step 5: Value Model Update:** Update the critic via regression: $L_V(\psi) = \mathbb{E}_{(x, y)}[(V_\psi(x) - r_\phi(x, y))^2]$.

**Step 6: Iteration and Rollout:** Repeat with new samples from the updated policy. Periodically evaluate human or synthetic preference metrics to ensure alignment progress.

Computational and practical considerations:

- **Training Scale:** The RLHF fine-tuning phase typically uses hundreds of thousands to millions of samples, requiring large-scale distributed training. Compute cost is dominated by sampling (policy forward passes) and reward scoring.
- **Stability:** PPO's clipping and KL regularization stabilize updates that would otherwise explode in such large parameter spaces.
- **Safety and Alignment:** The reward model embeds alignment objectives (helpfulness, harmlessness, honesty). KL regularization ensures fidelity to the pretrained model's linguistic priors.
- **Continuous Improvement:** Iterative retraining of reward models using newer policy outputs yields increasingly aligned systems, a process sometimes called iterative RLHF or alignment bootstrapping [22].

In summary, RLHF operationalizes RL at massive scale by combining:

- The policy for exploration and response generation,
- The reward for human alignment,
- The value for stability and variance control, and
- The reference for constraint and safety.

This synergy enables LLMs to internalize nuanced human feedback, forming the foundation for systems like ChatGPT, Anthropic's Claude, and Google's Gemini.

## 5. DPO: Direct Preference Optimization

RLHF with PPO is complex: it requires training a reward model, sampling from the policy during training, and tuning PPO hyperparameters. Direct Preference Optimization (DPO) [16] eliminates the reward model and the online RL loop, optimizing preferences directly.

Starting from the optimal policy:

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x, y)/\beta)}{Z(x)}
$$

Rearrange to express the reward:

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

Substitute this into the Bradley-Terry model for preferences:

$$
P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))
$$

$$
= \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \beta \log Z(x) \right)
$$

The prompt-dependent term $\beta \log Z(x)$ cancels:

$$
= \sigma\left( \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)
$$

Replacing the unknown $\pi^*$ with the parameterized policy $\pi_\theta$ and minimizing the negative log-likelihood of preferences yields the DPO loss:

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

Assumptions and limitations: DPO assumes the preference data follows the Bradley-Terry model and that the reference policy is a good approximation of the optimal policy's support. The implicit reward $r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is only defined up to a prompt-dependent constant. DPO is an offline method: it cannot explore beyond the preference data. It can be prone to over-optimization when the reference policy is weak or the data is noisy. Length and tokenization effects can introduce spurious correlations: longer responses may be preferred for reasons unrelated to quality, and DPO's implicit reward can exploit this.

Comparison of RLHF (PPO) and DPO:

| Dimension | RLHF (PPO) | DPO |
|-----------|-----------|-----|
| Reward model | Learned explicitly | Implicit via policy ratio |
| Sampling | Online (from current policy) | Offline (fixed preference data) |
| Reference policy | Used for KL penalty | Used for implicit reward baseline |
| Computational cost | High (reward model + PPO) | Moderate (single training stage) |
| Stability | Sensitive to hyperparameters | More stable in practice |
| Exploration | Yes, through sampling | No |

DPO should not be described as "the same as PPO without a critic." It is a different objective derived from a different starting point: the closed-form solution to a KL-regularized reward maximization problem. It optimizes an implicit reward, not an explicit one, and it does so offline.

## 6. RLVR: Reinforcement Learning with Verifiable Rewards

Reinforcement Learning with Verifiable Rewards (RLVR) replaces learned reward models with deterministic or programmatic verification. The reward is computed by a verifier:

$$
R(x, y) = \operatorname{Verify}(x, y)
$$

For mathematical problems, the verifier may check whether the final answer matches a ground-truth answer. For code, it may run unit tests. For formal reasoning, it may check proof validity.

The distinction from RLHF is the source of the reward signal: subjective human preferences versus objective verification. Verifiable rewards are low-noise and difficult to hack, provided the verifier is correct. But they are also narrow: domains without clear correctness criteria (open-ended writing, dialogue, ethical judgment) cannot be verified programmatically.

For a binary verifier, $R(x, y) \in \{0, 1\}$. The expected reward objective is:

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} [R(x, y)]
$$

The policy gradient is:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ R(x, y) \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t}) \right]
$$

This is the same score-function estimator as in classical RL, with the sequence-level reward substituting for the trajectory return.

The verifier may check the final answer only, or it may verify intermediate steps. **Outcome supervision** provides a single reward at the end. **Process supervision** provides rewards for each reasoning step [23]. Process supervision can improve credit assignment but requires step-level labels, which are expensive to obtain.

Challenges of RLVR include:

**Sparse rewards.** Binary rewards provide no gradient signal on problems the policy always fails or always solves. Group-based methods (GRPO) address this by comparing responses within a group, but if all responses in a group receive the same reward, the relative advantage is zero.

**Exploration.** Verifiable rewards require the policy to occasionally produce correct solutions. If the policy's pass@k on a problem is zero for all $k$ up to the group size, no learning signal is available. Curriculum learning and hint-based methods attempt to address this.

**Multiple valid solutions.** A verifier that checks only the final answer may accept incorrect reasoning that happens to reach the right result, or reject correct reasoning with a non-canonical format. Verifier design is a substantive challenge.

**Reward hacking.** Even deterministic verifiers can be exploited. A policy may learn to produce outputs that pass the verifier without genuinely solving the problem, for example by memorizing answer patterns or exploiting formatting conventions. This is a form of specification gaming.

**Generalization.** Improvements on verifiable tasks do not automatically transfer to non-verifiable domains. A model trained on mathematics with RLVR may not become better at writing, reasoning about ethics, or engaging in open-ended dialogue.

## 7. GRPO and Group-Based Policy Optimization

Group Relative Policy Optimization (GRPO), introduced in DeepSeekMath [17], is a critic-free alternative to PPO for language model optimization. Instead of training a value function to estimate advantages, GRPO samples multiple responses per prompt and computes relative advantages within the group.

The motivation is practical: value functions for long-horizon language generation are difficult to train accurately, and their approximation errors introduce bias into advantage estimates. Group-relative advantages avoid this by using the group itself as a baseline.

For a prompt $x$, sample $G$ responses $\{y_1, \ldots, y_G\}$ from $\pi_{\theta_{\text{old}}}$. Compute rewards $R_1, \ldots, R_G$. Define:

$$
\mu_R = \frac{1}{G} \sum_{i=1}^{G} R_i, \quad \sigma_R = \sqrt{\frac{1}{G} \sum_{i=1}^{G} (R_i - \mu_R)^2}
$$

The normalized advantage for response $i$ is:

$$
\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R + \varepsilon}
$$

where $\varepsilon$ is a small constant for numerical stability.

**Numerical example.** Suppose $G = 4$ responses with rewards $(1, 1, 0, 0)$. Then $\mu_R = 0.5$. The deviations are $(0.5, 0.5, -0.5, -0.5)$. The variance is $\frac{1}{4}(0.25 + 0.25 + 0.25 + 0.25) = 0.25$, so $\sigma_R = 0.5$. With $\varepsilon = 10^{-8}$, the normalized advantages are $(1, 1, -1, -1)$. The two correct responses receive positive advantages; the two incorrect responses receive negative advantages. If all responses receive the same reward (e.g., all 1 or all 0), $\sigma_R = 0$, and all advantages are zero (or undefined without $\varepsilon$). The group provides no relative signal, a key limitation of GRPO.

The token-level likelihood ratio is:

$$
r_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}
$$

The GRPO objective (schematic form) is:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t} \min\left( r_{i,t}(\theta) \hat{A}_i, \operatorname{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{\mathrm{KL}} \right]
$$

This is structurally similar to PPO, with two key differences: the advantage $\hat{A}_i$ is shared across all tokens in response $i$ (since rewards are sequence-level), and no critic is used. The KL term is typically an estimator of the divergence between $\pi_\theta$ and $\pi_{\text{ref}}$, computed per token. The normalization by $|y_i|$ (response length) is a design choice that affects the relative weighting of long and short responses. Different implementations may use different normalization schemes. Some variants remove length normalization entirely to avoid biasing toward longer outputs.

These methods should be classified along separate dimensions:

| Method | Algorithm | Reward Source | Online? | Critic? | Reference? |
|--------|-----------|---------------|---------|---------|------------|
| PPO (RLHF) | Clipped surrogate | Learned RM | Yes | Yes | Yes (KL) |
| DPO | Classification loss | Preferences | No | No | Yes (implicit) |
| GRPO | Clipped surrogate | Verifier or RM | Yes | No | Yes (KL) |
| RLVR | Reward paradigm | Verifier | Depends | Depends | Depends |

RLVR describes a reward paradigm, not an optimization algorithm. GRPO describes an optimization algorithm that can be used with verifiable or learned rewards. PPO describes an optimization algorithm that is typically used with learned rewards but can in principle use any reward signal. DPO describes an offline preference-optimization objective. A GRPO-style optimizer combined with verifiable rewards is a common configuration for reasoning models. A PPO-style optimizer with a learned reward model is the classical RLHF configuration. These are not mutually exclusive categories.

## 8. Additional Modern Methods and Connections

### 8.1 RLAIF

Reinforcement Learning from AI Feedback replaces human preference labels with AI-generated labels [19]. Constitutional AI uses a set of principles to guide the AI's critiques and revisions of its own outputs. RLAIF can scale preference data collection but inherits the limitations of the AI annotator: it may have systematic biases, and its judgments may not align with human preferences.

### 8.2 Process Reward Models and Outcome Reward Models

Outcome Reward Models (ORMs) assign a reward to the final answer. Process Reward Models (PRMs) assign rewards to intermediate reasoning steps [23]. Process supervision provides denser credit assignment and can improve reasoning, but requires step-level labels. The trade-off between label cost and credit assignment accuracy is an active area of research.

### 8.3 Rejection Sampling and Best-of-N

Best-of-N sampling generates $N$ responses and selects the one with the highest reward according to a verifier or reward model. This improves performance at inference time without updating the policy. It is complementary to training-time methods but does not improve the policy's underlying capability; it only selects better samples from the existing distribution.

### 8.4 Alternative Preference-Optimization Methods

**IPO (Identity Preference Optimization)** [24] derives a general objective from which DPO and RLHF are special cases, and proposes a variant that is more robust to over-optimization. **KTO (Kahneman-Tversky Optimization)** [25] uses prospect theory to define losses for desirable and undesirable outputs separately, removing the need for paired preferences. **ORPO (Odds Ratio Preference Optimization)** [26] combines SFT and preference alignment into a single objective, eliminating the need for a reference model.

### 8.5 Agentic Reinforcement Learning

RL can be applied to agents that use tools, execute multi-step actions, and interact with environments. The challenges include partial observability (the agent may not observe the full state), large and structured action spaces (tool calls, API invocations), long-horizon credit assignment, and safety constraints. Agentic RL is an active research frontier.

### 8.6 Embodied AI and Robot Learning

The same policy-gradient foundations apply to robot learning. Continuous action spaces require policies that output means and variances of Gaussian distributions, or more complex distributions. Sim-to-real transfer addresses the gap between simulated training and physical deployment. Imitation learning (behavior cloning) is often used to initialize policies, with RL fine-tuning for performance. Reward engineering, sample efficiency, and safety remain challenges.

## 9. A Unified Mathematical View

Across REINFORCE, PPO, RLHF, DPO, RLVR, and GRPO, several components are shared:

- **Policy parameterization:** All methods optimize a parameterized policy $\pi_\theta$.
- **Expected-return objective:** The underlying goal is to maximize expected reward.
- **Score-function gradient:** Policy gradient methods use $\nabla_\theta \log \pi_\theta$ to estimate gradients.
- **Baselines and advantages:** Variance reduction through baselines is universal.
- **Conservative updates:** PPO clipping and KL regularization serve similar purposes.

What differs is:

- **Reward source:** Human preferences, learned reward models, verifiers, or environment rewards.
- **Sampling regime:** Online (PPO, GRPO) vs. offline (DPO).
- **Gradient estimator:** Score-function vs. reparameterization vs. direct preference loss.
- **Critic:** Required in PPO, absent in GRPO and DPO.
- **Reference model:** Used in RLHF and GRPO for KL, implicit in DPO.

Not all methods are direct extensions of each other. DPO is derived from the closed-form solution to a KL-regularized problem, not from the policy gradient theorem. GRPO modifies PPO's advantage estimation but retains the clipped surrogate. RLVR is a reward paradigm that can be combined with various optimizers. The relationships are conceptual and mathematical, not hierarchical.

## 10. Practical and Scientific Limitations

**Reward hacking** occurs when the policy exploits flaws in the reward signal. Learned reward models are especially vulnerable: the policy may find adversarial examples that score highly without being genuinely good. Verifiers can be exploited if they are not robust.

**Reward-model overoptimization** is a related phenomenon: optimizing too hard against a learned reward model degrades true performance. The KL penalty mitigates this but does not eliminate it.

**Policy drift** occurs when the policy moves far from the reference distribution, reducing the reliability of reward estimates and potentially degrading capabilities. KL regularization constrains drift but trades off against reward maximization.

**Sparse and noisy rewards** make learning difficult. Binary verifiers provide no gradient signal on problems the policy cannot solve. Noisy human preferences introduce label noise.

**High variance** in policy-gradient estimates requires large batch sizes or variance-reduction techniques. Long horizons exacerbate this.

**Exploration failure** occurs when the policy never discovers high-reward behaviors. In language generation, this means the model never produces a correct solution to a hard problem.

**Data contamination** in reasoning benchmarks can inflate performance estimates. Test problems may appear in training data, making it unclear whether the model is reasoning or memorizing.

**Evaluation beyond benchmark scores** is essential. A model that improves on GSM8K may not be better at reasoning in general. Reward metrics can be gamed; human evaluation and adversarial testing are necessary complements.

These limitations are not merely engineering challenges. They reflect fundamental tensions in the RL formulation: the reward is a proxy for what we actually want, and optimizing a proxy too hard degrades true performance.

## 11. Reproducible Experiments and Implementation Intuition

**REINFORCE pseudocode:**

```
Input: policy pi_theta, environment, learning rate alpha
for episode = 1, 2, ... do
    Sample trajectory tau = (s_0, a_0, r_1, ..., s_T) using pi_theta
    Compute returns G_t for each t
    Update theta <- theta + alpha * sum_t G_t * grad_theta log pi_theta(a_t | s_t)
end for
```

**Actor-Critic with GAE:**

```
Input: policy pi_theta, value function V_phi, environment
for iteration = 1, 2, ... do
    Collect trajectories
    Compute TD residuals delta_t = r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t)
    Compute GAE advantages A_t = sum (gamma * lambda)^l * delta_{t+l}
    Update theta using A_t * grad_theta log pi_theta
    Update phi by minimizing (V_phi(s_t) - (A_t + V_phi(s_t)))^2
end for
```

**DPO training:**

```
Input: policy pi_theta, reference pi_ref, preference data (x, y_w, y_l), beta
for batch in data do
    Compute log ratio_w = log pi_theta(y_w|x) - log pi_ref(y_w|x)
    Compute log ratio_l = log pi_theta(y_l|x) - log pi_ref(y_l|x)
    Loss = -log sigmoid(beta * (log ratio_w - log ratio_l))
    Update theta by gradient descent on Loss
end for
```

**GRPO with group-relative rewards:**

```
Input: policy pi_theta, reference pi_ref, prompts, verifier, G, beta
for iteration = 1, 2, ... do
    for prompt x in batch do
        Sample G responses {y_i} from pi_theta_old
        Compute rewards R_i = Verify(x, y_i)
        Compute mu_R, sigma_R, and A_i = (R_i - mu_R) / (sigma_R + eps)
    end for
    for epoch = 1, ..., K do
        for each response do
            Compute ratio r_{i,t} for each token
            Compute clipped surrogate with A_i
            Compute KL with pi_ref
            Update theta by gradient descent on -J_GRPO
        end for
    end for
    Update theta_old = theta
end for
```

**Toy example.** Prompt: "What is 17 times 23?" Sample 4 responses:

- $y_1$: "391" (correct, $R_1 = 1$)
- $y_2$: "391" (correct, $R_2 = 1$)
- $y_3$: "381" (incorrect, $R_3 = 0$)
- $y_4$: "401" (incorrect, $R_4 = 0$)

$\mu_R = 0.5$, $\sigma_R = 0.5$, $\hat{A}_1 = \hat{A}_2 = 1$, $\hat{A}_3 = \hat{A}_4 = -1$. The policy is reinforced to produce "391" and suppressed from producing "381" and "401." If the model had produced "391" in some contexts and other correct forms in others, the verifier's normalization (e.g., stripping whitespace) would matter.

## 12. Open Research Questions

1. How can policy-gradient variance be reduced for long reasoning trajectories without introducing bias?
2. When does verifiable-reward training generalize beyond the verifier's domain?
3. How should process rewards be designed to balance credit assignment and label cost?
4. How can reward hacking be detected and mitigated in practice?
5. What is the relationship between exploration, inference-time search, and learning?
6. Can RL improve tool use and embodied interaction robustly?
7. How should researchers evaluate generalization rather than benchmark memorization?
8. What are the trade-offs between offline preference optimization and online RL?

These are active research questions. The field has made substantial progress, but the fundamental tension, that optimizing a proxy reward may degrade true performance, remains unresolved.

## 13. Conclusion

Policy optimization provides a mathematical language for connecting classical reinforcement learning to modern language model post-training. The score-function estimator, derived from the log-derivative identity, allows us to estimate gradients of expected return without knowing the environment's dynamics. Baselines and advantage functions reduce variance. Trust-region and clipped objectives constrain updates. KL regularization keeps policies close to a reference. Preference models and verifiers provide reward signals when the environment does not.

The methods surveyed here, including REINFORCE, PPO, RLHF, DPO, RLVR, and GRPO, are not a linear progression. They are different points in a design space defined by the reward source, the sampling regime, the gradient estimator, and the constraints on policy updates. Understanding their mathematical foundations clarifies both their power and their limitations. The same framework that lets a robot learn to grasp an object lets a language model learn to solve a differential equation. The same failure mode, reward hacking, appears in both domains. And the same open question, how to specify what we actually want, remains unanswered.

## 14. Glossary of Symbols and Concepts

| Symbol | Meaning |
|--------|---------|
| $\mathcal{M}$ | Markov Decision Process |
| $\mathcal{S}, \mathcal{A}$ | State and action spaces |
| $P(s' \mid s, a)$ | Transition probability |
| $R(s, a, s')$ | Reward function |
| $\gamma$ | Discount factor |
| $\pi_\theta(a \mid s)$ | Parameterized policy |
| $V^\pi(s), Q^\pi(s, a)$ | State-value and action-value functions |
| $A^\pi(s, a)$ | Advantage function |
| $G_t$ | Discounted return from time $t$ |
| $\tau$ | Trajectory |
| $r_t(\theta)$ | Likelihood ratio in PPO |
| $\beta$ | KL penalty coefficient |
| $\pi_{\text{ref}}$ | Reference policy |
| $r_\phi(x, y)$ | Reward model |
| $\hat{A}_i$ | Group-relative advantage in GRPO |

## 15. References

[1] Sutton, R. S., & Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press.

[2] Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. Advances in Neural Information Processing Systems, 12, 1057–1063.

[3] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems, 35.

[4] Rummery, G. A., & Niranjan, M. (1994). On-Line Q-Learning Using Connectionist Systems. Technical Report CUED/F-INFENG/TR 166, Cambridge University Engineering Department.

[5] Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. Machine Learning, 8, 279–292.

[6] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-Level Control Through Deep Reinforcement Learning. Nature, 518, 529–533.

[7] Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 8, 229–256.

[8] Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. IEEE Transactions on Systems, Man, and Cybernetics, 13(5), 834–846.

[9] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

[10] Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). Continuous Control with Deep Reinforcement Learning. arXiv:1509.02971.

[11] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. International Conference on Learning Representations.

[12] Kakade, S. (2001). A Natural Policy Gradient. Advances in Neural Information Processing Systems, 14.

[13] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust Region Policy Optimization. Proceedings of the 32nd International Conference on Machine Learning, 1889–1897.

[14] Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529, 484–489.

[15] Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the Game of Go Without Human Knowledge. Nature, 550, 354–359.

[16] Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Advances in Neural Information Processing Systems, 36.

[17] Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

[18] Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4), 324–345.

[19] Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.

[20] Wang, Y., Kordi, Y., Mishra, S., et al. (2022). Self-Instruct: Aligning Language Models with Self-Generated Instructions. arXiv:2212.10560.

[21] Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. Advances in Neural Information Processing Systems, 12, 1008–1014.

[22] Christiano, P. F., Leike, J., Brown, T. B., et al. (2017). Deep Reinforcement Learning from Human Preferences. Advances in Neural Information Processing Systems, 30.

[23] Lightman, H., Kosaraju, V., Burda, Y., et al. (2024). Let's Verify Step by Step. International Conference on Learning Representations.

[24] Azar, M. G., Guo, Z. D., Piot, B., et al. (2024). A General Theoretical Paradigm to Understand Learning from Human Preferences. International Conference on Artificial Intelligence and Statistics, 4447–4455.

[25] Ethayarajh, K., Xu, W., Muennighoff, N., et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. arXiv:2402.01306.

[26] Hong, J., Lee, N., & Thorne, J. (2024). ORPO: Monolithic Preference Optimization without Reference Model. Proceedings of EMNLP 2024, 11170–11189.
