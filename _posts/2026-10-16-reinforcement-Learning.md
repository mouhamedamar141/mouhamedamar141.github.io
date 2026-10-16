---
layout: post
title: "Reinforcement Learning: From Foundations to Modern Policy Optimization for Large Language Models"
date: 2026-10-16
description: "A comprehensive mathematical guide to reinforcement learning, from MDPs and the policy gradient theorem to REINFORCE, TRPO, PPO, RLHF, DPO, GRPO and RLVR for LLM post-training."
tags: [reinforcement-learning, policy-gradients, REINFORCE, TRPO, PPO, RLHF, DPO, GRPO, RLVR, LLMs]
categories: [Reinforcement Learning]
toc:
  sidebar: left
---

**Author:** Mouhamed Amar

## Abstract

Reinforcement learning (RL) has evolved from a niche technique for sequential decision-making into a central framework for aligning and fine-tuning large language models. This article gives a self-contained, derivation-first treatment of the mathematics that connects classical RL to modern language-model post-training.

We start from Markov decision processes and the Bellman equations, then spend most of the first half on **policy gradients**: the log-derivative trick, a full proof of the policy gradient theorem, REINFORCE and its failure modes, baselines, advantage functions, actor-critic methods and generalized advantage estimation. We then derive **importance sampling**, the performance difference lemma, **TRPO** (including its natural-gradient step) and **PPO** (including a case analysis of its clipped objective). In the second half, we formalize text generation as a token-level MDP, derive the **RLHF** pipeline, the KL-regularized optimal policy, **DPO**, **GRPO** and **RLVR**, and discuss the failure modes (reward hacking, over-optimization, verifier exploitation) that remain open research questions.

Every major objective is derived from its assumptions, and we distinguish mathematical guarantees from empirical regularities.

<p><b>How to navigate.</b> Every <a href="#sec-refs">citation</a> such as <a href="#ref-9">[9]</a> is a link: clicking it jumps to the matching entry in the <a href="#sec-refs">References</a>. Every entry of the table of contents below jumps to its section.</p>

## Table of Contents

- [1. Introduction](#sec-intro)
- [2. Mathematical foundations](#sec-foundations)
  - [2.1 The agent-environment loop](#sec-loop)
  - [2.2 Markov decision processes](#sec-mdp)
  - [2.3 Policies, returns, value functions and advantage](#sec-values)
  - [2.4 Bellman equations and contraction](#sec-bellman)
  - [2.5 Dynamic programming](#sec-dp)
  - [2.6 Learning from experience: MC, TD, SARSA, Q-learning](#sec-model-free)
  - [2.7 Exploration versus exploitation](#sec-exploration)
  - [2.8 On-policy vs. off-policy, model-based vs. model-free](#sec-onoff)
- [3. Deep value-based methods](#sec-deep-value)
- [4. Policy gradient methods](#sec-pg)
  - [4.1 Objective and policy parameterizations](#sec-pg-objective)
  - [4.2 The log-derivative (score-function) trick](#sec-score)
  - [4.3 The policy gradient theorem, with proofs](#sec-pgt)
  - [4.4 REINFORCE](#sec-reinforce)
  - [4.5 Problems of REINFORCE](#sec-reinforce-problems)
  - [4.6 Variance reduction: reward-to-go, baselines, advantages](#sec-variance)
  - [4.7 Actor-critic methods](#sec-actor-critic)
  - [4.8 Generalized advantage estimation (GAE)](#sec-gae)
  - [4.9 Natural policy gradient](#sec-npg)
- [5. Importance sampling, TRPO and PPO](#sec-trust)
  - [5.1 Importance sampling](#sec-is)
  - [5.2 The performance difference lemma and the surrogate objective](#sec-pdl)
  - [5.3 Trust Region Policy Optimization (TRPO)](#sec-trpo)
  - [5.4 Proximal Policy Optimization (PPO)](#sec-ppo)
- [6. Continuous control, model-based RL and embodied agents](#sec-continuous)
- [7. RL for large language models: RLHF](#sec-llm)
  - [7.1 Language generation as an MDP](#sec-llm-mdp)
  - [7.2 The four models: policy, reference, reward, value](#sec-roles)
  - [7.3 The reward model](#sec-rm)
  - [7.4 The value model and advantage estimation](#sec-vm)
  - [7.5 PPO for RLHF: the full loop](#sec-rlhf-loop)
- [8. DPO: Direct Preference Optimization](#sec-dpo)
- [9. RLVR and GRPO](#sec-rlvr-grpo)
- [10. A unified mathematical view](#sec-unified)
- [11. Practical and scientific limitations](#sec-limits)
- [12. Reproducible experiments and implementation intuition](#sec-impl)
- [13. Open research questions](#sec-open)
- [14. Conclusion](#sec-conclusion)
- [15. Glossary of symbols](#sec-glossary)
- [16. References](#sec-refs)

<h2 id="sec-intro">1. Introduction</h2>

Reinforcement learning is a branch of machine learning in which an agent learns to make sequential decisions by interacting with an environment, with the goal of maximizing cumulative reward. Unlike supervised learning, there is no dataset of correct answers: the agent must **explore** to discover which actions pay off and **exploit** what it has learned.

The successes are well known. AlphaGo defeated the world champion at Go [[14]](#ref-14), and AlphaZero mastered Go, chess and shogi through self-play [[15]](#ref-15). Deep Q-Networks reached human-level control from raw pixels [[6]](#ref-6). In robotics, RL has produced dexterous manipulation and locomotion policies [[10]](#ref-10). And in language modeling, RL now sits at the center of how models are aligned and taught to reason [[3]](#ref-3), [[16]](#ref-16), [[17]](#ref-17).

This article argues that **policy optimization** is a unifying mathematical framework behind these results. The same score-function estimator that lets a robot learn to grasp lets a language model learn to solve mathematical problems. The same advantage function that reduces variance in continuous control reduces it in token-level credit assignment. The same KL-regularized objective appears, with small modifications, in TRPO, PPO-based RLHF, DPO and GRPO.

But unification should not be confused with identity. RLHF is not just PPO applied to text. DPO is not just RLHF without a reward model. GRPO is not simply PPO without a critic. Each method makes distinct assumptions, optimizes a distinct objective and fails in distinct ways. Our goal is to derive each one from its foundations and to be explicit about what is guaranteed and what is merely observed.

**Roadmap.** Section 2 and 3 cover foundations and value-based methods. Section 4 is a detailed study of policy gradients (the theorem, REINFORCE, variance reduction, GAE, natural gradients). Section 5 derives importance sampling, TRPO and PPO. Section 6 briefly covers continuous control, model-based RL and embodied agents. Sections 7 to 9 move to language models: RLHF, DPO, RLVR and GRPO. Sections 10 to 14 compare, criticize and summarize.

**Notation.** $s$ is a state, $a$ an action, $r$ a reward, $\gamma\in[0,1)$ the discount factor, $\pi_\theta$ a policy with parameters $\theta$, and $\tau=(s_0,a_0,r_0,s_1,\dots)$ a trajectory. For language models, $x$ is a prompt and $y$ a completion. A glossary is given in [Section 15](#sec-glossary).

<h2 id="sec-foundations">2. Mathematical foundations</h2>

<h3 id="sec-loop">2.1 The agent-environment loop</h3>

At each time step the agent observes a state, chooses an action according to its policy, the environment transitions to a new state, and the agent receives a reward. The agent uses this feedback to improve its policy.

- **Agent:** the learner and decision-maker.
- **Environment:** everything the agent interacts with; it defines the transition dynamics.
- **State $s$:** a representation of the environment sufficient to make a decision.
- **Action $a$:** a decision taken by the agent.
- **Reward $r$:** a scalar feedback signal after each action.
- **Policy $\pi$:** a mapping from states to (a distribution over) actions. It can be a lookup table or a deep network, deterministic or stochastic.
- **Trajectory (episode, rollout):** the sequence $(s_0,a_0,r_0,s_1,a_1,r_1,\dots)$ of one interaction.
- **Replay buffer:** a store of past transitions reused for training (off-policy methods).

<h3 id="sec-mdp">2.2 Markov decision processes</h3>

RL problems are formalized as a **Markov decision process** (MDP) $\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)$ [[1]](#ref-1):

- $\mathcal S$ and $\mathcal A$ are the state and action spaces,
- $P(s'\mid s,a)$ is the transition probability,
- $R(s,a,s')$ is the reward function,
- $\gamma\in[0,1)$ is the discount factor.

The **Markov property** states that the next state and reward depend only on the current state and action. A policy $\pi(a\mid s)$ and the dynamics induce a distribution over trajectories:

$$
p_\theta(\tau)=\rho_0(s_0)\prod_{t=0}^{\infty}\pi_\theta(a_t\mid s_t)\,P(s_{t+1}\mid s_t,a_t),
$$

where $\rho_0$ is the initial-state distribution. This factorization, in which the policy terms and the dynamics terms separate, is the key to the policy gradient theorem (Section 4.3).

<h3 id="sec-values">2.3 Policies, returns, value functions and advantage</h3>

The **return** from time $t$ is the discounted sum of future rewards:

$$
G_t=\sum_{k=0}^{\infty}\gamma^k\,r_{t+k+1}.
$$

The discount $\gamma$ controls the horizon: small values make the agent myopic, values close to 1 encourage long-term planning. The agent maximizes the expected return

$$
J(\pi)=\mathbb E_{\tau\sim p_\pi}\Big[\sum_{t=0}^{\infty}\gamma^t r_t\Big].
$$

The **state-value function** and **action-value function** are

$$
V^\pi(s)=\mathbb E_\pi\big[G_t\mid s_t=s\big],\qquad Q^\pi(s,a)=\mathbb E_\pi\big[G_t\mid s_t=s,a_t=a\big],
$$

and they are related by $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$. The **advantage function**

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

measures how much better action $a$ is than the policy's average action in state $s$. By construction, $\mathbb E_{a\sim\pi}[A^\pi(s,a)]=0$. The advantage is the central quantity of modern policy gradient methods because it centers the learning signal and thus reduces variance.

<h3 id="sec-bellman">2.4 Bellman equations and contraction</h3>

Since $G_t=r_{t+1}+\gamma G_{t+1}$, taking conditional expectations gives the **Bellman expectation equations**:

$$
V^\pi(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V^\pi(s')\big],
$$

$$
Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)\Big[R(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')Q^\pi(s',a')\Big].
$$

The **Bellman optimality equations** characterize the optimal functions $V^*(s)=\max_\pi V^\pi(s)$ and $Q^*$:

$$
Q^*(s,a)=\sum_{s'}P(s'\mid s,a)\Big[R(s,a,s')+\gamma\max_{a'}Q^*(s',a')\Big],\qquad \pi^*(s)=\arg\max_aQ^*(s,a).
$$

**Why iterating Bellman updates converges.** Define the operator $(\mathcal T^\pi V)(s)=\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)[R+\gamma V(s')]$. For two value functions $V,V'$,

$$
\big|(\mathcal T^\pi V)(s)-(\mathcal T^\pi V')(s)\big|=\gamma\Big|\sum_a\pi(a\mid s)\sum_{s'}P(s'\mid s,a)\big(V(s')-V'(s')\big)\Big|\le\gamma\,\|V-V'\|_\infty,
$$

because the weights $\pi(a\mid s)P(s'\mid s,a)$ sum to one. Hence $\mathcal T^\pi$ is a $\gamma$-**contraction** in the sup norm. By the Banach fixed-point theorem it has a unique fixed point $V^\pi$, and repeated application converges to it geometrically. The same argument applies to the optimality operator [[46]](#ref-46). With function approximation this guarantee is lost, which is the source of many instabilities discussed later.

<h3 id="sec-dp">2.5 Dynamic programming</h3>

When $P$ and $R$ are known, dynamic programming (DP) [[46]](#ref-46), [[47]](#ref-47) solves the MDP exactly:

- **Policy evaluation** iterates $V_{k+1}=\mathcal T^\pi V_k$ until convergence to $V^\pi$.
- **Policy improvement** acts greedily: $\pi'(s)=\arg\max_a\sum_{s'}P(s'\mid s,a)[R+\gamma V^\pi(s')]$. The policy improvement theorem guarantees $V^{\pi'}\ge V^\pi$.
- **Policy iteration** alternates the two; **value iteration** applies the optimality operator directly:

$$
V_{k+1}(s)=\max_a\sum_{s'}P(s'\mid s,a)\big[R(s,a,s')+\gamma V_k(s')\big].
$$

DP is exact and converges for finite MDPs, but it needs the model and scales poorly (curse of dimensionality).

<h3 id="sec-model-free">2.6 Learning from experience: MC, TD, SARSA, Q-learning</h3>

**Monte Carlo (MC)** methods average observed returns, needing only complete episodes and no model:

$$
V(s)\approx\frac1{N(s)}\sum_{i=1}^{N(s)}G^{(i)}.
$$

They are unbiased but have high variance and cannot learn online within an episode.

**Temporal-difference (TD) learning** bootstraps: it updates an estimate toward another estimate, after a single step:

$$
V(s_t)\leftarrow V(s_t)+\alpha\,\delta_t,\qquad \delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t).
$$

The **TD error** $\delta_t$ is positive when the outcome was better than predicted. TD has lower variance than MC but is biased (it relies on the current estimate), and it works for continuing tasks.

**SARSA** [[4]](#ref-4) is on-policy TD control, learning from the actual next action:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big[r_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)\big].
$$

**Q-learning** [[5]](#ref-5) is off-policy: it learns about the greedy policy regardless of how data were collected:

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big[r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\big].
$$

In the tabular case Q-learning converges to $Q^*$ if all state-action pairs are visited infinitely often and the step sizes satisfy $\sum_t\alpha_t=\infty$, $\sum_t\alpha_t^2<\infty$.

<h3 id="sec-exploration">2.7 Exploration versus exploitation</h3>

An agent that only exploits what it knows can lock onto a suboptimal behavior; one that only explores wastes reward. Standard strategies:

- **$\epsilon$-greedy:** act greedily with probability $1-\epsilon$, randomly otherwise.
- **Softmax (Boltzmann):** $P(a\mid s)=\dfrac{e^{Q(s,a)/\tau}}{\sum_be^{Q(s,b)/\tau}}$, with temperature $\tau$.
- **Upper confidence bound (UCB):** $a_t=\arg\max_a\big[Q(a)+c\sqrt{\ln t/N(a)}\big]$, favoring uncertain actions.
- **Entropy regularization:** add $\beta\,\mathcal H(\pi(\cdot\mid s))$ to the objective, used in A3C [[31]](#ref-31) and SAC [[29]](#ref-29) (Section 6).

<h3 id="sec-onoff">2.8 On-policy vs. off-policy, model-based vs. model-free</h3>

Let $\pi$ be the **target policy** (being learned) and $\mu$ the **behavior policy** (generating data). On-policy learning has $\pi=\mu$; off-policy learning has $\pi\ne\mu$. Off-policy data requires correction through **importance sampling** (Section 5.1) [[43]](#ref-43).

| Property | On-policy | Off-policy |
|---|---|---|
| Data | Fresh samples from the current policy | Any past or exploratory data |
| Bias | Low | Potentially high (distribution mismatch) |
| Variance | Moderate | High (importance weights) |
| Sample efficiency | Low | High (replay buffers) |
| Stability | High | Needs care |
| Examples | SARSA, REINFORCE, A2C, TRPO, PPO | Q-learning, DQN, DDPG, TD3, SAC |

**Model-based** methods learn or use a model of $P$ and $R$ to plan or to generate synthetic experience (Section 6); **model-free** methods learn values or policies directly from experience tuples $(s,a,r,s')$. Model-free methods are simpler and more general but less sample-efficient.

<h2 id="sec-deep-value">3. Deep value-based methods</h2>

Tabular methods are infeasible with images or continuous states. **Deep Q-Networks** (DQN) [[6]](#ref-6) approximate $Q_\theta(s,a)$ with a neural network trained to minimize the TD error on transitions sampled from a replay buffer $\mathcal D$:

$$
\mathcal L(\theta)=\mathbb E_{(s,a,r,s')\sim\mathcal D}\Big[\big(r+\gamma\max_{a'}Q_{\bar\theta}(s',a')-Q_\theta(s,a)\big)^2\Big].
$$

Two ideas stabilize learning: **experience replay** breaks the correlation between consecutive samples, and the **target network** $Q_{\bar\theta}$ (updated slowly) prevents the regression target from moving at every step. Extensions include:

- **Double DQN** [[37]](#ref-37) reduces overestimation by decoupling selection and evaluation: $y=r+\gamma Q_{\bar\theta}\big(s',\arg\max_{a'}Q_\theta(s',a')\big)$.
- **Dueling networks** [[38]](#ref-38) write $Q(s,a)=V(s)+A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')$.
- **Prioritized replay** [[39]](#ref-39) samples with probability $P(i)\propto|\delta_i|^\alpha$, corrected by importance weights $w_i=(N\,P(i))^{-\beta}$.
- **Rainbow** [[40]](#ref-40) combines several improvements (multi-step returns, distributional values, noisy exploration).

Limitations: DQN needs a discrete action space (because of the $\max$), and it combines function approximation, bootstrapping and off-policy data, the "deadly triad", which can diverge. These limitations motivate policy-based methods, the subject of the next section.

<h2 id="sec-pg">4. Policy gradient methods</h2>

Value-based methods derive a policy indirectly (act greedily on $Q$). **Policy-based** methods optimize a parameterized policy $\pi_\theta(a\mid s)$ directly by gradient ascent on the expected return. They handle continuous and stochastic action spaces naturally, give smooth policy improvement (a small change in $\theta$ changes the action probabilities only slightly) and are the foundation of RLHF.

<h3 id="sec-pg-objective">4.1 Objective and policy parameterizations</h3>

The objective is

$$
J(\theta)=\mathbb E_{\tau\sim p_\theta}\big[R(\tau)\big],\qquad R(\tau)=\sum_{t=0}^{\infty}\gamma^tr_t.\tag{4.1}
$$

Two common parameterizations:

- **Softmax policy** (discrete actions): $\pi_\theta(a\mid s)=\dfrac{e^{h_\theta(s,a)}}{\sum_be^{h_\theta(s,b)}}$ with action preferences $h_\theta$ (linear $\theta^\top\phi(s,a)$ or a network). Then
$$
\nabla_\theta\log\pi_\theta(a\mid s)=\nabla_\theta h_\theta(s,a)-\sum_b\pi_\theta(b\mid s)\nabla_\theta h_\theta(s,b),
$$
which is "the features of the chosen action minus the policy-averaged features".
- **Gaussian policy** (continuous actions): $\pi_\theta(a\mid s)=\mathcal N\big(\mu_\theta(s),\sigma^2\big)$. Then $\nabla_\theta\log\pi_\theta(a\mid s)=\dfrac{a-\mu_\theta(s)}{\sigma^2}\nabla_\theta\mu_\theta(s)$. In practice $\sigma$ is also learned (usually through $\log\sigma$), and actions may be squashed with $\tanh$ to respect bounds.

For language models, $\pi_\theta(y_t\mid x,y_{<t})$ is the softmax over the vocabulary produced by the transformer.

<h3 id="sec-score">4.2 The log-derivative (score-function) trick</h3>

We want $\nabla_\theta J(\theta)=\nabla_\theta\int p_\theta(\tau)R(\tau)\,d\tau$. The difficulty is that $\theta$ appears inside the distribution we sample from, so we cannot simply push the gradient inside an expectation. The **log-derivative identity**

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)
$$

fixes this: it converts the gradient of an integral into an expectation we can estimate by sampling,

$$
\nabla_\theta J(\theta)=\int p_\theta(\tau)\,\nabla_\theta\log p_\theta(\tau)\,R(\tau)\,d\tau=\mathbb E_{\tau\sim p_\theta}\big[\nabla_\theta\log p_\theta(\tau)\,R(\tau)\big].\tag{4.2}
$$

A useful companion fact is that the expected score is zero (the **EGLP lemma**): for any distribution $p_\theta(x)$,

$$
\mathbb E_{x\sim p_\theta}\big[\nabla_\theta\log p_\theta(x)\big]=\int\nabla_\theta p_\theta(x)\,dx=\nabla_\theta\int p_\theta(x)\,dx=\nabla_\theta 1=0.\tag{4.3}
$$

This will be used twice: to drop past rewards and to add baselines.

<h3 id="sec-pgt">4.3 The policy gradient theorem, with proofs</h3>

**Step 1: the dynamics disappear.** Taking the logarithm of the trajectory distribution of [Section 2.2](#sec-mdp),

$$
\log p_\theta(\tau)=\log\rho_0(s_0)+\sum_{t=0}^{\infty}\Big[\log\pi_\theta(a_t\mid s_t)+\log P(s_{t+1}\mid s_t,a_t)\Big].
$$

The initial distribution and the transition probabilities do not depend on $\theta$, so

$$
\nabla_\theta\log p_\theta(\tau)=\sum_{t=0}^{\infty}\nabla_\theta\log\pi_\theta(a_t\mid s_t).
$$

Plugging this into (4.2):

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim p_\theta}\Big[\Big(\sum_{t=0}^{\infty}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\Big)R(\tau)\Big].\tag{4.4}
$$

This is remarkable: we can estimate the gradient of the return **without knowing or differentiating the environment dynamics**.

**Step 2: causality (reward-to-go).** Split $R(\tau)=\sum_{k}\gamma^kr_k$. Consider a term with a reward $r_k$ at time $k<t$. Since $r_k$ and $s_t$ are determined before $a_t$ is drawn, conditioning on the history $h_t=(s_0,a_0,\dots,s_t)$ gives

$$
\mathbb E\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,r_k\big]=\mathbb E\Big[r_k\;\mathbb E_{a_t\sim\pi_\theta(\cdot\mid s_t)}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big]\Big]=0
$$

by (4.3). Hence only rewards from time $t$ onward matter for the term at time $t$:

$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim p_\theta}\Big[\sum_{t=0}^{\infty}\nabla_\theta\log\pi_\theta(a_t\mid s_t)\sum_{k\ge t}\gamma^kr_k\Big]
=\mathbb E_{\tau\sim p_\theta}\Big[\sum_{t=0}^{\infty}\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,G_t\Big].\tag{4.5}
$$

**Step 3: from return to action-value.** By the tower property, $\mathbb E[G_t\mid s_t,a_t]=Q^{\pi_\theta}(s_t,a_t)$, and $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ is a function of $(s_t,a_t)$ only. Therefore:

**Policy Gradient Theorem** [[2]](#ref-2).
$$
\nabla_\theta J(\theta)=\mathbb E_{\tau\sim p_\theta}\Big[\sum_{t=0}^{\infty}\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,Q^{\pi_\theta}(s_t,a_t)\Big].\tag{4.6}
$$

Intuition: the update increases the log-probability of an action in proportion to how good that action turned out to be, and decreases it when the action was worse than expected.

*Remark (the $\gamma^t$ factor).* The exact gradient of the discounted objective carries the weight $\gamma^t$, which down-weights decisions taken late. Almost all practical implementations drop it, effectively optimizing a different (undiscounted-state-distribution) objective. This is a known, deliberate approximation and it matters little in practice.

**Alternative proof (recursion on value functions) and the state-visitation form** [[2]](#ref-2). Write $V^\pi(s)=\sum_a\pi(a\mid s)Q^\pi(s,a)$ and differentiate with the product rule:

$$
\nabla_\theta V^\pi(s)=\sum_a\Big[\nabla_\theta\pi(a\mid s)\,Q^\pi(s,a)+\pi(a\mid s)\,\nabla_\theta Q^\pi(s,a)\Big].
$$

Since $Q^\pi(s,a)=\sum_{s'}P(s'\mid s,a)[R+\gamma V^\pi(s')]$ and $P,R$ do not depend on $\theta$, we have $\nabla_\theta Q^\pi(s,a)=\gamma\sum_{s'}P(s'\mid s,a)\nabla_\theta V^\pi(s')$. This gives a recursion for $\nabla_\theta V^\pi(s)$ in terms of $\nabla_\theta V^\pi(s')$. Unrolling it forward in time and evaluating at the start state yields

$$
\nabla_\theta J(\theta)=\sum_s d^\pi(s)\sum_a Q^\pi(s,a)\,\nabla_\theta\pi_\theta(a\mid s),\qquad d^\pi(s)=\sum_{t=0}^{\infty}\gamma^t\Pr(s_t=s\mid\pi),
$$

with $d^\pi$ the (unnormalized) discounted state-visitation distribution. Using $\nabla\pi=\pi\nabla\log\pi$ recovers (4.6). Both proofs show the same essential fact: no derivative of the state distribution $d^\pi$ appears, even though it depends on $\theta$.

<h3 id="sec-reinforce">4.4 REINFORCE</h3>

REINFORCE [[7]](#ref-7) is the Monte Carlo instantiation of (4.5): sample trajectories with the current policy, compute the returns, and ascend the gradient estimate.

$$
\hat g=\frac1N\sum_{i=1}^{N}\sum_{t=0}^{T-1}\nabla_\theta\log\pi_\theta\big(a_t^{(i)}\mid s_t^{(i)}\big)\,G_t^{(i)},\qquad \theta\leftarrow\theta+\alpha\,\hat g.
$$

```
for iteration = 1, 2, ... do
    sample N trajectories with pi_theta
    for each trajectory and each t: compute G_t = sum_{k>=t} gamma^(k-t) r_k
    g_hat = (1/N) sum_i sum_t grad log pi(a_t|s_t) * G_t
    theta <- theta + alpha * g_hat
end for
```

**Why it works.** $\hat g$ is an unbiased estimator of $\nabla_\theta J$ (by (4.5)). It is a *return-weighted maximum likelihood* update: for each visited $(s_t,a_t)$ it takes a supervised-learning step toward repeating $a_t$, with weight $G_t$. Good trajectories are imitated more, bad ones less (or avoided when $G_t$ is negative).

<h3 id="sec-reinforce-problems">4.5 Problems of REINFORCE</h3>

1. **High variance.** $\hat g$ sums many noisy terms, and the return $G_t$ aggregates the randomness of all future actions and transitions. Variance grows with the horizon, so many trajectories are needed per update.
2. **Weak signal without centering.** If all returns are positive, every sampled action is reinforced, even bad ones. *Example:* two actions with expected returns 100 and 101. With raw returns both actions are pushed up by about 100 units of signal, and the 1-unit difference that actually matters is buried in noise. Subtracting a baseline of about 100.5 turns the signal into $\pm0.5$.
3. **Credit assignment.** All actions in a trajectory share the same return, so a good action in a bad episode is penalized (and vice versa). Reward-to-go helps (it removes past rewards) but does not solve it.
4. **Sample inefficiency (on-policy).** After one gradient step the data are stale and must be discarded, because they were generated by an older policy.
5. **Step-size sensitivity.** A step of fixed size in parameter space can change the policy's behavior drastically in some regions and negligibly in others, and a single bad step can collapse performance (the data distribution changes with the policy). This is exactly what TRPO and PPO address (Section 5).
6. **Episodic requirement.** Full returns require episodes to terminate (or truncation with bootstrapping).
7. **Local optima.** Gradient ascent on a non-concave objective converges only to local optima.

<h3 id="sec-variance">4.6 Variance reduction: reward-to-go, baselines, advantages</h3>

**Reward-to-go.** Using $G_t$ instead of the full return $R(\tau)$ is exactly the causality step (4.5) and removes irrelevant terms with no bias.

**Baselines.** Subtract any function $b(s_t)$ that does not depend on the action:

$$
\nabla_\theta J(\theta)=\mathbb E\Big[\sum_t\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big].
$$

*Proof that this is unbiased.* For fixed $s_t$, by (4.3),
$\mathbb E_{a_t\sim\pi}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,b(s_t)\big]=b(s_t)\,\mathbb E_{a_t}\big[\nabla_\theta\log\pi_\theta(a_t\mid s_t)\big]=0.$ $\square$

For a scalar parameter, the variance-minimizing constant baseline is $b^*=\dfrac{\mathbb E[(\nabla\log\pi)^2G]}{\mathbb E[(\nabla\log\pi)^2]}$, a gradient-weighted average return. In practice one uses the state value $b(s)=V^\pi(s)$, which is close to optimal and easy to learn.

**Advantage.** Choosing $b=V^\pi$ replaces $Q^\pi$ (or $G_t$) with the advantage:

$$
\nabla_\theta J(\theta)=\mathbb E\Big[\sum_t\gamma^t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,A^\pi(s_t,a_t)\Big].\tag{4.7}
$$

The advantage is positive for better-than-average actions and negative for worse-than-average ones, which is precisely the signal we want. Since $V^\pi$ is unknown, we estimate it with a learned critic.

<h3 id="sec-actor-critic">4.7 Actor-critic methods</h3>

Actor-critic methods maintain two components: an **actor** $\pi_\theta$ (the policy) and a **critic** $V_w$ (or $Q_w$) that estimates values. The critic is trained by regression on TD targets:

$$
L(w)=\mathbb E\Big[\big(r_t+\gamma V_w(s_{t+1})-V_w(s_t)\big)^2\Big],
$$

and the actor is updated with the critic's TD error as an advantage estimate, since $\mathbb E[\delta_t\mid s_t,a_t]=A^\pi(s_t,a_t)$ when $V_w=V^\pi$:

$$
\theta\leftarrow\theta+\alpha\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\delta_t,\qquad \delta_t=r_t+\gamma V_w(s_{t+1})-V_w(s_t).
$$

The critic lowers variance but introduces **bias** (its estimates are imperfect), the classic bias-variance trade-off. The idea goes back to the adaptive heuristic critic [[8]](#ref-8). Practical variants include **A3C** [[31]](#ref-31), which runs asynchronous workers sharing a global model, and its synchronous version **A2C**, which averages gradients over parallel environments before each update. Actor-critic can be seen as *generalized policy iteration*: the critic performs approximate policy evaluation and the actor performs approximate policy improvement.

<h3 id="sec-gae">4.8 Generalized advantage estimation (GAE)</h3>

The $k$-step advantage estimator is $\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}$, which equals $\sum_{l=0}^{k-1}\gamma^lr_{t+l}+\gamma^kV(s_{t+k})-V(s_t)$. Small $k$ means low variance and high bias; $k\to\infty$ gives the Monte Carlo advantage $G_t-V(s_t)$ (no bias from the critic, high variance). GAE [[11]](#ref-11) takes an exponentially weighted average over all $k$:

$$
\hat A_t^{\text{GAE}(\gamma,\lambda)}=(1-\lambda)\sum_{k=1}^{\infty}\lambda^{k-1}\hat A_t^{(k)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}.\tag{4.8}
$$

*Derivation.* Expanding $\hat A^{(k)}_t$ in TD errors and exchanging the order of summation, the coefficient of $\delta_{t+l}$ is $\gamma^l(1-\lambda)\sum_{k>l}\lambda^{k-1}=\gamma^l\lambda^l$. $\square$

It satisfies the backward recursion $\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}$, computed in a single backward pass over a rollout. The parameter $\lambda\in[0,1]$ interpolates: $\lambda=0$ gives the one-step TD error $\delta_t$ (low variance, high bias), $\lambda=1$ gives $G_t-V(s_t)$ (high variance, low bias). Typical values are $\gamma=0.99$, $\lambda=0.95$.

<h3 id="sec-npg">4.9 Natural policy gradient</h3>

Vanilla gradient ascent measures step size in **parameter space**, but what matters is the change in the **policy distribution**. The same $\|\Delta\theta\|$ can change the policy a lot or very little depending on the parameterization. The **natural policy gradient** [[12]](#ref-12), [[42]](#ref-42), [[48]](#ref-48) measures steps with the KL divergence. For small steps,

$$
D_{\mathrm{KL}}\big(\pi_\theta\,\|\,\pi_{\theta+\Delta}\big)\approx\tfrac12\,\Delta^\top F(\theta)\,\Delta,\qquad F(\theta)=\mathbb E\big[\nabla_\theta\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)^\top\big],
$$

where $F$ is the Fisher information matrix. Maximize the linearized objective under a KL budget:

$$
\max_\Delta\ g^\top\Delta\quad\text{s.t.}\quad\tfrac12\Delta^\top F\Delta\le\delta,\qquad g=\nabla_\theta J.
$$

With a Lagrange multiplier $\lambda$, stationarity gives $g=\lambda F\Delta$, i.e. $\Delta=\tfrac1\lambda F^{-1}g$. Enforcing the constraint with equality gives $\lambda=\sqrt{g^\top F^{-1}g/(2\delta)}$, so

$$
\Delta\theta=\sqrt{\frac{2\delta}{g^\top F^{-1}g}}\;F^{-1}g.\tag{4.9}
$$

The update direction $F^{-1}g$ is invariant to reparameterization of the policy. Inverting $F$ is too expensive for large networks, which motivates the approximations used in TRPO (next section).

<h2 id="sec-trust">5. Importance sampling, TRPO and PPO</h2>

<h3 id="sec-is">5.1 Importance sampling</h3>

Policy gradient estimates are on-policy: after every update the data are stale. **Importance sampling** (IS) lets us reuse data generated by another distribution. For distributions $p,q$ with $q(x)>0$ wherever $p(x)f(x)\ne0$,

$$
\mathbb E_{x\sim p}\big[f(x)\big]=\int p(x)f(x)\,dx=\int q(x)\frac{p(x)}{q(x)}f(x)\,dx=\mathbb E_{x\sim q}\Big[\frac{p(x)}{q(x)}f(x)\Big].
$$

The ratio $p/q$ re-weights samples from $q$ so that the estimator is unbiased for the expectation under $p$. Its variance is

$$
\mathrm{Var}_{x\sim q}\Big[\tfrac pqf\Big]=\mathbb E_{q}\Big[\big(\tfrac pq\big)^2f^2\Big]-\big(\mathbb E_p[f]\big)^2,
$$

which blows up when $p/q$ is large somewhere, i.e. when the two distributions differ.

**Application to trajectories.** For a target policy $\pi$ and behavior policy $\mu$, the ratio of trajectory probabilities is

$$
\frac{p_\pi(\tau)}{p_\mu(\tau)}=\prod_{t=0}^{T-1}\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}=\prod_{t}\rho_t,\qquad \rho_t=\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)},
$$

because the initial distribution and the transition probabilities cancel [[43]](#ref-43). Off-policy value estimates become $V^\pi(s_t)=\mathbb E_\mu[\prod_{k\ge t}\rho_k\,G_t]$. The product of ratios makes the variance **grow exponentially with the horizon**, so exact trajectory-level IS is unusable for long horizons. Practical remedies are *per-decision* IS (each reward only uses the ratios up to its own time), *truncation* of the ratios and *self-normalization*. PPO and TRPO take a different route: they **keep the policies close** so that ratios stay near 1, and they use only the single-step ratio $\rho_t$.

<h3 id="sec-pdl">5.2 The performance difference lemma and the surrogate objective</h3>

To justify the surrogate used in TRPO and PPO we need to relate the performance $J(\tilde\pi)$ of a new policy to that of the old one. Let $J(\pi)=\mathbb E_{s_0}[V^\pi(s_0)]$.

**Performance difference lemma** [[27]](#ref-27).
$$
J(\tilde\pi)=J(\pi)+\mathbb E_{\tau\sim p_{\tilde\pi}}\Big[\sum_{t=0}^{\infty}\gamma^tA^\pi(s_t,a_t)\Big].\tag{5.1}
$$

*Proof.* Since $A^\pi(s_t,a_t)=\mathbb E_{s_{t+1}}\big[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)\mid s_t,a_t\big]$, along a trajectory of $\tilde\pi$ the sum telescopes:

$$
\mathbb E_{\tilde\pi}\Big[\sum_{t}\gamma^t\big(r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)\big)\Big]=\mathbb E_{\tilde\pi}\Big[\sum_t\gamma^tr_t\Big]-\mathbb E[V^\pi(s_0)]=J(\tilde\pi)-J(\pi).\ \square
$$

Writing the expectation over states with the visitation distribution $d^{\tilde\pi}$ gives $J(\tilde\pi)=J(\pi)+\sum_sd^{\tilde\pi}(s)\sum_a\tilde\pi(a\mid s)A^\pi(s,a)$. This tells us that **any policy update that has non-negative expected advantage under the *new* state distribution improves the policy**. The problem is that $d^{\tilde\pi}$ is unknown before the update. The standard trick is to replace $d^{\tilde\pi}$ with the old $d^{\pi}$, which gives the **surrogate objective**:

$$
L_\pi(\tilde\pi)=J(\pi)+\sum_sd^{\pi}(s)\sum_a\tilde\pi(a\mid s)A^\pi(s,a)=J(\pi)+\mathbb E_{s\sim d^\pi,\,a\sim\pi}\Big[\frac{\tilde\pi(a\mid s)}{\pi(a\mid s)}A^\pi(s,a)\Big],\tag{5.2}
$$

where the last step is **importance sampling** from the old policy, exactly the single-step ratio of Section 5.1. This surrogate can be estimated from data collected with $\pi$. It matches the true objective to first order at $\tilde\pi=\pi$:

$$
L_\pi(\pi)=J(\pi),\qquad\nabla_{\tilde\theta}L_{\pi}(\pi_{\tilde\theta})\big|_{\tilde\theta=\theta}=\nabla_\theta J(\theta),
$$

(the second identity follows from $\nabla\tilde\pi/\pi=\nabla\log\pi$ and (4.7)). So the surrogate is a faithful local model of $J$, but it is accurate only while the new policy stays **close** to the old one. This is the entire motivation for trust regions.

<h3 id="sec-trpo">5.3 Trust Region Policy Optimization (TRPO)</h3>

**Theory: monotonic improvement.** Let $\varepsilon=\max_{s,a}|A^\pi(s,a)|$ and let $\alpha=\max_sD_{\mathrm{TV}}\big(\pi(\cdot\mid s),\tilde\pi(\cdot\mid s)\big)$ be the maximal total-variation distance. Schulman et al. [[13]](#ref-13) prove

$$
J(\tilde\pi)\ \ge\ L_\pi(\tilde\pi)-\frac{4\varepsilon\gamma}{(1-\gamma)^2}\,\alpha^2,
$$

and since $D_{\mathrm{TV}}^2\le D_{\mathrm{KL}}$,

$$
J(\tilde\pi)\ \ge\ L_\pi(\tilde\pi)-C\,D_{\mathrm{KL}}^{\max}(\pi,\tilde\pi),\qquad C=\frac{4\varepsilon\gamma}{(1-\gamma)^2}.\tag{5.3}
$$

Define $M_i(\pi)=L_{\pi_i}(\pi)-C\,D_{\mathrm{KL}}^{\max}(\pi_i,\pi)$. Then $J(\pi_{i+1})\ge M_i(\pi_{i+1})$ with equality $J(\pi_i)=M_i(\pi_i)$, so maximizing $M_i$ at each iteration guarantees $J(\pi_{i+1})-J(\pi_i)\ge M_i(\pi_{i+1})-M_i(\pi_i)\ge0$: **monotonic improvement** (a minorization-maximization scheme).

**Practice.** The penalty coefficient $C$ is very large, giving tiny steps, and the maximum KL over all states is impractical. TRPO therefore uses a **hard constraint** on the *average* KL:

$$
\max_\theta\ \mathbb E_{s,a\sim\pi_{\theta_{\text{old}}}}\Big[\frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\hat A(s,a)\Big]\quad\text{s.t.}\quad\mathbb E_s\big[D_{\mathrm{KL}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\big)\big]\le\delta.\tag{5.4}
$$

This is solved with the natural-gradient machinery of Section 4.9: linearize the objective around $\theta_{\text{old}}$ (its gradient is the policy gradient $g$) and approximate the KL to second order with the Fisher matrix $F$, giving the step (4.9), $\Delta\theta=\sqrt{2\delta/(g^\top F^{-1}g)}\,F^{-1}g$. The implementation details are:

1. Compute $g$ from a batch of on-policy data with advantages from GAE.
2. Solve $Fx=g$ with the **conjugate gradient** algorithm using only Fisher-vector products (no explicit $F$ or $F^{-1}$).
3. Take the full step $\Delta\theta$ and run a **backtracking line search** (shrink by a factor $\kappa^j$) until the true surrogate improves and the true KL constraint holds.

TRPO is theoretically principled and stable but complex: it needs second-order information, conjugate gradient, and it is awkward to combine with architectures that share parameters between actor and critic, or with noise such as dropout.

<h3 id="sec-ppo">5.4 Proximal Policy Optimization (PPO)</h3>

PPO [[9]](#ref-9) aims to retain TRPO's reliable improvement with a **first-order** method that can run several epochs of minibatch SGD on the same batch of data. Let

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\qquad \rho_t(\theta_{\text{old}})=1,
$$

the importance ratio of Section 5.2. The unconstrained surrogate $\mathbb E_t[\rho_t\hat A_t]$ would encourage arbitrarily large changes. PPO's **clipped objective** removes the incentive to move the ratio outside $[1-\epsilon,1+\epsilon]$:

$$
L^{\mathrm{CLIP}}(\theta)=\mathbb E_t\Big[\min\Big(\rho_t(\theta)\hat A_t,\ \operatorname{clip}\big(\rho_t(\theta),1-\epsilon,1+\epsilon\big)\hat A_t\Big)\Big].\tag{5.5}
$$

**Case analysis.** The $\min$ makes $L^{\mathrm{CLIP}}$ a *pessimistic lower bound* of the unclipped objective.

- **$\hat A_t>0$ (good action).** The objective is $\min(\rho_t,1+\epsilon)\hat A_t$. It increases with $\rho_t$ until $\rho_t=1+\epsilon$, then becomes flat: the gradient is zero, so there is no reward for making the action even more likely.
- **$\hat A_t<0$ (bad action).** The objective is $\max(\rho_t,1-\epsilon)\hat A_t$, which improves as $\rho_t$ decreases until $\rho_t=1-\epsilon$, then is flat: no further incentive to suppress the action.
- **Asymmetry of the min.** If the policy moved in the *wrong* direction (e.g. $\hat A_t>0$ but $\rho_t<1-\epsilon$), the $\min$ selects the unclipped term, so the gradient is kept and the mistake can be corrected. Clipping only removes gradients that would push *further in the right direction*.

**Gradient.** For an unclipped sample, $\nabla_\theta\big(\rho_t\hat A_t\big)=\rho_t\,\hat A_t\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$, i.e. the policy gradient of (4.7) re-weighted by the ratio. For a clipped sample the gradient is zero.

**Full loss.** In an actor-critic implementation with a shared network, PPO optimizes

$$
L(\theta)=\mathbb E_t\Big[L^{\mathrm{CLIP}}_t(\theta)-c_1\big(V_\theta(s_t)-V_t^{\text{targ}}\big)^2+c_2\,\mathcal H\big[\pi_\theta(\cdot\mid s_t)\big]\Big],\tag{5.6}
$$

with value-loss coefficient $c_1$, entropy bonus $c_2$ (to prevent premature determinism), and targets $V^{\text{targ}}_t=\hat A_t+V_{\theta_{\text{old}}}(s_t)$.

**PPO-Penalty (adaptive KL).** The paper's alternative variant maximizes $\mathbb E_t[\rho_t\hat A_t-\beta D_{\mathrm{KL}}]$ and adapts $\beta$ to a target KL $d_{\text{targ}}$: if the measured KL is much below the target (by a factor of about 1.5), halve $\beta$; if it is above, double it. The clipped version became the default as it is simpler and works as well in practice.

**Algorithm.**

```
for iteration = 1, 2, ... do
    run pi_theta_old in N parallel environments for T steps
    compute advantages A_t with GAE(gamma, lambda) and value targets V_targ
    for epoch = 1..K do
        for each minibatch of size M:
            maximize L(theta) (eq. 5.6) with Adam
    theta_old <- theta
end for
```

Typical settings are $\epsilon\approx0.1$–$0.3$, $K$ of a few epochs (3–10), $\gamma=0.99$ and $\lambda=0.95$ [[9]](#ref-9).

**What PPO does and does not guarantee.** Clipping is a heuristic: it removes the incentive to leave the region but does not strictly enforce it. After several epochs of SGD the ratios can drift outside $[1-\epsilon,1+\epsilon]$, and a common safeguard is to stop the epochs early when the measured KL exceeds a threshold. Engstrom et al. [[41]](#ref-41) show that many code-level details (advantage normalization, value clipping, learning-rate annealing, orthogonal initialization, gradient clipping, observation normalization) account for a large share of PPO's performance and of its apparent trust-region behavior, so reported gains should be interpreted with care.

**Comparison.**

| | REINFORCE | A2C | TRPO | PPO |
|---|---|---|---|---|
| Critic | No (or baseline) | Yes | Yes | Yes |
| Data reuse | 1 gradient step | 1 gradient step | 1 constrained step | Several epochs |
| Step control | None | None | KL constraint (2nd order) | Clipped ratio (1st order) |
| Guarantee | Unbiased gradient | Biased by critic | Approximate monotonic improvement | Heuristic |

<h2 id="sec-continuous">6. Continuous control, model-based RL and embodied agents</h2>

**Deterministic policy gradient.** For a deterministic policy $a=\mu_\theta(s)$, the gradient of the return is [[28]](#ref-28)

$$
\nabla_\theta J(\theta)=\mathbb E_{s\sim d^\mu}\Big[\nabla_\theta\mu_\theta(s)\,\nabla_aQ^\mu(s,a)\big|_{a=\mu_\theta(s)}\Big],
$$

i.e. move the action in the direction that increases the critic's value. **DDPG** [[10]](#ref-10) combines this with a replay buffer, target networks and exploration noise. **TD3** [[30]](#ref-30) mitigates overestimation with three changes: two critics with a clipped double-Q target $y=r+\gamma\min_{i=1,2}Q_{\bar w_i}\big(s',\mu_{\bar\theta}(s')+\epsilon\big)$, target-policy smoothing noise $\epsilon$, and delayed actor updates.

**Soft Actor-Critic (SAC)** [[29]](#ref-29) optimizes the maximum-entropy objective

$$
J(\pi)=\sum_t\mathbb E_{(s_t,a_t)\sim\pi}\big[r(s_t,a_t)+\alpha\,\mathcal H(\pi(\cdot\mid s_t))\big],
$$

with a soft Bellman target $y=r+\gamma\big(\min_iQ_{\bar w_i}(s',a')-\alpha\log\pi(a'\mid s')\big)$, $a'\sim\pi$, and an actor loss $\mathbb E[\alpha\log\pi(a\mid s)-\min_iQ_{w_i}(s,a)]$ trained through the reparameterization $a=\tanh(\mu_\theta(s)+\sigma_\theta(s)\odot\xi)$, $\xi\sim\mathcal N(0,I)$. SAC is off-policy, sample-efficient and robust, and is a default choice for continuous control.

**Model-based RL.** Instead of learning only from real interaction, the agent learns $\hat P_\phi(s'\mid s,a)$ and $\hat R_\phi(s,a)$ (explicit or in a latent space) and uses them to plan or to generate synthetic data, improving sample efficiency at the price of **model bias**: errors compound along long simulated rollouts and the policy can exploit model flaws.

- **Dyna** [[35]](#ref-35) mixes real and model-generated updates.
- **World Models** [[45]](#ref-45) and **Dreamer** [[32]](#ref-32) learn a latent dynamics model and train the policy on "imagined" trajectories.
- **MBPO** [[34]](#ref-34) uses short model rollouts branched from real states to limit error accumulation.
- **MuZero** [[33]](#ref-33) learns a latent model sufficient for Monte Carlo tree search (representation, dynamics and prediction networks), without access to the game rules.

**Embodied AI and robot learning.** The same policy gradient foundations apply. Robots need continuous action spaces, sample-efficient and safe learning, and bridging the **sim-to-real gap** (e.g. through domain randomization). Behavior cloning is often used to initialize a policy that RL then fine-tunes. **Agentic RL** for tool-using language agents adds partial observability, structured action spaces (tool calls) and long-horizon credit assignment, and is an active frontier.

<h2 id="sec-llm">7. RL for large language models: RLHF</h2>

<h3 id="sec-llm-mdp">7.1 Language generation as an MDP</h3>

An autoregressive language model is a policy $\pi_\theta(y\mid x)=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})$ that generates a completion $y$ for a prompt $x$. The mapping to RL notation is:

| RL | Language modeling |
|---|---|
| State $s_t$ | Prompt plus prefix $(x,y_{<t})$ |
| Action $a_t$ | Next token $y_t$ |
| Policy | Next-token distribution over the vocabulary |
| Transition | Deterministic: append the token |
| Trajectory | A full completion $y_{1:T}$ |
| Reward | $r(x,y)$ of the whole completion (outcome) or per step (process) |

The reward of a completion is the sum of step-wise rewards, $r(x,y_{1:T})\equiv\sum_tr(s_t,a_t)$, which covers outcome-based (terminal-only) and process-based (step-wise) rewards. From here on we use the $x,y$ notation. The RL objective is

$$
\max_\theta\ \mathbb E_{x\sim\mathcal D,\ y\sim\pi_\theta(\cdot\mid x)}\big[r(x,y)\big].
$$

<h3 id="sec-roles">7.2 The four models: policy, reference, reward, value</h3>

RLHF as popularized by InstructGPT [[3]](#ref-3) (building on [[22]](#ref-22) and [[36]](#ref-36)) is a three-phase pipeline: **(1) supervised fine-tuning (SFT)** on human demonstrations; **(2) reward modeling** on human comparisons; **(3) RL optimization** against the reward model. Four models interact:

- **Policy $\pi_\theta$:** the LLM being optimized (a causal transformer), initialized from the SFT model. It generates responses and is updated by PPO.
- **Reference $\pi_{\text{ref}}$:** a frozen copy of the SFT model, used to penalize divergence: $\beta\,D_{\mathrm{KL}}(\pi_\theta\|\pi_{\text{ref}})$. It keeps the policy fluent, truthful and anchored to its prior, and mitigates reward hacking. It has the same architecture and vocabulary as the policy, so token probabilities are directly comparable.
- **Reward model $r_\phi$:** maps $(x,y)$ to a scalar; trained offline on preferences, then frozen.
- **Value model $V_\psi$:** the critic; estimates expected return for advantage computation; trained online with the policy.

The KL-regularized objective is

$$
\max_\theta\ \mathbb E_{x\sim\mathcal D,\ y\sim\pi_\theta}\Big[r_\phi(x,y)-\beta\,D_{\mathrm{KL}}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big].\tag{7.1}
$$

<h3 id="sec-rm">7.3 The reward model</h3>

**Architecture.** In InstructGPT the reward model is initialized from the SFT model with the language-modeling head (the unembedding projection to vocabulary logits) removed and replaced by a linear head producing a scalar from the final hidden state, $r_\phi(x,y)=w^\top h_T+b$. A 6B-parameter reward model was used to train policies up to 175B parameters [[3]](#ref-3).

**Preference model (Bradley-Terry).** Human comparisons are modeled by the Bradley-Terry model [[18]](#ref-18):

$$
P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)=\frac{e^{r_\phi(x,y_w)}}{e^{r_\phi(x,y_w)}+e^{r_\phi(x,y_l)}}.
$$

Only reward **differences** matter: adding a constant to $r_\phi$ leaves the probabilities unchanged, so the reward is identified only up to a shift. If the scores are equal the probability is $0.5$; if $r_\phi(x,y_w)\gg r_\phi(x,y_l)$ it tends to 1.

**Loss.** Maximum likelihood of the observed preferences gives the pairwise loss

$$
L_{\text{RM}}(\phi)=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big].\tag{7.2}
$$

**Data and batching.** Labelers rank $K$ completions per prompt ($K$ between 4 and 9 in InstructGPT), producing $\binom K2$ comparisons. Treating these as independent shuffled datapoints caused severe overfitting, because each completion appears in $K-1$ pairs and is seen many times. InstructGPT instead treats all $\binom K2$ comparisons of a prompt as **one batch element**, so each completion needs a single forward pass, and training is both cheaper and better calibrated [[3]](#ref-3):

$$
L_{\text{RM}}(\phi)=-\frac1{\binom K2}\,\mathbb E_{(x,y_w,y_l)\sim\mathcal D}\Big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\Big].
$$

**Normalization.** Since the reward is only defined up to a shift, a bias is added before RL so that labeler demonstrations have mean score 0, giving rewards a consistent baseline. Partial responses receive no reward; a scalar is applied only when a complete response has been generated.

**Scale and quality.** InstructGPT's reward-model dataset contained on the order of tens of thousands of prompts [[3]](#ref-3), while larger systems use hundreds of thousands of comparisons or more, increasingly supplemented with synthetic AI feedback [[19]](#ref-19), [[20]](#ref-20). Reward models are evaluated by accuracy at predicting held-out human preferences, commonly reported in the range of roughly 65-80% depending on domain and data quality. They must be monitored for drift because the policy's output distribution changes during training.

<h3 id="sec-vm">7.4 The value model and advantage estimation</h3>

The value model (critic) estimates the expected cumulative reward from a state, i.e. the probability-weighted average return over all continuations the policy could generate:

$$
V_\psi(s_t)\approx\mathbb E_{y_{t+1:T}\sim\pi_\theta(\cdot\mid s_t)}\big[R(s_t)\big],\qquad s_t=(x,y_{1:t}).
$$

It is typically initialized from the trained reward model [[3]](#ref-3) (with a scalar head predicting values) and trained online by regression:

$$
L_V(\psi)=\mathbb E\Big[\big(V_\psi(s_t)-\hat R_t\big)^2\Big],
$$

where $\hat R_t$ is a Monte Carlo return, or a bootstrapped / GAE target in token-level variants. Each target is one sampled trajectory, but minimizing a mean-squared error over many samples drives $V_\psi$ toward the conditional expectation, since $\arg\min_f\mathbb E[(f(x)-R)^2\mid x]=\mathbb E[R\mid x]$ [[21]](#ref-21).

**Reward model versus value model.** They answer different questions and coincide only in degenerate cases:

- The reward model gives a **realized** reward for a specific completion: *what counts as good?*
- The value model gives an **expectation** over possible futures: *how good is the future expected to be from this prefix?*

At a terminal state, or in a pure bandit setting with a single terminal reward, the expectation collapses and $V^\pi(s_T)=r_\phi(x,y_{1:T})$; at intermediate prefixes $V^\pi(s_t)=\mathbb E[r_\phi(x,y_{1:T})\mid s_t]\neq r_\phi(x,y_{1:T})$. This separation is what lets PPO assign credit across tokens: the advantage compares what happened against what was expected given the prefix. In the bandit form used by InstructGPT (one prompt, one completion, one reward) it reduces to $\hat A=r_\phi(x,y)-V_\psi(x)$, which is still valuable as a variance-reducing baseline. With token-level values, the advantage follows GAE (4.8) with $\delta_t=r_t+\gamma V_\psi(s_{t+1})-V_\psi(s_t)$.

<h3 id="sec-rlhf-loop">7.5 PPO for RLHF: the full loop</h3>

**Token-level KL shaping.** The KL penalty of (7.1) is applied per token, so the effective reward at step $t$ is

$$
r_t=-\beta\log\frac{\pi_\theta(y_t\mid x,y_{<t})}{\pi_{\text{ref}}(y_t\mid x,y_{<t})}+\mathbb 1[t=T]\;r_\phi(x,y),
$$

and the per-token log-ratio is a sampled estimate of the KL divergence. Without this term, optimizing against a learned reward model leads to degenerate, reward-hacked outputs.

**PPO objective.** With ratio $\rho_t(\theta)=\pi_\theta(y_t\mid s_t)/\pi_{\theta_{\text{old}}}(y_t\mid s_t)$ and advantages $\hat A_t$ from GAE, the policy is trained with the clipped loss (5.5), the value model with $L_V$, as in (5.6).

**PPO-ptx.** InstructGPT also mixes pretraining gradients into the RL objective to limit performance regressions on standard NLP benchmarks (the "alignment tax") and forgetting [[3]](#ref-3):

$$
\max_\theta\ \mathbb E_{(x,y)\sim\pi_\theta}\Big[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}\Big]+\lambda_{\text{ptx}}\,\mathbb E_{x\sim\mathcal D_{\text{pretrain}}}\big[\log\pi_\theta(x)\big],
$$

(InstructGPT writes the coefficient as $\gamma$; we use $\lambda_{\text{ptx}}$ to avoid confusion with the discount factor).

**The training loop.**

1. **Sample:** draw prompts $x_i$ and generate completions $y_i\sim\pi_\theta$.
2. **Score:** compute $r_\phi(x_i,y_i)$ and the per-token KL against $\pi_{\text{ref}}$.
3. **Advantages:** estimate $V_\psi$ and compute $\hat A_t$ by GAE.
4. **Policy update:** optimize the PPO loss for a few epochs on the batch.
5. **Value update:** regress $V_\psi$ on the return targets.
6. **Repeat** with fresh samples; periodically evaluate with human or held-out preference metrics. Iterating this process with newly collected preferences is sometimes called iterative RLHF [[22]](#ref-22).

**Practical considerations.** Cost is dominated by sampling and reward scoring; generating many tokens per rollout dominates wall-clock time. In distributed implementations, all four models are sharded across accelerators, but only the policy and value models receive gradients. When memory is tight, the value head can share the policy backbone, at the risk of interference between actor and critic gradients. Parameter-efficient methods such as LoRA are sometimes used to reduce the number of updated parameters.

<h2 id="sec-dpo">8. DPO: Direct Preference Optimization</h2>

RLHF with PPO is complex: it needs a reward model, online sampling and many hyperparameters. DPO [[16]](#ref-16) removes both the explicit reward model and the RL loop by solving the KL-regularized problem in closed form.

**Step 1: the optimal policy.** For a fixed prompt, the objective in (7.1) can be rewritten as

$$
\mathbb E_{y\sim\pi}\big[r(x,y)\big]-\beta D_{\mathrm{KL}}(\pi\|\pi_{\text{ref}})=-\beta\,D_{\mathrm{KL}}\big(\pi\,\|\,\pi^*\big)+\beta\log Z(x),
$$

where

$$
\pi^*(y\mid x)=\frac{\pi_{\text{ref}}(y\mid x)\,\exp\big(r(x,y)/\beta\big)}{Z(x)},\qquad Z(x)=\sum_y\pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big).
$$

(To check: $-\beta\,\mathbb E_\pi\big[\log\frac{\pi}{\pi_{\text{ref}}}-\frac r\beta\big]=-\beta\,\mathbb E_\pi\big[\log\frac{\pi}{\pi^*Z}\big]$.) Since $D_{\mathrm{KL}}\ge0$ with equality iff $\pi=\pi^*$, the maximizer is $\pi^*$.

**Step 2: invert for the reward.** Solving for $r$:

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x).
$$

**Step 3: plug into Bradley-Terry.** In $r(x,y_w)-r(x,y_l)$ the prompt-dependent term $\beta\log Z(x)$ **cancels**, leaving

$$
P(y_w\succ y_l\mid x)=\sigma\Big(\beta\log\frac{\pi^*(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi^*(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big).
$$

**Step 4: the loss.** Replacing $\pi^*$ with $\pi_\theta$ and minimizing the negative log-likelihood of the observed preferences gives

$$
\mathcal L_{\text{DPO}}(\theta)=-\mathbb E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big].\tag{8.1}
$$

**Gradient.** With the implicit reward $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$,

$$
\nabla_\theta\mathcal L_{\text{DPO}}=-\beta\,\mathbb E\Big[\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)\big(\nabla_\theta\log\pi_\theta(y_w\mid x)-\nabla_\theta\log\pi_\theta(y_l\mid x)\big)\Big].
$$

It raises the likelihood of the preferred answer and lowers that of the rejected one, weighted by how **wrong** the implicit reward currently is: pairs already ranked correctly contribute little.

**Assumptions and limitations.** DPO assumes preferences follow the Bradley-Terry model and that the reference policy covers the support of good responses. The implicit reward is defined only up to a prompt-dependent constant. DPO is an **offline** method: it cannot explore beyond the preference data, and can over-optimize when the reference is weak or the data are noisy. Length and tokenization effects can create spurious correlations (longer answers may be preferred for reasons unrelated to quality).

| Dimension | RLHF (PPO) | DPO |
|---|---|---|
| Reward model | Learned explicitly | Implicit through the policy ratio |
| Sampling | Online, from the current policy | Offline, fixed preference data |
| Reference policy | KL penalty | Baseline of the implicit reward |
| Cost | High (reward model + RL) | Moderate (one training stage) |
| Stability | Sensitive to hyperparameters | More stable in practice |
| Exploration | Yes | No |

DPO is **not** "PPO without a critic": it is a different objective, derived from the closed-form solution of a KL-regularized problem, optimizing an implicit reward offline. Related objectives include **IPO** [[24]](#ref-24) (a more over-optimization-robust variant from a general preference framework), **KTO** [[25]](#ref-25) (prospect-theoretic losses that need only desirable/undesirable labels, not pairs) and **ORPO** [[26]](#ref-26) (combines SFT and preference alignment without a reference model).

<h2 id="sec-rlvr-grpo">9. RLVR and GRPO</h2>

<h3 id="sec-rlvr">9.1 RLVR: verifiable rewards</h3>

**Reinforcement Learning with Verifiable Rewards (RLVR)** replaces a learned reward model with a deterministic or programmatic verifier $R(x,y)=\mathrm{Verify}(x,y)$: whether a final mathematical answer matches the ground truth, whether unit tests pass, whether a proof checks. For a binary verifier $R\in\{0,1\}$,

$$
J(\theta)=\mathbb E_{x\sim\mathcal D}\,\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}\big[R(x,y)\big],\qquad
\nabla_\theta J=\mathbb E\Big[R(x,y)\sum_{t=1}^{T}\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t})\Big],
$$

the same score-function estimator as in Section 4, with a sequence-level reward. **RLVR is a reward paradigm, not an algorithm**: it can be combined with PPO, GRPO or other optimizers.

*Outcome vs. process.* Outcome supervision rewards only the final answer; **process supervision** rewards each reasoning step [[23]](#ref-23). Process reward models (PRMs) give denser credit assignment but need step-level labels, which are expensive; outcome reward models (ORMs) are cheaper but sparser.

*Challenges.* (i) **Sparse rewards:** no gradient signal on problems the policy always fails or always solves. (ii) **Exploration:** if the policy never produces a correct answer, nothing can be learned. (iii) **Verifier design:** answer-only checks may accept wrong reasoning or reject correct answers in a non-canonical format. (iv) **Reward hacking:** policies can exploit verifiers, e.g. by pattern memorization or formatting tricks. (v) **Generalization:** gains on verifiable domains do not automatically transfer to open-ended tasks. Verifiable rewards are low-noise and hard to hack when the verifier is correct, but they are narrow.

<h3 id="sec-grpo">9.2 GRPO: group-relative policy optimization</h3>

GRPO, introduced in DeepSeekMath [[17]](#ref-17), is a **critic-free** variant of PPO. Value functions for long generations are hard to train and their errors bias advantage estimates; GRPO uses the group of responses to the same prompt as the baseline.

For each prompt $x$, sample $G$ responses $\{y_1,\dots,y_G\}$ from $\pi_{\theta_{\text{old}}}$ with rewards $R_1,\dots,R_G$. Define

$$
\mu_R=\frac1G\sum_{i=1}^GR_i,\qquad\sigma_R=\sqrt{\frac1G\sum_{i=1}^G(R_i-\mu_R)^2},\qquad \hat A_i=\frac{R_i-\mu_R}{\sigma_R+\varepsilon}.
$$

With the token-level ratio $\rho_{i,t}(\theta)=\dfrac{\pi_\theta(y_{i,t}\mid x,y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t}\mid x,y_{i,<t})}$ the objective is

$$
\mathcal J_{\text{GRPO}}(\theta)=\mathbb E\Bigg[\frac1G\sum_{i=1}^G\frac1{|y_i|}\sum_{t=1}^{|y_i|}\Big(\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)\Big)-\beta\,D_{\mathrm{KL}}\big(\pi_\theta\|\pi_{\text{ref}}\big)\Bigg].\tag{9.1}
$$

The KL term is estimated per token with the non-negative estimator $\frac{\pi_{\text{ref}}}{\pi_\theta}-\log\frac{\pi_{\text{ref}}}{\pi_\theta}-1$ [[17]](#ref-17). Compared with PPO, there are two key differences: the advantage $\hat A_i$ is **shared by all tokens** of response $i$ (the reward is sequence-level), and there is **no critic**. The $1/|y_i|$ normalization and the division by $\sigma_R$ are design choices that change how long and short responses, or easy and hard prompts, are weighted; implementations differ here.

**Numerical example.** For $G=4$ rewards $(1,1,0,0)$: $\mu_R=0.5$, the deviations are $(0.5,0.5,-0.5,-0.5)$, the variance is $0.25$ so $\sigma_R=0.5$, and the advantages are $(1,1,-1,-1)$. If all rewards in a group are identical, $\sigma_R=0$ and every advantage is zero: the group gives no signal, a key limitation of GRPO on prompts that are too easy or too hard.

<h3 id="sec-other">9.3 Related ideas</h3>

- **RLAIF / Constitutional AI** [[19]](#ref-19) replaces human preference labels with AI-generated labels guided by written principles. It scales preference collection but inherits the annotator's biases.
- **Best-of-$N$ / rejection sampling** generates $N$ responses and keeps the best according to a verifier or reward model. It improves outputs at inference time without changing the policy, so it selects better samples from the existing distribution rather than improving the underlying capability.

<h2 id="sec-unified">10. A unified mathematical view</h2>

Across REINFORCE, PPO, RLHF, DPO, RLVR and GRPO several ingredients recur: a **parameterized policy**, an **expected-return objective**, the **score-function gradient** $\nabla_\theta\log\pi_\theta$, **baselines and advantages** for variance reduction, and **conservative updates** (clipping, KL regularization). What differs is the reward source, the sampling regime, the critic, and the reference model.

| Method | Reward source | Sampling | Critic | Reference | Update rule |
|---|---|---|---|---|---|
| REINFORCE | Environment | Online | No (baseline optional) | No | Return-weighted score |
| A2C / A3C | Environment | Online | Yes | No | Advantage-weighted score |
| TRPO | Environment | Online | Yes | Old policy (KL constraint) | Natural gradient + line search |
| PPO | Environment or learned RM | Online | Yes | Old policy (clipping) | Clipped surrogate |
| RLHF (PPO) | Learned RM | Online | Yes | Frozen SFT (KL) | Clipped surrogate + KL reward |
| DPO | Preferences | Offline | No | Implicit | Classification loss |
| GRPO | Verifier or RM | Online | No | Frozen (KL) | Clipped surrogate + group advantage |
| RLVR | Verifier | Depends | Depends | Depends | Reward paradigm |

Not all of these are extensions of one another. DPO is derived from the closed-form solution of a KL-regularized problem, not from the policy gradient theorem. GRPO modifies PPO's advantage estimation but retains the clipped surrogate. RLVR can be combined with various optimizers. The relationships are conceptual and mathematical, not hierarchical.

<h2 id="sec-limits">11. Practical and scientific limitations</h2>

- **Reward hacking:** the policy exploits flaws in the reward signal. Learned reward models are vulnerable to adversarial outputs that score high without being good; verifiers can be exploited too.
- **Reward-model over-optimization:** optimizing too hard against a learned reward degrades true quality, with a characteristic rise-then-fall of the true reward as the KL from the reference grows [[44]](#ref-44). The KL penalty mitigates but does not eliminate it.
- **Policy drift** reduces the reliability of reward estimates and can degrade general capabilities; KL regularization trades drift against reward.
- **Sparse or noisy rewards:** binary verifiers give no signal on unsolvable problems, and human preferences are noisy.
- **High variance:** especially for long reasoning trajectories; it requires large batches or variance-reduction techniques.
- **Exploration failure:** the model never finds a correct solution to a hard problem.
- **Data contamination** can inflate benchmark scores; a model that improves on one benchmark may not reason better in general. Human and adversarial evaluation are necessary complements.

These are not only engineering issues: the reward is a **proxy** for what we actually want, and optimizing a proxy too hard degrades the true objective.

<h2 id="sec-impl">12. Reproducible experiments and implementation intuition</h2>

**REINFORCE with reward-to-go and baseline:**

```
Input: policy pi_theta, baseline b(s), learning rate alpha
for episode = 1, 2, ... do
    sample trajectory tau with pi_theta
    compute returns G_t = sum_{k>=t} gamma^(k-t) r_k
    theta <- theta + alpha * sum_t grad log pi(a_t|s_t) * (G_t - b(s_t))
end for
```

**Actor-critic with GAE:**

```
Input: policy pi_theta, value V_phi
for iteration = 1, 2, ... do
    collect trajectories
    delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t = delta_t + gamma*lambda*A_{t+1}     (backward pass)
    theta <- theta + alpha * A_t * grad log pi(a_t|s_t)
    phi   <- minimize (V(s_t) - (A_t + V_old(s_t)))^2
end for
```

**PPO:** see the algorithm box in [Section 5.4](#sec-ppo).

**DPO training:**

```
Input: policy pi_theta, frozen pi_ref, preference data (x, y_w, y_l), beta
for batch in data do
    lw = log pi_theta(y_w|x) - log pi_ref(y_w|x)
    ll = log pi_theta(y_l|x) - log pi_ref(y_l|x)
    loss = -log sigmoid(beta * (lw - ll))
    theta <- gradient step on loss
end for
```

**GRPO with verifiable rewards:**

```
Input: policy pi_theta, reference pi_ref, prompts, verifier, group size G, beta
for iteration = 1, 2, ... do
    for prompt x in batch do
        sample G responses {y_i} from pi_theta_old
        R_i = Verify(x, y_i)
        A_i = (R_i - mean(R)) / (std(R) + eps)
    end for
    for epoch = 1..K do
        for each response: compute token ratios, clipped surrogate with A_i, KL to pi_ref
        gradient step on -J_GRPO
    end for
    theta_old <- theta
end for
```

**Toy example.** Prompt: "What is 17 times 23?" Four sampled answers: "391" (correct), "391" (correct), "381" and "401" (incorrect). Then $R=(1,1,0,0)$, $\hat A=(1,1,-1,-1)$: the policy is reinforced toward "391" and pushed away from "381" and "401". Whether "391" and "391 " (trailing space) both count as correct depends on how the verifier normalizes answers, which is why verifier design matters.

<h2 id="sec-open">13. Open research questions</h2>

1. How can policy gradient variance be reduced for long reasoning trajectories without introducing bias?
2. When does verifiable-reward training generalize beyond the verifier's domain?
3. How should process rewards be designed to balance credit assignment and labeling cost?
4. How can reward hacking be detected and mitigated in practice?
5. What is the relationship between exploration, inference-time search and learning?
6. Can RL improve tool use and embodied interaction robustly?
7. How should researchers evaluate generalization rather than benchmark memorization?
8. What are the trade-offs between offline preference optimization and online RL?

<h2 id="sec-conclusion">14. Conclusion</h2>

Policy optimization provides a mathematical language that connects classical reinforcement learning to modern language-model post-training. The log-derivative identity yields gradients of expected return without knowing the environment's dynamics (Section 4.3). Baselines and advantages reduce variance (Sections 4.6-4.8). Importance sampling and the performance difference lemma justify reusing data and motivate trust regions (Sections 5.1-5.2), which TRPO enforces with a KL constraint and PPO approximates with clipping (Sections 5.3-5.4). KL regularization keeps a policy close to a reference, and preference models or verifiers provide rewards when the environment does not (Sections 7-9).

The methods surveyed here are not a linear progression but different points in a design space defined by the reward source, the sampling regime, the gradient estimator and the constraint on policy updates. Understanding their derivations clarifies both their power and their limits. The same framework lets a robot learn to grasp and a language model learn to solve a differential equation, and the same failure mode, reward hacking, appears in both. The same open question also remains: how to specify what we actually want.

<h2 id="sec-glossary">15. Glossary of symbols</h2>

| Symbol | Meaning |
|---|---|
| $\mathcal M$ | Markov decision process |
| $\mathcal S,\mathcal A$ | State and action spaces |
| $P(s'\mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward function |
| $\gamma$ | Discount factor |
| $\pi_\theta(a\mid s)$ | Parameterized policy |
| $\pi_{\text{ref}}$ | Reference policy |
| $V^\pi(s),\,Q^\pi(s,a)$ | State-value and action-value functions |
| $A^\pi(s,a)$ | Advantage function |
| $G_t$ | Discounted return from time $t$ |
| $\tau$ | Trajectory |
| $J(\theta)$ | Expected return objective |
| $\delta_t$ | TD error |
| $\hat A_t^{\text{GAE}}$ | Generalized advantage estimate, parameter $\lambda$ |
| $\rho_t(\theta)$ | Importance (likelihood) ratio |
| $\epsilon$ | PPO clipping parameter |
| $F(\theta)$ | Fisher information matrix |
| $\beta$ | KL penalty coefficient |
| $r_\phi(x,y)$ | Reward model |
| $V_\psi$ | Value model (critic) |
| $\hat A_i$ | Group-relative advantage in GRPO |

<h2 id="sec-refs">16. References</h2>

<a id="ref-1"></a>[1] Sutton, R. S., & Barto, A. G. (1998). Reinforcement Learning: An Introduction. MIT Press.

<a id="ref-2"></a>[2] Sutton, R. S., McAllester, D. A., Singh, S. P., & Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. Advances in Neural Information Processing Systems, 12, 1057-1063.

<a id="ref-3"></a>[3] Ouyang, L., Wu, J., Jiang, X., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. Advances in Neural Information Processing Systems, 35.

<a id="ref-4"></a>[4] Rummery, G. A., & Niranjan, M. (1994). On-Line Q-Learning Using Connectionist Systems. Technical Report CUED/F-INFENG/TR 166, Cambridge University Engineering Department.

<a id="ref-5"></a>[5] Watkins, C. J. C. H., & Dayan, P. (1992). Q-Learning. Machine Learning, 8, 279-292.

<a id="ref-6"></a>[6] Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-Level Control Through Deep Reinforcement Learning. Nature, 518, 529-533.

<a id="ref-7"></a>[7] Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning, 8, 229-256.

<a id="ref-8"></a>[8] Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems. IEEE Transactions on Systems, Man, and Cybernetics, 13(5), 834-846.

<a id="ref-9"></a>[9] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.

<a id="ref-10"></a>[10] Lillicrap, T. P., Hunt, J. J., Pritzel, A., et al. (2015). Continuous Control with Deep Reinforcement Learning. arXiv:1509.02971.

<a id="ref-11"></a>[11] Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. International Conference on Learning Representations.

<a id="ref-12"></a>[12] Kakade, S. (2001). A Natural Policy Gradient. Advances in Neural Information Processing Systems, 14.

<a id="ref-13"></a>[13] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust Region Policy Optimization. Proceedings of the 32nd International Conference on Machine Learning, 1889-1897.

<a id="ref-14"></a>[14] Silver, D., Huang, A., Maddison, C. J., et al. (2016). Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 529, 484-489.

<a id="ref-15"></a>[15] Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the Game of Go Without Human Knowledge. Nature, 550, 354-359.

<a id="ref-16"></a>[16] Rafailov, R., Sharma, A., Mitchell, E., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Advances in Neural Information Processing Systems, 36.

<a id="ref-17"></a>[17] Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

<a id="ref-18"></a>[18] Bradley, R. A., & Terry, M. E. (1952). Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons. Biometrika, 39(3/4), 324-345.

<a id="ref-19"></a>[19] Bai, Y., Kadavath, S., Kundu, S., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073.

<a id="ref-20"></a>[20] Wang, Y., Kordi, Y., Mishra, S., et al. (2022). Self-Instruct: Aligning Language Models with Self-Generated Instructions. arXiv:2212.10560.

<a id="ref-21"></a>[21] Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. Advances in Neural Information Processing Systems, 12, 1008-1014.

<a id="ref-22"></a>[22] Christiano, P. F., Leike, J., Brown, T. B., et al. (2017). Deep Reinforcement Learning from Human Preferences. Advances in Neural Information Processing Systems, 30.

<a id="ref-23"></a>[23] Lightman, H., Kosaraju, V., Burda, Y., et al. (2024). Let's Verify Step by Step. International Conference on Learning Representations.

<a id="ref-24"></a>[24] Azar, M. G., Guo, Z. D., Piot, B., et al. (2024). A General Theoretical Paradigm to Understand Learning from Human Preferences. International Conference on Artificial Intelligence and Statistics, 4447-4455.

<a id="ref-25"></a>[25] Ethayarajh, K., Xu, W., Muennighoff, N., et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization. arXiv:2402.01306.

<a id="ref-26"></a>[26] Hong, J., Lee, N., & Thorne, J. (2024). ORPO: Monolithic Preference Optimization without Reference Model. Proceedings of EMNLP 2024, 11170-11189.

<a id="ref-27"></a>[27] Kakade, S., & Langford, J. (2002). Approximately Optimal Approximate Reinforcement Learning. Proceedings of the 19th International Conference on Machine Learning, 267-274.

<a id="ref-28"></a>[28] Silver, D., Lever, G., Heess, N., Degris, T., Wierstra, D., & Riedmiller, M. (2014). Deterministic Policy Gradient Algorithms. Proceedings of the 31st International Conference on Machine Learning.

<a id="ref-29"></a>[29] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Proceedings of the 35th International Conference on Machine Learning.

<a id="ref-30"></a>[30] Fujimoto, S., van Hoof, H., & Meger, D. (2018). Addressing Function Approximation Error in Actor-Critic Methods. Proceedings of the 35th International Conference on Machine Learning.

<a id="ref-31"></a>[31] Mnih, V., Badia, A. P., Mirza, M., et al. (2016). Asynchronous Methods for Deep Reinforcement Learning. Proceedings of the 33rd International Conference on Machine Learning.

<a id="ref-32"></a>[32] Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). Dream to Control: Learning Behaviors by Latent Imagination. International Conference on Learning Representations.

<a id="ref-33"></a>[33] Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature, 588, 604-609.

<a id="ref-34"></a>[34] Janner, M., Fu, J., Zhang, M., & Levine, S. (2019). When to Trust Your Model: Model-Based Policy Optimization. Advances in Neural Information Processing Systems, 32.

<a id="ref-35"></a>[35] Sutton, R. S. (1990). Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. Proceedings of the 7th International Conference on Machine Learning, 216-224.

<a id="ref-36"></a>[36] Stiennon, N., Ouyang, L., Wu, J., et al. (2020). Learning to Summarize from Human Feedback. Advances in Neural Information Processing Systems, 33.

<a id="ref-37"></a>[37] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence.

<a id="ref-38"></a>[38] Wang, Z., Schaul, T., Hessel, M., et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. Proceedings of the 33rd International Conference on Machine Learning.

<a id="ref-39"></a>[39] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized Experience Replay. International Conference on Learning Representations.

<a id="ref-40"></a>[40] Hessel, M., Modayil, J., van Hasselt, H., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. Proceedings of the AAAI Conference on Artificial Intelligence.

<a id="ref-41"></a>[41] Engstrom, L., Ilyas, A., Santurkar, S., et al. (2020). Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO. International Conference on Learning Representations.

<a id="ref-42"></a>[42] Amari, S. (1998). Natural Gradient Works Efficiently in Learning. Neural Computation, 10(2), 251-276.

<a id="ref-43"></a>[43] Precup, D., Sutton, R. S., & Singh, S. (2000). Eligibility Traces for Off-Policy Policy Evaluation. Proceedings of the 17th International Conference on Machine Learning, 759-766.

<a id="ref-44"></a>[44] Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. Proceedings of the 40th International Conference on Machine Learning.

<a id="ref-45"></a>[45] Ha, D., & Schmidhuber, J. (2018). World Models. arXiv:1803.10122.

<a id="ref-46"></a>[46] Bellman, R. (1957). Dynamic Programming. Princeton University Press.

<a id="ref-47"></a>[47] Howard, R. A. (1960). Dynamic Programming and Markov Processes. MIT Press.

<a id="ref-48"></a>[48] Peters, J., & Schaal, S. (2008). Natural Actor-Critic. Neurocomputing, 71(7-9), 1180-1190.