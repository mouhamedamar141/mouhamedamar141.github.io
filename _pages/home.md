---

title: "Home"
layout: homelay
permalink: /
------------

<h1 class="home-hero">{{ site.name }}</h1>

<p class="home-hero-sub">{{ site.title }}, {{ site.institution }}</p>

<div class="chip-container" markdown="0">

<a href="{{ '/research' | relative_url }}" class="chip">Reinforcement Learning</a>

<a href="{{ '/research' | relative_url }}" class="chip">LLM Post-Training</a>

<a href="{{ '/research' | relative_url }}" class="chip">Agentic AI</a>

<a href="{{ '/research' | relative_url }}" class="chip">Robot Learning</a>

<a href="{{ '/research' | relative_url }}" class="chip">Vision-Language-Action</a>

<a href="{{ '/research' | relative_url }}" class="chip">AI Agents</a>

</div>

I am an AI/ML Engineer and researcher interested in building intelligent systems that can learn, reason, interact with their environment, and improve through feedback.

My current interests lie at the intersection of <strong>Reinforcement Learning, LLM post-training, Agentic AI, and Robot Learning</strong>. I am particularly interested in how reinforcement learning can be used to improve reasoning, decision-making, tool use, and interaction in intelligent agents.

<div class="callout callout-success" markdown="0">

<div class="callout-title">{% include icon.html name="award" class="callout-icon" %} Research Interests</div>

<p>Reinforcement Learning · LLM Post-Training · RLHF · RLVR · GRPO · Agentic AI · Reasoning · Vision-Language-Action Models · Robot Learning</p>

</div>

<div class="banner-frame" markdown="0">

<img src="{{ '/images/banner.webp' | relative_url }}" alt="Artificial Intelligence and Reinforcement Learning" width="1400" height="449" loading="lazy">

<div class="banner-caption">Exploring learning, reasoning, and interaction in intelligent systems.</div>

</div>

{% capture selected %}{% bibliography --query @*[selected=true] %}{% endcapture %}

{% if selected contains "pub-entry" %}

## Selected publications

<div class="section-card selected-pubs" markdown="0">

{{ selected }}

<p style="margin: var(--space-4) 0 0;"><a href="{{ '/publications' | relative_url }}">All publications &rarr;</a></p>

</div>

{% endif %}

## About me

I recently completed my Master's degree in <strong>Artificial Intelligence & Big Data</strong> at the École Supérieure Polytechnique de Dakar (ESP/UCAD), Senegal.

My background covers <strong>Machine Learning, Deep Learning, Natural Language Processing, Generative AI, and MLOps</strong>, with hands-on experience building and deploying intelligent systems.

I am currently deepening my research skills through the study and implementation of recent research papers, with a strong focus on <strong>Reinforcement Learning and post-training of language models</strong>.

My long-term research interests include developing intelligent agents that can <strong>reason, use tools, interact with environments, and learn from experience</strong>, with applications to agentic AI and robotics.

I am particularly interested in pursuing research at the intersection of <strong>Reinforcement Learning, LLMs, Vision-Language-Action models, and Robot Learning</strong>.
