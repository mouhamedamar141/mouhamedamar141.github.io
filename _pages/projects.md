
---
layout: page
title: Projects
permalink: /projects/
description: Research and engineering projects in Reinforcement Learning, LLM Post-Training, Robot Learning, Embodied AI, and LLM Architectures.
nav: true
nav_order: 3
display_categories:
  - reinforcement-learning
  - llm-post-training
  - robot-learning
  - llm-architectures
  - ai-engineering
  - embodied-ai
horizontal: false
---

<div class="projects">
{% if site.enable_project_categories and page.display_categories %}

  {% for category in page.display_categories %}

    {% assign categorized_projects = site.projects | where: "category", category %}
    {% assign sorted_projects = categorized_projects | sort: "importance" %}

    {% if sorted_projects.size > 0 %}
      {% case category %}
        {% when 'reinforcement-learning' %}
          {% assign category_name = 'Reinforcement Learning' %}
        {% when 'llm-post-training' %}
          {% assign category_name = 'LLM Post-Training & Alignment' %}
        {% when 'robot-learning' %}
          {% assign category_name = 'Robot Learning & Imitation Learning' %}
        {% when 'llm-architectures' %}
          {% assign category_name = 'LLM Architectures & Optimization' %}
        {% when 'ai-engineering' %}
          {% assign category_name = 'AI Engineering & RAG Systems' %}
        {% when 'embodied-ai' %}
          {% assign category_name = 'Embodied AI & Intelligent Agents' %}
        {% else %}
          {% assign category_name = category %}
      {% endcase %}

      <h2 class="category" id="{{ category }}">{{ category_name }}</h2>

      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-1 row-cols-md-2">
            {% for project in sorted_projects %}
              {% include projects_horizontal.liquid %}
            {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="row row-cols-1 row-cols-md-3">
          {% for project in sorted_projects %}
            {% include projects.liquid %}
          {% endfor %}
        </div>
      {% endif %}
    {% endif %}

  {% endfor %}

{% else %}

  {% assign sorted_projects = site.projects | sort: "importance" %}

  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% unless project.title == "Object Detection with YOLO and R-CNN" %}
        {% include projects.liquid %}
      {% endunless %}
    {% endfor %}
  </div>

{% endif %}
</div>
