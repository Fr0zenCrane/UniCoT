# **BATI**: **BA**gel-based Unified Chain of Thought Across **T**ext and **Image** 

[[Project]](https://github.com/SAIS-FUXI/projects) / [[Report]](technical_report.md) / [[Model]](https://huggingface.co)

## Overview

**BATI** is a Unified Chain-of-Thought (UniCoT) reasoning framework designed to empower Multimodal Large Language Models (MLLMs) to perform complex reasoning across both text and vision. By decomposing multimodal tasks into interpretable, modular steps and executing them sequentially or in parallel, BATI target to enable unified multimodal reasoning for a wide range of applications, including:

* Complex visual planning and editing
* Geometric and physics-consistent reasoning
* Verification of image and video generation outcomes

<!-- ## Pipeline Summary -->
The BATI reasoning pipeline consists of the following stages:

1. **Planning**: Decompose the complex task into a sequence of simpler, manageable subtasks.
2. **Stepwise Execution**: Execute each subtask using the unified model with step-by-step reasoning.
3. **Self-Check**: After completing each subtask, perform a validation check to ensure the intermediate result aligns with the intended goal.
4. **Final Result**: Aggregate the validated intermediate results to produce the final output.

<p align="center">
  <img src="assets/pipeline.png" width="800"/>
</p>

## Acknowledgement

- This project is based on [Bagel](https://github.com/ByteDance-Seed/Bagel) proposed by ByteDance-Seed team. Bagel is a powerful and popular unified model for multimodal understanding and generation, making it an ideal foundation and startup for this project. We thank the ByteDance-Seed team for their outstanding work, which has made BATI possible.
